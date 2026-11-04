"""System 39: Self-hosted inference platform. GPU capacity from first principles, and the
monthly cost of five ways to serve the same tokens."""

import math

from _lib import DAYS_PER_MONTH, HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch39"

# Workloads at today's volume (1x). Tokens per day.
EXTRACT_IN, EXTRACT_OUT = 1.2e9, 60e6  # batch document extraction, latency in hours
ASSIST_IN, ASSIST_OUT = 300e6, 40e6  # in-product assistant, interactive
ASSIST_HOURS = 8  # business hours carrying the interactive load
PEAK_FACTOR = 2.0  # peak hour against the business-hour average
PREFIX_HIT = 0.5  # share of assistant input served from prefix cache
VOLUMES = {"today": 1, "next_year": 3, "at_10x": 10}

# Planning assumptions for gpt-oss-120b on one H100 (weights fit on a single GPU).
PREFILL_MFU = 0.25  # of BF16 peak: H100 has no FP4 arithmetic, MoE routing costs more
DECODE_BW_EFF = 0.7  # achieved fraction of HBM bandwidth
ASSIST_BATCH, ASSIST_CTX = 32, 3_000  # concurrent streams and mean context, interactive
EXTRACT_BATCH, EXTRACT_CTX = 128, 4_000  # same, batch pool
UTIL_TARGET = 0.6  # plan interactive capacity to run at 60% of its ceiling at peak
GPUS_PER_NODE = 8


def capacity(H):
    m, g = H("model.gpt-oss-120b"), H("gpu.H100")
    kv_tok = (
        2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2
    )  # bytes, BF16, upper bound
    prefill = PREFILL_MFU * g["bf16_tflops"] * 1e12 / (2 * m["active_b"] * 1e9)

    def decode(batch, ctx):
        # Each token picks 4 of 128 experts per layer. With uniform routing, a batch of B tokens
        # touches 1 - (1 - 4/128)^B of the experts, so a step reads the shared weights, that
        # share of the expert weights, and every sequence's KV cache. Real routing is skewed,
        # which touches fewer experts, so this is conservative.
        touched = 1 - (1 - m["experts_per_token"] / m["experts"]) ** batch
        share = (1 - m["expert_param_share"]) + m["expert_param_share"] * touched
        step = (m["weights_gb"] * 1e9 * share + batch * ctx * kv_tok) / (
            g["bandwidth_tb_s"] * 1e12 * DECODE_BW_EFF
        )
        return batch / step, 1 / step

    d_assist, per_stream = decode(ASSIST_BATCH, ASSIST_CTX)
    d_extract, _ = decode(EXTRACT_BATCH, EXTRACT_CTX)
    return dict(
        kv_kb=kv_tok / 1024,
        prefill=prefill,
        d_assist=d_assist,
        per_stream=per_stream,
        d_extract=d_extract,
    )


def gpus_needed(c, k):
    """Interactive GPUs at peak, and GPU-hours a day for everything."""
    peak_in = ASSIST_IN * k / (ASSIST_HOURS * 3600) * PEAK_FACTOR * (1 - PREFIX_HIT)
    peak_out = ASSIST_OUT * k / (ASSIST_HOURS * 3600) * PEAK_FACTOR
    peak_busy = peak_in / c["prefill"] + peak_out / c["d_assist"]
    interactive = max(2, math.ceil(peak_busy / UTIL_TARGET))
    assist_hours = (
        ASSIST_IN * k * (1 - PREFIX_HIT) / c["prefill"] + ASSIST_OUT * k / c["d_assist"]
    ) / 3600
    extract_hours = (
        EXTRACT_IN * k / c["prefill"] + EXTRACT_OUT * k / c["d_extract"]
    ) / 3600
    return peak_busy, interactive, assist_hours, extract_hours


def nodes_for(interactive, assist_hours, extract_hours):
    # Enough nodes for the interactive peak, and for all GPU-hours at 75% average utilisation.
    by_peak = math.ceil(interactive / GPUS_PER_NODE)
    by_hours = math.ceil((assist_hours + extract_hours) / (GPUS_PER_NODE * 24 * 0.75))
    return max(by_peak, by_hours)


def build(P, H):
    c = capacity(H)
    son = P("llm.bedrock.claude-sonnet-5-5")
    hai = P("llm.bedrock.claude-haiku-5-5")
    oss = P("hosted.fireworks.gpt-oss-120b")
    p5 = P("gpu.aws.p5-48xlarge")
    lam = P("gpu.lambda.h100-8x")
    rp = P("gpu.runpod.h100-sxm")
    eks = P("infra.aws.eks.cluster")["hourly"] * HOURS_PER_MONTH
    cpu = (
        P("infra.aws.ec2.c7i-2xlarge")["hourly"] * HOURS_PER_MONTH
    )  # gateway, queue workers
    s3 = P("infra.aws.s3.standard")["storage_gb_month_first_50tb"]

    v = {
        "kv_kb": f"{c['kv_kb']:.0f}",
        "prefill": num(c["prefill"]),
        "d_assist": num(c["d_assist"]),
        "per_stream": f"{c['per_stream']:.0f}",
        "d_extract": num(c["d_extract"]),
        "assist_batch": str(ASSIST_BATCH),
        "extract_batch": str(EXTRACT_BATCH),
        "p5_hour": f"${p5['hourly']:.2f}",
        "p5_sp_hour": f"${p5['ec2_instance_savings_plan_1yr_no_upfront']:.2f}",
        "lambda_gpu_hour": f"${lam['per_gpu_hourly']:.2f}",
        "runpod_gpu_hour": f"${rp['per_gpu_hourly']:.2f}",
        "p5_gpu_hour": f"${p5['per_gpu_hourly']:.2f}",
        "oss_in": f"${oss['input']:.2f}",
        "oss_out": f"${oss['output']:.2f}",
    }
    # Unit cost of a self-hosted token at full utilisation, for the break-even discussion.
    v["unit_in_p5"] = usd(p5["per_gpu_hourly"] / (c["prefill"] * 3600 / 1e6))
    v["unit_out_p5"] = usd(p5["per_gpu_hourly"] / (c["d_extract"] * 3600 / 1e6))
    v["unit_in_lambda"] = usd(lam["per_gpu_hourly"] / (c["prefill"] * 3600 / 1e6))
    v["unit_out_lambda"] = usd(lam["per_gpu_hourly"] / (c["d_extract"] * 3600 / 1e6))

    options = {
        "Claude Sonnet 5.5 on Bedrock (today)": {},
        "Claude Haiku 5.5 on Bedrock": {},
        "gpt-oss-120b, hosted API": {},
        "Self-hosted, AWS p5 on-demand": {},
        "Self-hosted, AWS p5 1-year savings plan": {},
        "Self-hosted, 8x H100 at a GPU cloud": {},
        "Self-hosted, per-GPU cloud, autoscaled": {},
    }
    nodes_row, gpus_row = {}, {}
    for vol, k in VOLUMES.items():
        api_days = DAYS_PER_MONTH
        # Today's system: extraction through batch, assistant with prompt caching.
        extract_api = llm_cost(son, EXTRACT_IN * k, EXTRACT_OUT * k, batch=True)
        assist_api = llm_cost(
            son,
            ASSIST_IN * k * (1 - PREFIX_HIT),
            ASSIST_OUT * k,
            cache_read=ASSIST_IN * k * PREFIX_HIT,
        )
        options["Claude Sonnet 5.5 on Bedrock (today)"][vol] = (
            extract_api + assist_api
        ) * api_days
        extract_h = llm_cost(hai, EXTRACT_IN * k, EXTRACT_OUT * k, batch=True)
        assist_h = llm_cost(
            hai, ASSIST_IN * k * (1 - PREFIX_HIT), ASSIST_OUT * k, cache_read=ASSIST_IN * k * PREFIX_HIT
        )
        options["Claude Haiku 5.5 on Bedrock"][vol] = (extract_h + assist_h) * api_days
        v[f"{vol}.haiku_api"] = usd(options["Claude Haiku 5.5 on Bedrock"][vol])
        extract_oss = llm_cost(oss, EXTRACT_IN * k, EXTRACT_OUT * k, batch=True)
        assist_oss = llm_cost(
            oss,
            ASSIST_IN * k * (1 - PREFIX_HIT),
            ASSIST_OUT * k,
            cache_read=ASSIST_IN * k * PREFIX_HIT,
        )
        options["gpt-oss-120b, hosted API"][vol] = (extract_oss + assist_oss) * api_days

        peak_busy, interactive, ah, eh = gpus_needed(c, k)
        n = nodes_for(interactive, ah, eh)
        nodes_row[vol], gpus_row[vol] = n, interactive
        overhead = eks + 2 * cpu + 200 * s3  # cluster, gateway nodes, weights and logs
        options["Self-hosted, AWS p5 on-demand"][vol] = (
            n * p5["hourly"] * HOURS_PER_MONTH + overhead
        )
        options["Self-hosted, AWS p5 1-year savings plan"][vol] = (
            n * p5["ec2_instance_savings_plan_1yr_no_upfront"] * HOURS_PER_MONTH
            + overhead
        )
        options["Self-hosted, 8x H100 at a GPU cloud"][vol] = (
            n * lam["hourly"] * HOURS_PER_MONTH + overhead
        )
        # Interactive GPUs for 10 hours, two GPUs overnight, extraction at 75% utilisation.
        daily = interactive * 10 + 2 * 14 + eh / 0.75
        options["Self-hosted, per-GPU cloud, autoscaled"][vol] = (
            daily * DAYS_PER_MONTH * rp["per_gpu_hourly"] + overhead
        )
        v[f"{vol}.self_rp"] = usd(
            options["Self-hosted, per-GPU cloud, autoscaled"][vol]
        )
        v[f"{vol}.peak_busy"] = f"{peak_busy:.1f}"
        v[f"{vol}.interactive"] = str(interactive)
        v[f"{vol}.assist_hours"] = f"{ah:.0f}"
        v[f"{vol}.extract_hours"] = f"{eh:.0f}"
        v[f"{vol}.nodes"] = str(n)
        v[f"{vol}.api_today"] = usd(
            options["Claude Sonnet 5.5 on Bedrock (today)"][vol]
        )
        v[f"{vol}.oss_api"] = usd(options["gpt-oss-120b, hosted API"][vol])
        v[f"{vol}.self_od"] = usd(options["Self-hosted, AWS p5 on-demand"][vol])
        v[f"{vol}.self_sp"] = usd(
            options["Self-hosted, AWS p5 1-year savings plan"][vol]
        )
        v[f"{vol}.self_gc"] = usd(options["Self-hosted, 8x H100 at a GPU cloud"][vol])
        v[f"{vol}.idle_share"] = pct(1 - (ah + eh) / (n * GPUS_PER_NODE * 24))

    t = Table("Option", "Today (1x)", "Next year (3x)", "At 10x", align="lrrr")
    for name, vals in options.items():
        t.add(name, *(usd(vals[k]) for k in VOLUMES))
    t.add("H100 nodes needed (self-hosted)", *(str(nodes_row[k]) for k in VOLUMES))
    tables = {"options": t.md()}

    t = Table("Quantity", "Value", "How", align="lrl")
    t.add(
        "KV cache per token",
        f"{c['kv_kb']:.0f} KB",
        "2 x 36 layers x 8 KV heads x 64 dims x 2 bytes",
    )
    t.add(
        "Prefill per H100",
        f"{num(c['prefill'])} tokens/s",
        f"{PREFILL_MFU:.0%} of 989 TFLOPS / (2 x 5.1B active params)",
    )
    t.add(
        f"Decode per H100, batch {ASSIST_BATCH}",
        f"{num(c['d_assist'])} tokens/s",
        f"{ASSIST_BATCH} streams, {ASSIST_CTX:,}-token contexts, {DECODE_BW_EFF:.0%} of 3.35 TB/s",
    )
    t.add(
        "Speed per interactive stream",
        f"{c['per_stream']:.0f} tokens/s",
        "one token per decode step",
    )
    t.add(
        f"Decode per H100, batch {EXTRACT_BATCH}",
        f"{num(c['d_extract'])} tokens/s",
        f"{EXTRACT_BATCH} streams, {EXTRACT_CTX:,}-token contexts",
    )
    tables["capacity"] = t.md()

    t = Table("", "Today (1x)", "Next year (3x)", "At 10x", align="lrrr")
    t.add("Interactive GPUs busy at peak", *(v[f"{k}.peak_busy"] for k in VOLUMES))
    t.add(
        "Interactive GPUs provisioned (60% target, minimum 2)",
        *(v[f"{k}.interactive"] for k in VOLUMES),
    )
    t.add("Assistant GPU-hours a day", *(v[f"{k}.assist_hours"] for k in VOLUMES))
    t.add("Extraction GPU-hours a day", *(v[f"{k}.extract_hours"] for k in VOLUMES))
    t.add("8x H100 nodes", *(v[f"{k}.nodes"] for k in VOLUMES))
    t.add("GPU time idle over a day", *(v[f"{k}.idle_share"] for k in VOLUMES))
    tables["sizing"] = t.md()
    return {"tables": tables, "vars": v}
