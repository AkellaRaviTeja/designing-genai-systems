"""System 39: Self-hosted inference platform.

Everything here is arithmetic on datasheet figures, config files and list prices, plus a few
inputs that the chapter labels as assumptions. Nothing was measured. Where a conclusion turns on
how close a GPU gets to its datasheet ceiling, the module reports the break-even share of that
ceiling instead of assuming one.
"""

import math

from _lib import DAYS_PER_MONTH, HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch39"
HOURS_PER_YEAR = 8760

# Worked-scenario inputs (illustrative). Tokens per day at today's volume.
EXTRACT_IN, EXTRACT_OUT = 1.2e9, 60e6  # document extraction, deadline in hours
ASSIST_IN, ASSIST_OUT = 300e6, 40e6  # in-product assistant, interactive
ASSIST_HOURS = 8  # business hours carrying the interactive load
PEAK_FACTOR = 2.0  # peak hour against the business-hour average
DOC_MIN, DOC_MAX = 4_000, 10_000  # extraction prompt length range
EXTRACT_CTX = (DOC_MIN + DOC_MAX) // 2  # assumed mean: the middle of the range
BURST_DOCS = 400  # documents one extraction job submits at once
ASSIST_CTX = 3_000  # mean assistant prompt, tokens
SYSTEM_TOKENS = 1_800  # shared assistant system prompt
TURNS = 4  # assumed turns per conversation
STREAM_TARGET = 40  # tokens/s per assistant stream
ASSIST_BATCH = 32  # streams per interactive replica
SPARES = 1  # interactive replicas held for one failure
RESTRICTED_SHARE = 0.4  # assumed share of tokens from the restricted customers
VOLUMES = {"today": 1, "next_year": 3, "at_10x": 10}

# Assumptions about the server. Each is labelled in the chapter.
MEM_UTIL = (
    0.95  # --gpu-memory-utilization the vLLM gpt-oss recipe suggests for one H100
)
KV_BUDGET_GB = 10  # assumed KV cache space once activations and CUDA graphs are taken
PLAN_EFF = 0.5  # assumed share of the datasheet ceiling, used only to count batch GPUs
SPEC_K = 3  # drafted tokens verified per step in the speculative-decoding arithmetic
S3_GBPS = (
    12.5  # GB/s: the p5.4xlarge's 100 Gbps network, a floor for the weight download
)
OVERHEAD_S3_GB = 200  # weights, adapters and logs kept in S3


def model_bytes(m):
    """Bytes in the checkpoint by role, from config.json shapes. Biases and norms are ignored."""
    attn = m["layers"] * (
        2 * m["hidden"] * m["attention_heads"] * m["head_dim"]
        + 2 * m["hidden"] * m["kv_heads"] * m["head_dim"]
    )
    table = m["vocab"] * m["hidden"]  # embedding and lm_head are separate matrices
    router = m["layers"] * m["hidden"] * m["experts"]
    bf16 = 2 * (attn + 2 * table + router)
    experts = m["checkpoint_bytes"] - bf16  # MXFP4 expert weights are the rest
    # A decode step reads attention, router and lm_head; the embedding is a lookup.
    dense_read = 2 * (attn + table + router)
    return dict(
        total=m["checkpoint_bytes"],
        experts=experts,
        dense_read=dense_read,
        per_expert=experts / (m["experts"] * m["layers"]),
    )


def kv_layer_token(m, dtype_bytes):
    return 2 * m["kv_heads"] * m["head_dim"] * dtype_bytes  # K and V


def kv_seq(m, ctx, dtype_bytes, hybrid=True):
    """KV bytes for one sequence. With a hybrid allocator the sliding-window layers keep only
    the window; without it every layer keeps the whole context."""
    per = kv_layer_token(m, dtype_bytes)
    if not hybrid:
        return m["layers"] * ctx * per
    full = m["layers"] - m["sliding_layers"]
    return full * ctx * per + m["sliding_layers"] * min(ctx, m["sliding_window"]) * per


def touched(m, tokens):
    """Expected share of experts a step touches when `tokens` tokens each pick 4 of 128,
    with uniform routing. Real routing is skewed and touches fewer."""
    return 1 - (1 - m["experts_per_token"] / m["experts"]) ** tokens


def step_bytes(m, b, tokens, kv_read):
    return b["dense_read"] + b["experts"] * touched(m, tokens) + kv_read


def prefill_flops(m, ctx):
    """FLOPs per prompt token: 2 per active parameter, plus attention over the causal context
    (full layers see half the context on average, sliding layers the window)."""
    width = m["attention_heads"] * m["head_dim"]
    full = m["layers"] - m["sliding_layers"]
    attn = 4 * width * (full * ctx / 2 + m["sliding_layers"] * m["sliding_window"])
    return 2 * m["active_b"] * 1e9 + attn


def build(P, H):
    m, g = H("model.gpt-oss-120b"), H("gpu.H100")
    bw = g["bandwidth_tb_s"] * 1e12
    b = model_bytes(m)
    son = P("llm.bedrock.claude-sonnet-5-5")
    hai = P("llm.bedrock.claude-haiku-5-5")
    oss = P("hosted.fireworks.gpt-oss-120b")
    p5 = P("gpu.aws.p5-48xlarge")
    p1 = P("gpu.aws.p5-4xlarge")
    lam = P("gpu.lambda.h100-8x")
    eks = P("infra.aws.eks.cluster")["hourly"] * HOURS_PER_MONTH
    cpu = P("infra.aws.ec2.c7i-2xlarge")["hourly"] * HOURS_PER_MONTH
    s3 = P("infra.aws.s3.standard")["storage_gb_month_first_50tb"]
    overhead = (
        eks + 2 * cpu + OVERHEAD_S3_GB * s3
    )  # cluster, gateway and worker nodes, S3
    sp1 = p1["ec2_instance_savings_plan_1yr_no_upfront"]
    v = {}

    # ---- Memory -------------------------------------------------------------------------
    kv_full_bf16 = m["layers"] * kv_layer_token(m, 2)
    kv_hyb_bf16 = (m["layers"] - m["sliding_layers"]) * kv_layer_token(m, 2)
    kv_hyb_fp8 = kv_hyb_bf16 / 2
    weights_gb = b["total"] / 1e9
    left_gb = g["memory_gb"] * MEM_UTIL - weights_gb
    budget = KV_BUDGET_GB * 1e9

    def max_batch(ctx, dtype, hybrid):
        return int(budget // kv_seq(m, ctx, dtype, hybrid))

    a_kv = ASSIST_BATCH * kv_seq(m, ASSIST_CTX, 1)
    mem = {
        "a_full_bf16": ASSIST_BATCH * kv_seq(m, ASSIST_CTX, 2, hybrid=False),
        "a_fp8": a_kv,
        "e_full_bf16_128": 128 * kv_seq(m, EXTRACT_CTX, 2, hybrid=False),
        "e_hyb_bf16_128": 128 * kv_seq(m, EXTRACT_CTX, 2),
        "e_fp8_128": 128 * kv_seq(m, EXTRACT_CTX, 1),
    }
    eb_full = max_batch(EXTRACT_CTX, 2, False)
    eb_bf16 = max_batch(EXTRACT_CTX, 2, True)
    eb_fp8 = max_batch(EXTRACT_CTX, 1, True)
    eb_fp8_range = (max_batch(DOC_MAX, 1, True), max_batch(DOC_MIN, 1, True))
    prefix_room = (budget - a_kv) / kv_hyb_fp8
    v.update(
        {
            "weights_gb": f"{weights_gb:.1f}",
            "weights_gib": f"{b['total'] / 2**30:.1f}",
            "experts_gb": f"{b['experts'] / 1e9:.0f}",
            "dense_read_gb": f"{b['dense_read'] / 1e9:.1f}",
            "per_expert_mb": f"{b['per_expert'] / 1e6:.1f}",
            "mem_util": pct(MEM_UTIL),
            "mem_left_gb": f"{left_gb:.1f}",
            "kv_budget_gb": str(KV_BUDGET_GB),
            "runtime_gb": f"{g['memory_gb'] - weights_gb - KV_BUDGET_GB:.0f}",
            "kv_kb": f"{kv_full_bf16 / 1024:.0f}",
            "kv_hyb_kb": f"{kv_hyb_bf16 / 1024:.0f}",
            "kv_fp8_kb": f"{kv_hyb_fp8 / 1024:.0f}",
            "extract_ctx": f"{EXTRACT_CTX:,}",
            "eb_full": str(eb_full),
            "eb_bf16": str(eb_bf16),
            "eb_fp8": str(eb_fp8),
            "eb_fp8_lo": str(eb_fp8_range[0]),
            "eb_fp8_hi": str(eb_fp8_range[1]),
            "prefix_room": num(prefix_room),
            "assist_batch": str(ASSIST_BATCH),
        }
    )
    for k, x in mem.items():
        v[f"mem.{k}"] = f"{x / 1e9:.1f}"

    # ---- Ceilings: datasheet figures, no efficiency factor ---------------------------------
    def decode(batch, ctx, dtype=1, spec=0):
        kv = batch * kv_seq(m, ctx, dtype)
        step = step_bytes(m, b, batch * (1 + spec), kv) / bw
        return batch / step, 1 / step, step

    pf_extract = g["bf16_tflops"] * 1e12 / prefill_flops(m, EXTRACT_CTX)
    pf_assist = g["bf16_tflops"] * 1e12 / prefill_flops(m, ASSIST_CTX)
    d_assist, per_stream, step_a = decode(ASSIST_BATCH, ASSIST_CTX)
    d_extract, _, step_e = decode(eb_fp8, EXTRACT_CTX)
    d_extract_bf16, _, _ = decode(eb_bf16, EXTRACT_CTX, dtype=2)
    d_extract_fp8_same, _, _ = decode(eb_bf16, EXTRACT_CTX, dtype=1)
    v.update(
        {
            "pf_flops": f"{prefill_flops(m, EXTRACT_CTX) / 1e9:.1f}",
            "pf_extract": num(pf_extract),
            "pf_assist": num(pf_assist),
            "d_assist": num(d_assist),
            "per_stream": f"{per_stream:.0f}",
            "step_a_ms": f"{step_a * 1e3:.1f}",
            "d_extract": num(d_extract),
            "step_e_ms": f"{step_e * 1e3:.1f}",
            "d_extract_bf16": num(d_extract_bf16),
            "fp8_gain_batch": f"{d_extract / d_extract_bf16:.1f}",
            "fp8_gain_same": f"{d_extract_fp8_same / d_extract_bf16:.2f}",
            "touched_32": pct(touched(m, ASSIST_BATCH)),
            "stream_target": str(STREAM_TARGET),
            "itl_target_ms": f"{1000 / STREAM_TARGET:.0f}",
        }
    )

    # Batch size against the per-stream target: the bandwidth share each batch needs.
    t = Table(
        "Streams per GPU",
        "Step at the ceiling",
        "Per stream at the ceiling",
        "GPU output at the ceiling",
        f"Bandwidth share needed for {STREAM_TARGET} tokens/s",
        align="rrrrr",
    )
    for batch in (16, 32, 64):
        agg, ps, st = decode(batch, ASSIST_CTX)
        t.add(
            str(batch),
            f"{st * 1e3:.1f} ms",
            f"{ps:.0f} tokens/s",
            f"{num(agg)} tokens/s",
            pct(STREAM_TARGET / ps),
        )
        v[f"bw_need_{batch}"] = pct(STREAM_TARGET / ps)
    tables = {"streams": t.md()}

    t = Table("Quantity", "Value", "How", align="lrl")
    t.add(
        "Weights in memory",
        f"{weights_gb:.1f} GB",
        "checkpoint size; Hopper keeps MXFP4 experts packed and multiplies in BF16",
    )
    t.add(
        "Read per decode step, dense part",
        f"{b['dense_read'] / 1e9:.1f} GB",
        "attention, router and lm_head in BF16",
    )
    t.add(
        "Expert weights",
        f"{b['experts'] / 1e9:.0f} GB",
        f"{m['experts']} experts x {m['layers']} layers, {b['per_expert'] / 1e6:.1f} MB each",
    )
    t.add(
        "Prefill ceiling per H100",
        f"{num(pf_extract)} tokens/s",
        f"989 TFLOPS BF16 / {prefill_flops(m, EXTRACT_CTX) / 1e9:.1f} GFLOP per token at {EXTRACT_CTX:,} tokens",
    )
    t.add(
        f"Decode ceiling, {ASSIST_BATCH} streams of {ASSIST_CTX:,}",
        f"{num(d_assist)} tokens/s",
        f"3.35 TB/s; {touched(m, ASSIST_BATCH):.0%} of experts touched per step",
    )
    t.add(
        f"Decode ceiling, {eb_fp8} streams of {EXTRACT_CTX:,}",
        f"{num(d_extract)} tokens/s",
        f"the batch an FP8 KV cache fits in {KV_BUDGET_GB} GB",
    )
    t.add(
        f"Decode ceiling, {eb_bf16} streams, BF16 KV",
        f"{num(d_extract_bf16)} tokens/s",
        f"the batch a BF16 KV cache fits in {KV_BUDGET_GB} GB",
    )
    tables["capacity"] = t.md()

    # ---- The burst ----------------------------------------------------------------------
    burst = BURST_DOCS * EXTRACT_CTX
    v["burst_tokens"] = num(burst)
    v["burst_docs"] = str(BURST_DOCS)
    v["burst_s"] = f"{burst / (8 * pf_extract):.0f}"
    v["burst_s_quarter"] = f"{burst / (8 * pf_extract * 0.25):.0f}"
    v["burst_s4"] = f"{burst / (4 * pf_extract):.0f}"
    v["burst_per_replica"] = num(burst / 8)

    # ---- Prefix cache: inputs, not observations --------------------------------------------
    hist = (ASSIST_CTX - SYSTEM_TOKENS) / ((TURNS + 1) / 2)  # tokens each turn adds
    hit_floor = SYSTEM_TOKENS / ASSIST_CTX
    hit_affinity = 1 - hist / ASSIST_CTX
    hit_rr8 = hit_floor + (hit_affinity - hit_floor) / 8
    v.update(
        {
            "turns": str(TURNS),
            "hist_tokens": f"{hist:.0f}",
            "hit_floor": pct(hit_floor),
            "hit_affinity": pct(hit_affinity),
            "hit_rr8": pct(hit_rr8),
            "uncached_floor": pct(1 - hit_floor),
            "uncached_affinity": pct(1 - hit_affinity),
        }
    )
    hit = (
        hit_floor  # the planning value: what any routing keeps if the prefix is stable
    )
    v["assist_prefill_hours"] = f"{ASSIST_IN * (1 - hit) / pf_assist / 3600:.1f}"

    # ---- Speculative decoding ---------------------------------------------------------------
    def accept_for(speedup, ratio, k=SPEC_K):
        need = speedup * ratio  # mean tokens per step needed
        lo, hi = 0.0, 1.0
        for _ in range(60):
            a = (lo + hi) / 2
            if sum(a**i for i in range(k + 1)) < need:
                lo = a
            else:
                hi = a
        return lo

    _, _, st_spec = decode(ASSIST_BATCH, ASSIST_CTX, spec=SPEC_K)
    ratio_a = st_spec / step_a
    _, _, st_spec_e = decode(eb_fp8, EXTRACT_CTX, spec=SPEC_K)
    ratio_e = st_spec_e / step_e
    v.update(
        {
            "spec_k": str(SPEC_K),
            "spec_tokens": str(ASSIST_BATCH * (SPEC_K + 1)),
            "spec_touched": pct(touched(m, ASSIST_BATCH * (SPEC_K + 1))),
            "spec_ratio": f"{ratio_a:.2f}",
            "spec_ratio_batch": f"{ratio_e:.2f}",
            "spec_accept_even": pct(accept_for(1.0, ratio_a)),
            "spec_accept_15": pct(accept_for(1.5, ratio_a)),
            "spec_flops_ms": f"{ASSIST_BATCH * (SPEC_K + 1) * prefill_flops(m, ASSIST_CTX) / (g['bf16_tflops'] * 1e12) * 1e3:.1f}",
        }
    )

    # ---- Hosted costs per day ---------------------------------------------------------------
    def hosted(p, k, share=1.0, cache=True):
        x = llm_cost(p, EXTRACT_IN * k * share, EXTRACT_OUT * k * share, batch=True)
        a_in = ASSIST_IN * k * share
        y = llm_cost(
            p,
            a_in * (1 - hit) if cache else a_in,
            ASSIST_OUT * k * share,
            cache_read=a_in * hit if cache else 0,
        )
        return x, y

    # ---- Break-even: what a GPU-hour must produce to match the hosted price ---------------------
    batch_in = oss["input"] * (1 - oss["batch_discount"])
    batch_out = oss["output"] * (1 - oss["batch_discount"])
    ex_h, as_h = hosted(oss, 1)
    ext_ceiling_h = (EXTRACT_IN / pf_extract + EXTRACT_OUT / d_extract) / 3600
    ast_ceiling_h = (ASSIST_IN * (1 - hit) / pf_assist + ASSIST_OUT / d_assist) / 3600
    rates = [
        ("od", "AWS p5.4xlarge, on-demand", p1["hourly"]),
        (
            "csp",
            "AWS p5.4xlarge, Compute Savings Plan",
            p1["compute_savings_plan_1yr_no_upfront"],
        ),
        ("sp", "AWS p5.4xlarge, EC2 Instance Savings Plan", sp1),
        ("lam", "Lambda 8x H100, per GPU (outside the boundary)", lam["per_gpu_hourly"]),
    ]
    t = Table(
        "GPU-hour",
        "Price",
        "Extraction input",
        "Extraction output",
        "Extraction mix",
        "Assistant mix",
        align="lrrrrr",
    )

    def need(price, per_m, ceiling):
        tok_s = price / per_m * 1e6 / 3600
        return tok_s, tok_s / ceiling

    for key, name, price in rates:
        i_s, i_f = need(price, batch_in, pf_extract)
        o_s, o_f = need(price, batch_out, d_extract)
        ext_f = ext_ceiling_h * price / ex_h
        ast_f = ast_ceiling_h * price / as_h
        t.add(
            name,
            f"${price:.2f}",
            f"{num(i_s)}/s ({pct(i_f)})",
            f"{num(o_s)}/s ({pct(o_f)})",
            pct(ext_f),
            pct(ast_f),
        )
        v[f"be.{key}.in_tok"] = num(i_s)
        v[f"be.{key}.in_pct"] = pct(i_f)
        v[f"be.{key}.out_tok"] = num(o_s)
        v[f"be.{key}.out_pct"] = pct(o_f)
        v[f"be.{key}.ext"] = pct(ext_f)
        v[f"be.{key}.blend"] = pct(
            (ext_ceiling_h + ast_ceiling_h) * price / (ex_h + as_h)
        )
        v[f"be.{key}.ast"] = pct(ast_f)
    tables["breakeven"] = t.md()
    v["batch_in"] = f"${batch_in:.3f}"
    v["batch_out"] = f"${batch_out:.2f}"
    v["oss_in"] = f"${oss['input']:.2f}"
    v["oss_out"] = f"${oss['output']:.2f}"
    v["ext_ceiling_h"] = f"{ext_ceiling_h:.1f}"
    v["ast_ceiling_h"] = f"{ast_ceiling_h:.1f}"
    v["stand_out_tok"] = num(sp1 / oss["output"] * 1e6 / 3600)
    v["stand_out_pct"] = pct(sp1 / oss["output"] * 1e6 / 3600 / d_assist)

    # ---- Sizing ----------------------------------------------------------------------------
    def fleet(k, share=1.0):
        peak_out = ASSIST_OUT * k * share / (ASSIST_HOURS * 3600) * PEAK_FACTOR
        streams = peak_out / STREAM_TARGET  # Little's law: rate / per-stream speed
        inter = math.ceil(streams / ASSIST_BATCH) + SPARES
        ext_h = ext_ceiling_h * k * share
        batch = max(1, math.ceil(ext_h / (24 * PLAN_EFF)))
        work_h = ext_h + ast_ceiling_h * k * share
        return streams, inter, ext_h, batch, work_h

    def self_cost(gpus, rate):
        return gpus * rate * HOURS_PER_MONTH + overhead

    options = {
        "Claude Sonnet 5.5 on Bedrock (today)": {},
        "Claude Haiku 5.5 on Bedrock, if it passed": {},
        "gpt-oss-120b, hosted API (not compliant)": {},
        "Self-hosted, p5.48xlarge nodes, on-demand": {},
        "Self-hosted, p5.48xlarge nodes, savings plan": {},
        "Self-hosted, p5.4xlarge, on-demand": {},
        "Self-hosted, p5.4xlarge, savings plan": {},
        f"Restricted {pct(RESTRICTED_SHARE)} self-hosted (p5.4xlarge, savings plan), rest on hosted API": {},
    }
    split = list(options)[-1]
    rows = {
        k: {}
        for k in (
            "streams",
            "inter",
            "ext_h",
            "batch",
            "gpus",
            "nodes",
            "busy",
            "busy_plan",
            "split_gpus",
        )
    }
    for vol, k in VOLUMES.items():
        d = DAYS_PER_MONTH
        options["Claude Sonnet 5.5 on Bedrock (today)"][vol] = sum(hosted(son, k)) * d
        options["Claude Haiku 5.5 on Bedrock, if it passed"][vol] = (
            sum(hosted(hai, k)) * d
        )
        api = sum(hosted(oss, k)) * d
        options["gpt-oss-120b, hosted API (not compliant)"][vol] = api
        streams, inter, ext_h, batch, work_h = fleet(k)
        gpus = inter + batch
        nodes = math.ceil(gpus / 8)
        options["Self-hosted, p5.48xlarge nodes, on-demand"][vol] = self_cost(
            nodes, p5["hourly"]
        )
        options["Self-hosted, p5.48xlarge nodes, savings plan"][vol] = self_cost(
            nodes, p5["ec2_instance_savings_plan_1yr_no_upfront"]
        )
        options["Self-hosted, p5.4xlarge, on-demand"][vol] = self_cost(
            gpus, p1["hourly"]
        )
        options["Self-hosted, p5.4xlarge, savings plan"][vol] = self_cost(gpus, sp1)
        s_streams, s_inter, _, s_batch, s_work = fleet(k, RESTRICTED_SHARE)
        v[f"{vol}.split_streams"] = f"{s_streams:.0f}"
        v[f"{vol}.split_inter"] = str(s_inter)
        v[f"{vol}.split_batch"] = str(s_batch)
        v[f"{vol}.split_busy"] = pct(s_work / ((s_inter + s_batch) * 24))
        s_gpus = s_inter + s_batch
        options[split][vol] = (
            self_cost(s_gpus, sp1) + sum(hosted(oss, k, 1 - RESTRICTED_SHARE)) * d
        )
        rows["streams"][vol] = f"{streams:.0f}"
        rows["inter"][vol] = str(inter)
        rows["ext_h"][vol] = f"{ext_h:.0f}"
        rows["batch"][vol] = str(batch)
        rows["gpus"][vol] = str(gpus)
        rows["nodes"][vol] = str(nodes)
        rows["busy"][vol] = pct(work_h / (gpus * 24))
        rows["busy_plan"][vol] = pct(work_h / PLAN_EFF / (gpus * 24))
        rows["split_gpus"][vol] = str(s_gpus)
        node_busy = work_h / (nodes * 8 * 24)
        sp_cost = options["Self-hosted, p5.4xlarge, savings plan"][vol]
        v[f"{vol}.streams"] = rows["streams"][vol]
        v[f"{vol}.inter"] = rows["inter"][vol]
        v[f"{vol}.batch"] = rows["batch"][vol]
        v[f"{vol}.gpus"] = rows["gpus"][vol]
        v[f"{vol}.nodes"] = rows["nodes"][vol]
        v[f"{vol}.busy"] = rows["busy"][vol]
        v[f"{vol}.busy_plan"] = rows["busy_plan"][vol]
        v[f"{vol}.idle"] = pct(1 - work_h / (gpus * 24))
        v[f"{vol}.inter_share"] = pct(inter / gpus)
        v[f"{vol}.node_busy"] = pct(node_busy)
        v[f"{vol}.work_h"] = f"{work_h:.0f}"
        v[f"{vol}.split_gpus"] = rows["split_gpus"][vol]
        v[f"{vol}.api_today"] = usd(
            options["Claude Sonnet 5.5 on Bedrock (today)"][vol]
        )
        v[f"{vol}.haiku_api"] = usd(
            options["Claude Haiku 5.5 on Bedrock, if it passed"][vol]
        )
        v[f"{vol}.oss_api"] = usd(api)
        v[f"{vol}.node_od"] = usd(
            options["Self-hosted, p5.48xlarge nodes, on-demand"][vol]
        )
        v[f"{vol}.node_sp"] = usd(
            options["Self-hosted, p5.48xlarge nodes, savings plan"][vol]
        )
        v[f"{vol}.self_od"] = usd(options["Self-hosted, p5.4xlarge, on-demand"][vol])
        v[f"{vol}.self_sp"] = usd(sp_cost)
        v[f"{vol}.split"] = usd(options[split][vol])
        v[f"{vol}.sp_vs_api"] = f"{sp_cost / api:.1f}"
        v[f"{vol}.split_vs_api"] = f"{options[split][vol] / api:.1f}"
        v[f"{vol}.api_buys_gpus"] = f"{api / (sp1 * HOURS_PER_MONTH):.1f}"
        v[f"{vol}.commit_all"] = usd(gpus * sp1 * HOURS_PER_YEAR)
        v[f"{vol}.commit_split"] = usd(s_gpus * sp1 * HOURS_PER_YEAR)

    t = Table("Option", "Today (1x)", "Next year (3x)", "At 10x", align="lrrr")
    for name, vals in options.items():
        t.add(name, *(usd(vals[k]) for k in VOLUMES))
    t.add("H100s self-hosted, all tokens", *(rows["gpus"][k] for k in VOLUMES))
    t.add(
        "H100s self-hosted, restricted share only",
        *(rows["split_gpus"][k] for k in VOLUMES),
    )
    tables["options"] = t.md()

    t = Table("", "Today (1x)", "Next year (3x)", "At 10x", align="lrrr")
    t.add(
        f"Assistant streams at peak ({STREAM_TARGET} tokens/s each)",
        *(rows["streams"][k] for k in VOLUMES),
    )
    t.add(
        f"Interactive GPUs ({ASSIST_BATCH} streams each, plus {SPARES} spare)",
        *(rows["inter"][k] for k in VOLUMES),
    )
    t.add(
        "Extraction GPU-hours a day at the ceiling",
        *(rows["ext_h"][k] for k in VOLUMES),
    )
    t.add(
        f"Batch GPUs (assumed {pct(PLAN_EFF)} of the ceiling, minimum 1)",
        *(rows["batch"][k] for k in VOLUMES),
    )
    t.add("H100s in all", *(rows["gpus"][k] for k in VOLUMES))
    t.add(
        "Paid GPU time the work needs, at the ceiling",
        *(rows["busy"][k] for k in VOLUMES),
    )
    t.add(
        f"Paid GPU time the work needs, at {pct(PLAN_EFF)} (assumed)",
        *(rows["busy_plan"][k] for k in VOLUMES),
    )
    tables["sizing"] = t.md()

    # ---- Prices and other figures the prose quotes ---------------------------------------------
    sonnet_geo = son["geo_input"] / son["input"]
    v.update(
        {
            "p5_hour": f"${p5['hourly']:.2f}",
            "p5_sp_hour": f"${p5['ec2_instance_savings_plan_1yr_no_upfront']:.2f}",
            "p5_month": usd(p5["hourly"] * HOURS_PER_MONTH),
            "p1_hour": f"${p1['hourly']:.2f}",
            "p1_sp_hour": f"${sp1:.2f}",
            "p1_csp_hour": f"${p1['compute_savings_plan_1yr_no_upfront']:.2f}",
            "p1_sp_month": usd(sp1 * HOURS_PER_MONTH),
            "p1_csp_off": pct(
                1 - p1["compute_savings_plan_1yr_no_upfront"] / p1["hourly"]
            ),
            "p1_sp_off": pct(1 - sp1 / p1["hourly"]),
            "p1_day_od_10h": usd(p1["hourly"] * 10),
            "p1_day_sp": usd(sp1 * 24),
            "node_commit": usd(
                p5["ec2_instance_savings_plan_1yr_no_upfront"] * HOURS_PER_YEAR
            ),
            "node_day_od": usd(p5["hourly"] * 24),
            "lambda_gpu_hour": f"${lam['per_gpu_hourly']:.2f}",
            "restricted_share": pct(RESTRICTED_SHARE),
            "plan_eff": pct(PLAN_EFF),
            "sonnet_geo": pct(sonnet_geo - 1),
            "today.api_today_geo": usd(
                sum(hosted(son, 1)) * DAYS_PER_MONTH * sonnet_geo
            ),
            "overhead": usd(overhead),
            "spare_month": usd(sp1 * HOURS_PER_MONTH),
            "canary_od_2d": usd(p1["hourly"] * 48),
            "load_s": f"{weights_gb / S3_GBPS:.0f}",
        }
    )

    # LoRA fine-tune inside the boundary, as arithmetic: 6 FLOPs per active parameter per token
    # (forward and backward) is an upper bound for LoRA, which skips most weight gradients.
    ft_tokens = 50_000 * EXTRACT_CTX
    ft_h = 6 * m["active_b"] * 1e9 * ft_tokens / (8 * g["bf16_tflops"] * 1e12) / 3600
    v["ft_tokens"] = num(ft_tokens)
    v["ft_hours"] = f"{ft_h:.1f}"
    v["ft_hours_tenth"] = f"{ft_h * 10:.0f}"
    return {"tables": tables, "vars": v}
