"""System 8: Production RAG with hybrid search and reranking. Worked scenario numbers and
monthly cost."""

import math

from _lib import HOURS_PER_MONTH, WEEKS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch08"

SCALES = {
    # queries per month (launch is derived below), corpus tokens, Postgres instance and
    # read replicas, storage GB, app tasks, ingestion task-hours per month
    "launch": dict(
        queries=None,
        corpus=96e6,
        db="db-r6g-large-ap-south-1",
        replicas=0,
        disk_gb=50,
        tasks=2,
        ingest_h=60,
    ),
    "growth": dict(
        queries=3e6,
        corpus=400e6,
        db="db-r6g-2xlarge-ap-south-1",
        replicas=0,
        disk_gb=200,
        tasks=4,
        ingest_h=250,
    ),
    "large": dict(
        queries=30e6,
        corpus=1e9,
        db="db-r6g-2xlarge-ap-south-1",
        replicas=1,
        disk_gb=500,
        tasks=16,
        ingest_h=800,
    ),
}

# Launch traffic: hub and operations staff, plus the customer support desk.
OPS_STAFF, OPS_ACTIVE, OPS_PER_WEEK = 4_000, 0.6, 10
AGENTS, AGENT_PER_DAY, WORKING_DAYS = 300, 30, 22
PAGES, PAGE_TOKENS = 30_000, 1_600
PDFS, PDF_TOKENS = 6_000, 8_000

CHUNK_TOKENS = 400
DIMS = 1_536
MONTHLY_CHURN = 0.10
QUESTION = 40

# Generation: the grounded answer contract from System 7, six chunks after reranking.
SYSTEM_PROMPT = 500
CHUNKS_SENT = 6
OUTPUT = 220
V0_CHUNKS = 5
WIDE_CHUNKS = 20  # the "just send more" experiment in the ranking break

# Reranking: fused candidates, each scored with the question.
CANDIDATES = 30
V2_CANDIDATES = 50

# Query rewriting on Haiku 5.5 with a cached glossary prefix, skipped for identifier queries.
REWRITE_SHARE = 0.55
REWRITE_PREFIX = 2_000
REWRITE_IN = 60
REWRITE_OUT = 90
BUSINESS_HOURS = WORKING_DAYS * 10
CACHE_WRITES_PER_HOUR = 12  # worst case: the 5-minute cache expires between every write

# Evaluation: the golden set runs through the whole pipeline, judged by Sonnet 5.5.
GOLDEN = 400
EVAL_RUNS = 30
JUDGE_IN, JUDGE_OUT = 3_500, 200

# Self-hosted reranker sizing (rule of thumb inputs, stated in the chapter).
BUSY_SHARE = 0.85  # share of queries in business hours
PEAK_FACTOR = 2  # peak over the business-hour average
GPU_EFFICIENCY = 0.35  # share of datasheet BF16 an encoder reaches at these lengths
MIN_GPUS = 2


def build(P, H):
    haiku = P("llm.anthropic.claude-haiku-5-5")
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    emb = P("embed.openai.text-embedding-3-small")
    pro = P("rerank.cohere.rerank-5-pro")
    fast = P("rerank.cohere.rerank-5-fast")
    l4 = P("gpu.aws.g6-xlarge")
    far = P("infra.aws.fargate.linux-x86-ap-south-1")
    disk_rate = P("vector.aws.rds-postgres-db-r6g-xlarge")
    gpu, rr = H("gpu.L4"), H("model.bge-reranker-v2-m3")

    launch_q = (
        OPS_STAFF * OPS_ACTIVE * OPS_PER_WEEK * WEEKS_PER_MONTH
        + AGENTS * AGENT_PER_DAY * WORKING_DAYS
    )
    SCALES["launch"]["queries"] = launch_q

    gen_in = SYSTEM_PROMPT + CHUNKS_SENT * CHUNK_TOKENS + QUESTION
    v0_in = SYSTEM_PROMPT + V0_CHUNKS * CHUNK_TOKENS + QUESTION
    wide_in = SYSTEM_PROMPT + WIDE_CHUNKS * CHUNK_TOKENS + QUESTION
    rr_tokens = CANDIDATES * (CHUNK_TOKENS + QUESTION)
    rr_tokens_v2 = V2_CANDIDATES * (CHUNK_TOKENS + QUESTION)

    q_gen = llm_cost(haiku, gen_in, OUTPUT)
    q_v0 = llm_cost(haiku, v0_in, OUTPUT)
    q_wide = llm_cost(haiku, wide_in, OUTPUT)
    q_rr = rr_tokens * pro["input"] / 1e6
    q_rr_fast = rr_tokens * fast["input"] / 1e6
    q_rr_v2 = rr_tokens_v2 * pro["input"] / 1e6
    q_rw_cached = llm_cost(haiku, REWRITE_IN, REWRITE_OUT, cache_read=REWRITE_PREFIX)
    q_rw_uncached = llm_cost(haiku, REWRITE_PREFIX + REWRITE_IN, REWRITE_OUT)
    q_emb = (1 + REWRITE_SHARE) * QUESTION * emb["input"] / 1e6
    q_pipeline = q_gen + q_rr + REWRITE_SHARE * q_rw_cached + q_emb

    flops = 2 * rr["params_b"] * 1e9 * rr_tokens
    qps_per_gpu = gpu["bf16_tflops"] * 1e12 * GPU_EFFICIENCY / flops

    corpus_launch = PAGES * PAGE_TOKENS + PDFS * PDF_TOKENS
    assert corpus_launch == SCALES["launch"]["corpus"]

    v = {
        "pages": f"{PAGES:,}",
        "pdfs": f"{PDFS:,}",
        "page_tokens": f"{PAGES * PAGE_TOKENS / 1e6:.0f}M",
        "pdf_tokens": f"{PDFS * PDF_TOKENS / 1e6:.0f}M",
        "ops_q": num(OPS_STAFF * OPS_ACTIVE * OPS_PER_WEEK * WEEKS_PER_MONTH),
        "agent_q": num(AGENTS * AGENT_PER_DAY * WORKING_DAYS),
        "gen_in": f"{gen_in:,}",
        "v0_in": f"{v0_in:,}",
        "wide_in": f"{wide_in:,}",
        "wide_ratio": f"{wide_in / v0_in:.1f}",
        "out_tokens": f"{OUTPUT}",
        "rr_tokens": f"{rr_tokens:,}",
        "rr_tokens_v2": f"{rr_tokens_v2:,}",
        "rr_to_gen": f"{rr_tokens / gen_in:.1f}",
        "q_gen": usd(q_gen),
        "q_v0": usd(q_v0),
        "q_wide": usd(q_wide),
        "q_rr": usd(q_rr),
        "q_rr_fast": usd(q_rr_fast),
        "q_rr_v2": usd(q_rr_v2),
        "q_rr_share": pct(q_rr / q_pipeline),
        "q_rw": usd(q_rw_cached),
        "q_rw_uncached": usd(q_rw_uncached),
        "rw_saving": pct(1 - q_rw_cached / q_rw_uncached),
        "q_pipeline": usd(q_pipeline),
        "haiku_in": f"${haiku['input']:.2f}",
        "haiku_out": f"${haiku['output']:.2f}",
        "haiku_cache_read": f"${haiku['cache_read']:.2f}",
        "emb_price": f"${emb['input']:.2f}",
        "rr_pro_price": f"${pro['input']:.2f}",
        "rr_fast_price": f"${fast['input']:.2f}",
        "l4_hourly": f"${l4['hourly']:.2f}",
        "l4_monthly": usd(l4["hourly"] * HOURS_PER_MONTH),
        "rr_params": f"{rr['params_b'] * 1e3:.0f}M",
        "rr_tflop": f"{flops / 1e12:.0f}",
        "l4_tflops": f"{gpu['bf16_tflops']}",
        "gpu_eff": pct(GPU_EFFICIENCY),
        "qps_per_gpu": f"{qps_per_gpu:.1f}",
    }

    cost = Table(
        "Line item",
        "Launch",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rerank_t = Table(
        "Reranker",
        "Launch",
        "Growth",
        "Large",
        align="lrrr",
    )
    rows: dict[str, dict[str, float]] = {}
    rr_rows: dict[str, dict[str, float]] = {}
    for name, s in SCALES.items():
        q = s["queries"]
        chunks = s["corpus"] / CHUNK_TOKENS
        db = P(f"infra.aws.rds-postgres.{s['db']}")
        rewrites = q * REWRITE_SHARE
        cache_writes = BUSINESS_HOURS * CACHE_WRITES_PER_HOUR
        r = {
            "gen": q * q_gen,
            "rerank": q * q_rr,
            "rewrite": rewrites * q_rw_cached
            + cache_writes * REWRITE_PREFIX * haiku["cache_write"] / 1e6,
            "embed": s["corpus"] * MONTHLY_CHURN * emb["input"] / 1e6 + q * q_emb,
            "db": (db["hourly_multi_az"] + s["replicas"] * db["hourly"])
            * HOURS_PER_MONTH
            + s["disk_gb"]
            * (
                disk_rate["gp3_storage_gb_month_multi_az"]
                + s["replicas"] * disk_rate["gp3_storage_gb_month"]
            ),
            "app": s["tasks"]
            * HOURS_PER_MONTH
            * (1 * far["vcpu_hourly"] + 2 * far["gb_hourly"]),
            "ingest": s["ingest_h"] * (1 * far["vcpu_hourly"] + 2 * far["gb_hourly"]),
            "eval": EVAL_RUNS
            * GOLDEN
            * (q_pipeline + llm_cost(sonnet, JUDGE_IN, JUDGE_OUT)),
        }
        r["total"] = sum(r.values())
        rows[name] = r

        peak_qps = BUSY_SHARE * q / (BUSINESS_HOURS * 3600) * PEAK_FACTOR
        gpus = max(MIN_GPUS, math.ceil(peak_qps / qps_per_gpu))
        rr_rows[name] = {
            "pro": q * q_rr,
            "fast": q * q_rr_fast,
            "gpu_flat": gpus * HOURS_PER_MONTH * l4["hourly"],
            "gpu_scaled": (
                gpus * BUSINESS_HOURS + MIN_GPUS * (HOURS_PER_MONTH - BUSINESS_HOURS)
            )
            * l4["hourly"],
            "gpus": gpus,
            "peak": peak_qps,
        }

        v[f"{name}.queries"] = num(q)
        v[f"{name}.corpus"] = num(s["corpus"])
        v[f"{name}.chunks"] = num(chunks)
        v[f"{name}.index_gb"] = f"{chunks * DIMS * 4 * 1.5 / 1e9:.1f}"
        v[f"{name}.embed_once"] = usd(s["corpus"] * emb["input"] / 1e6)
        v[f"{name}.total"] = usd(r["total"])
        v[f"{name}.per_query"] = usd(r["total"] / q)
        v[f"{name}.gen"] = usd(r["gen"])
        v[f"{name}.rerank"] = usd(r["rerank"])
        v[f"{name}.rerank_share"] = pct(r["rerank"] / r["total"])
        v[f"{name}.gen_share"] = pct(r["gen"] / r["total"])
        v[f"{name}.fixed_share"] = pct(
            (r["db"] + r["app"] + r["ingest"] + r["eval"]) / r["total"]
        )
        v[f"{name}.peak_qps"] = f"{peak_qps:.2g}"
        v[f"{name}.gpus"] = f"{gpus}"
        v[f"{name}.rr_fast"] = usd(rr_rows[name]["fast"])
        v[f"{name}.rr_gpu_flat"] = usd(rr_rows[name]["gpu_flat"])
        v[f"{name}.rr_gpu_scaled"] = usd(rr_rows[name]["gpu_scaled"])
        fast_total = r["total"] - r["rerank"] + rr_rows[name]["fast"]
        v[f"{name}.fast_total"] = usd(fast_total)
        v[f"{name}.fast_saving"] = pct(1 - fast_total / r["total"])

    g = SCALES["growth"]
    how = {
        "gen": f"{num(g['queries'])} queries x {usd(q_gen)} (Haiku 5.5, {gen_in:,} in, {OUTPUT} out)",
        "rerank": f"{CANDIDATES} candidates x {CHUNK_TOKENS + QUESTION} tokens x {v['rr_pro_price']}/M (Rerank 5 Pro)",
        "rewrite": f"{pct(REWRITE_SHARE)} of queries on Haiku 5.5, {REWRITE_PREFIX:,}-token glossary prefix cached",
        "embed": f"{pct(MONTHLY_CHURN)} of the corpus re-embedded, plus one or two query embeddings",
        "db": f"RDS {g['db'].removesuffix('-ap-south-1').replace('-', '.')} Multi-AZ, Mumbai, {g['disk_gb']} GB gp3",
        "app": f"{g['tasks']} Fargate tasks, 1 vCPU and 2 GB each, always on",
        "ingest": f"Sync and parse worker, {g['ingest_h']} task-hours a month",
        "eval": f"{EVAL_RUNS} runs x {GOLDEN} questions, full pipeline plus a Sonnet 5.5 judge",
    }
    labels = {
        "gen": "Generation",
        "rerank": "Reranking",
        "rewrite": "Query rewriting",
        "embed": "Embeddings",
        "db": "Postgres (vectors and full text)",
        "app": "Retrieval and answer service",
        "ingest": "Ingestion worker",
        "eval": "Evaluation runs",
    }
    for k, label in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in SCALES), how[k])
    cost.add(
        "**Total per month**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), ""
    )
    cost.add(
        "Cost per query",
        *(usd(rows[s]["total"] / SCALES[s]["queries"]) for s in SCALES),
        "",
    )
    cost.add(
        "Queries per month",
        *(num(SCALES[s]["queries"]) for s in SCALES),
        "Launch: hub staff plus the support desk; growth and large are 10x and 100x",
    )

    rerank_t.add("Rerank 5 Pro API", *(usd(rr_rows[s]["pro"]) for s in SCALES))
    rerank_t.add("Rerank 5 Fast API", *(usd(rr_rows[s]["fast"]) for s in SCALES))
    rerank_t.add(
        "Self-hosted on L4, sized for peak, always on",
        *(usd(rr_rows[s]["gpu_flat"]) for s in SCALES),
    )
    rerank_t.add(
        "Self-hosted on L4, peak fleet in business hours only",
        *(usd(rr_rows[s]["gpu_scaled"]) for s in SCALES),
    )
    rerank_t.add(
        "L4 GPUs at peak (minimum 2)", *(str(rr_rows[s]["gpus"]) for s in SCALES)
    )
    rerank_t.add(
        "Peak queries per second", *(f"{rr_rows[s]['peak']:.2g}" for s in SCALES)
    )
    return {"tables": {"monthly": cost.md(), "rerank": rerank_t.md()}, "vars": v}
