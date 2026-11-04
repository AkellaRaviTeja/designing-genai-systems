"""System 7: Simple RAG over a company wiki. Worked scenario numbers and monthly cost."""

from _lib import (
    DAYS_PER_MONTH,
    HOURS_PER_MONTH,
    WEEKS_PER_MONTH,
    Table,
    llm_cost,
    num,
    pct,
    usd,
)

ID = "ch07"

SCALES = {
    # staff, wiki pages, Postgres instance, multi-AZ, storage GB, app tasks, task size.
    # sync_min: billed minutes per 15-minute sync run (Fargate bills per second with a
    # 1-minute minimum, so 1 is the floor); reconcile_min: the daily page-ID reconcile.
    # Both run times are assumptions, not measurements.
    "pilot": dict(
        staff=1_200,
        pages=8_000,
        db="db-t4g-medium-ap-south-1",
        multi_az=False,
        disk_gb=20,
        tasks=2,
        vcpu=0.5,
        mem_gb=1,
        sync_min=1,
        reconcile_min=5,
    ),
    "growth": dict(
        staff=10_000,
        pages=40_000,
        db="db-r6g-large-ap-south-1",
        multi_az=True,
        disk_gb=100,
        tasks=2,
        vcpu=1,
        mem_gb=2,
        sync_min=2,
        reconcile_min=15,
    ),
    "large": dict(
        staff=50_000,
        pages=150_000,
        db="db-r6g-2xlarge-ap-south-1",
        multi_az=True,
        disk_gb=300,
        tasks=4,
        vcpu=1,
        mem_gb=2,
        sync_min=5,
        reconcile_min=60,
    ),
}

WEEKLY_ACTIVE = 0.4
QUESTIONS_PER_WEEK = 4
TOKENS_PER_PAGE = 1_600
CHUNK_TOKENS = 400  # structure-aware chunks, including the heading path header
DIMS = 1_536
MONTHLY_CHURN = 0.10  # assumption: share of pages edited in a month, re-embedded
SYNC_RUNS_PER_DAY = 96  # every 15 minutes
HEADER_TOKENS = 20  # heading-path prefix per chunk, inside CHUNK_TOKENS
SYSTEM_PROMPT = 450
CHUNKS_PER_QUERY = 5
QUESTION = 40
OUTPUT = 220
QUERY_EMBED = 40
GOLDEN = 150
EVAL_RUNS = 40  # assumption: per month, one per prompt or retrieval change plus nightly
JUDGE_IN, JUDGE_OUT = 3_000, 200  # judge from another model family (F4)
SMALL_CORPUS = 50 * TOKENS_PER_PAGE  # the 50-page variant


def build(P, H):
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    haiku = P("llm.anthropic.claude-haiku-5-5")
    judge = P("llm.openai.gpt-6-1-terra")
    emb = P("embed.openai.text-embedding-3-small")
    far = P("infra.aws.fargate.linux-x86-ap-south-1")
    disk = P("vector.aws.rds-postgres-db-r6g-xlarge")["gp3_storage_gb_month_multi_az"]
    disk_single = P("vector.aws.rds-postgres-db-r6g-xlarge")["gp3_storage_gb_month"]

    in_tokens = SYSTEM_PROMPT + CHUNKS_PER_QUERY * CHUNK_TOKENS + QUESTION
    q_sonnet = llm_cost(sonnet, in_tokens, OUTPUT)
    q_haiku = llm_cost(haiku, in_tokens, OUTPUT)
    q_judge = llm_cost(judge, JUDGE_IN, JUDGE_OUT)
    # Variant: the whole 50-page corpus in a cached prompt on Sonnet 5.5.
    small_prefix = SMALL_CORPUS + SYSTEM_PROMPT
    q_small = llm_cost(sonnet, QUESTION, OUTPUT, cache_read=small_prefix)
    small_write_1h = small_prefix * sonnet["cache_write_1h"] / 1e6

    v = {
        "in_tokens": f"{in_tokens:,}",
        "out_tokens": f"{OUTPUT}",
        "q_sonnet": usd(q_sonnet),
        "q_haiku": usd(q_haiku),
        "q_ratio": f"{q_sonnet / q_haiku:.0f}",
        "sonnet_in": f"${sonnet['input']:.2f}",
        "sonnet_out": f"${sonnet['output']:.2f}",
        "haiku_in": f"${haiku['input']:.2f}",
        "haiku_out": f"${haiku['output']:.2f}",
        "emb_price": f"${emb['input']:.2f}",
        "q_in_cost": usd(in_tokens * sonnet["input"] / 1e6),
        "q_out_cost": usd(OUTPUT * sonnet["output"] / 1e6),
        "q_judge": usd(q_judge),
        "judge_run": usd(GOLDEN * (q_sonnet + q_judge)),
        "q_saving": usd(q_sonnet - q_haiku),
        "out_share": pct(OUTPUT * sonnet["output"] / 1e6 / q_sonnet),
        "header_q": f"{HEADER_TOKENS * CHUNKS_PER_QUERY}",
        "header_share": pct(HEADER_TOKENS * CHUNKS_PER_QUERY / in_tokens),
        "lc_threshold": num(haiku["long_context_threshold_tokens"]),
        "small_corpus": num(SMALL_CORPUS),
        "q_small": usd(q_small),
        "small_write_1h": usd(small_write_1h),
        "sync_runs": num(SYNC_RUNS_PER_DAY * DAYS_PER_MONTH),
    }

    cost = Table(
        "Line item",
        "Pilot",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rows: dict[str, dict[str, float]] = {}
    for name, s in SCALES.items():
        queries = s["staff"] * WEEKLY_ACTIVE * QUESTIONS_PER_WEEK * WEEKS_PER_MONTH
        corpus = s["pages"] * TOKENS_PER_PAGE
        chunks = corpus / CHUNK_TOKENS
        db = P(f"infra.aws.rds-postgres.{s['db']}")
        ingest_h = (
            SYNC_RUNS_PER_DAY * DAYS_PER_MONTH * s["sync_min"]
            + DAYS_PER_MONTH * s["reconcile_min"]
        ) / 60
        r = {
            "gen": queries * q_sonnet,
            "embed": (corpus * MONTHLY_CHURN + queries * QUERY_EMBED)
            * emb["input"]
            / 1e6,
            "db": (db["hourly_multi_az"] if s["multi_az"] else db["hourly"])
            * HOURS_PER_MONTH
            + s["disk_gb"] * (disk if s["multi_az"] else disk_single),
            "app": s["tasks"]
            * HOURS_PER_MONTH
            * (s["vcpu"] * far["vcpu_hourly"] + s["mem_gb"] * far["gb_hourly"]),
            "ingest": ingest_h * (1 * far["vcpu_hourly"] + 2 * far["gb_hourly"]),
            "eval": EVAL_RUNS
            * GOLDEN
            * (q_sonnet + q_judge),
        }
        r["total"] = sum(r.values())
        r["queries"] = queries
        r["haiku_gen"] = queries * q_haiku
        r["haiku_total"] = r["total"] - r["gen"] + r["haiku_gen"]
        rows[name] = r
        v[f"{name}.queries"] = num(queries)
        v[f"{name}.corpus"] = num(corpus)
        v[f"{name}.chunks"] = num(chunks)
        v[f"{name}.index_mb"] = num(chunks * DIMS * 4 * 1.5 / 1e6)
        v[f"{name}.embed_once"] = usd(corpus * emb["input"] / 1e6)
        v[f"{name}.total"] = usd(r["total"])
        v[f"{name}.per_query"] = usd(r["total"] / queries)
        v[f"{name}.gen"] = usd(r["gen"])
        v[f"{name}.gen_share"] = pct(r["gen"] / r["total"])
        v[f"{name}.haiku_total"] = usd(r["haiku_total"])
        v[f"{name}.haiku_per_query"] = usd(r["haiku_total"] / queries)
        v[f"{name}.saving"] = pct(1 - r["haiku_total"] / r["total"])
        v[f"{name}.staff"] = f"{s['staff']:,}"
        v[f"{name}.pages"] = f"{s['pages']:,}"
        v[f"{name}.ingest_h"] = num(ingest_h)
        v[f"{name}.churn_pages"] = num(s["pages"] * MONTHLY_CHURN)
        v[f"{name}.haiku_saving_month"] = usd(r["gen"] - r["haiku_gen"])

    g = SCALES["growth"]
    how = {
        "gen": f"{num(rows['growth']['queries'])} queries x {usd(q_sonnet)} (Sonnet 5.5)",
        "embed": f"{pct(MONTHLY_CHURN)} of pages re-embedded (assumed), plus query embeddings",
        "db": f"RDS {g['db'].removesuffix('-ap-south-1').replace('-', '.')} Multi-AZ, Mumbai, {g['disk_gb']} GB gp3",
        "app": f"{g['tasks']} Fargate tasks, {g['vcpu']} vCPU and {g['mem_gb']} GB each, always on",
        "ingest": f"{SYNC_RUNS_PER_DAY} runs a day x {g['sync_min']} min, plus a {g['reconcile_min']}-min daily reconcile; 1 vCPU, 2 GB",
        "eval": f"{EVAL_RUNS} runs x {GOLDEN} questions, answer (Sonnet 5.5) plus judge (GPT-6.1 Terra)",
    }
    labels = {
        "gen": "Generation",
        "embed": "Embeddings",
        "db": "Postgres with pgvector",
        "app": "API and Slack bot",
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
        *(usd(rows[s]["total"] / rows[s]["queries"]) for s in SCALES),
        "",
    )
    cost.add(
        "Queries per month",
        *(num(rows[s]["queries"]) for s in SCALES),
        "staff x 40% weekly active x 4 x 4.35",
    )

    opt = Table("", "Pilot", "Growth", "Large", align="lrrr")
    opt.add("Total with Sonnet 5.5", *(usd(rows[s]["total"]) for s in SCALES))
    opt.add("Total with Haiku 5.5", *(usd(rows[s]["haiku_total"]) for s in SCALES))
    opt.add(
        "Saving", *(pct(1 - rows[s]["haiku_total"] / rows[s]["total"]) for s in SCALES)
    )
    return {"tables": {"monthly": cost.md(), "haiku": opt.md()}, "vars": v}
