"""System 10: Multi-tenant RAG for a SaaS product. Worked scenario numbers and monthly cost.

Lanternly, a team wiki sold as SaaS, adds an "Ask" panel built on System 8's retrieval and
System 9's follow-up handling. Every count, share, size and rate below that is not a price is a
worked-scenario assumption, labelled as such in the chapter.
"""

import math

import ch08
import ch09
from _lib import HOURS_PER_MONTH, WEEKS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch10"

# Tenant tiers at launch (assumptions): workspaces, seats each, pages each.
TIERS = {
    "small": dict(tenants=2_000, seats=25, pages=400),
    "mid": dict(tenants=380, seats=300, pages=2_000),
    "enterprise": dict(tenants=20, seats=4_000, pages=40_000),
}
BANK = dict(
    seats=20_000, pages=200_000
)  # the largest enterprise tenant, inside the tier

# Growth and large scales multiply every tier's tenant count (assumption).
SCALES = {
    # tenant multiplier, app tasks, sync and ingestion tasks
    "launch": dict(mult=1, tasks=4, workers=2),
    "growth": dict(mult=3, tasks=8, workers=4),
    "large": dict(mult=10, tasks=24, workers=8),
}

# Usage (assumptions).
WEEKLY_ACTIVE = 0.30  # share of seats that use Ask in a week
TURNS_PER_WEEK = 5  # turns (questions and follow-ups) per active seat
FOLLOWUP_SHARE = 0.35  # share of turns that are follow-ups and pay for condensation
BUSY_SHARE, BUSY_HOURS, PEAK_FACTOR = 0.85, 220, 2  # as System 8

# Corpus and storage.
PAGE_TOKENS = 1_600
CHUNK_TOKENS = ch08.CHUNK_TOKENS
CHUNKS_PER_PAGE = PAGE_TOKENS // CHUNK_TOKENS
DIMS = ch08.DIMS
VEC_BYTES = DIMS * 4
TEXT_BYTES = CHUNK_TOKENS * 4  # about 4 bytes of text per token (rule of thumb)
OTHER_BYTES = 2_000  # tsvector, readers array, keys and their indexes (assumption)
HNSW_OVERHEAD = 1.5  # vectors plus graph, rule of thumb as System 7
MONTHLY_CHURN = 0.10

# Physical layout (version 1 on).
POOL_MAX_CHUNKS = (
    10_000  # tenants at or below this share the pooled partition, exact scan
)
HOT_GB_PER_SHARD = 32  # HNSW memory budget per db.r6g.2xlarge (64 GiB), half its memory

# pgvector settings from its README.
EF_SEARCH = 40
MAX_SCAN_TUPLES = 20_000
DENSE_CANDIDATES = 50  # System 8's dense list length

# Permissions (assumptions).
BANK_RESTRICTED = 0.40  # share of the bank's pages restricted to groups
GUEST_PAGES = 50  # pages shared with an external guest in the bank
SUBTREE_PAGES = 5_000  # pages under one top-level page whose permissions change

# Pipeline per turn: System 8 without the glossary rewrite, System 9's condensation.
SYSTEM_PROMPT = ch08.SYSTEM_PROMPT
CHUNKS_SENT = ch08.CHUNKS_SENT
QUESTION = ch08.QUESTION
OUTPUT = ch08.OUTPUT
CANDIDATES = ch08.CANDIDATES
HISTORY = ch09.WINDOW * ch09.TURN_QA + ch09.STANDALONE  # what a follow-up adds
CONDENSE_IN = (
    ch09.CONDENSE_SYSTEM + ch09.SUMMARY + ch09.WINDOW * ch09.TURN_QA + QUESTION
)
CONDENSE_OUT = ch09.CONDENSE_OUT

# Evaluation: golden set on Lanternly's own and opted-in workspaces, plus the leak suite.
GOLDEN = 400
LEAK_TENANTS, LEAK_USERS, LEAK_PROBES = 3, 12, 600
EVAL_RUNS = 30
JUDGE_IN, JUDGE_OUT = ch08.JUDGE_IN, ch08.JUDGE_OUT

# Heavy usage for the billing break (assumptions).
SUPPORT_TURNS_PER_DAY, WORKING_DAYS = 30, 22
SCRIPT_TURNS_PER_SECOND = 10


def build(P, H):
    haiku = P("llm.anthropic.claude-haiku-5-5")
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    emb = P("embed.openai.text-embedding-3-small")
    pro = P("rerank.cohere.rerank-5-pro")
    fast = P("rerank.cohere.rerank-5-fast")
    far = P("infra.aws.fargate.linux-x86")
    db = P("infra.aws.rds-postgres.db-r6g-2xlarge")
    disk = P("vector.aws.rds-postgres-db-r6g-xlarge")["gp3_storage_gb_month_multi_az"]

    gen_in = SYSTEM_PROMPT + CHUNKS_SENT * CHUNK_TOKENS + QUESTION
    q_gen_first = llm_cost(haiku, gen_in, OUTPUT)
    q_gen_follow = llm_cost(haiku, gen_in + HISTORY, OUTPUT)
    q_condense = llm_cost(haiku, CONDENSE_IN, CONDENSE_OUT)
    rr_tokens = CANDIDATES * (CHUNK_TOKENS + QUESTION)
    q_rr = rr_tokens * pro["input"] / 1e6
    q_rr_fast = rr_tokens * fast["input"] / 1e6
    q_emb = (QUESTION + FOLLOWUP_SHARE * ch09.STANDALONE) * emb["input"] / 1e6
    q_gen = (1 - FOLLOWUP_SHARE) * q_gen_first + FOLLOWUP_SHARE * (
        q_gen_follow + q_condense
    )
    q_turn = q_gen + q_rr + q_emb  # metered model cost of an average turn
    q_judge = llm_cost(sonnet, JUDGE_IN, JUDGE_OUT)

    turns_per_seat = WEEKLY_ACTIVE * TURNS_PER_WEEK * WEEKS_PER_MONTH

    def chunks_of(pages: int) -> int:
        return pages * CHUNKS_PER_PAGE

    def tier_totals(mult: int) -> dict:
        t = {"tenants": 0, "seats": 0, "chunks": 0, "pooled": 0, "dedicated": 0}
        for tier in TIERS.values():
            n = tier["tenants"] * mult
            c = chunks_of(tier["pages"])
            t["tenants"] += n
            t["seats"] += n * tier["seats"]
            t["chunks"] += n * c
            t["pooled" if c <= POOL_MAX_CHUNKS else "dedicated"] += n * c
        return t

    launch = tier_totals(1)
    small_chunks = chunks_of(TIERS["small"]["pages"])
    small_share = small_chunks / launch["chunks"]
    bank_chunks = chunks_of(BANK["pages"])
    guest_share = GUEST_PAGES * CHUNKS_PER_PAGE / bank_chunks
    bytes_per_chunk = VEC_BYTES + TEXT_BYTES + OTHER_BYTES

    v = {
        "small_tenants": f"{TIERS['small']['tenants']:,}",
        "mid_tenants": f"{TIERS['mid']['tenants']:,}",
        "ent_tenants": f"{TIERS['enterprise']['tenants']:,}",
        "small_seats": f"{TIERS['small']['seats']:,}",
        "mid_seats": f"{TIERS['mid']['seats']:,}",
        "ent_seats": f"{TIERS['enterprise']['seats']:,}",
        "small_pages": f"{TIERS['small']['pages']:,}",
        "mid_pages": f"{TIERS['mid']['pages']:,}",
        "ent_pages": f"{TIERS['enterprise']['pages']:,}",
        "small_chunks": f"{small_chunks:,}",
        "mid_chunks": f"{chunks_of(TIERS['mid']['pages']):,}",
        "ent_chunks": num(chunks_of(TIERS["enterprise"]["pages"])),
        "bank_seats": f"{BANK['seats']:,}",
        "bank_pages": f"{BANK['pages']:,}",
        "bank_chunks": num(bank_chunks),
        "weekly_active": pct(WEEKLY_ACTIVE),
        "turns_per_week": f"{TURNS_PER_WEEK}",
        "turns_per_seat": f"{turns_per_seat:.1f}",
        "followup_share": pct(FOLLOWUP_SHARE),
        "vec_bytes": f"{VEC_BYTES:,}",
        "bytes_per_chunk": f"{bytes_per_chunk:,}",
        "pool_max": f"{POOL_MAX_CHUNKS:,}",
        "pool_max_pages": f"{POOL_MAX_CHUNKS // CHUNKS_PER_PAGE:,}",
        "pool_scan_mb": f"{POOL_MAX_CHUNKS * VEC_BYTES / 1e6:.0f}",
        "hot_gb_shard": f"{HOT_GB_PER_SHARD}",
        "ef_search": f"{EF_SEARCH}",
        "max_scan": f"{MAX_SCAN_TUPLES:,}",
        "dense_candidates": f"{DENSE_CANDIDATES}",
        "small_share": f"{small_share * 100:.3f}%",
        "small_ef": f"{EF_SEARCH * small_share:.3f}",
        "small_iter": f"{MAX_SCAN_TUPLES * small_share:.1f}",
        "small_needed": num(DENSE_CANDIDATES / small_share),
        "bank_restricted": pct(BANK_RESTRICTED),
        "guest_pages": f"{GUEST_PAGES}",
        "guest_share": f"{guest_share * 100:.3f}%",
        "guest_iter": f"{MAX_SCAN_TUPLES * guest_share:.0f}",
        "guest_chunks": f"{GUEST_PAGES * CHUNKS_PER_PAGE}",
        "subtree_pages": f"{SUBTREE_PAGES:,}",
        "subtree_rows": num(SUBTREE_PAGES * CHUNKS_PER_PAGE),
        "gen_in": f"{gen_in:,}",
        "history": f"{HISTORY}",
        "out_tokens": f"{OUTPUT}",
        "rr_tokens": f"{rr_tokens:,}",
        "candidates": f"{CANDIDATES}",
        "q_gen_first": usd(q_gen_first),
        "q_gen_follow": usd(q_gen_follow),
        "q_condense": usd(q_condense),
        "q_rr": usd(q_rr),
        "q_emb": usd(q_emb),
        "q_turn": usd(q_turn),
        "rr_share": pct(q_rr / q_turn),
        "turns_per_dollar": num(1 / q_turn),
        "golden": f"{GOLDEN}",
        "leak_tenants": f"{LEAK_TENANTS}",
        "leak_users": f"{LEAK_USERS}",
        "leak_probes": f"{LEAK_PROBES}",
        "eval_runs": f"{EVAL_RUNS}",
        "haiku_in": f"${haiku['input']:.2f}",
        "haiku_out": f"${haiku['output']:.2f}",
        "rr_price": f"${pro['input']:.2f}",
    }

    # Noisy neighbour: one tenant's script against the organisation's shared limits.
    v["script_tps"] = f"{SCRIPT_TURNS_PER_SECOND}"
    v["script_itpm"] = num(SCRIPT_TURNS_PER_SECOND * gen_in * 60)
    v["script_rr_tpm"] = num(SCRIPT_TURNS_PER_SECOND * rr_tokens * 60)
    v["script_month"] = usd(SCRIPT_TURNS_PER_SECOND * 3600 * 24 * 30.4 * q_turn)
    v["script_day"] = usd(SCRIPT_TURNS_PER_SECOND * 3600 * 24 * q_turn)

    support_turns = SUPPORT_TURNS_PER_DAY * WORKING_DAYS
    v["support_per_day"] = f"{SUPPORT_TURNS_PER_DAY}"
    v["support_turns"] = f"{support_turns:,}"
    v["support_seat"] = usd(support_turns * q_turn)
    v["avg_seat_metered"] = usd(turns_per_seat * q_turn)
    v["support_ratio"] = f"{support_turns / turns_per_seat:.0f}"

    cost = Table(
        "Line item",
        "Launch",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rows: dict[str, dict[str, float]] = {}
    for name, s in SCALES.items():
        t = tier_totals(s["mult"])
        turns = t["seats"] * turns_per_seat
        corpus_tokens = t["chunks"] * CHUNK_TOKENS
        hot_gb = t["dedicated"] * VEC_BYTES * HNSW_OVERHEAD / 1e9
        shards = max(1, math.ceil(hot_gb / HOT_GB_PER_SHARD))
        disk_gb = (
            t["chunks"] * bytes_per_chunk
            + t["dedicated"] * VEC_BYTES * (HNSW_OVERHEAD - 1)
        ) / 1e9
        disk_alloc = math.ceil(disk_gb * 1.3 / 50) * 50  # 30% headroom, in 50 GB steps
        peak_qps = BUSY_SHARE * turns / (BUSY_HOURS * 3600) * PEAK_FACTOR
        task_month = HOURS_PER_MONTH * (1 * far["vcpu_hourly"] + 2 * far["gb_hourly"])
        r = {
            "gen": turns * q_gen,
            "rerank": turns * q_rr,
            "embed": corpus_tokens * MONTHLY_CHURN * emb["input"] / 1e6 + turns * q_emb,
            "db": shards * db["hourly_multi_az"] * HOURS_PER_MONTH + disk_alloc * disk,
            "app": s["tasks"] * task_month,
            "workers": s["workers"] * task_month,
            "eval": EVAL_RUNS
            * (
                GOLDEN * (q_gen_first + q_rr + q_judge)
                + LEAK_PROBES * (q_gen_first + q_rr)
            ),
        }
        r["total"] = sum(r.values())
        r["turns"], r["seats"], r["chunks"] = turns, t["seats"], t["chunks"]
        r["shards"], r["disk_alloc"], r["hot_gb"] = shards, disk_alloc, hot_gb
        r["db_inst"] = shards * db["hourly_multi_az"] * HOURS_PER_MONTH
        r["db_disk"] = disk_alloc * disk
        r["disk_gb"] = disk_gb
        rows[name] = r
        v[f"{name}.tenants"] = f"{t['tenants']:,}"
        v[f"{name}.seats"] = num(t["seats"])
        v[f"{name}.turns"] = num(turns)
        v[f"{name}.chunks"] = num(t["chunks"])
        v[f"{name}.pooled"] = num(t["pooled"])
        v[f"{name}.dedicated"] = num(t["dedicated"])
        v[f"{name}.hot_gb"] = f"{hot_gb:.0f}"
        v[f"{name}.disk_gb"] = num(disk_gb)
        v[f"{name}.disk_alloc"] = f"{disk_alloc:,}"
        v[f"{name}.shards"] = f"{shards}"
        v[f"{name}.peak_qps"] = f"{peak_qps:.2g}"
        v[f"{name}.total"] = usd(r["total"])
        v[f"{name}.per_turn"] = usd(r["total"] / turns)
        v[f"{name}.per_seat"] = usd(r["total"] / t["seats"])
        v[f"{name}.metered_share"] = pct(
            (r["gen"] + r["rerank"] + r["embed"]) / r["total"]
        )
        v[f"{name}.rerank"] = usd(r["rerank"])
        v[f"{name}.gen"] = usd(r["gen"])
        v[f"{name}.db"] = usd(r["db"])
        v[f"{name}.db_share"] = pct(r["db"] / r["total"])
        # O2: Rerank 5 Fast instead of Pro, same candidates.
        v[f"{name}.fast_saving"] = usd(turns * (q_rr - q_rr_fast))
        # Half-precision vectors: half the hot HNSW memory, so fewer shards.
        shards_half = max(1, math.ceil(hot_gb / 2 / HOT_GB_PER_SHARD))
        v[f"{name}.shards_half"] = f"{shards_half}"
        v[f"{name}.half_saving"] = usd(
            (shards - shards_half) * db["hourly_multi_az"] * HOURS_PER_MONTH
        )
        # Peak re-checks: one batched permission query per turn.
        v[f"{name}.rechecks"] = f"{peak_qps:.2g}"

    L = rows["launch"]
    v["launch.vec_raw_gb"] = f"{L['chunks'] * VEC_BYTES / 1e9:.0f}"
    v["launch.dedicated_tenants"] = f"{TIERS['enterprise']['tenants']}"
    v["rr_fast_price"] = f"${fast['input']:.2f}"
    v["q_rr_fast"] = usd(q_rr_fast)
    v["shard_month"] = usd(db["hourly_multi_az"] * HOURS_PER_MONTH)
    v["shard_hourly"] = f"${db['hourly_multi_az']:.3f}"
    v["cache_min"] = "512"
    leak_run = LEAK_PROBES * (q_gen_first + q_rr)
    golden_run = GOLDEN * (q_gen_first + q_rr + q_judge)
    v["eval_leak_share"] = pct(leak_run / (leak_run + golden_run))
    v["system_prompt"] = f"{SYSTEM_PROMPT}"

    g = rows["growth"]
    how = {
        "gen": f"Haiku 5.5: {usd(q_gen_first)} a first question; a follow-up {usd(q_gen_follow)} plus {usd(q_condense)} to condense; {pct(FOLLOWUP_SHARE)} of turns are follow-ups",
        "rerank": f"{CANDIDATES} candidates x {CHUNK_TOKENS + QUESTION} tokens x {v['rr_price']}/M (Rerank 5 Pro)",
        "embed": f"{pct(MONTHLY_CHURN)} of {num(g['chunks'] * CHUNK_TOKENS)} corpus tokens re-embedded, plus query embeddings",
        "db": f"{g['shards']} RDS db.r6g.2xlarge Multi-AZ shards, {g['disk_alloc']:,} GB gp3 in all",
        "app": f"{SCALES['growth']['tasks']} Fargate tasks, 1 vCPU and 2 GB each, always on",
        "workers": f"{SCALES['growth']['workers']} tasks for ingestion and permission sync, always on",
        "eval": f"{EVAL_RUNS} runs x ({GOLDEN} judged questions + {LEAK_PROBES} leak probes checked in code)",
    }
    labels = {
        "gen": "Generation and condensation",
        "rerank": "Reranking",
        "embed": "Embeddings",
        "db": "Postgres shards",
        "app": "Ask service",
        "workers": "Ingestion and permission sync",
        "eval": "Evaluation and leak suite",
    }
    for k, label in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in SCALES), how[k])
    cost.add(
        "**Total per month**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), ""
    )
    cost.add(
        "Cost per seat", *(usd(rows[s]["total"] / rows[s]["seats"]) for s in SCALES), ""
    )
    cost.add(
        "Cost per turn", *(usd(rows[s]["total"] / rows[s]["turns"]) for s in SCALES), ""
    )
    cost.add(
        "Turns per month",
        *(num(rows[s]["turns"]) for s in SCALES),
        f"seats x {pct(WEEKLY_ACTIVE)} weekly active x {TURNS_PER_WEEK} x 4.35",
    )

    # Per-tenant attribution at launch. Metered lines by usage. The database: instance cost
    # half by hot HNSW memory (dedicated tenants only) and half by turns; disk by bytes stored.
    # Compute by turns. These keys are a choice, stated in the chapter.
    shared_compute = L["app"] + L["workers"]
    attr = Table(
        "Tenant",
        "Seats",
        "Turns a month",
        "Metered",
        "Database share",
        "Compute share",
        "Total",
        "Per seat",
        align="lrrrrrrr",
    )
    examples = [
        ("Small workspace", "small", TIERS["small"]["seats"], TIERS["small"]["pages"], turns_per_seat),
        ("Mid-size workspace", "mid", TIERS["mid"]["seats"], TIERS["mid"]["pages"], turns_per_seat),
        ("The bank", "bank", BANK["seats"], BANK["pages"], turns_per_seat),
        ("Small workspace, support desk", "support", TIERS["small"]["seats"], TIERS["small"]["pages"], support_turns),
    ]
    for label, key, seats, pages, tps in examples:
        turns = seats * tps
        chunks = chunks_of(pages)
        dedicated = chunks > POOL_MAX_CHUNKS
        metered = turns * q_turn + chunks * CHUNK_TOKENS * MONTHLY_CHURN * emb["input"] / 1e6
        hot = chunks * VEC_BYTES * HNSW_OVERHEAD / 1e9 if dedicated else 0.0
        stored = (
            chunks * bytes_per_chunk
            + (chunks * VEC_BYTES * (HNSW_OVERHEAD - 1) if dedicated else 0)
        ) / 1e9
        dbshare = (
            L["db_inst"] / 2 * hot / L["hot_gb"]
            + L["db_inst"] / 2 * turns / L["turns"]
            + L["db_disk"] * stored / L["disk_gb"]
        )
        compute = shared_compute * turns / L["turns"]
        total = metered + dbshare + compute
        attr.add(label, f"{seats:,}", num(turns), usd(metered), usd(dbshare), usd(compute), usd(total), usd(total / seats))
        v[f"attr_{key}_total"] = usd(total)
        v[f"attr_{key}_seat"] = usd(total / seats)
        v[f"attr_{key}_db_share"] = pct(dbshare / total)
        v[f"attr_{key}_metered_share"] = pct(metered / total)
        v[f"attr_{key}_hnsw_share"] = pct(L["db_inst"] / 2 * hot / L["hot_gb"] / total)

    v["attr_compute"] = usd(shared_compute)
    # Turns a month at which reranking would overtake the database at launch.
    v["rr_overtakes_db"] = num(L["db"] / q_rr)
    v["q_turn_exact"] = f"${q_turn:.5f}"
    # Exact-scan break-even: 50 ms over the largest pooled tenant's rows.
    v["us_per_row"] = f"{50e3 / POOL_MAX_CHUNKS:.0f}"
    # Iterative HNSW finds max_scan x readable / partition rows; 50 needs readable >= N / 400.
    v["readable_ratio"] = f"{MAX_SCAN_TUPLES // DENSE_CANDIDATES}"
    v["bank_cutoff"] = f"{bank_chunks * DENSE_CANDIDATES // MAX_SCAN_TUPLES:,}"
    # Rule of three: zero failures in n probes bounds the rate below about 3/n at 95%.
    v["small_gap"] = f"{DENSE_CANDIDATES / MAX_SCAN_TUPLES / small_share:.0f}"
    v["leak_bound"] = f"{3 / LEAK_PROBES * 100:.1f}%"
    v["subtree_acl_rows"] = f"{SUBTREE_PAGES:,}"

    return {"tables": {"monthly": cost.md(), "tenants": attr.md()}, "vars": v}
