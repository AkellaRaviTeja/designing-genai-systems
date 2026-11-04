"""System 9: Conversational RAG with follow-ups. Worked scenario numbers and monthly cost.

Builds on System 7: same company, wiki, infrastructure and generation model. Every rate,
share and length below that is not a price is a worked-scenario assumption.
"""

import ch07
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

ID = "ch09"

# Thread shape (assumptions).
FOLLOWUP_SHARE = 0.35  # share of first questions that get at least one follow-up
FOLLOWUPS_PER_THREAD = 3  # mean follow-ups in those threads
LONG_THREAD = 15  # turns in a long thread, for the growth arithmetic
STANDALONE_SHARE = (
    0.25  # share of follow-ups that start a new topic (no history needed)
)

# Token sizes. System 7's per-turn sizes, plus what the thread adds.
SYSTEM = ch07.SYSTEM_PROMPT
CHUNKS = ch07.CHUNKS_PER_QUERY * ch07.CHUNK_TOKENS
QUESTION = ch07.QUESTION
ANSWER = ch07.OUTPUT
TURN_QA = QUESTION + ANSWER  # one previous turn kept as text
TURN_FULL = (
    QUESTION + CHUNKS + ANSWER
)  # one previous turn kept with its chunks (version 1)
STANDALONE = 40  # the condensed, standalone question
WINDOW = 3  # previous turns kept verbatim (version 2 on)
SUMMARY = 200  # running summary of older turns

# Condenser and summariser on Haiku 5.5.
CONDENSE_SYSTEM = 300
CONDENSE_OUT = 50  # JSON: standalone question plus a new-topic flag
SUMMARISE_IN = 300 + SUMMARY + TURN_QA
SUMMARISE_OUT = SUMMARY

# Evaluation: System 7's set plus a conversation set, tagged (assumed split of the 200 turns).
CONV_THREADS = 80
CONV_TURNS = 200
SLICES = {"dependent": 100, "new_topic": 50, "about_answer": 30, "correction": 20}
assert sum(SLICES.values()) == CONV_TURNS

# Thread state: the prompt history at about 4 bytes of text per token (rule of thumb).
STATE_BYTES = (SUMMARY + WINDOW * TURN_QA) * 4 + 2_000  # history text plus keys and metadata
MSG_CAP = 300  # tokens kept per stored message; longer pasted text is truncated in history
RETENTION_DAYS = 90


def ci(n: int) -> float:
    """Worst-case 95% interval half-width in points, at a score of 50%."""
    return 1.96 * (0.25 / n) ** 0.5 * 100
EVAL_RUNS = ch07.EVAL_RUNS


def turn_input(n: int, per_turn: int, window: int | None = None) -> int:
    """Generation input tokens on turn n (1-based) of a thread."""
    prev = n - 1
    if window is None:
        hist = prev * per_turn
    else:
        hist = min(prev, window) * per_turn + (SUMMARY if prev > window else 0)
    extra = STANDALONE if (n > 1 and window is not None) else 0
    return SYSTEM + hist + CHUNKS + QUESTION + extra


def thread_input(turns: int, per_turn: int, window: int | None = None) -> int:
    return sum(turn_input(n, per_turn, window) for n in range(1, turns + 1))


def build(P, H):
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    haiku = P("llm.anthropic.claude-haiku-5-5")
    judge = P("llm.openai.gpt-6-1-terra")
    emb = P("embed.openai.text-embedding-3-small")
    far = P("infra.aws.fargate.linux-x86-ap-south-1")
    rds_disk = P("vector.aws.rds-postgres-db-r6g-xlarge")

    def gen(tokens: int) -> float:
        return llm_cost(sonnet, tokens, ANSWER)

    # Per-turn generation cost: version 1 (full history with chunks) against version 2 on.
    v1_t5, v1_t10 = turn_input(5, TURN_FULL), turn_input(10, TURN_FULL)
    v2_t5, v2_t10 = turn_input(5, TURN_QA, WINDOW), turn_input(10, TURN_QA, WINDOW)
    first = turn_input(1, TURN_QA, WINDOW)
    long_v1 = thread_input(LONG_THREAD, TURN_FULL)
    long_v2 = thread_input(LONG_THREAD, TURN_QA, WINDOW)
    long_v1_cost = sum(gen(turn_input(n, TURN_FULL)) for n in range(1, LONG_THREAD + 1))
    long_v2_cost = sum(
        gen(turn_input(n, TURN_QA, WINDOW)) for n in range(1, LONG_THREAD + 1)
    )

    # Condenser input grows with the same window: system + summary + window + new message.
    condense_in = CONDENSE_SYSTEM + SUMMARY + WINDOW * TURN_QA + QUESTION
    q_condense = llm_cost(haiku, condense_in, CONDENSE_OUT)
    q_summarise = llm_cost(haiku, SUMMARISE_IN, SUMMARISE_OUT)

    # A typical follow-up (turn 2 to 4) and the mean over the assumed thread shape.
    mean_turn = 1 + (FOLLOWUPS_PER_THREAD + 1) / 2  # mean position of a follow-up
    followup_in = turn_input(round(mean_turn), TURN_QA, WINDOW)
    q_first = gen(first)
    q_follow = gen(followup_in)
    # A follow-up that starts a new topic is answered without history.
    q_follow_mix = (1 - STANDALONE_SHARE) * q_follow + STANDALONE_SHARE * gen(
        first + STANDALONE
    )

    # Caching the history prefix (O1): the most a cache read could save on one follow-up.
    hist_prefix = SYSTEM + SUMMARY + WINDOW * TURN_QA
    cache_max = hist_prefix * (sonnet["input"] - sonnet["cache_read"]) / 1e6
    cache_write_extra = hist_prefix * (sonnet["cache_write"] - sonnet["input"]) / 1e6
    # Break-even share of requests that must read the cache: each hit saves input - read,
    # each miss pays write - input extra.
    hit_save = sonnet["input"] - sonnet["cache_read"]
    miss_cost = sonnet["cache_write"] - sonnet["input"]
    cache_breakeven = miss_cost / (miss_cost + hit_save)

    v = {
        "turn_qa": f"{TURN_QA}",
        "turn_full": f"{TURN_FULL:,}",
        "first_in": f"{first:,}",
        "v1_t5": f"{v1_t5:,}",
        "v1_t10": f"{v1_t10:,}",
        "v2_t5": f"{v2_t5:,}",
        "v2_t10": f"{v2_t10:,}",
        "v1_t10_cost": usd(gen(v1_t10)),
        "v2_t10_cost": usd(gen(v2_t10)),
        "first_cost": usd(q_first),
        "long": f"{LONG_THREAD}",
        "long_v1": num(long_v1),
        "long_v2": num(long_v2),
        "long_v1_cost": usd(long_v1_cost),
        "long_v2_cost": usd(long_v2_cost),
        "long_ratio": f"{long_v1_cost / long_v2_cost:.1f}",
        "window": f"{WINDOW}",
        "summary": f"{SUMMARY}",
        "condense_in": f"{condense_in:,}",
        "condense_out": f"{CONDENSE_OUT}",
        "q_condense": usd(q_condense),
        "q_condense_sonnet": usd(llm_cost(sonnet, condense_in, CONDENSE_OUT)),
        "q_summarise": usd(q_summarise),
        "followup_in": f"{followup_in:,}",
        "q_follow": usd(q_follow),
        "q_follow_total": usd(q_follow + q_condense),
        "condense_share": pct(q_condense / (q_follow + q_condense)),
        "hist_prefix": f"{hist_prefix:,}",
        "cache_max": usd(cache_max),
        "cache_write_extra": usd(cache_write_extra),
        "cache_min": f"{512}",
        "cache_breakeven": pct(cache_breakeven),
        "cache_hit_save": f"${hit_save:.2f}",
        "cache_miss_cost": f"${miss_cost:.2f}",
        "state_kb": f"{STATE_BYTES / 1e3:.0f}",
        "msg_cap": f"{MSG_CAP}",
        "slice_dep": f"{SLICES['dependent']}",
        "slice_new": f"{SLICES['new_topic']}",
        "slice_about": f"{SLICES['about_answer']}",
        "slice_fix": f"{SLICES['correction']}",
        "ci_dep": f"{ci(SLICES['dependent']):.0f}",
        "ci_new": f"{ci(SLICES['new_topic']):.0f}",
        "ci_conv": f"{ci(CONV_TURNS):.1f}",
        "followup_share": pct(FOLLOWUP_SHARE),
        "followups_per_thread": f"{FOLLOWUPS_PER_THREAD}",
        "standalone_share": pct(STANDALONE_SHARE),
        "conv_threads": f"{CONV_THREADS}",
        "conv_turns": f"{CONV_TURNS}",
        "haiku_in": f"${haiku['input']:.2f}",
        "sonnet_in": f"${sonnet['input']:.2f}",
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
    for name, s in ch07.SCALES.items():
        firsts = (
            s["staff"] * ch07.WEEKLY_ACTIVE * ch07.QUESTIONS_PER_WEEK * WEEKS_PER_MONTH
        )
        follows = firsts * FOLLOWUP_SHARE * FOLLOWUPS_PER_THREAD
        # Turns that fall out of the window each need one summary update.
        long_tail = max(0, FOLLOWUPS_PER_THREAD - WINDOW)
        summaries = firsts * FOLLOWUP_SHARE * long_tail
        corpus = s["pages"] * ch07.TOKENS_PER_PAGE
        db = P(f"infra.aws.rds-postgres.{s['db']}")
        ingest_h = (
            ch07.SYNC_RUNS_PER_DAY * DAYS_PER_MONTH * s["sync_min"]
            + DAYS_PER_MONTH * s["reconcile_min"]
        ) / 60
        q_judge = llm_cost(judge, ch07.JUDGE_IN, ch07.JUDGE_OUT)
        r = {
            "gen_first": firsts * q_first,
            "gen_follow": follows * q_follow_mix,
            "condense": follows * q_condense + summaries * q_summarise,
            "embed": (
                corpus * ch07.MONTHLY_CHURN + (firsts + follows) * ch07.QUERY_EMBED
            )
            * emb["input"]
            / 1e6,
            "db": (db["hourly_multi_az"] if s["multi_az"] else db["hourly"])
            * HOURS_PER_MONTH
            + s["disk_gb"]
            * (
                rds_disk["gp3_storage_gb_month_multi_az"]
                if s["multi_az"]
                else rds_disk["gp3_storage_gb_month"]
            ),
            "app": s["tasks"]
            * HOURS_PER_MONTH
            * (s["vcpu"] * far["vcpu_hourly"] + s["mem_gb"] * far["gb_hourly"]),
            "ingest": ingest_h * (1 * far["vcpu_hourly"] + 2 * far["gb_hourly"]),
            "eval": EVAL_RUNS
            * (
                ch07.GOLDEN * (q_first + q_judge)
                + CONV_TURNS * (q_follow + q_condense + q_judge)
            ),
        }
        r["total"] = sum(r.values())
        r["turns"] = firsts + follows
        r["firsts"] = firsts
        r["follows"] = follows
        rows[name] = r
        v[f"{name}.firsts"] = num(firsts)
        v[f"{name}.follows"] = num(follows)
        v[f"{name}.turns"] = num(firsts + follows)
        v[f"{name}.total"] = usd(r["total"])
        v[f"{name}.per_turn"] = usd(r["total"] / (firsts + follows))
        v[f"{name}.added"] = usd(r["gen_follow"] + r["condense"])
        v[f"{name}.condense"] = usd(r["condense"])
        v[f"{name}.cache_cap"] = usd(follows * (1 - STANDALONE_SHARE) * cache_max)
        v[f"{name}.condense_sonnet"] = usd(follows * llm_cost(sonnet, condense_in, CONDENSE_OUT))
        v[f"{name}.state_mb_month"] = num(firsts * STATE_BYTES / 1e6)
        v[f"{name}.state_mb_held"] = num(firsts * STATE_BYTES / 1e6 * RETENTION_DAYS / DAYS_PER_MONTH)

    v["turn_ratio"] = f"{rows['pilot']['turns'] / rows['pilot']['firsts']:.2f}"

    g = rows["growth"]
    how = {
        "gen_first": f"{num(g['firsts'])} first questions x {usd(q_first)} (Sonnet 5.5, as System 7)",
        "gen_follow": f"{num(g['follows'])} follow-ups, {pct(STANDALONE_SHARE)} without history, the rest at {usd(q_follow)}",
        "condense": f"Haiku 5.5: {usd(q_condense)} per follow-up, {usd(q_summarise)} per summary update",
        "embed": "As System 7, plus one query embedding per follow-up",
        "db": f"As System 7; thread state adds {num(g['firsts'] * STATE_BYTES / 1e6)} MB a month",
        "app": "As System 7",
        "ingest": "As System 7",
        "eval": f"{EVAL_RUNS} runs x ({ch07.GOLDEN} first questions + {CONV_TURNS} conversation turns), judged",
    }
    labels = {
        "gen_first": "Generation, first questions",
        "gen_follow": "Generation, follow-ups",
        "condense": "Condensation and summaries",
        "embed": "Embeddings",
        "db": "Postgres with pgvector",
        "app": "API and Slack bot",
        "ingest": "Ingestion worker",
        "eval": "Evaluation runs",
    }
    for k, label in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in ch07.SCALES), how[k])
    cost.add(
        "**Total per month**",
        *(f"**{usd(rows[s]['total'])}**" for s in ch07.SCALES),
        "",
    )
    cost.add(
        "Cost per turn",
        *(usd(rows[s]["total"] / rows[s]["turns"]) for s in ch07.SCALES),
        "",
    )
    cost.add(
        "Turns per month",
        *(num(rows[s]["turns"]) for s in ch07.SCALES),
        f"System 7's questions, plus {pct(FOLLOWUP_SHARE)} x {FOLLOWUPS_PER_THREAD} follow-ups",
    )

    growth_t = Table(
        "Turn",
        "Version 1: history with chunks",
        "Version 2: window and summary",
        align="lrr",
    )
    for n in (1, 2, 3, 5, 10, 15):
        growth_t.add(
            str(n),
            f"{turn_input(n, TURN_FULL):,} tokens, {usd(gen(turn_input(n, TURN_FULL)))}",
            f"{turn_input(n, TURN_QA, WINDOW):,} tokens, {usd(gen(turn_input(n, TURN_QA, WINDOW)))}",
        )
    growth_t.add(
        f"**Whole {LONG_THREAD}-turn thread**",
        f"**{num(long_v1)} tokens, {usd(long_v1_cost)}**",
        f"**{num(long_v2)} tokens, {usd(long_v2_cost)}**",
    )
    return {"tables": {"monthly": cost.md(), "growth": growth_t.md()}, "vars": v}
