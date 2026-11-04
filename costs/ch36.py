"""System 36: LLM gateway. Worked scenario numbers and monthly cost.

Halden, a B2B software company in AWS us-east-1, has five features that call hosted models,
built by four teams. A platform team puts one gateway in front of every model call. Every
volume, size, share, limit, duration and money value below that is not a price is an
assumption, labelled as such in the chapter. Nothing here was measured: provider availability,
gateway overhead and failover success are what production shows, so this module computes token
rates against limits, costs, burn rates and the break-even values a decision turns on.
"""

from _lib import HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch36"

# Traffic shape (assumptions): a B2B product used in business hours, F2's rule for the peak.
WORKING_DAYS = 22
BUSINESS_HOURS = 8
BUSY_SHARE = 0.8  # share of a day's interactive traffic inside business hours
PEAK_FACTOR = 2  # busiest hour against the business-hours average

# Features at the growth scale (assumptions). Tokens per request: a shared prefix (tools,
# system prompt) cached per feature, conversation history cached per conversation, new input
# uncached, output.
FEATURES = {
    # key: label, team, model, kind, requests a month, shared prefix, history, new, out
    "support": dict(
        label="Support assistant",
        team="Support tools",
        model="sonnet",
        kind="interactive",
        req=400_000,
        prefix=3_000,
        history=2_000,
        new=1_500,
        out=400,
    ),
    "search": dict(
        label="Search answers",
        team="Search",
        model="haiku",
        kind="interactive",
        req=1_500_000,
        prefix=2_000,
        history=0,
        new=3_000,
        out=300,
    ),
    "extract": dict(
        label="Document extraction",
        team="Documents",
        model="haiku",
        kind="async",
        req=2_000_000,
        prefix=1_500,
        history=0,
        new=4_000,
        out=500,
    ),
    "drafts": dict(
        label="Email drafts",
        team="Growth",
        model="sonnet",
        kind="interactive",
        req=150_000,
        prefix=1_500,
        history=0,
        new=2_000,
        out=600,
    ),
    "evals": dict(
        label="Evaluation runs",
        team="CI, all teams",
        model="sonnet",
        kind="offline",
        req=100_000,
        prefix=0,
        history=0,
        new=6_000,
        out=300,
    ),
}
JUDGE_IN, JUDGE_OUT = (
    2_500,
    200,
)  # each evaluation request is judged once, GPT-6.1 Terra

SCALES = {
    # traffic multiplier, gateway tasks (1 vCPU, 2 GB each)
    "small": dict(mult=0.3, tasks=3),
    "growth": dict(mult=1.0, tasks=3),
    "large": dict(mult=5.0, tasks=6),
}
TASK_VCPU, TASK_GB = 1, 2
LOG_BYTES = (
    1_500  # one usage record per request: ids, model, provider, tokens, cost, timings
)
EGRESS_TASKS, EGRESS_VCPU, EGRESS_GB = 3, 0.5, 1
LOG_MONTHS = 13  # retained a year and a month, so a year-on-year comparison has data

# Organisation limits on the Claude API (assumptions: read your own on the console; the
# rate-limits page lists 2M ITPM and 400K OTPM for both models at the Start tier).
LIMITS = {
    "sonnet": dict(itpm=4_000_000, otpm=800_000),
    "haiku": dict(itpm=4_000_000, otpm=800_000),
}

# Break 1: a backfill (assumptions).
BACKFILL_DOCS = 150_000  # a new customer's archive, submitted at once
BACKFILL_WORKERS = 100  # extraction workers, each with one call in flight
CALL_SECONDS = 6  # one extraction call, end to end
BATCH_SHARE = (
    0.4  # share of extraction that can wait for the batch API (backfills, re-runs)
)
HEADROOM = 0.25  # share of each limit no class may plan to use
BURST = 1.5  # an interactive feature's bucket: its busiest-hour rate times this, for minute bursts
OFFLINE_SYNC_SHARE = 0.10  # share of Sonnet ITPM left for synchronous release-gate runs
EVAL_SWEEP = 20_000  # one full evaluation sweep, requests (assumed)
EVALS_BATCH_SHARE = 0.8  # share of evaluation requests that can wait for the batch API
MAX_TASKS = {"growth": 4, "large": 8}  # most gateway tasks that may run, deploy surge included

# Break 2: an outage (assumptions; scenario inputs, not a statement about any provider).
OUTAGE_HOURS = 2
CLIENT_ATTEMPTS = 3  # the caller's SDK: one try and two retries
GATEWAY_ATTEMPTS = 3  # version 0's proxy: the same
RETRY_BUDGET = 0.10  # version 2: retries at most a tenth of first attempts
CACHE_MINUTES = 5  # the default cache lifetime
AVAIL = 0.995  # an availability for the arithmetic only, not a figure for any provider

# Break 3: a leaked key (assumption: an attacker runs at the organisation's limits).
LEAK_HOURS_UNNOTICED = 14  # a Friday evening to Saturday morning

# Break 4: attribution.
UNTAGGED_SHARE = (
    0.12  # share of version 0 spend with no feature: scripts, notebooks, shared keys
)


def per_minute_peak(req_month: float) -> float:
    """Requests a minute in the busiest hour of a business day."""
    day = req_month / WORKING_DAYS
    return day * BUSY_SHARE / BUSINESS_HOURS * PEAK_FACTOR / 60


def req_cost(p, f, warm=True):
    """USD for one request. Warm: shared prefix and history are cache reads. Cold: they are
    written again, as after a switch to a provider that has never seen them."""
    cached = f["prefix"] + f["history"]
    if warm:
        return llm_cost(p, f["new"], f["out"], cache_read=cached)
    return llm_cost(p, f["new"], f["out"], cache_write=cached)


def req_cost_naive(p, f):
    """What a gateway that prices every input token at the input price would record."""
    return llm_cost(p, f["prefix"] + f["history"] + f["new"], f["out"])


def build(P, H):
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    haiku = P("llm.anthropic.claude-haiku-5-5")
    br_sonnet = P("llm.bedrock.claude-sonnet-5-5")
    judge = P("llm.openai.gpt-6-1-terra")
    far = P("infra.aws.fargate.linux-x86")
    valkey = P("infra.aws.elasticache.cache-r7g-large")
    db = P("infra.aws.rds-postgres.db-t4g-medium")
    s3 = P("infra.aws.s3.standard")
    models = {"sonnet": sonnet, "haiku": haiku}

    v: dict[str, str] = {}
    g = FEATURES

    # Per-request and monthly model cost per feature, growth.
    q = {k: req_cost(models[f["model"]], f) for k, f in g.items()}
    q_judge = llm_cost(judge, JUDGE_IN, JUDGE_OUT)
    month = {k: g[k]["req"] * q[k] for k in g}
    month["evals"] += g["evals"]["req"] * q_judge
    model_total_g = sum(month.values())

    feat = Table(
        "Feature (team)",
        "Model",
        "Kind",
        "Requests a month",
        "Tokens in: cached + new",
        "Out",
        "Cost a request",
        "Model cost a month",
        align="lllrrrrr",
    )
    for k, f in g.items():
        name = "Sonnet 5.5" if f["model"] == "sonnet" else "Haiku 5.5"
        if k == "evals":
            name += " + GPT-6.1 Terra judge"
        cached = f["prefix"] + f["history"]
        feat.add(
            f"{f['label']} ({f['team']})",
            name,
            f["kind"],
            num(f["req"]),
            f"{cached:,} + {f['new']:,}",
            f"{f['out']}",
            usd(q[k] + (q_judge if k == "evals" else 0)),
            usd(month[k]),
        )
    feat.add(
        "**All five**",
        "",
        "",
        f"**{num(sum(f['req'] for f in g.values()))}**",
        "",
        "",
        "",
        f"**{usd(model_total_g)}**",
    )
    for k, f in g.items():
        v[f"{k}.req"] = num(f["req"])
        v[f"{k}.q"] = usd(q[k])
        v[f"{k}.month"] = usd(month[k])
        v[f"{k}.prefix"] = f"{f['prefix']:,}"
        v[f"{k}.history"] = f"{f['history']:,}"
        v[f"{k}.new"] = f"{f['new']:,}"
        v[f"{k}.out"] = f"{f['out']}"
        v[f"{k}.peak_min"] = f"{per_minute_peak(f['req']):.0f}"
    v["model_total"] = usd(model_total_g)
    v["requests"] = num(sum(f["req"] for f in g.values()))
    v["working_days"] = f"{WORKING_DAYS}"
    v["business_hours"] = f"{BUSINESS_HOURS}"
    v["busy_share"] = pct(BUSY_SHARE)
    v["peak_factor"] = f"{PEAK_FACTOR}"

    # --- Break 1: one class takes the shared limit.
    lim_h = LIMITS["haiku"]
    peak = {k: per_minute_peak(f["req"]) for k, f in g.items()}
    # Only uncached input counts toward ITPM for current Claude models; warm caches assumed.
    itpm = {k: peak[k] * g[k]["new"] for k in g}
    otpm = {k: peak[k] * g[k]["out"] for k in g}
    haiku_peak_itpm = itpm["search"] + itpm["extract"]
    bf_rate = BACKFILL_WORKERS * 60 / CALL_SECONDS  # requests a minute
    bf_itpm = bf_rate * g["extract"]["new"]
    bf_minutes = BACKFILL_DOCS / bf_rate
    with_bf = haiku_peak_itpm + bf_itpm
    search_429_req = (
        peak["search"] * bf_minutes
    )  # search requests while the bucket is drained
    v.update(
        {
            "lim_itpm": num(lim_h["itpm"]),
            "lim_otpm": num(lim_h["otpm"]),
            "search_itpm": num(itpm["search"]),
            "extract_itpm": num(itpm["extract"]),
            "haiku_peak_itpm": num(haiku_peak_itpm),
            "haiku_peak_share": pct(haiku_peak_itpm / lim_h["itpm"]),
            "bf_docs": num(BACKFILL_DOCS),
            "bf_workers": f"{BACKFILL_WORKERS}",
            "call_seconds": f"{CALL_SECONDS}",
            "bf_rate": num(bf_rate),
            "bf_itpm": num(bf_itpm),
            "bf_share": pct(bf_itpm / lim_h["itpm"]),
            "bf_minutes": f"{bf_minutes:.0f}",
            "with_bf": num(with_bf),
            "with_bf_x": f"{with_bf / lim_h['itpm']:.1f}",
            "search_429_req": num(search_429_req),
            "bf_be_workers": f"{(lim_h['itpm'] - haiku_peak_itpm) / (60 / CALL_SECONDS * g['extract']['new']):.0f}",
            "search_cached_itpm": num(
                peak["search"] * (g["search"]["new"] + g["search"]["prefix"])
            ),
        }
    )
    # Version 1: budgets per class. Interactive gets its peak; async gets what is left after
    # headroom; the batch API takes what can wait.
    plan = lim_h["itpm"] * (1 - HEADROOM)
    search_bucket = itpm["search"] * BURST
    async_budget = plan - search_bucket
    async_rate = (
        async_budget / g["extract"]["new"]
    )  # requests a minute the budget allows
    bf_minutes_v1 = BACKFILL_DOCS / max(async_rate - peak["extract"], 1)
    extract_batch_saving = month["extract"] * BATCH_SHARE * haiku["batch_discount"]
    v.update(
        {
            "headroom": pct(HEADROOM),
            "plan_itpm": num(plan),
            "async_budget": num(async_budget),
            "burst": f"{BURST:g}",
            "search_bucket": num(search_bucket),
            "local_share_g": f"1/{MAX_TASKS['growth']}",
            "max_tasks_g": f"{MAX_TASKS['growth']}",
            "max_tasks_l": f"{MAX_TASKS['large']}",
            "async_rate": num(async_rate),
            "bf_minutes_v1": f"{bf_minutes_v1:.0f}",
            "bf_hours_v1": f"{bf_minutes_v1 / 60:.1f}",
            "batch_share": pct(BATCH_SHARE),
            "batch_saving": usd(extract_batch_saving),
            "bf_sync_cost": usd(BACKFILL_DOCS * q["extract"]),
            "bf_batch_cost": usd(
                BACKFILL_DOCS * q["extract"] * (1 - haiku["batch_discount"])
            ),
        }
    )

    # Sonnet: interactive buckets, and the offline class (evaluation).
    lim_s = LIMITS["sonnet"]
    son_inter = (itpm["support"] + itpm["drafts"]) * BURST
    sweep_itpm = BACKFILL_WORKERS * 60 / CALL_SECONDS * g["evals"]["new"]
    offline_sync = lim_s["itpm"] * OFFLINE_SYNC_SHARE
    offline_rate = offline_sync / g["evals"]["new"]
    evals_sonnet = g["evals"]["req"] * q["evals"]
    evals_batch_saving = evals_sonnet * EVALS_BATCH_SHARE * sonnet["batch_discount"]
    v.update(
        {
            "son_inter_bucket": num(son_inter),
            "son_inter_share": pct(son_inter / lim_s["itpm"]),
            "eval_sweep": num(EVAL_SWEEP),
            "sweep_itpm": num(sweep_itpm),
            "sweep_x": f"{sweep_itpm / lim_s['itpm']:.1f}",
            "offline_sync_share": pct(OFFLINE_SYNC_SHARE),
            "offline_sync": num(offline_sync),
            "offline_rate": num(offline_rate),
            "evals_batch_share": pct(EVALS_BATCH_SHARE),
            "evals_batch_saving": usd(evals_batch_saving),
        }
    )

    # --- Break 2: a provider outage.
    sup = g["support"]
    sup_peak_hour = peak["support"] * 60
    failed = sup_peak_hour * OUTAGE_HOURS
    v.update(
        {
            "outage_hours": f"{OUTAGE_HOURS}",
            "sup_peak_hour": num(sup_peak_hour, 3),
            "failed": num(failed, 3),
            "drafts_failed": num(peak["drafts"] * 60 * OUTAGE_HOURS),
            "client_attempts": f"{CLIENT_ATTEMPTS}",
            "gateway_attempts": f"{GATEWAY_ATTEMPTS}",
            "amplify": f"{CLIENT_ATTEMPTS * GATEWAY_ATTEMPTS}",
            "storm_min": num(peak["support"] * CLIENT_ATTEMPTS * GATEWAY_ATTEMPTS),
            "retry_budget": pct(RETRY_BUDGET),
            "budget_amp": f"{1 + RETRY_BUDGET:.1f}",
            "avail": f"{AVAIL * 100:.1f}%",
            "avail_min": f"{(1 - AVAIL) * 30.4 * 24 * 60:.0f}",
            "avail_bus_min": f"{(1 - AVAIL) * WORKING_DAYS * BUSINESS_HOURS * 60:.0f}",
            "avail_both": f"{(1 - (1 - AVAIL) ** 2) * 100:.4f}%",
            "avail_both_min": f"{(1 - AVAIL) ** 2 * 30.4 * 24 * 60:.1f}",
        }
    )
    # Cache forfeited on a switch: every request in the first cache lifetime writes its
    # prefix and history again on the new provider.
    geo = {
        "input": br_sonnet["geo_input"],
        "output": br_sonnet["geo_output"],
        "cache_read": br_sonnet["geo_cache_read"],
        "cache_write": br_sonnet["geo_cache_write"],
    }
    warm = llm_cost(geo, sup["new"], sup["out"], cache_read=sup["prefix"] + sup["history"])
    # Cold for a conversation: the shared prefix is already re-written by the first request
    # after the switch; the conversation's own history is written again.
    cold = llm_cost(geo, sup["new"], sup["out"], cache_read=sup["prefix"], cache_write=sup["history"])
    window = peak["support"] * CACHE_MINUTES
    switch_extra = window * (cold - warm) + sup["prefix"] * (geo["cache_write"] - geo["cache_read"]) / 1e6
    cold_itpm = peak["support"] * (sup["new"] + sup["history"]) + sup["prefix"]
    ceil_itpm = peak["support"] * (sup["new"] + sup["prefix"] + sup["history"])
    warm_itpm = peak["support"] * sup["new"]
    v.update(
        {
            "q_warm": usd(warm),
            "q_cold": usd(cold),
            "cold_x": f"{cold / warm:.1f}",
            "cache_minutes": f"{CACHE_MINUTES}",
            "switch_window": num(window),
            "switch_extra": usd(switch_extra),
            "warm_itpm": num(warm_itpm),
            "cold_itpm": num(cold_itpm),
            "cold_itpm_x": f"{cold_itpm / warm_itpm:.1f}",
            "ceil_itpm": num(ceil_itpm),
            "ceil_itpm_x": f"{ceil_itpm / warm_itpm:.1f}",
            "fb_itpm": num(
                ceil_itpm
                + peak["drafts"] * (g["drafts"]["new"] + g["drafts"]["prefix"])
            ),
            "q_claude_warm": usd(q["support"]),
            "fb_otpm": num(otpm["support"] + otpm["drafts"]),
        }
    )

    # --- Break 3: a leaked key at the organisation's limits.
    def burn_minute(p, lim):
        # Cache writes count toward ITPM and cost more than input, so they set the ceiling.
        return (lim["itpm"] * p["cache_write"] + lim["otpm"] * p["output"]) / 1e6

    burn_s = burn_minute(sonnet, LIMITS["sonnet"])
    burn_h = burn_minute(haiku, LIMITS["haiku"])
    burn_hour = (burn_s + burn_h) * 60
    leak = burn_hour * LEAK_HOURS_UNNOTICED
    working_day = {k: month[k] / WORKING_DAYS for k in g}
    # A stolen drafts identity is held to the drafts feature's own bucket, at the write price.
    id_burn = (itpm["drafts"] * BURST * sonnet["cache_write"] + otpm["drafts"] * BURST * sonnet["output"]) / 1e6
    v.update(
        {
            "sonnet_itpm": num(LIMITS["sonnet"]["itpm"]),
            "sonnet_otpm": num(LIMITS["sonnet"]["otpm"]),
            "burn_s_min": usd(burn_s),
            "burn_h_min": usd(burn_h),
            "burn_hour": usd(burn_hour),
            "leak_hours": f"{LEAK_HOURS_UNNOTICED}",
            "leak": usd(leak),
            "leak_share": pct(leak / model_total_g),
            "leak_day": usd(burn_hour * 24),
            "cap_hours": f"{model_total_g / burn_hour:.0f}",
            "drafts_day": usd(working_day["drafts"]),
            "drafts_cap": usd(2 * working_day["drafts"]),
            "id_burn_hour": usd(id_burn * 60),
            "drafts_cap_hours": f"{2 * working_day['drafts'] / id_burn / 60:.0f}",
            "burn_s_1h_min": usd((LIMITS["sonnet"]["itpm"] * sonnet["cache_write_1h"] + LIMITS["sonnet"]["otpm"] * sonnet["output"]) / 1e6),
        }
    )

    # --- Break 4: attribution.
    naive = {k: req_cost_naive(models[g[k]["model"]], g[k]) for k in g}
    naive_month = sum(g[k]["req"] * naive[k] for k in g) + g["evals"]["req"] * q_judge
    v.update(
        {
            "sup_naive": usd(naive["support"]),
            "sup_naive_x": f"{naive['support'] / q['support']:.1f}",
            "naive_total": usd(naive_month),
            "naive_over": pct(naive_month / model_total_g - 1),
            "search_naive_x": f"{naive['search'] / q['search']:.1f}",
            "extract_naive_x": f"{naive['extract'] / q['extract']:.1f}",
            "untagged": pct(UNTAGGED_SHARE),
            "untagged_usd": usd(UNTAGGED_SHARE * model_total_g),
            "sonnet_cache_read_ratio": pct(sonnet["cache_read"] / sonnet["input"]),
            "br_geo_premium": pct(br_sonnet["geo_input"] / br_sonnet["input"] - 1),
        }
    )

    # Attribution table: what each approach records for one support request.
    attr = Table(
        "How cost is computed", "Support request", "Month, all features", align="lrr"
    )
    attr.add(
        "Every input token at the input price", usd(naive["support"]), usd(naive_month)
    )
    attr.add(
        "From the usage fields, with cache reads at the read price",
        usd(q["support"]),
        usd(model_total_g),
    )
    attr.add(
        "Same request served on Bedrock, Geo or in-region price",
        usd(
            llm_cost(
                {
                    "input": br_sonnet["geo_input"],
                    "output": br_sonnet["geo_output"],
                    "cache_read": br_sonnet["geo_cache_read"],
                },
                sup["new"],
                sup["out"],
                cache_read=sup["prefix"] + sup["history"],
            )
        ),
        "",
    )

    # --- Cost of the gateway itself, three scales.
    cost = Table(
        "Line item",
        "Small",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rows = {}
    for s, sc in SCALES.items():
        reqs = sum(f["req"] for f in g.values()) * sc["mult"]
        log_gb = reqs * LOG_BYTES / 1e9 * LOG_MONTHS
        r = {
            "gw": sc["tasks"]
            * HOURS_PER_MONTH
            * (TASK_VCPU * far["vcpu_hourly"] + TASK_GB * far["gb_hourly"]),
            "egress": EGRESS_TASKS
            * HOURS_PER_MONTH
            * (EGRESS_VCPU * far["vcpu_hourly"] + EGRESS_GB * far["gb_hourly"]),
            "valkey": 2 * valkey["hourly_valkey"] * HOURS_PER_MONTH,
            "db": db["hourly_multi_az"] * HOURS_PER_MONTH,
            "logs": log_gb * s3["storage_gb_month_first_50tb"]
            + reqs
            / 1000
            * s3["put_per_1000"]
            / 1000,  # records written in batches of 1,000
        }
        r["total"] = sum(r.values())
        r["model"] = model_total_g * sc["mult"]
        rows[s] = r
        v[f"{s}.gw_total"] = usd(r["total"])
        v[f"{s}.model"] = usd(r["model"])
        v[f"{s}.share"] = f"{r['total'] / r['model'] * 100:.1f}%"
        v[f"{s}.requests"] = num(reqs)
        v[f"{s}.per_m"] = usd(r["total"] / reqs * 1e6)
    labels = {
        "gw": (
            "Gateway tasks",
            f"{SCALES['growth']['tasks']} Fargate tasks of {TASK_VCPU} vCPU and {TASK_GB} GB across three zones (six at large)",
        ),
        "egress": (
            "Egress proxy",
            f"{EGRESS_TASKS} Fargate tasks of {EGRESS_VCPU} vCPU and {EGRESS_GB} GB, one per zone",
        ),
        "valkey": (
            "Valkey",
            "Two cache.r7g.large nodes: rate-limit buckets, breaker state, budgets",
        ),
        "db": (
            "Postgres",
            "db.t4g.medium Multi-AZ: keys, routes, budgets, price table, usage ledger rollups",
        ),
        "logs": (
            "Usage records in S3",
            f"{LOG_BYTES:,} bytes a request, kept {LOG_MONTHS} months, written in batches",
        ),
    }
    for k, (label, how) in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in SCALES), how)
    cost.add("**Gateway total**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), "")
    cost.add(
        "Model spend through it",
        *(usd(rows[s]["model"]) for s in SCALES),
        "The feature table, scaled",
    )
    cost.add(
        "Gateway as a share of model spend",
        *(f"{rows[s]['total'] / rows[s]['model'] * 100:.1f}%" for s in SCALES),
        "",
    )
    cost.add(
        "Gateway cost per million requests",
        *(
            usd(
                rows[s]["total"]
                / (sum(f["req"] for f in g.values()) * SCALES[s]["mult"])
                * 1e6
            )
            for s in SCALES
        ),
        "",
    )

    v["batch_saving_share"] = pct(extract_batch_saving / model_total_g)
    v["batch_be_share"] = pct(rows["growth"]["total"] / (month["extract"] * haiku["batch_discount"]))

    return {
        "tables": {
            "features": feat.md(),
            "attribution": attr.md(),
            "monthly": cost.md(),
        },
        "vars": v,
    }
