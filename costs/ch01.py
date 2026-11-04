"""System 1: Support ticket classifier and router. Worked scenario numbers and monthly cost.

Paybridge is a worked scenario. Every volume, share, time and money value below that is not
a price is an assumption, labelled as such in the chapter. Nothing here was measured: the
coverage and precision a model reaches are what the golden set decides, so this module only
computes costs per ticket and the break-even values those results must clear.
"""

import math

from _lib import DAYS_PER_MONTH, HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch01"

# Volumes (assumptions). Growth is Paybridge's whole volume today; pilot is one product line.
SCALES = {
    "pilot": dict(tickets=8_000, tasks=2, vcpu=0.5, mem_gb=1, multi_az=False),
    "growth": dict(tickets=60_000, tasks=2, vcpu=0.5, mem_gb=1, multi_az=False),
    "large": dict(tickets=250_000, tasks=2, vcpu=1, mem_gb=2, multi_az=True),
}
WORKING_DAYS = 22  # a month of weekdays
BUSINESS_HOURS = 10  # 9:00 to 19:00 IST
IN_HOURS_SHARE = 0.85  # share of tickets created in business hours
PEAK_FACTOR = 2  # busiest hour against the business-hours average (F2's rule of thumb)

# Categories and queues (assumptions).
CATEGORIES = 14
QUEUES = 6

# People's time and what a mistake costs (assumptions).
AGENT_HOURLY = 15.0  # USD, loaded cost of a support agent in Pune
TRIAGE_MIN = 1.0  # minutes for a person to read and route one ticket
WRONG_QUEUE_MIN = 2.0  # minutes the wrong team spends before sending it back
MISROUTE = (
    2.00  # USD, the support lead's value: agent time plus the customer's extra wait
)
PRECISION_TARGET = 0.95  # share of auto-routed tickets in the right queue (target)
COVERAGE_TARGET = (
    0.70  # share of tickets auto-routed (the support lead's goal, a target)
)
MISROUTE_CASES = (None, 1.00, 2.00, 4.00)  # None = agent time only

# Tokens per ticket (assumptions; count your own with the token-counting endpoint).
SYSTEM = 2_400  # instructions, 14 category definitions, two short examples each
TICKET = 450  # subject and body after quoted replies and signatures are stripped
TICKET_CAP = 2_000  # longer bodies are cut before the call
JSON_OUT = 60  # category, confidence, language, a one-line reason
THINK = 150  # thinking tokens at low effort, billed as output (assumption; read usage)
FREE_TEXT_OUT = 20  # version 0's one-word answer, plus thinking

# Evaluation (assumptions).
GOLDEN = 2_000
HALF = GOLDEN // 2  # tuning half and held-out half
DOUBLE_LABEL = HALF  # the held-out half is labelled twice and adjudicated
MIN_PER_CATEGORY = 100  # stratified: at least this many per category in the whole set
LABEL_MIN = 1.0  # minutes for a person to assign a category to one ticket (assumption)
ENABLE_ROUTED = 30  # routed tuning tickets a category needs, with zero errors, to be enabled
TRUE_P_EXAMPLE = 0.97  # a category's true precision, for the pass-probability example
HUMAN_ERR_EXAMPLE = 0.05  # example human misroute rate; measure it from the history
SURGE = 3  # payroll-week rate against an ordinary peak (assumption)
EVAL_RUNS = 20  # a month: one per prompt or model change, plus weekly

# Sizing assumption for the second stage of the cascade; replace with the measured share.
ESCALATED = 0.30

CACHE_TTL_MIN = 5


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion p on n trials."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build(P, H):
    models = {
        "Haiku 5.5": P("llm.anthropic.claude-haiku-5-5"),
        "Sonnet 5.5": P("llm.anthropic.claude-sonnet-5-5"),
        "Opus 5.5": P("llm.anthropic.claude-opus-5-5"),
        "Fable 5.1": P("llm.anthropic.claude-fable-5-1"),
    }
    haiku, sonnet = models["Haiku 5.5"], models["Sonnet 5.5"]
    far = P("infra.aws.fargate.linux-x86-ap-south-1")
    db = P("infra.aws.rds-postgres.db-t4g-medium-ap-south-1")
    disk = P("vector.aws.rds-postgres-db-r6g-xlarge")

    out = JSON_OUT + THINK
    inp = SYSTEM + TICKET

    def q(m: dict, cached_hit: float | None = None) -> float:
        """Cost of one classification. cached_hit: probability the prefix is in cache."""
        if cached_hit is None:
            return llm_cost(m, inp, out)
        hit = llm_cost(m, TICKET, out, cache_read=SYSTEM)
        miss = llm_cost(m, TICKET, out, cache_write=SYSTEM)
        return cached_hit * hit + (1 - cached_hit) * miss

    # People's costs per ticket and the break-even precision.
    triage = TRIAGE_MIN / 60 * AGENT_HOURLY
    misroute_time = (WRONG_QUEUE_MIN + TRIAGE_MIN) / 60 * AGENT_HOURLY

    def breakeven(c_mis: float) -> float:
        return 1 - triage / c_mis

    p_star = breakeven(MISROUTE)
    save_auto = (
        triage - (1 - PRECISION_TARGET) * MISROUTE
    )  # per auto-routed ticket at target

    # Arrivals and cache warmth (Poisson arrivals, an assumption): a request finds the
    # prefix cached if the previous one came within the cache lifetime.
    def rates(tickets: float) -> tuple[float, float]:
        in_min = WORKING_DAYS * BUSINESS_HOURS * 60
        all_min = DAYS_PER_MONTH * 24 * 60
        lam_in = tickets * IN_HOURS_SHARE / in_min
        lam_out = tickets * (1 - IN_HOURS_SHARE) / (all_min - in_min)
        return lam_in, lam_out

    def hit_rate(tickets: float, share: float = 1.0) -> float:
        lam_in, lam_out = rates(tickets * share)
        h_in = 1 - math.exp(-lam_in * CACHE_TTL_MIN)
        h_out = 1 - math.exp(-lam_out * CACHE_TTL_MIN)
        return IN_HOURS_SHARE * h_in + (1 - IN_HOURS_SHARE) * h_out

    g = SCALES["growth"]["tickets"]
    lam_in_g, lam_out_g = rates(g)
    hit_g = hit_rate(g)

    v = {
        "categories": f"{CATEGORIES}",
        "queues": f"{QUEUES}",
        "agent_hourly": f"${AGENT_HOURLY:.0f}",
        "triage_min": f"{TRIAGE_MIN:g}",
        "wrong_min": f"{WRONG_QUEUE_MIN:g}",
        "triage": usd(triage),
        "misroute_time": usd(misroute_time),
        "misroute": usd(MISROUTE),
        "p_star": f"{p_star * 100:.1f}%",
        "p_star_time": pct(breakeven(misroute_time)),
        "precision_target": pct(PRECISION_TARGET),
        "coverage_target": pct(COVERAGE_TARGET),
        "save_auto": usd(save_auto),
        "miss_allow": f"${(1 - PRECISION_TARGET) * MISROUTE:.2f}",
        "system": f"{SYSTEM:,}",
        "ticket": f"{TICKET}",
        "ticket_cap": f"{TICKET_CAP:,}",
        "json_out": f"{JSON_OUT}",
        "think": f"{THINK}",
        "out": f"{out}",
        "inp": f"{inp:,}",
        "golden": f"{GOLDEN:,}",
        "surge": f"{SURGE}",
        "half": f"{HALF}",
        "min_per_cat": f"{MIN_PER_CATEGORY}",
        "min_per_half": f"{MIN_PER_CATEGORY // 2}",
        "double_label": f"{DOUBLE_LABEL}",
        "eval_runs": f"{EVAL_RUNS}",
        "escalated": pct(ESCALATED),
        "cache_ttl": f"{CACHE_TTL_MIN}",
        "growth_lam_in": f"{lam_in_g:.1f}",
        "growth_gap_in": f"{60 / lam_in_g:.0f}",
        "growth_lam_out": f"{lam_out_g:.2f}",
        "growth_hit": pct(hit_g),
        "triage_hours": num(g * TRIAGE_MIN / 60),
        "triage_fte": f"{g * TRIAGE_MIN / 60 / (WORKING_DAYS * 8):.1f}",
        "triage_month": usd(g * triage),
        "v0_cost": usd(llm_cost(haiku, inp, FREE_TEXT_OUT + THINK)),
        "q_haiku": usd(q(haiku)),
        "q_haiku_cached": usd(q(haiku, hit_g)),
        "q_sonnet": usd(q(sonnet)),
        "q_sonnet_cached": usd(q(sonnet, hit_rate(g, ESCALATED))),
        "haiku_think_share": pct(THINK * haiku["output"] / 1e6 / q(haiku)),
        "ci_lo": f"{wilson(PRECISION_TARGET, int(HALF * COVERAGE_TARGET))[0] * 100:.1f}%",
        "ci_hi": f"{wilson(PRECISION_TARGET, int(HALF * COVERAGE_TARGET))[1] * 100:.1f}%",
        "n_auto": f"{int(HALF * COVERAGE_TARGET)}",
        "cat_lo": f"{wilson(PRECISION_TARGET, int(MIN_PER_CATEGORY / 2 * COVERAGE_TARGET))[0] * 100:.0f}%",
        "cat_n": f"{int(MIN_PER_CATEGORY / 2 * COVERAGE_TARGET)}",
        "enable_n": f"{ENABLE_ROUTED}",
        "enable_bound": pct(1 - 3 / ENABLE_ROUTED),
        "enable_pass": pct(TRUE_P_EXAMPLE**ENABLE_ROUTED),
        "true_p": pct(TRUE_P_EXAMPLE),
        "enable_60": pct(1 - 3 / 60),
        "label_hours": f"{(GOLDEN + DOUBLE_LABEL) * LABEL_MIN / 60:.0f}",
        "human_err": pct(HUMAN_ERR_EXAMPLE),
        "p_star_human": f"{(p_star - HUMAN_ERR_EXAMPLE) * 100:.1f}%",
        "net_per_10": usd(SCALES["growth"]["tickets"] * 0.10 * save_auto),
        "label_min": f"{LABEL_MIN:g}",
        "save_auto_h": usd(save_auto + HUMAN_ERR_EXAMPLE * MISROUTE),
        "extra_sonnet_c_h": f"{(q(sonnet, hit_g) - q(haiku, hit_g)) / (save_auto + HUMAN_ERR_EXAMPLE * MISROUTE) * 100:.1f}",
        "extra_fable_c_h": f"{(q(models['Fable 5.1'], hit_g) - q(haiku, hit_g)) / (save_auto + HUMAN_ERR_EXAMPLE * MISROUTE) * 100:.1f}",
        "escalated_tune": f"{int(HALF * ESCALATED)}",
        "rare_share": pct(MIN_PER_CATEGORY / 2 / HALF),
        "per_point_auto": usd(g * 0.01 * MISROUTE),
        "per_point_auto_n": num(g * 0.01),
    }

    # Peak traffic against the helpdesk's update limit.
    for name, s in SCALES.items():
        lam_in, _ = rates(s["tickets"])
        v[f"{name}.tickets"] = num(s["tickets"])
        v[f"{name}.peak_min"] = f"{lam_in * PEAK_FACTOR:.0f}"
        v[f"{name}.surge_min"] = f"{lam_in * PEAK_FACTOR * SURGE:.0f}"
        v[f"{name}.per_10pts"] = usd(s["tickets"] * 0.10 * triage)
        v[f"{name}.hit"] = pct(hit_rate(s["tickets"]))
        v[f"{name}.sonnet_hit"] = pct(hit_rate(s["tickets"], ESCALATED))
    v["pilot.gap_in"] = f"{60 / rates(SCALES['pilot']['tickets'])[0]:.0f}"
    v["pilot.gap_out_min"] = f"{1 / rates(SCALES['pilot']['tickets'])[1]:.0f}"

    # Break-even precision against the value put on a misroute.
    be = Table(
        "Cost of one misroute", "Break-even precision", "Read it as", align="lrl"
    )
    for c in MISROUTE_CASES:
        c_mis = misroute_time if c is None else c
        label = f"{usd(c_mis)} (agent time only)" if c is None else usd(c_mis)
        p = breakeven(c_mis)
        be.add(
            label,
            f"{p * 100:.1f}%",
            f"Auto-routing loses money below {p * 100:.1f}% precision",
        )

    # Model cost per ticket, and the extra coverage a dearer model must add over Haiku.
    mt = Table(
        "Model",
        "Per ticket, uncached",
        "Per ticket, cached at growth",
        "Extra coverage to pay for itself",
        "Same, cached",
        align="lrrrr",
    )
    base_u, base_c = q(haiku), q(haiku, hit_g)
    for name, m in models.items():
        qu, qc = q(m), q(m, hit_g)
        if name == "Haiku 5.5":
            extra_u = extra_c = "(baseline)"
        else:
            extra_u = f"{(qu - base_u) / save_auto * 100:.1f} points"
            extra_c = f"{(qc - base_c) / save_auto * 100:.1f} points"
        mt.add(name, usd(qu), usd(qc), extra_u, extra_c)
        key = name.split()[0].lower()
        v[f"q_{key}_u"] = usd(qu)
        v[f"q_{key}_c"] = usd(qc)
        v[f"extra_{key}"] = f"{(qu - base_u) / save_auto * 100:.1f}"
        v[f"extra_{key}_c"] = f"{(qc - base_c) / save_auto * 100:.1f}"
        v[f"ratio_{key}"] = f"{qu / base_u:.0f}"
    # Cascade: the share of escalated tickets Sonnet must auto-route at target to pay.
    v["cascade_breakeven"] = f"{q(sonnet, hit_rate(g, ESCALATED)) / save_auto * 100:.1f}%"
    v["cascade_breakeven_u"] = f"{q(sonnet) / save_auto * 100:.1f}%"
    v["triage_over_fable"] = f"{triage / q(models['Fable 5.1']):.0f}"
    v["triage_over_haiku"] = num(triage / base_u)

    # Evaluation: every golden ticket through Haiku and Sonnet (the cascade is simulated
    # offline from the two runs), at the batch price, without caching.
    eval_run = GOLDEN * (
        llm_cost(haiku, inp, out, batch=True) + llm_cost(sonnet, inp, out, batch=True)
    )
    compare = GOLDEN * sum(llm_cost(m, inp, out, batch=True) for m in models.values())
    v["eval_run"] = usd(eval_run)
    v["compare_run"] = usd(compare)
    v["compare_run_std"] = usd(
        GOLDEN * sum(llm_cost(m, inp, out) for m in models.values())
    )

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
        t = s["tickets"]
        h1, h2 = hit_rate(t), hit_rate(t, ESCALATED)
        r = {
            "haiku": t * q(haiku, h1),
            "sonnet": t * ESCALATED * q(sonnet, h2),
            "eval": EVAL_RUNS * eval_run,
            "app": s["tasks"]
            * HOURS_PER_MONTH
            * (s["vcpu"] * far["vcpu_hourly"] + s["mem_gb"] * far["gb_hourly"]),
            "db": (db["hourly_multi_az"] if s["multi_az"] else db["hourly"])
            * HOURS_PER_MONTH
            + 20
            * (
                disk["gp3_storage_gb_month_multi_az"]
                if s["multi_az"]
                else disk["gp3_storage_gb_month"]
            ),
        }
        r["total"] = sum(r.values())
        rows[name] = r
        v[f"{name}.total"] = usd(r["total"])
        v[f"{name}.models"] = usd(r["haiku"] + r["sonnet"])
        v[f"{name}.per_ticket"] = usd(r["total"] / t)
        v[f"{name}.sonnet_all"] = usd(t * q(sonnet, h1))
        v[f"{name}.haiku_all"] = usd(r["haiku"])
        v[f"{name}.triage_all"] = usd(t * triage)
        v[f"{name}.model_share_triage"] = f"{(r['haiku'] + r['sonnet']) / (t * triage) * 100:.1f}%"
        v[f"{name}.cache_saving"] = usd(
            t * (q(haiku) - q(haiku, h1)) + t * ESCALATED * (q(sonnet) - q(sonnet, h2))
        )
    gr = rows["growth"]
    how = {
        "haiku": f"{num(g)} tickets x {usd(q(haiku, hit_g))} (system prompt cached on {pct(hit_g)} of calls)",
        "sonnet": f"{pct(ESCALATED)} of tickets (sizing assumption) x {usd(q(sonnet, hit_rate(g, ESCALATED)))}",
        "eval": f"{EVAL_RUNS} runs x {GOLDEN:,} tickets through both models, batch price",
        "app": "2 Fargate tasks, webhook receiver and workers",
        "db": "db.t4g.medium, jobs, decisions and labels; 20 GB",
    }
    labels = {
        "haiku": "First stage, Haiku 5.5",
        "sonnet": "Second stage, Sonnet 5.5",
        "eval": "Evaluation runs",
        "app": "Classifier service",
        "db": "Postgres",
    }
    for k, label in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in SCALES), how[k])
    cost.add(
        "**Total per month**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), ""
    )
    cost.add(
        "Cost per ticket",
        *(usd(rows[s]["total"] / SCALES[s]["tickets"]) for s in SCALES),
        "",
    )

    stake = Table("Per month", "Pilot", "Growth", "Large", align="lrrr")
    stake.add("Tickets", *(num(SCALES[s]["tickets"]) for s in SCALES))
    stake.add(
        "Human triage of every ticket",
        *(usd(SCALES[s]["tickets"] * triage) for s in SCALES),
    )
    stake.add(
        "Each 10 points of coverage",
        *(usd(SCALES[s]["tickets"] * 0.10 * triage) for s in SCALES),
    )
    stake.add(
        "Each point of precision lost on auto-routed tickets, at 70% coverage",
        *(
            usd(SCALES[s]["tickets"] * COVERAGE_TARGET * 0.01 * MISROUTE)
            for s in SCALES
        ),
    )
    stake.add(
        "Both model stages (from the table above)",
        *(usd(rows[s]["haiku"] + rows[s]["sonnet"]) for s in SCALES),
    )
    v["growth.per_point_70"] = usd(g * COVERAGE_TARGET * 0.01 * MISROUTE)

    return {
        "tables": {
            "breakeven": be.md(),
            "models": mt.md(),
            "monthly": cost.md(),
            "stakes": stake.md(),
        },
        "vars": v,
    }
