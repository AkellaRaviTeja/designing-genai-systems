"""System 43: evaluation platform. Worked scenario numbers and monthly cost.

Halden (System 36's company) builds one evaluation platform for its four model-backed product
features. The features, their volumes, the evaluation request's token profile, the judge and the
offline class's rate budget are System 36's and are imported, not restated. Every suite size,
run count, judge accuracy, flip rate, traffic share, labelling time, staffing and experiment input
below is an assumption, labelled as such in the chapter. Nothing here was measured: this module
computes what a judge's error does to a score, what noise does to a gate, what a stale suite
cannot see, how long an online experiment must run, and what the platform costs.
"""

from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from math import comb, sqrt

import ch36
from _lib import (
    sig,
    DAYS_PER_MONTH,
    HOURS_PER_MONTH,
    WEEKS_PER_MONTH,
    Table,
    llm_cost,
    num,
    pct,
    usd,
)

ID = "ch43"

PRODUCT = ["support", "search", "extract", "drafts"]  # System 36's features with suites
EV = ch36.FEATURES[
    "evals"
]  # one evaluation request: 6,000 tokens in, 300 out, Sonnet 5.5

# Version 0 (assumptions).
SUITE_V0 = 200  # examples per feature, one overall score
PR_RUNS = (
    15  # changes a month per feature that touch a prompt, model, retrieval or tools
)
V0_THRESHOLD = (
    3  # System 7's starting threshold: fail if more than this many pass -> fail
)

# Break 1: the judge (assumptions; calibration measures the real values).
SENS = 0.95  # share of truly good answers the judge passes
SPEC = 0.70  # share of truly bad answers the judge fails
P_BEFORE, P_AFTER = 0.85, 0.75  # true pass rates before and after a regression
SPEC_NEW = 0.80  # the same judge after its provider updates it
FAIL_RATE = (
    0.15  # share of a feature's outputs that are truly bad, for calibration sizing
)
CAL_N = 400  # calibration examples per judged criterion, stratified by the judge's verdict
CAL_RANDOM = 200  # the sample a team takes first, as traffic comes
J_TARGET = 0.70  # lower 95% bound of sensitivity + specificity - 1 a judge must reach
CRITERIA = 2  # judged criteria per feature
LABELLERS = 2
LABEL_MIN = 3.0  # minutes for a person to label one output against a rubric
LABEL_HOURLY = 60.0  # USD, loaded cost of the domain experts who label
RECAL_MONTHS = (
    3  # recalibrate each criterion quarterly, and on any judge or rubric change
)

# Break 2: the gate (assumptions).
FLIP = 0.06  # share of examples whose verdict differs between two runs of one configuration
SLICES = 4  # gated slices per feature, a cap
METRICS_V0 = 2  # gated metrics per slice once teams add slices in version 0
ALPHA = 0.05
POWER = 0.80
SLICE_SIZES = (100, 200, 300, 400)
FLIPS = (0.0, 0.02, 0.06)
SUITE_SLICE = 300  # examples per slice in the final design
ALERT_P = (
    0.01  # alert when a count this large has at most this chance under the design rate
)

# Break 3: the suite (assumptions).
NEW_SHARE_TRAFFIC = 0.15  # a slice that grew after launch
NEW_SHARE_SUITE = 0.03
NEW_FAIL_BEFORE, NEW_FAIL_AFTER = 0.10, 0.40
REVIEW_WEEK = 50  # production outputs per feature reviewed by a person each week
PROD_JUDGE_DAY = 100  # production outputs per feature judged each day
HELD_OUT = 0.2  # share of each slice never used to tune a prompt

# Break 4: an online experiment, email drafts from Sonnet 5.5 to Haiku 5.5 (assumptions).
SENT_RATE = 0.50  # share of drafts sent with light edits
MARGIN = 0.02  # the largest drop the Growth team will accept for the saving
Z_ALPHA_1 = 1.645  # one-sided 5%
Z_POWER = 0.8416  # 80% power
USERS = 2_000  # people who use drafts in a month
ICC = 0.05  # correlation between one user's drafts
CV = 1.0  # coefficient of variation of drafts per user

# Optimisation (assumption): the judge's instructions and rubric, the same for every output.
RUBRIC_TOKENS = 1_500

# Scales: the number of features with suites. Evaluation cost follows features and changes,
# not traffic.
SCALES = {"small": 2, "growth": 4, "large": 16}
PLATFORM_FTE = {"small": 0.5, "growth": 1.0, "large": 2.0}  # engineers running it
ENGINEER_MONTH = (
    15_000.0  # USD, loaded monthly cost of a platform engineer (assumption)
)
RUNNER_TASKS, RUNNER_VCPU, RUNNER_GB = 2, 1, 2  # scheduler and runner, always on
LABEL_TASKS = 1  # the labelling tool
OUTPUT_KB = 20  # one stored output with its verdicts and trace
KEEP_MONTHS = 13


def p1(x: float) -> str:
    """A rate to one decimal place, rounded half up, so differences of shown values add up."""
    d = Decimal(str(round(x * 100, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{d}%"


def usd3(x: float) -> str:
    """A unit price to three significant figures, so a reader can multiply it out."""
    return "$" + f"{x:.3g}"


def strata() -> tuple[float, float, float]:
    """The judge's fail share, and the truly bad share among outputs it failed and passed."""
    pf = FAIL_RATE * SPEC + (1 - FAIL_RATE) * (1 - SENS)
    return pf, FAIL_RATE * SPEC / pf, FAIL_RATE * (1 - SPEC) / (1 - pf)


def split_hw(n_fail: int, n_pass: int) -> tuple[float, float]:
    """95% half-widths of specificity and sensitivity from people's labels on n_fail outputs
    the judge failed and n_pass it passed, weighted back by the judge's fail share (delta
    method; the share itself is known from the judge's verdicts on all traffic)."""
    pf, bf, bp = strata()
    pp = 1 - pf
    a, d = pp * bp, pf * bf  # truly bad and passed, truly bad and failed
    gp, gf = pp * (1 - bp), pf * (1 - bf)
    vbf, vbp = bf * (1 - bf) / n_fail, bp * (1 - bp) / n_pass
    v_spec = (pf * a / (a + d) ** 2) ** 2 * vbf + (pp * d / (a + d) ** 2) ** 2 * vbp
    g = gp + gf
    v_sens = (pp * gf / g**2) ** 2 * vbp + (pf * gp / g**2) ** 2 * vbf
    return 1.96 * sqrt(v_spec), 1.96 * sqrt(v_sens)


def best_split(n: int) -> tuple[int, float, float]:
    """The judge-failed sample size that minimises the specificity interval."""
    best = min(range(5, n - 4, 5), key=lambda k: split_hw(k, n - k)[0])
    return (best, *split_hw(best, n - best))


def binom_sf(n: int, p: float, k: int) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def binom_pmf(n: int, q: float) -> list[tuple[int, float]]:
    if q == 0:
        return [(0, 1.0)]
    top = min(n, int(n * q * 4 + 30))
    return [(i, comb(n, i) * q**i * (1 - q) ** (n - i)) for i in range(top + 1)]


@lru_cache(maxsize=None)
def critical(total: int, alpha: float) -> int:
    """Smallest b that an exact one-sided McNemar test rejects with b + c = total."""
    for k in range(total + 1):
        if binom_sf(total, 0.5, k) <= alpha:
            return k
    return total + 1


def power(n: int, flip: float, r: int, alpha: float) -> float:
    """Chance the exact McNemar test rejects on a slice of n with r extra pass -> fail
    examples, when noise flips each example either way with probability flip / 2."""
    # A regression of expected size r is itself random: each example regresses with
    # probability r / n, on top of the noise.
    db = binom_pmf(n, min(1.0, flip / 2 + r / n))
    dc = binom_pmf(n, flip / 2)
    return sum(
        pb * pc for b, pb in db for c, pc in dc if b >= critical(b + c, alpha)
    )


def resolution(n: int, flip: float, alpha: float) -> int:
    """Smallest observed regression (extra pass -> fail examples) caught with POWER."""
    for r in range(n + 1):
        if power(n, flip, r, alpha) >= POWER:
            return r
    return n


def true_pts(r: int, n: int) -> str:
    """An observed regression of r examples as points of the true pass rate, through a judge
    at the calibration bound J_TARGET."""
    return f"{r / n / J_TARGET * 100:.0f}"


def alert_count(n: int, p: float) -> int:
    """Smallest count with at most ALERT_P chance under Binomial(n, p)."""
    for k in range(n + 1):
        if binom_sf(n, p, k) <= ALERT_P:
            return k
    return n


def build(P, H):
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    haiku = P("llm.anthropic.claude-haiku-5-5")
    judge = P("llm.openai.gpt-6-1-terra")
    far = P("infra.aws.fargate.linux-x86")
    db = P("infra.aws.rds-postgres.db-t4g-medium")
    s3 = P("infra.aws.s3.standard")

    v: dict[str, str] = {}
    n_feat = len(PRODUCT)

    # One example: the feature's call at System 36's evaluation profile, plus one judgement.
    q_sys = llm_cost(sonnet, EV["new"], EV["out"])
    q_judge = llm_cost(judge, ch36.JUDGE_IN, ch36.JUDGE_OUT)
    q_ex = q_sys + q_judge
    q_ex_batch = llm_cost(sonnet, EV["new"], EV["out"], batch=True) + llm_cost(
        judge, ch36.JUDGE_IN, ch36.JUDGE_OUT, batch=True
    )
    q_judge_batch = llm_cost(judge, ch36.JUDGE_IN, ch36.JUDGE_OUT, batch=True)
    v.update(
        {
            "features": f"{n_feat}",
            "ev_in": f"{EV['new']:,}",
            "ev_out": f"{EV['out']}",
            "judge_in": f"{ch36.JUDGE_IN:,}",
            "judge_out": f"{ch36.JUDGE_OUT}",
            "q_ex": usd3(q_ex),
            "q_judge": usd3(q_judge),
            "ch36_evals": num(EV["req"]),
            "ch36_batch_share": pct(ch36.EVALS_BATCH_SHARE),
        }
    )

    # --- Version 0: every gate runs the candidate and the last release; every suite nightly.
    nightly = DAYS_PER_MONTH
    v0_examples = n_feat * (2 * PR_RUNS + nightly) * SUITE_V0
    v0_cost = v0_examples * q_ex
    v.update(
        {
            "suite_v0": f"{SUITE_V0}",
            "pr_runs": f"{PR_RUNS}",
            "pr_month": f"{n_feat * PR_RUNS}",
            "v0_threshold": f"{V0_THRESHOLD}",
            "v0_examples": num(v0_examples),
            "v0_cost": usd(v0_cost),
            "nightly": f"{nightly:g}",
            "terra_cache_ratio": pct(judge["cache_read"] / judge["input"]),
        }
    )

    # --- Break 1: the judge.
    def observed(p, s=SENS, c=SPEC):
        return p * s + (1 - p) * (1 - c)

    j = SENS + SPEC - 1
    ob, oa = observed(P_BEFORE), observed(P_AFTER)
    ob_new = observed(P_BEFORE, c=SPEC_NEW)
    fails_random = CAL_RANDOM * FAIL_RATE
    hw_random = 1.96 * sqrt(SPEC * (1 - SPEC) / fails_random)
    hw_random_n = 1.96 * sqrt(SPEC * (1 - SPEC) / (CAL_N * FAIL_RATE))
    n_fail, hw_spec, hw_sens = best_split(CAL_N)
    n_pass = CAL_N - n_fail
    hw_j = sqrt(hw_spec**2 + hw_sens**2)
    pf, bf, bp = strata()
    n_bad = n_fail * bf + n_pass * bp
    r_drift = resolution(round(n_bad), FLIP, ALPHA / 2)
    rg = (oa + SPEC - 1) / j  # Rogan-Gladen correction of the "after" score
    cal_hours = CAL_N * LABELLERS * LABEL_MIN / 60
    v.update(
        {
            "sens": pct(SENS),
            "spec": pct(SPEC),
            "p_before": pct(P_BEFORE),
            "p_after": pct(P_AFTER),
            "drop": f"{(P_BEFORE - P_AFTER) * 100:.0f}",
            "obs_before": p1(ob),
            "obs_after": p1(oa),
            "obs_drop": f"{float(p1(ob)[:-1]) - float(p1(oa)[:-1]):.1f}",
            "j": f"{j:.2f}",
            "j_target": f"{J_TARGET:.1f}",
            "j_point": f"{J_TARGET + hw_j:.1f}",
            "hw_j": f"{hw_j:.2f}",
            "n_factor": f"{1 / j**2:.1f}",
            "bad_pass": pct(1 - SPEC),
            "spec_new": pct(SPEC_NEW),
            "obs_before_new": p1(ob_new),
            "judge_shift": f"{float(p1(ob)[:-1]) - float(p1(ob_new)[:-1]):.1f}",
            "fail_rate": pct(FAIL_RATE),
            "cal_n": f"{CAL_N}",
            "cal_random": f"{CAL_RANDOM}",
            "cal_hw_random_n": f"{hw_random_n * 100:.0f}",
            "cal_n_fail": f"{n_fail}",
            "cal_n_pass": f"{n_pass}",
            "cal_hw_sens": f"{hw_sens * 100:.0f}",
            "cal_bad": f"{n_bad:.0f}",
            "judge_fail_share": pct(pf),
            "missed_in_pass": pct(bp),
            "drift_pts": f"{r_drift / n_bad * 100:.0f}",
            "cal_event": usd(cal_hours * LABEL_HOURLY),
            "cal_fails_random": f"{fails_random:.0f}",
            "cal_hw_random": f"{hw_random * 100:.0f}",
            "cal_hw_enriched": f"{hw_spec * 100:.0f}",
            "rg_after": f"{rg * 100:.0f}%",
            "label_min": f"{LABEL_MIN:g}",
            "label_hourly": f"${LABEL_HOURLY:.0f}",
            "labellers": f"{LABELLERS}",
            "cal_hours": f"{cal_hours:.0f}",
            "criteria": f"{CRITERIA}",
            "recal_months": f"{RECAL_MONTHS}",
        }
    )

    # --- Break 2: the gate.
    exp_noise = SUITE_V0 * FLIP / 2
    false_fail = binom_sf(SUITE_V0, FLIP / 2, V0_THRESHOLD + 1)

    def flip_at(target):
        lo, hi = 0.0, 0.3
        for _ in range(60):
            mid = (lo + hi) / 2
            if binom_sf(SUITE_V0, mid / 2, V0_THRESHOLD + 1) < target:
                lo = mid
            else:
                hi = mid
        return lo

    tests_v0 = SLICES * METRICS_V0
    family = 1 - (1 - ALPHA) ** tests_v0
    alpha_each = (
        ALPHA / SLICES
    )  # one verdict per example: a family of one test per slice
    v.update(
        {
            "flip": pct(FLIP),
            "exp_noise": f"{exp_noise:.0f}",
            "false_fail": pct(false_fail),
            "flip_be": f"{flip_at(ALPHA) * 100:.1f}%",
            "flip_half": f"{flip_at(0.5) * 100:.1f}%",
            "false_fail_month": f"{n_feat * PR_RUNS * false_fail:.0f}",
            "slices": f"{SLICES}",
            "metrics_v0": f"{METRICS_V0}",
            "tests_v0": f"{tests_v0}",
            "family": pct(family),
            "alpha_each": f"{alpha_each:.4f}",
            "power": pct(POWER),
        }
    )
    gate = Table(
        "Examples in the slice",
        *(f"Flip rate {pct(f)}" for f in FLIPS),
        align="l" + "r" * len(FLIPS),
    )
    for n in SLICE_SIZES:
        cells = []
        for f in FLIPS:
            r = resolution(n, f, alpha_each)
            cells.append(f"{r} ({true_pts(r, n)} points)")
            v[f"res_{n}_{int(f * 100)}"] = f"{r}"
            v[f"res_{n}_{int(f * 100)}_pts"] = true_pts(r, n)
        gate.add(f"{n}", *cells)
    v["flip_lo"] = pct(FLIPS[1])
    held = int(SUITE_SLICE * HELD_OUT)
    v["held_out"] = pct(HELD_OUT)
    v["held_n"] = f"{held}"
    # Held-out against the rest of the slice is an unpaired comparison of two pass rates.
    se = sqrt(P_BEFORE * (1 - P_BEFORE) * (1 / held + 1 / (SUITE_SLICE - held)))
    v["held_pts"] = f"{(1.96 + Z_POWER) * se / J_TARGET * 100:.0f}"

    suite_final = SLICES * SUITE_SLICE
    lim_s = ch36.LIMITS["sonnet"]["itpm"]
    offline_rate = lim_s * ch36.OFFLINE_SYNC_SHARE / EV["new"]
    gate_min = suite_final / offline_rate
    two_itpm = 2 * suite_final / 20 * EV["new"]  # two gates at once inside 20 minutes
    pr_day = n_feat * PR_RUNS / ch36.WORKING_DAYS
    false_month = n_feat * PR_RUNS * ALPHA
    v.update(
        {
            "suite_slice": f"{SUITE_SLICE}",
            "suite_final": f"{suite_final:,}",
            "offline_rate": f"{offline_rate:.0f}",
            "offline_share": pct(ch36.OFFLINE_SYNC_SHARE),
            "gate_min": f"{gate_min:.0f}",
            "all_suites_min": f"{gate_min * n_feat:.0f}",
            "two_itpm": num(two_itpm),
            "two_share": pct(two_itpm / lim_s),
            "pr_day": f"{pr_day:.1f}",
            "false_month": f"{false_month:.0f}",
            "override_alert": f"{alert_count(PR_RUNS, ALPHA)}",
        }
    )

    # --- Break 3: the suite against the traffic.
    d_fail = NEW_FAIL_AFTER - NEW_FAIL_BEFORE
    new_in_suite = SUITE_V0 * NEW_SHARE_SUITE
    weeks_uniform = SUITE_SLICE / (REVIEW_WEEK * NEW_SHARE_TRAFFIC)
    weeks_targeted = SUITE_SLICE / REVIEW_WEEK
    v.update(
        {
            "new_traffic": pct(NEW_SHARE_TRAFFIC),
            "new_suite": pct(NEW_SHARE_SUITE),
            "new_in_suite": f"{new_in_suite:.0f}",
            "new_fail_before": pct(NEW_FAIL_BEFORE),
            "new_fail_after": pct(NEW_FAIL_AFTER),
            "suite_moves": f"{NEW_SHARE_SUITE * d_fail * 100:.1f}",
            "suite_moves_obs": f"{NEW_SHARE_SUITE * d_fail * j * 100:.1f}",
            "prod_moves": f"{NEW_SHARE_TRAFFIC * d_fail * 100:.1f}",
            "review_week": f"{REVIEW_WEEK}",
            "weeks_uniform": f"{weeks_uniform:.0f}",
            "weeks_targeted": f"{weeks_targeted:.0f}",
            "prod_judge_day": f"{PROD_JUDGE_DAY}",
        }
    )

    # --- Break 4: an online experiment on drafts.
    dr = ch36.FEATURES["drafts"]
    q_dr_s = llm_cost(sonnet, dr["new"], dr["out"], cache_read=dr["prefix"])
    q_dr_h = llm_cost(haiku, dr["new"], dr["out"], cache_read=dr["prefix"])
    saving = dr["req"] * (q_dr_s - q_dr_h)
    n_arm = (Z_ALPHA_1 + Z_POWER) ** 2 * 2 * SENT_RATE * (1 - SENT_RATE) / MARGIN**2
    per_day = dr["req"] / ch36.WORKING_DAYS
    days_draft = 2 * n_arm / per_day
    users_arm = USERS / 2
    per_user_day = dr["req"] / USERS / ch36.WORKING_DAYS
    k = CV**2 + 1  # unequal cluster sizes inflate the design effect
    # n_eff = U m / (1 + (k m - 1) rho) >= n  =>  m >= n (1 - rho) / (U - n k rho)
    m_need = n_arm * (1 - ICC) / (users_arm - n_arm * k * ICC)
    days_user = m_need / per_user_day
    icc_be = users_arm / (n_arm * k)
    icc_be_small = users_arm / 3 / (n_arm * k)
    v.update(
        {
            "dr_q_sonnet": usd3(q_dr_s),
            "dr_q_haiku": usd3(q_dr_h),
            "dr_saving": f"${sig(saving, 3):,.0f}",
            "dr_req": num(dr["req"]),
            "sent_rate": pct(SENT_RATE),
            "margin": f"{MARGIN * 100:.0f}",
            "n_arm": num(n_arm),
            "dr_day": num(per_day),
            "days_draft": f"{days_draft:.1f}",
            "users": num(USERS),
            "per_user_month": f"{dr['req'] / USERS:.0f}",
            "icc": f"{ICC:g}",
            "cv": f"{CV:g}",
            "m_need": f"{m_need:.0f}",
            "days_user": f"{days_user:.0f}",
            "icc_be": f"{icc_be:.3f}",
            "icc_be_small": f"{icc_be_small:.3f}",
            "draft_be": usd(saving / (MARGIN * dr["req"])),
        }
    )

    # --- Monthly cost at three scales.
    cost = Table(
        "Line item",
        "Small",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rows = {}
    for s, f in SCALES.items():
        pr_ex = f * PR_RUNS * suite_final
        aa_ex = f * WEEKS_PER_MONTH * suite_final
        r = {
            "gates": pr_ex * q_ex,
            "aa": aa_ex * q_ex_batch,
            "prod": f * DAYS_PER_MONTH * PROD_JUDGE_DAY * q_judge_batch,
            "drift": f * CRITERIA * CAL_N * q_judge_batch,
            "cal": f * CRITERIA * cal_hours / RECAL_MONTHS * LABEL_HOURLY,
            "review": f * REVIEW_WEEK * WEEKS_PER_MONTH * LABEL_MIN / 60 * LABEL_HOURLY,
            "infra": (RUNNER_TASKS + LABEL_TASKS)
            * HOURS_PER_MONTH
            * (RUNNER_VCPU * far["vcpu_hourly"] + RUNNER_GB * far["gb_hourly"])
            + db["hourly_multi_az"] * HOURS_PER_MONTH
            + (pr_ex + aa_ex)
            * OUTPUT_KB
            / 1e6
            * KEEP_MONTHS
            * s3["storage_gb_month_first_50tb"],
            "eng": PLATFORM_FTE[s] * ENGINEER_MONTH,
        }
        r["model"] = r["gates"] + r["aa"] + r["prod"] + r["drift"]
        r["people"] = r["cal"] + r["review"]
        r["total"] = r["model"] + r["people"] + r["infra"] + r["eng"]
        r["hours"] = r["people"] / LABEL_HOURLY
        r["examples"] = pr_ex + aa_ex
        r["sync_share"] = pr_ex / (pr_ex + aa_ex)
        rows[s] = r
        v[f"{s}.total"] = usd(r["total"])
        v[f"{s}.model"] = usd(r["model"])
        v[f"{s}.people"] = usd(r["people"])
        v[f"{s}.eng"] = usd(r["eng"])
        v[f"{s}.hours"] = f"{r['hours']:.0f}"
        v[f"{s}.people_share"] = pct((r["people"] + r["eng"]) / r["total"])
        v[f"{s}.examples"] = num(r["examples"])
        v[f"{s}.features"] = f"{f}"
    lines = {
        "gates": (
            "Gate runs",
            f"{PR_RUNS} changes a feature a month x {suite_final:,} examples, candidate only, synchronous, at {usd3(q_ex)} each",
        ),
        "aa": (
            "Weekly A/A runs",
            f"The release configuration re-run weekly, {suite_final:,} examples, batch API",
        ),
        "prod": (
            "Production judging",
            f"{PROD_JUDGE_DAY} outputs a feature a day, judge only, batch API",
        ),
        "drift": (
            "Judge drift re-runs",
            f"The {CAL_N}-output calibration set of each criterion re-judged monthly, batch API",
        ),
        "cal": (
            "Calibration labels",
            f"{CRITERIA} criteria a feature, {CAL_N} outputs, {LABELLERS} people, {LABEL_MIN:g} min each, quarterly",
        ),
        "review": (
            "Review queue",
            f"{REVIEW_WEEK} production outputs a feature a week, one person",
        ),
        "infra": (
            "Runner, labelling tool, Postgres, S3",
            "Three small Fargate tasks, db.t4g.medium Multi-AZ, outputs kept 13 months",
        ),
        "eng": (
            "Platform engineering",
            f"{PLATFORM_FTE['growth']:g} engineer at an assumed ${ENGINEER_MONTH:,.0f} a month (0.5 small, 2 large)",
        ),
    }
    for key, (label, how) in lines.items():
        cost.add(label, *(usd(rows[s][key]) for s in SCALES), how)
    cost.add(
        "**Platform total**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), ""
    )
    cost.add(
        "Of which labelling",
        *(usd(rows[s]["people"]) for s in SCALES),
        f"At an assumed ${LABEL_HOURLY:.0f} an hour",
    )
    cost.add("Expert hours a month", *(f"{rows[s]['hours']:.0f}" for s in SCALES), "")
    cost.add("Features with suites", *(f"{SCALES[s]}" for s in SCALES), "")

    g = rows["growth"]
    # One-off: growing each suite from version 0's size, and the first calibration.
    build_ex = (suite_final - SUITE_V0) * n_feat
    build_hours = build_ex * LABEL_MIN / 60 + n_feat * CRITERIA * cal_hours
    judge_calls = (
        n_feat * (PR_RUNS + WEEKS_PER_MONTH) * suite_final
        + n_feat * DAYS_PER_MONTH * PROD_JUDGE_DAY
    )
    batch_calls = n_feat * (WEEKS_PER_MONTH * suite_final + DAYS_PER_MONTH * PROD_JUDGE_DAY)
    sync_calls = judge_calls - batch_calls
    o1 = (
        RUBRIC_TOKENS
        * (judge["input"] - judge["cache_read"])
        * (sync_calls + batch_calls * (1 - judge["batch_discount"]))
        / 1e6
    )
    v.update(
        {
            "g_gates": usd(g["gates"]),
            "g_aa": usd(g["aa"]),
            "g_aa_sync": usd(n_feat * WEEKS_PER_MONTH * suite_final * q_ex),
            "g_prod": usd(g["prod"]),
            "g_drift": usd(g["drift"]),
            "g_vs_ch36": pct(g["examples"] / EV["req"]),
            "g_sync_share": pct(g["sync_share"]),
            "v0_vs_final": f"{g['model'] / v0_cost:.1f}",
            "rubric_tokens": f"{RUBRIC_TOKENS:,}",
            "o1_saving": usd(o1),
            "run_less_saving": usd(g["gates"] * (n_feat - 1)),
            "eng_month": f"${ENGINEER_MONTH:,.0f}",
            "build_examples": num(build_ex),
            "build_hours": f"{build_hours:,.0f}",
            "build_cost": usd(build_hours * LABEL_HOURLY),
            "build_weeks": f"{(suite_final - SUITE_V0) / REVIEW_WEEK:.0f}",
        }
    )

    return {
        "tables": {"gate": gate.md(), "monthly": cost.md()},
        "vars": v,
    }
