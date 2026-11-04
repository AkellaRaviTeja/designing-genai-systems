"""System 17: Tool-calling assistant. Worked scenario numbers and monthly cost.

Paybridge, System 1's worked scenario, gives its support agents an assistant that answers
questions about a ticket by calling Paybridge's internal APIs as tools. Every volume, size,
share, time and money value below that is not a price is an assumption, labelled as such in the
chapter. Nothing here was measured: how often the model picks the right tool, writes valid
arguments or proposes a wrong action is what the task set decides, so this module computes
token counts, costs per task and the break-even values those results must clear.
"""

import math

import ch01
from _lib import HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch17"

# Use (assumptions). Agents open the assistant on a share of System 1's tickets.
USE_SHARE = 0.30
WORKING_DAYS = ch01.WORKING_DAYS
BUSINESS_HOURS = ch01.BUSINESS_HOURS  # agents work these hours; so does the assistant
SCALES = {
    "pilot": dict(tasks=2, vcpu=0.5, mem_gb=1, multi_az=False),
    "growth": dict(tasks=2, vcpu=0.5, mem_gb=1, multi_az=False),
    "large": dict(tasks=3, vcpu=1, mem_gb=2, multi_az=True),
}
AGENT_HOURLY = ch01.AGENT_HOURLY  # USD, loaded

# Tool-use system prompt the API adds when tools are present, Sonnet 5.5 and Haiku 5.5 with
# tool_choice auto or none (Anthropic's tool use pricing table, checked 2026-11-04).
TOOL_SYS = 286

# Version 0: every endpoint of the internal API, generated from its OpenAPI spec (assumption).
V0_TOOLS = 22
V0_TOKENS_PER_TOOL = (
    480  # generated descriptions, every parameter documented, nested schemas
)

# Version 1 on: the hand-written catalogue. Tokens per definition are assumptions; count your
# own with the token-counting endpoint.
TOOLS = [
    # name, kind, tokens, what it does
    (
        "get_customer",
        "read",
        210,
        "A customer: plan, state registrations, payroll calendar",
    ),
    (
        "list_payroll_runs",
        "read",
        230,
        "Recent runs for a customer, with status and error codes",
    ),
    (
        "get_payroll_run",
        "read",
        240,
        "One run: steps, errors by employee, bank file status",
    ),
    (
        "get_filing_status",
        "read",
        250,
        "PF, ESI and TDS filings for a period, with portal acknowledgements",
    ),
    (
        "search_help_articles",
        "read",
        200,
        "Search Paybridge's help centre; returns titles, links and snippets",
    ),
    (
        "get_ticket_history",
        "read",
        190,
        "A customer's last 20 tickets: subject, status, solving group",
    ),
    (
        "add_internal_note",
        "write",
        210,
        "Add an internal note to this ticket (agents only see it)",
    ),
    (
        "resend_payslips",
        "write",
        260,
        "Resend a run's payslips to employees' email addresses on file",
    ),
    (
        "retry_payroll_run",
        "write",
        290,
        "Re-run a failed payroll run from its failed step",
    ),
]
V1_TOOLS = len(TOOLS)
V1_DEFS = sum(t[2] for t in TOOLS)

# Prompt and task shape (assumptions).
SYSTEM = 1_200  # role, rules for tools and confirmation, answer format
TICKET = (
    600  # the ticket's subject, body and last reply, fetched by the service, not a tool
)
QUESTION = 60  # the agent's question
TOOL_USE_OUT = 120  # one step's output: a short preamble, thinking at low effort, the tool_use block
ANSWER_OUT = 250  # the final answer to the agent
TOOL_STEPS = 3  # tool rounds in a typical task; one more model call writes the answer
CALLS = TOOL_STEPS + 1
RESULT_V0 = 2_500  # a raw API response, every field
RESULT_V1 = 600  # a tool result shaped for the model: the fields it needs
RESULT_CAP = 1_500  # hard cap on any tool result (version 3)
STEP_CAP = 8  # model calls per task, the final answer and any truncation retry included (version 3)
MAX_TOOLS_PER_STEP = 3  # tool calls run per model response; more get an is_error result (version 3)
MAX_OUT = 1_024  # max_tokens on every call, the ceiling for the worst case
RUNAWAY_STEPS = 25  # assumed: a task stuck retrying a failing tool, version 2 with no cap
ERROR_PAGE = 1_500  # an HTML error page returned as a tool result in version 0

# Arguments (version 2): what one invalid tool call costs.
INVALID_RATES = (
    0.01,
    0.05,
)  # shares of tool calls rejected by validation, for the arithmetic

# Side effects (version 4): what a confirmation costs and what a wrong action costs.
CONFIRM_SECONDS = 10  # an agent reading a confirmation card and clicking (assumption)
WRITE_SHARE = 0.20  # share of tasks that propose a confirmed write, resend or retry (assumption); notes excluded
CATCH_RATES = (1.0, 0.5)  # share of wrong proposals an agent catches on the card (assumption)
WRONG_ACTION = {
    # name: (USD cost of one wrong execution, why), all the support lead's assumptions
    "add_internal_note": (0.25, "an agent deletes or corrects the note"),
    "resend_payslips": (
        40.0,
        "a customer's employees receive a duplicate or the wrong month's payslips; support time and an apology",
    ),
    "retry_payroll_run": (
        500.0,
        "a second bank file for the same run; unwinding it takes the customer's and Paybridge's finance teams",
    ),
}

# Evaluation (assumptions).
TASK_SET = 150  # scripted tasks with recorded tool fixtures
EVAL_RUNS = 20  # a month: one per prompt, tool or model change, plus weekly
JUDGE_IN, JUDGE_OUT = 2_500, 200  # judge of the final answer, another model family (F4)

# Audit and conversation storage (rule of thumb: about 4 bytes of text per token).
BYTES_PER_TOKEN = 4
RETENTION_DAYS = 90
DISK_GB = 20


def task_tokens(defs: int, result: int, calls: int, tools_present: bool = True):
    """Input and output tokens per model call for one task, uncached.

    Call n (1-based) sends the prefix plus every earlier step's tool_use output and result."""
    prefix = (TOOL_SYS if tools_present else 0) + defs + SYSTEM + TICKET + QUESTION
    out = []
    for n in range(1, calls + 1):
        inp = prefix + (n - 1) * (TOOL_USE_OUT + result)
        o = ANSWER_OUT if n == calls else TOOL_USE_OUT
        out.append((inp, o))
    return out


def task_cost_uncached(p, defs, result, calls):
    return sum(llm_cost(p, i, o) for i, o in task_tokens(defs, result, calls))


def task_cost_cached(p, defs, result, calls, hit):
    """Automatic caching: the breakpoint follows the last block, so call n reads what call n-1
    sent and writes only what is new. The shared prefix (tool prompt, definitions, system)
    is a read on a cache hit and a write on a miss; `hit` is the chance the previous task
    left it warm."""
    shared = TOOL_SYS + defs + SYSTEM
    cost = 0.0
    prev_in = 0
    for n, (inp, o) in enumerate(task_tokens(defs, result, calls), start=1):
        if n == 1:
            tail = inp - shared
            cost += hit * llm_cost(p, 0, 0, cache_read=shared, cache_write=tail)
            cost += (1 - hit) * llm_cost(p, 0, 0, cache_write=inp)
        else:
            cost += llm_cost(p, 0, 0, cache_read=prev_in, cache_write=inp - prev_in)
        cost += llm_cost(p, 0, o)
        prev_in = inp
    return cost


def hit_rate(tasks_month: float) -> float:
    """Chance a task finds the shared prefix cached: another task started within the 5-minute
    lifetime. Arrivals treated as random within business hours."""
    rate = tasks_month / (WORKING_DAYS * BUSINESS_HOURS * 3600)
    return 1 - math.exp(-rate * 300)


def build(P, H):
    sonnet = P("llm.anthropic.claude-sonnet-5-5")
    haiku = P("llm.anthropic.claude-haiku-5-5")
    judge = P("llm.openai.gpt-6-1-terra")
    far = P("infra.aws.fargate.linux-x86-ap-south-1")
    db = P("infra.aws.rds-postgres.db-t4g-medium-ap-south-1")
    disk = P("vector.aws.rds-postgres-db-r6g-xlarge")

    v0_defs = V0_TOOLS * V0_TOKENS_PER_TOOL
    v0 = task_tokens(v0_defs, RESULT_V0, CALLS)
    v1 = task_tokens(V1_DEFS, RESULT_V1, CALLS)
    v0_in = sum(i for i, _ in v0)
    v1_in = sum(i for i, _ in v1)
    v0_out = sum(o for _, o in v0)
    q_v0 = task_cost_uncached(sonnet, v0_defs, RESULT_V0, CALLS)
    q_v1_u = task_cost_uncached(sonnet, V1_DEFS, RESULT_V1, CALLS)
    defs_share_v0 = CALLS * (TOOL_SYS + v0_defs) / v0_in

    tasks = {s: ch01.SCALES[s]["tickets"] * USE_SHARE for s in SCALES}
    hits = {s: hit_rate(tasks[s]) for s in SCALES}
    q_v1 = {
        s: task_cost_cached(sonnet, V1_DEFS, RESULT_V1, CALLS, hits[s]) for s in SCALES
    }
    q_v1_g = q_v1["growth"]
    q_v0_cached_g = task_cost_cached(sonnet, v0_defs, RESULT_V0, CALLS, hits["growth"])

    # Cost of the definitions alone on one model call, uncached and as a cache read.
    def_call_v0 = (TOOL_SYS + v0_defs) * sonnet["input"] / 1e6
    def_call_v1_read = (TOOL_SYS + V1_DEFS) * sonnet["cache_read"] / 1e6
    def_call_v1_write = (TOOL_SYS + V1_DEFS + SYSTEM) * sonnet["cache_write"] / 1e6

    # Version 2: one invalid call costs one more model call (the error result and a retry).
    mid_in = v1[1][0]  # a typical second call's input
    extra_step = llm_cost(
        sonnet, 0, TOOL_USE_OUT, cache_read=mid_in, cache_write=TOOL_USE_OUT + 80
    )
    calls_month_g = tasks["growth"] * TOOL_STEPS

    # Version 3: a runaway task against a capped one.
    # Version 2's sizes, cached, with the error page as every result and no cap.
    runaway = task_cost_cached(sonnet, V1_DEFS, ERROR_PAGE, RUNAWAY_STEPS, hits["growth"])
    # Worst case under version 3: every call hits max_tokens and every step returns the most
    # tool results allowed, each at the cap. Cached as the ordinary task is.
    shared = TOOL_SYS + V1_DEFS + SYSTEM
    worst_step = MAX_OUT + MAX_TOOLS_PER_STEP * RESULT_CAP
    worst_in = [shared + TICKET + QUESTION + (n - 1) * worst_step for n in range(1, STEP_CAP + 1)]
    capped = llm_cost(sonnet, 0, 0, cache_read=shared, cache_write=worst_in[0] - shared)
    for a, b in zip(worst_in, worst_in[1:]):
        capped += llm_cost(sonnet, 0, 0, cache_read=a, cache_write=b - a)
    capped += llm_cost(sonnet, 0, STEP_CAP * MAX_OUT)
    capped_in_last = worst_in[-1]
    none_rewrite = (capped_in_last - shared) * sonnet["cache_write"] / 1e6
    # One more step of a runaway, early and late.
    def step_at(n):
        rows = task_tokens(V1_DEFS, ERROR_PAGE, RUNAWAY_STEPS)
        return llm_cost(sonnet, 0, TOOL_USE_OUT, cache_read=rows[n - 2][0], cache_write=rows[n - 1][0] - rows[n - 2][0])
    tasks_hour_peak = tasks["growth"] / (WORKING_DAYS * BUSINESS_HOURS) * ch01.PEAK_FACTOR
    runaway_in_last = task_tokens(V1_DEFS, ERROR_PAGE, RUNAWAY_STEPS)[-1][0]

    # Version 4: confirmation break-even per write tool.
    confirm_cost = CONFIRM_SECONDS / 3600 * AGENT_HOURLY
    breakeven = {k: confirm_cost / c for k, (c, _) in WRONG_ACTION.items()}
    half = CATCH_RATES[1]
    breakeven_half = {k: confirm_cost / (c * half) for k, (c, _) in WRONG_ACTION.items()}

    # Break-even agent time saved per task: what the assistant costs, in agent seconds.
    def secs(x):
        return x / AGENT_HOURLY * 3600

    # Haiku 5.5 on the same task (O9), cached at growth's hit rate.
    q_haiku_g = task_cost_cached(haiku, V1_DEFS, RESULT_V1, CALLS, hits["growth"])

    v = {
        "use_share": pct(USE_SHARE),
        "tool_sys": f"{TOOL_SYS}",
        "v0_tools": f"{V0_TOOLS}",
        "v0_per_tool": f"{V0_TOKENS_PER_TOOL}",
        "v0_defs": f"{v0_defs:,}",
        "v1_tools": f"{V1_TOOLS}",
        "v1_defs": f"{V1_DEFS:,}",
        "v1_reads": f"{sum(1 for t in TOOLS if t[1] == 'read')}",
        "v1_writes": f"{sum(1 for t in TOOLS if t[1] == 'write')}",
        "system": f"{SYSTEM:,}",
        "ticket": f"{TICKET}",
        "question": f"{QUESTION}",
        "tool_use_out": f"{TOOL_USE_OUT}",
        "answer_out": f"{ANSWER_OUT}",
        "tool_steps": f"{TOOL_STEPS}",
        "calls": f"{CALLS}",
        "result_v0": f"{RESULT_V0:,}",
        "result_v1": f"{RESULT_V1}",
        "result_cap": f"{RESULT_CAP:,}",
        "step_cap": f"{STEP_CAP}",
        "v0_first_in": f"{v0[0][0]:,}",
        "v0_last_in": f"{v0[-1][0]:,}",
        "v0_task_in": f"{v0_in:,}",
        "v0_task_out": f"{v0_out:,}",
        "v1_first_in": f"{v1[0][0]:,}",
        "v1_last_in": f"{v1[-1][0]:,}",
        "v1_task_in": f"{v1_in:,}",
        "q_v0": usd(q_v0),
        "q_v0_cached": usd(q_v0_cached_g),
        "q_v1_u": usd(q_v1_u),
        "q_v1": usd(q_v1_g),
        "v0_over_v1": f"{q_v0 / q_v1_g:.0f}",
        "defs_share_v0": pct(defs_share_v0),
        "def_call_v0": usd(def_call_v0),
        "def_call_v1_read": usd(def_call_v1_read),
        "def_call_v1_write": usd(def_call_v1_write),
        "def_task_v0": usd(def_call_v0 * CALLS),
        "cache_min": "512",
        "extra_step": usd(extra_step),
        "calls_month_g": num(calls_month_g),
        "invalid_lo": pct(INVALID_RATES[0]),
        "invalid_hi": pct(INVALID_RATES[1]),
        "invalid_lo_n": num(calls_month_g * INVALID_RATES[0]),
        "invalid_hi_n": num(calls_month_g * INVALID_RATES[1]),
        "invalid_hi_cost": usd(calls_month_g * INVALID_RATES[1] * extra_step),
        "invalid_hi_share": pct(INVALID_RATES[1] * TOOL_STEPS * extra_step / q_v1_g),
        "runaway_steps": f"{RUNAWAY_STEPS}",
        "error_page": f"{ERROR_PAGE:,}",
        "runaway": usd(runaway),
        "runaway_over": f"{runaway / q_v1_g:.0f}",
        "runaway_in_last": f"{runaway_in_last:,}",
        "capped": f"${capped:.2f}",
        "capped_in_last": f"{capped_in_last:,}",
        "confirm_seconds": f"{CONFIRM_SECONDS}",
        "confirm_cost": usd(confirm_cost),
        "write_share": pct(WRITE_SHARE),
        "secs_v1": f"{secs(q_v1_g):.0f}",
        "secs_v0": f"{secs(q_v0):.0f}",
        "q_haiku": usd(q_haiku_g),
        "sonnet_over_haiku": f"{q_v1_g / q_haiku_g:.0f}",
        "task_set": f"{TASK_SET}",
        "eval_runs": f"{EVAL_RUNS}",
        "retention": f"{RETENTION_DAYS}",
        "agent_hourly": usd(AGENT_HOURLY),
        "carry_call": usd(TOOL_STEPS * (RESULT_V1 + TOOL_USE_OUT) * sonnet["cache_read"] / 1e6),
        "none_rewrite": usd(none_rewrite),
        "max_tools_step": f"{MAX_TOOLS_PER_STEP}",
        "max_out": f"{MAX_OUT:,}",
        "step_cost_early": usd(step_at(2)),
        "step_cost_late": usd(step_at(RUNAWAY_STEPS)),
        "tasks_hour_peak": num(tasks_hour_peak),
        "upstream_storm": num(tasks_hour_peak * RUNAWAY_STEPS),
        "rule3_months": f"{3 / breakeven['retry_payroll_run'] / (tasks['growth'] * WRITE_SHARE):.0f}",
        "be_retry_half_one_in": num(1 / breakeven_half["retry_payroll_run"]),
        "be_resend_half_one_in": num(1 / breakeven_half["resend_payslips"]),
        "catch_half": pct(half),
        "haiku_secs": f"{secs(q_v1_g - q_haiku_g):.1f}",
        "gate_bound_set": pct(3 / TASK_SET),
        "gate_bound_inj": pct(3 / 20),
        "reorder_call": usd(sum(i for i, _ in v1) / CALLS * sonnet["cache_write"] / 1e6),
        "confirm_vs_retry": f"{tasks['growth'] * WRITE_SHARE * confirm_cost / WRONG_ACTION['retry_payroll_run'][0]:.1f}",
        "ci_tasks": f"{1.96 * (0.25 / TASK_SET) ** 0.5 * 100:.0f}",
        "rule3_retry": num(3 / breakeven["retry_payroll_run"]),
        "rule3_resend": num(3 / breakeven["resend_payslips"]),
        "confirm_month_g": usd(tasks["growth"] * WRITE_SHARE * confirm_cost),
    }
    for k, (c, _) in WRONG_ACTION.items():
        short = {
            "add_internal_note": "note",
            "resend_payslips": "resend",
            "retry_payroll_run": "retry",
        }[k]
        v[f"wrong_{short}"] = usd(c)
        be = breakeven[k]
        v[f"be_{short}"] = f"{be * 100:.2g}%" if be >= 0.001 else f"{be * 100:.1g}%"
        v[f"be_{short}_one_in"] = num(1 / be)

    # Tool catalogue table.
    cat = Table("Tool", "Kind", "Tokens (assumed)", "What it does", align="llrl")
    for name, kind, tok, what in TOOLS:
        cat.add(f"`{name}`", kind, f"{tok}", what)
    cat.add(
        "**All nine**",
        "",
        f"**{V1_DEFS:,}**",
        f"Plus the {TOOL_SYS}-token tool-use system prompt the API adds",
    )

    # Per-call token table, version 0 against version 1.
    calls_t = Table(
        "Model call", "Version 0 input", "Version 1 input", "Output", align="lrrr"
    )
    for n, ((i0, o0), (i1, _)) in enumerate(zip(v0, v1), start=1):
        calls_t.add(
            f"{n}" + (" (answer)" if n == CALLS else ""), f"{i0:,}", f"{i1:,}", f"{o0}"
        )
    calls_t.add(
        "**Task**",
        f"**{v0_in:,}, {usd(q_v0)}**",
        f"**{v1_in:,}, {usd(q_v1_u)} uncached, {usd(q_v1_g)} cached**",
        f"**{v0_out:,}**",
    )

    # Confirmation break-even table.
    conf = Table(
        "Write tool",
        "Cost of one wrong execution (assumed)",
        "Confirmation pays above, agent catches every error",
        f"Agent catches {pct(half)}",
        "Decision",
        align="lrrrl",
    )
    decisions = {
        "add_internal_note": "No confirmation; plain text only, logged, shown in the answer",
        "resend_payslips": "Confirm every call",
        "retry_payroll_run": "Confirm every call",
    }

    def fmt(be):
        bes = f"{be * 100:.2g}%" if be >= 0.001 else f"{be * 100:.1g}%"
        return f"{bes} (1 in {num(1 / be)})" if be < 1 else "never"

    for k, (c, why) in WRONG_ACTION.items():
        conf.add(f"`{k}`", f"{usd(c)}: {why}", fmt(breakeven[k]), fmt(breakeven_half[k]), decisions[k])

    # Monthly cost.
    cost = Table(
        "Line item",
        "Pilot",
        "Growth",
        "Large",
        "How it is calculated (growth)",
        align="lrrrl",
    )
    rows = {}
    q_judge = llm_cost(judge, JUDGE_IN, JUDGE_OUT)
    eval_task = task_cost_uncached(
        sonnet, V1_DEFS, RESULT_V1, CALLS
    )  # fixtures run cold
    for s, sc in SCALES.items():
        t = tasks[s]
        confirms = t * WRITE_SHARE
        tokens_month = t * (v1_in + sum(o for _, o in v1))
        r = {
            "model": t * q_v1[s],
            "app": sc["tasks"]
            * HOURS_PER_MONTH
            * (sc["vcpu"] * far["vcpu_hourly"] + sc["mem_gb"] * far["gb_hourly"]),
            "db": (db["hourly_multi_az"] if sc["multi_az"] else db["hourly"])
            * HOURS_PER_MONTH
            + DISK_GB
            * (
                disk["gp3_storage_gb_month_multi_az"]
                if sc["multi_az"]
                else disk["gp3_storage_gb_month"]
            ),
            "eval": EVAL_RUNS * TASK_SET * (eval_task + q_judge),
        }
        r["total"] = sum(r.values())
        rows[s] = r
        v[f"{s}.tasks"] = num(t)
        v[f"{s}.hit"] = "over 99%" if hits[s] > 0.99 else pct(hits[s])
        v[f"{s}.q"] = usd(q_v1[s])
        v[f"{s}.model"] = usd(r["model"])
        v[f"{s}.total"] = usd(r["total"])
        v[f"{s}.per_task"] = usd(r["total"] / t)
        v[f"{s}.secs"] = f"{secs(r['total'] / t):.0f}"
        v[f"{s}.confirms"] = num(confirms)
        v[f"{s}.confirm_hours"] = num(confirms * CONFIRM_SECONDS / 3600)
        v[f"{s}.v0_model"] = usd(t * q_v0)
        v[f"{s}.audit_gb"] = (
            f"{tokens_month * BYTES_PER_TOKEN / 1e9 * RETENTION_DAYS / 30.4:.1f}"
        )
        v[f"{s}.haiku_model"] = usd(
            t * task_cost_cached(haiku, V1_DEFS, RESULT_V1, CALLS, hits[s])
        )
    g = rows["growth"]
    how = {
        "model": f"{num(tasks['growth'])} tasks x {usd(q_v1_g)} (Sonnet 5.5, {CALLS} calls, cached, prefix hit rate over 99%)",
        "app": "Fargate: the loop, the tool executor, the confirmation API; two tasks (three at large)",
        "db": f"db.t4g.medium: conversations, proposals, audit log; {DISK_GB} GB",
        "eval": f"{EVAL_RUNS} runs x {TASK_SET} tasks on fixtures, uncached, plus a judge (GPT-6.1 Terra)",
    }
    labels = {
        "model": "Model calls",
        "app": "Assistant service",
        "db": "Postgres",
        "eval": "Evaluation runs",
    }
    for k, label in labels.items():
        cost.add(label, *(usd(rows[s][k]) for s in SCALES), how[k])
    cost.add(
        "**Total per month**", *(f"**{usd(rows[s]['total'])}**" for s in SCALES), ""
    )
    cost.add(
        "Tasks per month",
        *(num(tasks[s]) for s in SCALES),
        f"{pct(USE_SHARE)} of System 1's tickets",
    )
    cost.add("Cost per task", *(usd(rows[s]["total"] / tasks[s]) for s in SCALES), "")
    cost.add(
        "Break-even agent time saved per task",
        *(f"{secs(rows[s]['total'] / tasks[s]):.0f} s" for s in SCALES),
        f"Cost per task at {usd(AGENT_HOURLY)} an hour of agent time",
    )

    return {
        "tables": {
            "catalogue": cat.md(),
            "calls": calls_t.md(),
            "confirm": conf.md(),
            "monthly": cost.md(),
        },
        "vars": v,
    }
