"""System 28: Call-centre voice agent. Per-minute and monthly cost of the cascaded pipeline.

Every constant below is an assumption of the worked example, not a measurement. Replace them
with numbers from your own call records and traces.
"""

from _lib import DAYS_PER_MONTH, HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch28"

SCALES = {"pilot": 500, "target": 12_000, "large": 60_000}  # inbound calls per day

CONTAINMENT = 0.60  # share of calls the agent completes without a person
AHT_CONTAINED = 3.5  # minutes, calls the agent completes
AI_MIN_TRANSFERRED = 1.5  # minutes the agent spends before handing over
HUMAN_MIN = (
    6.0  # minutes a transferred call stays on Twilio after the handover (queue + talk)
)
TURNS_CONTAINED = 8
TURNS_TRANSFERRED = 4
VERIFY_TURN = 3  # turn at which identity is usually verified

# F2 rule for business traffic: 80% of the day in about 8 hours, peak hour about twice the
# average of those hours.
BUSY_SHARE, BUSY_HOURS, PEAK_FACTOR = 0.80, 8, 2.0
PEAK_HOUR_SHARE = BUSY_SHARE / BUSY_HOURS * PEAK_FACTOR
QUOTA_HEADROOM = 2.0  # vendor quotas negotiated to this multiple of the peak
CALLS_PER_NODE = 40  # planning number for one orchestration node; immaterial to cost

PREFIX = 2_500  # tool definitions + system prompt, identical across calls
PER_TURN_NEW = 150  # caller utterance + tool result added per turn
PER_TURN_OUT = 110  # agent reply + any tool call
SPOKEN_PER_TURN = 45  # of which spoken aloud: about two short sentences
CHARS_PER_OUT_TOKEN = 4.2  # spoken characters per output token, English
FIXED_CHARS_PER_CALL = 400  # greeting, disclosures, verification prompts, fillers
AGENT_TALK_SHARE = 0.5  # share of an AI minute in which the agent is speaking

RECORDING_MB_PER_MIN = 0.48  # 8 kHz mono, 64 kbit/s
RETENTION_MONTHS = 12
ANALYTICS_PROMPT, ANALYTICS_OUT = 500, 180  # post-call summary and disposition


def turns_cost(p: dict, turns: int, cached: bool) -> tuple[float, int, int]:
    """Model cost for one call, plus input tokens sent and tokens read from cache.

    With caching, the prefix (tools + system prompt) is the same bytes on every call, so it
    stays warm across calls and is read, not written, even on the first turn. Each turn
    writes only what it adds.
    """
    total = 0.0
    sent = read = 0
    for t in range(1, turns + 1):
        history = (PER_TURN_NEW + PER_TURN_OUT) * (t - 1)
        if cached:
            prior = PREFIX + history
            write = PER_TURN_NEW + (PER_TURN_OUT if t > 1 else 0)
            prior -= PER_TURN_OUT if t > 1 else 0
            total += llm_cost(p, 0, PER_TURN_OUT, cache_read=prior, cache_write=write)
            read += prior
            sent += prior + write
        else:
            inp = PREFIX + history + PER_TURN_NEW
            total += llm_cost(p, inp, PER_TURN_OUT)
            sent += inp
    return total, sent, read


def tool_switch_cost(p: dict) -> tuple[float, int]:
    """Extra cost when the tool list changes at verification.

    Tools come first in the cache order, so a new tool list invalidates everything after it.
    The post-verification prefix is shared across calls and stays warm, but this call's
    history so far must be written again instead of read.
    """
    history = (PER_TURN_NEW + PER_TURN_OUT) * (VERIFY_TURN - 1)
    return history * (p["cache_write"] - p["cache_read"]) / 1e6, history


def build(P, H):
    sonnet, haiku = (
        P("llm.anthropic.claude-sonnet-5-5"),
        P("llm.anthropic.claude-haiku-5-5"),
    )
    stt_entry = P("voice.stt.deepgram.nova-3-streaming")
    stt, stt_multi = stt_entry["price"], stt_entry["price_multilingual"]
    tts = P("voice.tts.elevenlabs.flash-turbo")["price"]
    tts_alt = P("voice.tts.deepgram.aura-2")["price"]
    tw = P("telephony.twilio.us")
    media = P("telephony.twilio.media-streams")["price"]
    node = P("infra.aws.ec2.c7i-2xlarge")["hourly"]
    s3 = P("infra.aws.s3.standard")["storage_gb_month_first_50tb"]

    ai_min = CONTAINMENT * AHT_CONTAINED + (1 - CONTAINMENT) * AI_MIN_TRANSFERRED
    turns = CONTAINMENT * TURNS_CONTAINED + (1 - CONTAINMENT) * TURNS_TRANSFERRED

    llm_nc, sent_nc, _ = turns_cost(sonnet, TURNS_CONTAINED, cached=False)
    llm_c, sent_c, read_c = turns_cost(sonnet, TURNS_CONTAINED, cached=True)
    llm_h, _, _ = turns_cost(haiku, TURNS_CONTAINED, cached=True)
    llm_c_tr, _, _ = turns_cost(sonnet, TURNS_TRANSFERRED, cached=True)
    switch, switch_tokens = tool_switch_cost(sonnet)
    per_turn = lambda c: c / TURNS_CONTAINED  # noqa: E731

    gen_chars_turn = SPOKEN_PER_TURN * CHARS_PER_OUT_TOKEN
    analytics_in = lambda n: ANALYTICS_PROMPT + n * (PER_TURN_NEW + PER_TURN_OUT)  # noqa: E731
    # Batch is not HIPAA-eligible on the Claude API, so post-call analysis runs at standard price.
    analytics_call = CONTAINMENT * llm_cost(
        haiku, analytics_in(TURNS_CONTAINED), ANALYTICS_OUT
    ) + (1 - CONTAINMENT) * llm_cost(
        haiku, analytics_in(TURNS_TRANSFERRED), ANALYTICS_OUT
    )

    tel_min = tw["inbound_local"] + media
    after_transfer = (tw["inbound_local"] + tw["sip_or_app"]) * HUMAN_MIN

    v = {
        "ai_min": f"{ai_min:.1f}",
        "turns": f"{turns:.1f}",
        "llm_call_nocache": usd(llm_nc),
        "llm_call_cache": usd(llm_c),
        "llm_call_haiku": usd(llm_h),
        "tokens_sent": f"{sent_nc:,}",
        "cache_share": pct(read_c / sent_c),
        "cache_saving": pct(1 - llm_c / llm_nc),
        "switch_tokens": f"{switch_tokens:,}",
        "switch_cost": usd(switch),
        "switch_share": pct(switch / llm_c),
        "gen_chars_call": f"{gen_chars_turn * TURNS_CONTAINED:,.0f}",
        "fixed_chars_call": f"{FIXED_CHARS_PER_CALL:,}",
        "fixed_share": pct(
            FIXED_CHARS_PER_CALL
            / (FIXED_CHARS_PER_CALL + gen_chars_turn * TURNS_CONTAINED)
        ),
        "tts_price": f"${tts:.2f}",
        "tts_alt_price": f"${tts_alt:.3f}",
        "stt_price": f"${stt:.4f}",
        "stt_multi_price": f"${stt_multi:.4f}",
        "analytics_in": f"{analytics_in(TURNS_CONTAINED):,}",
        "peak_share": pct(PEAK_HOUR_SHARE),
        "human_min": f"{HUMAN_MIN:.0f}",
        "after_transfer": usd(after_transfer),
    }

    def per_min(
        model_call: float,
        synth_fixed: bool,
        tts_rate: float = tts,
        stt_rate: float = stt,
    ) -> dict:
        call_llm = per_turn(model_call) * turns
        call_chars = gen_chars_turn * turns + (
            FIXED_CHARS_PER_CALL if synth_fixed else 0
        )
        return {
            "Telephony (inbound local + media stream)": tel_min,
            "Speech to text (streamed for the whole call)": stt_rate,
            "Model": call_llm / ai_min,
            "Text to speech": call_chars / 1000 * tts_rate / ai_min,
        }

    final = per_min(llm_c, synth_fixed=False)
    naive = per_min(llm_nc, synth_fixed=True)
    cheap = per_min(llm_h, synth_fixed=False, tts_rate=tts_alt)
    t = Table(
        "Per AI-handled minute",
        "Version 1 (no caching, all speech synthesised)",
        "Final design",
        "With Haiku 5.5 and Aura-2",
        align="lrrr",
    )
    for k in final:
        t.add(k, usd(naive[k]), usd(final[k]), usd(cheap[k]))
    t.add(
        "**Total per minute**",
        f"**{usd(sum(naive.values()))}**",
        f"**{usd(sum(final.values()))}**",
        f"**{usd(sum(cheap.values()))}**",
    )
    tables = {"per-minute": t.md()}
    total_min = sum(final.values())
    v["min_naive"] = usd(sum(naive.values()))
    v["min_final"] = usd(total_min)
    v["min_cheap"] = usd(sum(cheap.values()))
    v["tts_share_final"] = pct(final["Text to speech"] / total_min)
    v["tel_share_final"] = pct(
        final["Telephony (inbound local + media stream)"] / total_min
    )
    v["stt_share_final"] = pct(
        final["Speech to text (streamed for the whole call)"] / total_min
    )
    v["model_share_final"] = pct(final["Model"] / total_min)
    v["call_final"] = usd(total_min * ai_min)
    v["multi_extra"] = usd(
        (stt_multi - stt) * ai_min * SCALES["target"] * DAYS_PER_MONTH
    )
    stt_pause = stt * AGENT_TALK_SHARE
    v["stt_pause_saving"] = usd(stt_pause)
    v["stt_pause_share"] = pct(stt_pause / total_min)
    v["agent_talk_share"] = pct(AGENT_TALK_SHARE)

    # Cost of one call by outcome, final design, including post-call analysis.
    audio_min = tel_min + stt
    contained = (
        audio_min * AHT_CONTAINED
        + llm_c
        + gen_chars_turn * TURNS_CONTAINED / 1000 * tts
        + llm_cost(haiku, analytics_in(TURNS_CONTAINED), ANALYTICS_OUT)
    )
    transferred = (
        audio_min * AI_MIN_TRANSFERRED
        + llm_c_tr
        + gen_chars_turn * TURNS_TRANSFERRED / 1000 * tts
        + llm_cost(haiku, analytics_in(TURNS_TRANSFERRED), ANALYTICS_OUT)
    )
    v["contained_call"] = usd(contained)
    v["transferred_call"] = usd(transferred)
    v["transferred_call_tel"] = usd(transferred + after_transfer)

    # Monthly at three scales.
    m = Table(
        "Line item",
        "Pilot",
        "Target",
        "Large",
        "How it is calculated (target)",
        align="lrrrl",
    )
    rows = {}
    for name, calls in SCALES.items():
        month_calls = calls * DAYS_PER_MONTH
        minutes = month_calls * ai_min
        peak_calls = calls * PEAK_HOUR_SHARE
        peak_conc = peak_calls * ai_min / 60
        quota = peak_conc * QUOTA_HEADROOM
        nodes = max(2, -(-peak_conc // CALLS_PER_NODE) + 1)
        rec_gb = minutes * RECORDING_MB_PER_MIN / 1000
        r = {k: val * minutes for k, val in final.items()}
        r["Orchestration nodes (planning number)"] = nodes * node * HOURS_PER_MONTH
        r["Recordings (12 months kept)"] = rec_gb * RETENTION_MONTHS * s3
        r["Post-call analytics"] = month_calls * analytics_call
        r["Telephony after transfer"] = month_calls * (1 - CONTAINMENT) * after_transfer
        r_total = sum(r.values())
        rows[name] = (r, r_total, minutes, peak_conc, nodes)
        v[f"{name}.calls"] = f"{calls:,}"
        v[f"{name}.minutes"] = num(minutes)
        v[f"{name}.peak_calls"] = f"{peak_calls:,.0f}"
        v[f"{name}.peak"] = f"{peak_conc:.0f}"
        v[f"{name}.quota"] = f"{quota:,.0f}"
        v[f"{name}.nodes"] = f"{nodes:.0f}"
        v[f"{name}.total"] = usd(r_total)
        v[f"{name}.per_call"] = usd(r_total / month_calls)
        v[f"{name}.nodes_share"] = pct(
            r["Orchestration nodes (planning number)"] / r_total
        )
    tr, _, tmin, tpeak, tnodes = rows["target"]
    how = {
        "Telephony (inbound local + media stream)": f"{num(tmin)} AI minutes x {usd(final['Telephony (inbound local + media stream)'])}",
        "Speech to text (streamed for the whole call)": f"{num(tmin)} minutes x {usd(stt)}, monolingual model chosen by keypad",
        "Model": f"Sonnet 5.5 with caching, {usd(final['Model'])} a minute",
        "Text to speech": "Generated speech only; fixed phrases pre-rendered. ElevenLabs Flash, per character",
        "Orchestration nodes (planning number)": f"{tnodes:.0f} x c7i.2xlarge for a peak of {tpeak:.0f} concurrent calls",
        "Recordings (12 months kept)": "64 kbit/s mono in S3, steady state after a year",
        "Post-call analytics": "Haiku 5.5 summary per call, standard API (batch is not HIPAA-eligible)",
        "Telephony after transfer": f"Transferred calls stay on Twilio: {HUMAN_MIN:.0f} min x (inbound + SIP)",
    }
    for k in tr:
        m.add(k, *(usd(rows[s][0][k]) for s in SCALES), how[k])
    m.add("**Total per month**", *(f"**{usd(rows[s][1])}**" for s in SCALES), "")
    m.add(
        "Cost per call",
        *(usd(rows[s][1] / (SCALES[s] * DAYS_PER_MONTH)) for s in SCALES),
        "",
    )
    tables["monthly"] = m.md()
    return {"tables": tables, "vars": v}
