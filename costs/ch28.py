"""System 28: Call-centre voice agent. Per-minute and monthly cost of the cascaded pipeline."""

from _lib import DAYS_PER_MONTH, HOURS_PER_MONTH, Table, llm_cost, num, pct, usd

ID = "ch28"

SCALES = {"pilot": 500, "target": 12_000, "large": 60_000}  # inbound calls per day

CONTAINMENT = 0.60  # share of calls the agent completes without a person
AHT_CONTAINED = 3.5  # minutes, calls the agent completes
AI_MIN_TRANSFERRED = 1.5  # minutes the agent spends before handing over
TURNS_CONTAINED = 8
TURNS_TRANSFERRED = 4
PEAK_HOUR_SHARE = 0.12  # of the day's calls arrive in the busiest hour
CALLS_PER_NODE = 40  # concurrent calls one orchestration node carries with headroom

PREFIX = 2_500  # system prompt + tool definitions, stable
PER_TURN_NEW = 150  # caller utterance + tool result added per turn
PER_TURN_OUT = 110  # agent reply + any tool call
SPOKEN_PER_TURN = 45  # of which spoken aloud: about two short sentences
CHARS_PER_OUT_TOKEN = 4.2  # spoken characters per output token, English
FIXED_SPEECH_SHARE = 0.30  # greeting, disclosures, confirmations: pre-rendered audio

RECORDING_MB_PER_MIN = 0.48  # 8 kHz mono, 64 kbit/s
RETENTION_MONTHS = 12
ANALYTICS_IN, ANALYTICS_OUT = 1_400, 180  # post-call summary and disposition, batch


def turns_cost(p: dict, turns: int, cached: bool) -> tuple[float, int, int]:
    """Model cost for one call, plus input tokens sent and tokens read from cache."""
    total = 0.0
    sent = read = 0
    for t in range(1, turns + 1):
        history = (PER_TURN_NEW + PER_TURN_OUT) * (t - 1)
        if cached:
            # Everything up to the previous turn is read from cache; the new turn is written.
            prior = PREFIX + history if t > 1 else 0
            write = PER_TURN_NEW + (PREFIX if t == 1 else PER_TURN_OUT)
            total += llm_cost(p, 0, PER_TURN_OUT, cache_read=prior, cache_write=write)
            read += prior
            sent += prior + write
        else:
            inp = PREFIX + history + PER_TURN_NEW
            total += llm_cost(p, inp, PER_TURN_OUT)
            sent += inp
    return total, sent, read


def build(P, H):
    sonnet, haiku = (
        P("llm.anthropic.claude-sonnet-5-5"),
        P("llm.anthropic.claude-haiku-5-5"),
    )
    stt = P("voice.stt.deepgram.nova-3-streaming")["price"]
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
    per_turn = lambda c: c / TURNS_CONTAINED  # noqa: E731

    chars_per_turn = SPOKEN_PER_TURN * CHARS_PER_OUT_TOKEN
    v = {
        "ai_min": f"{ai_min:.1f}",
        "turns": f"{turns:.1f}",
        "llm_call_nocache": usd(llm_nc),
        "llm_call_cache": usd(llm_c),
        "llm_call_haiku": usd(llm_h),
        "tokens_sent": f"{sent_nc:,}",
        "cache_share": pct(read_c / sent_c),
        "cache_saving": pct(1 - llm_c / llm_nc),
        "chars_call": f"{chars_per_turn * TURNS_CONTAINED:,.0f}",
        "tts_call": usd(chars_per_turn * TURNS_CONTAINED / 1000 * tts),
        "tts_price": f"${tts:.2f}",
        "tts_alt_price": f"${tts_alt:.2f}",
        "stt_price": f"${stt:.4f}",
    }

    # Per AI-handled minute, final design (Sonnet 5.5 with caching, fixed phrases pre-rendered).
    def per_min(
        model_call: float,
        tts_share: float = 1 - FIXED_SPEECH_SHARE,
        tts_rate: float = tts,
    ) -> dict:
        llm_per_turn = per_turn(model_call)
        call_llm = llm_per_turn * turns
        call_chars = chars_per_turn * turns * tts_share
        return {
            "Telephony (inbound local + media stream)": tw["inbound_local"] + media,
            "Speech to text (streamed for the whole call)": stt,
            "Model": call_llm / ai_min,
            "Text to speech": call_chars / 1000 * tts_rate / ai_min,
        }

    final = per_min(llm_c)
    naive = per_min(llm_nc, tts_share=1.0)
    cheap = per_min(llm_h, tts_rate=tts_alt)
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
    v["min_naive"] = usd(sum(naive.values()))
    v["min_final"] = usd(sum(final.values()))
    v["min_cheap"] = usd(sum(cheap.values()))
    v["tts_share_final"] = pct(final["Text to speech"] / sum(final.values()))
    v["model_share_final"] = pct(final["Model"] / sum(final.values()))
    v["call_final"] = usd(sum(final.values()) * ai_min)

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
        minutes = calls * DAYS_PER_MONTH * ai_min
        peak_conc = calls * PEAK_HOUR_SHARE * ai_min / 60
        nodes = max(2, -(-peak_conc // CALLS_PER_NODE) + 1)
        rec_gb = calls * DAYS_PER_MONTH * ai_min * RECORDING_MB_PER_MIN / 1000
        r = {k: val * minutes for k, val in final.items()}
        r["Orchestration nodes"] = nodes * node * HOURS_PER_MONTH
        r["Recordings (12 months kept)"] = rec_gb * RETENTION_MONTHS * s3
        r["Post-call analytics (batch)"] = (
            calls
            * DAYS_PER_MONTH
            * llm_cost(haiku, ANALYTICS_IN, ANALYTICS_OUT, batch=True)
        )
        r_total = sum(r.values())
        rows[name] = (r, r_total, minutes, peak_conc, nodes)
        v[f"{name}.calls"] = f"{calls:,}"
        v[f"{name}.minutes"] = num(minutes)
        v[f"{name}.peak"] = f"{peak_conc:.0f}"
        v[f"{name}.nodes"] = f"{nodes:.0f}"
        v[f"{name}.total"] = usd(r_total)
        v[f"{name}.per_call"] = usd(r_total / (calls * DAYS_PER_MONTH))
    tr, _, tmin, tpeak, tnodes = rows["target"]
    how = {
        "Telephony (inbound local + media stream)": f"{num(tmin)} AI minutes x {usd(final['Telephony (inbound local + media stream)'])}",
        "Speech to text (streamed for the whole call)": f"{num(tmin)} minutes x {usd(stt)}",
        "Model": f"Sonnet 5.5 with caching, {usd(final['Model'])} a minute",
        "Text to speech": f"{pct(1 - FIXED_SPEECH_SHARE)} of speech synthesised, ElevenLabs Flash",
        "Orchestration nodes": f"{tnodes:.0f} x c7i.2xlarge for a peak of {tpeak:.0f} concurrent calls",
        "Recordings (12 months kept)": "64 kbit/s mono in S3, steady state after a year",
        "Post-call analytics (batch)": "Haiku 5.5 summary per call, batch API",
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
