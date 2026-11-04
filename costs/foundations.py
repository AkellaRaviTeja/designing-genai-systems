"""Price reference tables and the arithmetic used in the foundation chapters."""

from _lib import HOURS_PER_MONTH, Table, num, usd, usd_exact

ID = "f"

# The models the book treats as current, grouped by tier. Tiers are the book's judgement:
# "frontier" for the hardest reasoning and agent work, "workhorse" for most product
# features, "small" for high-volume classification, extraction and routing.
CURRENT = [
    ("frontier", "llm.anthropic.claude-fable-5-1", "Claude Fable 5.1"),
    ("frontier", "llm.openai.gpt-6-astra", "GPT-6 Astra"),
    ("frontier", "llm.anthropic.claude-opus-5-5", "Claude Opus 5.5"),
    ("frontier", "llm.google.gemini-4-argon", "Gemini 4 Argon"),
    ("workhorse", "llm.anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("workhorse", "llm.openai.gpt-6-1-sol", "GPT-6.1 Sol"),
    ("workhorse", "llm.google.gemini-3-1-pro-preview", "Gemini 3.1 Pro (preview)"),
    ("workhorse", "llm.openai.gpt-6-1-terra", "GPT-6.1 Terra"),
    ("workhorse", "llm.google.gemini-3-8-flash", "Gemini 3.8 Flash"),
    ("small", "llm.anthropic.claude-haiku-5-5", "Claude Haiku 5.5"),
    ("small", "llm.openai.gpt-6-luna", "GPT-6 Luna"),
    ("small", "llm.google.gemini-3-5-flash-lite", "Gemini 3.5 Flash-Lite"),
    ("small", "llm.google.gemini-3-1-flash-lite", "Gemini 3.1 Flash-Lite"),
]

HOSTED = [
    ("hosted.fireworks.gpt-oss-120b", "gpt-oss-120b", "Fireworks"),
    ("hosted.groq.gpt-oss-120b", "gpt-oss-120b", "Groq"),
    ("hosted.groq.gpt-oss-20b", "gpt-oss-20b", "Groq"),
    ("hosted.together.deepseek-v4-1-flash", "DeepSeek V4.1 Flash", "Together"),
    ("hosted.together.deepseek-v4-pro-0813", "DeepSeek V4 Pro", "Together"),
    ("hosted.together.qwen3-5-9b", "Qwen3.5 9B", "Together"),
    ("hosted.groq.qwen3-8-27b", "Qwen3.8 27B", "Groq"),
]

GPUS = [
    ("gpu.aws.g6-xlarge", "AWS g6.xlarge"),
    ("gpu.aws.g5-xlarge", "AWS g5.xlarge"),
    ("gpu.aws.g6e-xlarge", "AWS g6e.xlarge"),
    ("gpu.aws.g6e-12xlarge", "AWS g6e.12xlarge"),
    ("gpu.aws.p5-48xlarge", "AWS p5.48xlarge"),
    ("gpu.aws.p5en-48xlarge", "AWS p5en.48xlarge"),
    ("gpu.aws.p6-b200-48xlarge", "AWS p6-b200.48xlarge"),
    ("gpu.gcp.a3-highgpu-8g", "GCP a3-highgpu-8g"),
    ("gpu.coreweave.hgx-h100", "CoreWeave HGX H100"),
    ("gpu.coreweave.hgx-h200", "CoreWeave HGX H200"),
    ("gpu.coreweave.hgx-b200", "CoreWeave HGX B200"),
    ("gpu.lambda.h100-8x", "Lambda 8x H100"),
    ("gpu.lambda.b200-8x", "Lambda 8x B200"),
    ("gpu.runpod.h100-sxm", "RunPod H100 SXM"),
    ("gpu.runpod.h200", "RunPod H200"),
]


def price_note(p: dict) -> str:
    notes = []
    for k, v in p.items():
        if k.startswith("price_from_") and isinstance(v, dict):
            date = k.removeprefix("price_from_").replace("_", "-")
            notes.append(
                f"rises to {usd_exact(v['input'])} / {usd_exact(v['output'])} on {date}"
            )
    if "long_context_threshold_tokens" in p:
        notes.append(
            f"higher rate above {p['long_context_threshold_tokens'] // 1000}K-token prompts"
        )
    return "; ".join(notes)


def build(P, H):
    tables, vars_ = {}, {}

    # LLM API price table, by tier.
    t = Table(
        "Tier",
        "Model",
        "Input",
        "Output",
        "Cache read",
        "Batch",
        "Note",
        align="llrrrrl",
    )
    for tier, key, name in CURRENT:
        p = P(key)
        t.add(
            tier,
            name,
            usd_exact(p["input"]),
            usd_exact(p["output"]),
            usd_exact(p["cache_read"]),
            f"-{p['batch_discount'] * 100:.0f}%",
            price_note(p),
        )
    tables["llm-prices"] = t.md()

    t = Table("Model", "Provider", "Input", "Output", align="llrr")
    for key, name, vendor in HOSTED:
        p = P(key)
        t.add(name, vendor, usd_exact(p["input"]), usd_exact(p["output"]))
    tables["hosted-prices"] = t.md()

    t = Table("Model", "Kind", "Price per 1M tokens", align="llr")
    for key, name in [
        ("embed.openai.text-embedding-3-small", "OpenAI text-embedding-3-small"),
        ("embed.openai.text-embedding-3-large", "OpenAI text-embedding-3-large"),
        ("embed.voyage.voyage-4-lite", "Voyage 4 Lite"),
        ("embed.voyage.voyage-4", "Voyage 4"),
        ("embed.voyage.voyage-4-large", "Voyage 4 Large"),
        ("embed.cohere.embed-5-fast", "Cohere Embed 5 Fast"),
        ("embed.cohere.embed-5-pro", "Cohere Embed 5 Pro"),
    ]:
        t.add(name, "embedding", usd_exact(P(key)["input"]))
    for key, name in [
        ("rerank.cohere.rerank-5-fast", "Cohere Rerank 5 Fast"),
        ("rerank.cohere.rerank-5-pro", "Cohere Rerank 5 Pro"),
        ("rerank.voyage.rerank-3-lite", "Voyage Rerank 3 Lite"),
        ("rerank.voyage.rerank-3", "Voyage Rerank 3"),
    ]:
        t.add(name, "rerank", usd_exact(P(key)["input"]))
    tables["embed-prices"] = t.md()

    t = Table(
        "Instance",
        "GPUs",
        "Per hour",
        "Per GPU-hour",
        "Per month (730 h)",
        align="llrrr",
    )
    for key, name in GPUS:
        g = P(key)
        n = g.get("gpus", 1)
        hourly = g.get("hourly") or g["per_gpu_hourly"] * n
        t.add(
            name,
            f"{n}x {g['gpu']}",
            usd_exact(round(hourly, 2)),
            usd_exact(round(hourly / n, 2)),
            usd(hourly * HOURS_PER_MONTH),
        )
    tables["gpu-prices"] = t.md()

    t = Table("Service", "Kind", "Price", align="llr")
    t.add(
        "Deepgram Nova-3 streaming",
        "speech to text",
        f"{usd_exact(P('voice.stt.deepgram.nova-3-streaming')['price'])} / min",
    )
    t.add(
        "AssemblyAI Universal streaming",
        "speech to text",
        f"{usd_exact(P('voice.stt.assemblyai.universal-streaming')['price'])} / hour",
    )
    t.add(
        "ElevenLabs Scribe v2 realtime",
        "speech to text",
        f"{usd_exact(P('voice.stt.elevenlabs.scribe-v2-realtime')['price'])} / hour",
    )
    t.add(
        "ElevenLabs Flash",
        "text to speech",
        f"{usd_exact(P('voice.tts.elevenlabs.flash-turbo')['price'])} / 1K chars",
    )
    t.add(
        "Cartesia Sonic (Scale plan)",
        "text to speech",
        f"{usd_exact(P('voice.tts.cartesia.sonic-scale')['price'])} / 1K chars",
    )
    t.add(
        "Deepgram Aura-2",
        "text to speech",
        f"{usd_exact(P('voice.tts.deepgram.aura-2')['price'])} / 1K chars",
    )
    tw = P("telephony.twilio.us")
    t.add(
        "Twilio US local inbound",
        "telephony",
        f"{usd_exact(tw['inbound_local'])} / min",
    )
    t.add(
        "Twilio US toll-free inbound",
        "telephony",
        f"{usd_exact(tw['inbound_toll_free'])} / min",
    )
    t.add(
        "Twilio Media Streams",
        "telephony",
        f"{usd_exact(P('telephony.twilio.media-streams')['price'])} / min",
    )
    tables["voice-prices"] = t.md()

    # Prompt caching break-even (F5, O1): requests sharing a prefix before caching pays.
    t = Table(
        "Model",
        "Write premium",
        "Read price",
        "Cost of 10 calls, cached",
        "Saving on the prefix",
        align="lrrrr",
    )
    for key, name in [
        ("llm.anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5"),
        ("llm.anthropic.claude-opus-5-5", "Claude Opus 5.5"),
        ("llm.anthropic.claude-haiku-5-5", "Claude Haiku 5.5"),
        ("llm.openai.gpt-6-1-sol", "GPT-6.1 Sol"),
    ]:
        p = P(key)
        w, r = p["cache_write"] / p["input"], p["cache_read"] / p["input"]
        ten = w + 9 * r
        t.add(
            name,
            f"{w:.2f}x",
            f"{r:g}x",
            f"{ten:.2f}x vs 10x",
            f"{(1 - ten / 10) * 100:.0f}%",
        )
    tables["cache-breakeven"] = t.md()
    son = P("llm.anthropic.claude-sonnet-5-5")
    vars_["sonnet.in"] = usd_exact(son["input"])
    vars_["sonnet.out"] = usd_exact(son["output"])
    vars_["sonnet.read"] = usd_exact(son["cache_read"])
    vars_["sonnet.write"] = usd_exact(son["cache_write"])
    hai = P("llm.anthropic.claude-haiku-5-5")
    vars_["haiku.in"] = usd_exact(hai["input"])
    vars_["haiku.out"] = usd_exact(hai["output"])
    opus = P("llm.anthropic.claude-opus-5-5")
    vars_["opus.in"] = usd_exact(opus["input"])
    vars_["opus.out"] = usd_exact(opus["output"])
    fable = P("llm.anthropic.claude-fable-5-1")
    vars_["fable.in"] = usd_exact(fable["input"])
    vars_["fable.out"] = usd_exact(fable["output"])
    vars_["ratio.frontier_small"] = f"{fable['input'] / hai['input']:.0f}"

    # F2 worked cost: a 2,000-in / 400-out request on Sonnet 5.5.
    c_in, c_out = 2000 * son["input"] / 1e6, 400 * son["output"] / 1e6
    vars_["f2.req_in"] = usd(c_in)
    vars_["f2.req_out"] = usd(c_out)
    vars_["f2.req_total"] = usd(c_in + c_out)
    vars_["f2.req_month"] = usd((c_in + c_out) * 1_000_000)
    vars_["f2.out_share"] = f"{c_out / (c_in + c_out) * 100:.0f}%"

    # F2 GPU memory arithmetic.
    m70, m8, h100 = H("model.llama-3-3-70b"), H("model.llama-3-1-8b"), H("gpu.H100")
    kv70 = (
        2 * m70["layers"] * m70["kv_heads"] * m70["head_dim"] * 2
    )  # bytes per token, FP16
    kv8 = 2 * m8["layers"] * m8["kv_heads"] * m8["head_dim"] * 2
    vars_["f2.kv70_kb"] = f"{kv70 / 1024:.0f}"
    vars_["f2.kv8_kb"] = f"{kv8 / 1024:.0f}"
    vars_["f2.kv70_8k_gb"] = f"{kv70 * 8192 / 1e9:.1f}"
    vars_["f2.w70_fp16"] = f"{m70['params_b'] * 2:.0f}"
    vars_["f2.w70_fp8"] = f"{m70['params_b']:.0f}"
    vars_["f2.w70_int4"] = f"{m70['params_b'] * 0.5:.0f}"
    vars_["f2.h100_bw"] = f"{h100['bandwidth_tb_s']}"
    vars_["f2.decode_ceiling_70_fp8_tp2"] = (
        f"{2 * h100['bandwidth_tb_s'] * 1e12 / (m70['params_b'] * 1e9):.0f}"
    )
    # Datasheet ceiling, no efficiency factor: real servers reach some fraction of it.
    vars_["f2.prefill_tok_s"] = num(
        h100["fp8_tflops"] * 1e12 / (2 * m70["params_b"] * 1e9)
    )
    vars_["f2.h100_fp8"] = f"{h100['fp8_tflops']:,}"
    p5 = P("gpu.aws.p5-48xlarge")
    vars_["f2.p5_month"] = usd(p5["hourly"] * HOURS_PER_MONTH)
    vars_["f2.p5_hour"] = usd_exact(p5["hourly"])

    # Output to input price ratio across the current models, from the price table.
    ratios = sorted(P(key)["output"] / P(key)["input"] for _, key, _ in CURRENT)
    vars_["ratio.out_in_low"] = f"{ratios[0]:g}"
    vars_["ratio.out_in_high"] = f"{ratios[-1]:.1f}".removesuffix(".0")

    # F4: one judged run of 500 examples, 3,000 tokens in and 200 out, on Sonnet 5.5.
    vars_["f4.judge_run"] = usd(500 * (3000 * son["input"] + 200 * son["output"]) / 1e6)

    # F5 O20: one L4 against one H100 per GPU-hour.
    l4 = P("gpu.aws.g6-xlarge")
    vars_["f5.l4_hour"] = usd_exact(round(l4["hourly"], 2))
    vars_["f5.h100_gpu_hour"] = usd_exact(round(p5["hourly"] / p5["gpus"], 2))
    vars_["f5.h100_l4_ratio"] = f"{p5['hourly'] / p5['gpus'] / l4['hourly']:.1f}"
    return {"tables": tables, "vars": vars_}
