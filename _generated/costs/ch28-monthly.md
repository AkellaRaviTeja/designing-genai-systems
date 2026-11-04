| Line item | Pilot | Target | Large | How it is calculated (target) |
| :-- | --: | --: | --: | :-- |
| Telephony (inbound local + media stream) | $529 | $13,000 | $64,000 | 980K AI minutes x $0.013 |
| Speech to text (streamed for the whole call) | $316 | $7,600 | $38,000 | 980K minutes x $0.0077 |
| Model | $273 | $6,600 | $33,000 | Sonnet 5.5 with caching, $0.0067 a minute |
| Text to speech | $515 | $12,000 | $62,000 | 70% of speech synthesised, ElevenLabs Flash |
| Orchestration nodes | $521 | $782 | $2,600 | 3 x c7i.2xlarge for a peak of 65 concurrent calls |
| Recordings (12 months kept) | $5.44 | $130 | $652 | 64 kbit/s mono in S3, steady state after a year |
| Post-call analytics (batch) | $1.75 | $42 | $210 | Haiku 5.5 summary per call, batch API |
| **Total per month** | **$2,200** | **$40,000** | **$200,000** |  |
| Cost per call | $0.14 | $0.11 | $0.11 |  |
