| Line item | Pilot | Target | Large | How it is calculated (target) |
| :-- | --: | --: | --: | :-- |
| Telephony (inbound local + media stream) | $529 | $13,000 | $64,000 | 980K AI minutes x $0.013 |
| Speech to text (streamed for the whole call) | $316 | $7,600 | $38,000 | 980K minutes x $0.0077, monolingual model chosen by keypad |
| Model | $199 | $4,800 | $24,000 | Sonnet 5.5 with caching, $0.0049 a minute |
| Text to speech | $735 | $18,000 | $88,000 | Generated speech only; fixed phrases pre-rendered. ElevenLabs Flash, per character |
| Orchestration nodes (planning number) | $521 | $1,000 | $3,900 | 4 x c7i.2xlarge for a peak of 108 concurrent calls |
| Recordings (12 months kept) | $5.44 | $130 | $652 | 64 kbit/s mono in S3, steady state after a year |
| Post-call analytics | $4.66 | $112 | $559 | Haiku 5.5 summary per call, standard API (batch is not HIPAA-eligible) |
| Telephony after transfer | $456 | $11,000 | $55,000 | Transferred calls stay on Twilio: 6 min x (inbound + SIP) |
| **Total per month** | **$2,800** | **$55,000** | **$270,000** |  |
| Cost per call | $0.18 | $0.15 | $0.15 |  |
