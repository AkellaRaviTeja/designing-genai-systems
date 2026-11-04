| Line item | Pilot | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Generation, first questions | $60 | $500 | $2,500 | 70K first questions x $0.0072 (Sonnet 5.5, as System 7) |
| Generation, follow-ups | $71 | $588 | $2,900 | 73K follow-ups, 25% without history, the rest at $0.0083 |
| Condensation and summaries | $1.38 | $11 | $57 | Haiku 5.5: $0.00016 per follow-up, $0.00018 per summary update |
| Embeddings | $0.039 | $0.24 | $1.05 | As System 7, plus one query embedding per follow-up |
| Postgres with pgvector | $64 | $397 | $1,600 | As System 7; thread state adds 410 MB a month |
| API and Slack bot | $38 | $76 | $151 | As System 7 |
| Ingestion worker | $2.65 | $5.44 | $14 | As System 7 |
| Evaluation runs | $167 | $167 | $167 | 40 runs x (150 first questions + 200 conversation turns), judged |
| **Total per month** | **$403** | **$1,700** | **$7,400** |  |
| Cost per turn | $0.024 | $0.012 | $0.01 |  |
| Turns per month | 17K | 140K | 710K | System 7's questions, plus 35% x 3 follow-ups |
