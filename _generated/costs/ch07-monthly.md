| Line item | Pilot | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Generation | $60 | $500 | $2,500 | 70K queries x $0.0072 (Sonnet 5.5) |
| Embeddings | $0.032 | $0.18 | $0.76 | 10% of pages re-embedded (assumed), plus query embeddings |
| Postgres with pgvector | $64 | $397 | $1,600 | RDS db.r6g.large Multi-AZ, Mumbai, 100 GB gp3 |
| API and Slack bot | $38 | $76 | $151 | 2 Fargate tasks, 1 vCPU and 2 GB each, always on |
| Ingestion worker | $2.65 | $5.44 | $14 | 96 runs a day x 2 min, plus a 15-min daily reconcile; 1 vCPU, 2 GB |
| Evaluation runs | $67 | $67 | $67 | 40 runs x 150 questions, answer (Sonnet 5.5) plus judge (GPT-6.1 Terra) |
| **Total per month** | **$231** | **$1,000** | **$4,300** |  |
| Cost per query | $0.028 | $0.015 | $0.012 |  |
| Queries per month | 8,400 | 70K | 350K | staff x 40% weekly active x 4 x 4.35 |
