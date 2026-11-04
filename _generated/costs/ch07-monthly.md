| Line item | Pilot | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Generation | $60 | $500 | $2,500 | 70K queries x $0.0072 (Sonnet 5.5) |
| Embeddings | $0.032 | $0.18 | $0.76 | 10% of the corpus re-embedded, plus query embeddings |
| Postgres with pgvector | $64 | $397 | $1,600 | RDS db.r6g.large Multi-AZ, Mumbai, 100 GB gp3 |
| API and Slack bot | $38 | $76 | $151 | 2 Fargate tasks, 1 vCPU and 2 GB each, always on |
| Ingestion worker | $1.04 | $4.15 | $16 | Sync worker, 80 task-hours a month |
| Evaluation runs | $91 | $91 | $91 | 40 runs x 150 questions, answer plus judge |
| **Total per month** | **$254** | **$1,100** | **$4,300** |  |
| Cost per query | $0.03 | $0.015 | $0.012 |  |
| Queries per month | 8,400 | 70K | 350K | staff x 40% weekly active x 4 x 4.35 |
