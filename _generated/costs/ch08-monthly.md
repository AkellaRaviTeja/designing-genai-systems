| Line item | Launch | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Generation | $122 | $1,200 | $12,000 | 3M queries x $0.0004 (Haiku 5.5, 2,940 in, 220 out) |
| Reranking | $200 | $2,000 | $20,000 | 30 candidates x 440 tokens x $0.05/M (Rerank 5 Pro) |
| Query rewriting | $15 | $139 | $1,400 | 65% of queries on Haiku 5.5, 2,000-token glossary prefix cached |
| Embeddings | $0.59 | $4.76 | $42 | 10% of the corpus re-embedded, plus one or two query embeddings |
| Postgres (vectors and full text) | $386 | $1,500 | $2,400 | RDS db.r6g.2xlarge Multi-AZ, Mumbai, 200 GB gp3 |
| Retrieval and answer service | $76 | $151 | $606 | 4 Fargate tasks, 1 vCPU and 2 GB each, always on |
| Ingestion worker | $3.11 | $13 | $41 | Sync and parse worker, 250 task-hours a month |
| Evaluation runs | $121 | $121 | $121 | 30 runs x 400 questions, full pipeline plus a Sonnet 5.5 judge |
| **Total per month** | **$923** | **$5,200** | **$37,000** |  |
| Cost per query | $0.0031 | $0.0017 | $0.0012 |  |
| Queries per month | 300K | 3M | 30M | Launch: hub staff plus the support desk; growth and large are 10x and 100x |
