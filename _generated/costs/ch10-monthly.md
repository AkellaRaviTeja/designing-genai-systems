| Line item | Launch | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Generation and condensation | $776 | $2,300 | $7,800 | Haiku 5.5: $0.0004 a first question; a follow-up $0.00049 plus $0.00016 to condense; 35% of turns are follow-ups |
| Reranking | $1,100 | $3,200 | $11,000 | 30 candidates x 440 tokens x $0.05/M (Rerank 5 Pro) |
| Embeddings | $9.27 | $28 | $93 | 10% of 11B corpus tokens re-embedded, plus query embeddings |
| Postgres shards | $1,300 | $4,000 | $13,000 | 3 RDS db.r6g.2xlarge Multi-AZ shards, 400 GB gp3 in all |
| Ask service | $144 | $288 | $865 | 8 Fargate tasks, 1 vCPU and 2 GB each, always on |
| Ingestion and permission sync | $72 | $144 | $288 | 4 tasks for ingestion and permission sync, always on |
| Evaluation and leak suite | $140 | $140 | $140 | 30 runs x (400 judged questions + 600 leak probes checked in code) |
| **Total per month** | **$3,500** | **$10,000** | **$33,000** |  |
| Cost per seat | $0.015 | $0.014 | $0.014 |  |
| Cost per turn | $0.0022 | $0.0021 | $0.0021 |  |
| Turns per month | 1.6M | 4.8M | 16M | seats x 30% weekly active x 5 x 4.35 |
