| Line item | Small | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Gateway tasks | $108 | $108 | $216 | 3 Fargate tasks of 1 vCPU and 2 GB across three zones (six at large) |
| Egress proxy | $54 | $54 | $54 | 3 Fargate tasks of 0.5 vCPU and 1 GB, one per zone |
| Valkey | $256 | $256 | $256 | Two cache.r7g.large nodes: rate-limit buckets, breaker state, budgets |
| Postgres | $94 | $94 | $94 | db.t4g.medium Multi-AZ: keys, routes, budgets, price table, usage ledger rollups |
| Usage records in S3 | $0.56 | $1.88 | $9.41 | 1,500 bytes a request, kept 13 months, written in batches |
| **Gateway total** | **$513** | **$514** | **$630** |  |
| Model spend through it | $2,500 | $8,400 | $42,000 | The feature table, scaled |
| Gateway as a share of model spend | 20.3% | 6.1% | 1.5% |  |
| Gateway cost per million requests | $412 | $124 | $30 |  |
