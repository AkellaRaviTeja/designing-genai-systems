| Line item | Pilot | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| First stage, Haiku 5.5 | $1.81 | $11 | $44 | 60K tickets x $0.00018 (system prompt cached on 97% of calls) |
| Second stage, Sonnet 5.5 | $15 | $69 | $253 | 30% of tickets (sizing assumption) x $0.0038 |
| Evaluation runs | $164 | $164 | $164 | 20 runs x 2,000 tickets through both models, batch price |
| Classifier service | $38 | $38 | $76 | 2 Fargate tasks, webhook receiver and workers |
| Postgres | $64 | $64 | $127 | db.t4g.medium, jobs, decisions and labels; 20 GB |
| **Total per month** | **$282** | **$345** | **$663** |  |
| Cost per ticket | $0.035 | $0.0057 | $0.0027 |  |
