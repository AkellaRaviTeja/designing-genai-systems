| Line item | Pilot | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Model calls | $44 | $270 | $1,100 | 18K tasks x $0.015 (Sonnet 5.5, 4 calls, cached, prefix hit rate over 99%) |
| Assistant service | $38 | $38 | $114 | Fargate: the loop, the tool executor, the confirmation API; two tasks (three at large) |
| Postgres | $64 | $64 | $127 | db.t4g.medium: conversations, proposals, audit log; 20 GB |
| Evaluation runs | $156 | $156 | $156 | 20 runs x 150 tasks on fixtures, uncached, plus a judge (GPT-6.1 Terra) |
| **Total per month** | **$302** | **$528** | **$1,500** |  |
| Tasks per month | 2,400 | 18K | 75K | 30% of System 1's tickets |
| Cost per task | $0.13 | $0.029 | $0.02 |  |
| Break-even agent time saved per task | 30 s | 7 s | 5 s | Cost per task at $15 an hour of agent time |
