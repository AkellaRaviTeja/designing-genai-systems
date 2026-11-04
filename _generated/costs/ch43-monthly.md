| Line item | Small | Growth | Large | How it is calculated (growth) |
| :-- | --: | --: | --: | :-- |
| Gate runs | $666 | $1,300 | $5,300 | 15 changes a feature a month x 1,200 examples, candidate only, synchronous, at $0.0185 each |
| Weekly A/A runs | $97 | $193 | $773 | The release configuration re-run weekly, 1,200 examples, batch API |
| Production judging | $11 | $21 | $85 | 100 outputs a feature a day, judge only, batch API |
| Judge drift re-runs | $2.80 | $5.60 | $22 | The 400-output calibration set of each criterion re-judged monthly, batch API |
| Calibration labels | $3,200 | $6,400 | $26,000 | 2 criteria a feature, 400 outputs, 2 people, 3 min each, quarterly |
| Review queue | $1,300 | $2,600 | $10,000 | 50 production outputs a feature a week, one person |
| Runner, labelling tool, Postgres, S3 | $203 | $203 | $205 | Three small Fargate tasks, db.t4g.medium Multi-AZ, outputs kept 13 months |
| Platform engineering | $7,500 | $15,000 | $30,000 | 1 engineer at an assumed $15,000 a month (0.5 small, 2 large) |
| **Platform total** | **$13,000** | **$26,000** | **$72,000** |  |
| Of which labelling | $4,500 | $9,000 | $36,000 | At an assumed $60 an hour |
| Expert hours a month | 75 | 150 | 601 |  |
| Features with suites | 2 | 4 | 16 |  |
