| Write tool | Cost of one wrong execution (assumed) | Confirmation pays above, agent catches every error | Agent catches 50% | Decision |
| :-- | --: | --: | --: | :-- |
| `add_internal_note` | $0.25: an agent deletes or corrects the note | 17% (1 in 6) | 33% (1 in 3) | No confirmation; plain text only, logged, shown in the answer |
| `resend_payslips` | $40: a customer's employees receive a duplicate or the wrong month's payslips; support time and an apology | 0.1% (1 in 960) | 0.21% (1 in 480) | Confirm every call |
| `retry_payroll_run` | $500: a second bank file for the same run; unwinding it takes the customer's and Paybridge's finance teams | 0.008% (1 in 12K) | 0.02% (1 in 6,000) | Confirm every call |
