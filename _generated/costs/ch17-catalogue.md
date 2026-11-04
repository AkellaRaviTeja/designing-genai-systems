| Tool | Kind | Tokens (assumed) | What it does |
| :-- | :-- | --: | :-- |
| `get_customer` | read | 210 | A customer: plan, state registrations, payroll calendar |
| `list_payroll_runs` | read | 230 | Recent runs for a customer, with status and error codes |
| `get_payroll_run` | read | 240 | One run: steps, errors by employee, bank file status |
| `get_filing_status` | read | 250 | PF, ESI and TDS filings for a period, with portal acknowledgements |
| `search_help_articles` | read | 200 | Search Paybridge's help centre; returns titles, links and snippets |
| `get_ticket_history` | read | 190 | A customer's last 20 tickets: subject, status, solving group |
| `add_internal_note` | write | 210 | Add an internal note to this ticket (agents only see it) |
| `resend_payslips` | write | 260 | Resend a run's payslips to employees' email addresses on file |
| `retry_payroll_run` | write | 290 | Re-run a failed payroll run from its failed step |
| **All nine** |  | **2,080** | Plus the 286-token tool-use system prompt the API adds |
