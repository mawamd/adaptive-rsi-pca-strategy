# ETF 2-Hour Gainers (Alpaca SIP)

Ranks ETFs by percent change over the last **2 hours** using **Alpaca Market Data v2**.

## Quick start
1. Add `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` as **repo secrets**.
2. Ensure your Alpaca account has **SIP** market data (or switch workflow/env to `iex`).
3. Edit `etfs.txt` to control the universe.
4. Run the workflow via **Actions → workflow_dispatch**, or let the schedule run.

Outputs:
- CSV artifact: `etf_gainers_2h.csv`
- Job Summary (top 15 table)
