# TradingAgents Row Reproduction

This workflow reproduces only the `Ours / TradingAgents` row for:

- tickers: `AAPL`, `GOOGL`, `AMZN`
- start: `2024-01-01`
- end: `2024-04-11`

It does not implement the rule-based baselines (`B&H`, `MACD`, `KDJ&RSI`, `ZMR`, `SMA`).

## Assumptions

The original paper does not ship the authors' daily decision log or exact execution ledger. This script therefore makes the execution rule explicit:

- TradingAgents runs once for each trading day.
- The configured analyst set is `market`, `news`, and `fundamentals`.
  The Sentiment/Social analyst is intentionally disabled to avoid Reddit /
  StockTwits rate limits.
- Analysts for the same ticker/date run concurrently with
  `analyst_concurrency_limit: 3`; dates still run in chronological order.
- The final rating is mapped to exposure:
  - `Buy`: `+1.00`
  - `Overweight`: `+0.75`
  - `Hold`: `0.00`
  - `Underweight`: `-0.75`
  - `Sell`: `-1.00`
- The exposure decided on day `t` is applied to close-to-close return from day `t` to day `t+1`.
- Metrics are computed from those daily strategy returns:
  - `CR%`: cumulative return
  - `ARR%`: annualized return with 252 trading days
  - `SR`: annualized Sharpe ratio, zero risk-free rate
  - `MDD%`: maximum drawdown

Because LLM outputs, data vendors, and missing paper artifacts differ from the original experiment, this is a reproducible local approximation rather than an exact paper reproduction.

## Files

- `configs/tradingagents_repro.json`: experiment settings
- `scripts/reproduce_tradingagents_table.py`: command-line runner
- `tradingagents/backtesting/tradingagents_repro.py`: reusable implementation
- `results/reproduce_tradingagents_parallel_nosocial_2024_01_01_2024_04_11/decisions.csv`: cached daily decisions
- `results/reproduce_tradingagents_parallel_nosocial_2024_01_01_2024_04_11/tradingagents_metrics.csv`: final metrics
- `results/reproduce_tradingagents_parallel_nosocial_2024_01_01_2024_04_11/tradingagents_table.md`: markdown table row

## Run

Use the conda environment created for this project.

First do a small smoke run. This calls the LLM for only the first trading day of each ticker:

```powershell
cd C:\Users\27898\Desktop\TS\Agent\TradingAgents
conda activate tradingagents
python scripts/reproduce_tradingagents_table.py --run-agents --limit-dates 1
```

The default reproduction config already disables the Sentiment Analyst and runs
the remaining analysts in parallel. For a faster and more stable approximation,
use the fast config, which also uses Mistral on OpenRouter because it has passed
both plain and structured-output smoke tests:

```powershell
python scripts/reproduce_tradingagents_table.py --config configs/tradingagents_repro_fast.json --run-agents --limit-dates 1
```

Full fast run:

```powershell
python scripts/reproduce_tradingagents_table.py --config configs/tradingagents_repro_fast.json --run-agents
```

Then run the full experiment:

```powershell
python scripts/reproduce_tradingagents_table.py --run-agents
```

If the run is interrupted, run the same command again. Existing rows in `decisions.csv` are skipped by default.

After decisions are cached, recompute metrics without any LLM calls:

```powershell
python scripts/reproduce_tradingagents_table.py
```

Check progress without calling the LLM:

```powershell
python scripts/reproduce_tradingagents_table.py --status
```
