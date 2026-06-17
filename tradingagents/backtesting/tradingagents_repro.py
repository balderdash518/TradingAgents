from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

RATING_TO_EXPOSURE_LONG_SHORT = {
    "Buy": 1.0,
    "Overweight": 0.75,
    "Hold": 0.0,
    "Underweight": -0.75,
    "Sell": -1.0,
}

RATING_TO_EXPOSURE_LONG_ONLY = {
    "Buy": 1.0,
    "Overweight": 0.75,
    "Hold": 0.0,
    "Underweight": 0.0,
    "Sell": 0.0,
}


@dataclass(frozen=True)
class ReproConfig:
    tickers: list[str]
    start: str
    end: str
    output_dir: Path
    provider: str = "openrouter"
    quick_model: str = "mistralai/mistral-small-3.2-24b-instruct"
    deep_model: str = "mistralai/mistral-small-3.2-24b-instruct"
    selected_analysts: tuple[str, ...] = ("market", "news", "fundamentals")
    analyst_concurrency_limit: int = 3
    long_short: bool = True
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    temperature: float | None = 0.0
    llm_timeout: float = 120
    llm_max_retries: int = 3
    date_max_retries: int = 2
    date_retry_sleep_seconds: float = 30
    asset_type: str = "stock"


def load_config(path: Path) -> ReproConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected_analysts = tuple(
        data.get("selected_analysts", ["market", "news", "fundamentals"])
    )
    return ReproConfig(
        tickers=list(data["tickers"]),
        start=data["start"],
        end=data["end"],
        output_dir=Path(data.get("output_dir", "results/reproduce_tradingagents")),
        provider=data.get("provider", "openrouter"),
        quick_model=data.get("quick_model", "mistralai/mistral-small-3.2-24b-instruct"),
        deep_model=data.get("deep_model", "mistralai/mistral-small-3.2-24b-instruct"),
        selected_analysts=selected_analysts,
        analyst_concurrency_limit=int(
            data.get(
                "analyst_concurrency_limit",
                min(3, max(1, len(selected_analysts))),
            )
        ),
        long_short=bool(data.get("long_short", True)),
        max_debate_rounds=int(data.get("max_debate_rounds", 1)),
        max_risk_discuss_rounds=int(data.get("max_risk_discuss_rounds", 1)),
        temperature=data.get("temperature", 0.0),
        llm_timeout=float(data.get("llm_timeout", 120)),
        llm_max_retries=int(data.get("llm_max_retries", 3)),
        date_max_retries=int(data.get("date_max_retries", 2)),
        date_retry_sleep_seconds=float(data.get("date_retry_sleep_seconds", 30)),
        asset_type=data.get("asset_type", "stock"),
    )


def decision_cache_path(output_dir: Path) -> Path:
    return output_dir / "decisions.csv"


def metrics_path(output_dir: Path) -> Path:
    return output_dir / "tradingagents_metrics.csv"


def markdown_path(output_dir: Path) -> Path:
    return output_dir / "tradingagents_table.md"


def load_decision_cache(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(row["ticker"], row["date"]): row for row in csv.DictReader(f)}


def append_decision(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = ["ticker", "date", "rating", "exposure", "final_trade_decision"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def download_close(ticker: str, start: str, end: str) -> pd.Series:
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    df = yf.download(
        ticker,
        start=start,
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No price data downloaded for {ticker}.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = ticker
    return close


def trading_dates(ticker: str, start: str, end: str) -> list[str]:
    close = download_close(ticker, start, end)
    return [d.strftime("%Y-%m-%d") for d in close.index]


def make_agent_config(cfg: ReproConfig) -> dict[str, Any]:
    from tradingagents.default_config import DEFAULT_CONFIG

    agent_config = DEFAULT_CONFIG.copy()
    agent_config["llm_provider"] = cfg.provider
    agent_config["quick_think_llm"] = cfg.quick_model
    agent_config["deep_think_llm"] = cfg.deep_model
    agent_config["backend_url"] = None
    agent_config["max_debate_rounds"] = cfg.max_debate_rounds
    agent_config["max_risk_discuss_rounds"] = cfg.max_risk_discuss_rounds
    agent_config["temperature"] = cfg.temperature
    agent_config["llm_timeout"] = cfg.llm_timeout
    agent_config["llm_max_retries"] = cfg.llm_max_retries
    agent_config["analyst_concurrency_limit"] = cfg.analyst_concurrency_limit
    agent_config["results_dir"] = str(cfg.output_dir / "agent_logs")
    agent_config["memory_log_path"] = str(cfg.output_dir / "memory" / "trading_memory.md")
    agent_config["data_cache_dir"] = str(cfg.output_dir / "cache")
    agent_config.setdefault("data_vendors", {})
    agent_config["data_vendors"].update(
        {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        }
    )
    return agent_config


def run_agents(cfg: ReproConfig, resume: bool = True, limit_dates: int | None = None) -> Path:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = decision_cache_path(cfg.output_dir)
    cached = load_decision_cache(cache_file) if resume else {}
    exposure_map = RATING_TO_EXPOSURE_LONG_SHORT if cfg.long_short else RATING_TO_EXPOSURE_LONG_ONLY

    graph = TradingAgentsGraph(
        selected_analysts=list(cfg.selected_analysts),
        debug=False,
        config=make_agent_config(cfg),
    )

    for ticker in cfg.tickers:
        dates = trading_dates(ticker, cfg.start, cfg.end)
        if limit_dates is not None:
            dates = dates[:limit_dates]
        for trade_date in dates:
            if (ticker, trade_date) in cached:
                print(f"[skip] {ticker} {trade_date} already cached")
                continue
            print(f"[run] {ticker} {trade_date}")
            final_state = None
            rating = "Hold"
            for attempt in range(1, cfg.date_max_retries + 2):
                try:
                    final_state, rating = graph.propagate(
                        ticker, trade_date, asset_type=cfg.asset_type
                    )
                    break
                except Exception as exc:
                    if attempt > cfg.date_max_retries:
                        raise
                    print(
                        f"[retry] {ticker} {trade_date} failed on attempt {attempt}: "
                        f"{type(exc).__name__}: {exc}. Sleeping "
                        f"{cfg.date_retry_sleep_seconds:.0f}s before retry."
                    )
                    time.sleep(cfg.date_retry_sleep_seconds)
            exposure = exposure_map.get(rating, 0.0)
            append_decision(
                cache_file,
                {
                    "ticker": ticker,
                    "date": trade_date,
                    "rating": rating,
                    "exposure": exposure,
                    "final_trade_decision": final_state.get("final_trade_decision", "")
                    if final_state
                    else "",
                },
            )
            cached[(ticker, trade_date)] = {"rating": rating, "exposure": str(exposure)}
    return cache_file


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return max(0.0, float(-drawdown.min()))


def _sharpe(returns: pd.Series) -> float:
    std = float(returns.std(ddof=1))
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(252))


def compute_metrics_for_ticker(
    ticker: str,
    decisions: dict[tuple[str, str], dict[str, str]],
    start: str,
    end: str,
    long_short: bool,
) -> dict[str, Any]:
    close = download_close(ticker, start, end)
    daily_price_returns = close.pct_change().shift(-1).dropna()
    exposure_map = RATING_TO_EXPOSURE_LONG_SHORT if long_short else RATING_TO_EXPOSURE_LONG_ONLY

    rows = []
    for date, price_return in daily_price_returns.items():
        date_str = date.strftime("%Y-%m-%d")
        row = decisions.get((ticker, date_str))
        if row is None:
            raise RuntimeError(
                f"Missing TradingAgents decision for {ticker} {date_str}. "
                "Run with --run-agents first, or reduce the date range."
            )
        rating = row.get("rating", "Hold")
        exposure = float(row.get("exposure") or exposure_map.get(rating, 0.0))
        rows.append((date, exposure * float(price_return)))

    returns = pd.Series([r for _, r in rows], index=[d for d, _ in rows], name=ticker)
    equity = (1.0 + returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    annualized_return = float(equity.iloc[-1] ** (252 / len(returns)) - 1.0)
    return {
        "ticker": ticker,
        "model": "TradingAgents",
        "CR%": cumulative_return * 100,
        "ARR%": annualized_return * 100,
        "SR": _sharpe(returns),
        "MDD%": _max_drawdown(equity) * 100,
        "days": len(returns),
    }


def compute_metrics(cfg: ReproConfig) -> pd.DataFrame:
    decisions = load_decision_cache(decision_cache_path(cfg.output_dir))
    rows = [
        compute_metrics_for_ticker(ticker, decisions, cfg.start, cfg.end, cfg.long_short)
        for ticker in cfg.tickers
    ]
    df = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_path(cfg.output_dir), index=False)
    markdown_path(cfg.output_dir).write_text(metrics_to_markdown(df), encoding="utf-8")
    return df


def decision_status(cfg: ReproConfig) -> pd.DataFrame:
    decisions = load_decision_cache(decision_cache_path(cfg.output_dir))
    rows = []
    for ticker in cfg.tickers:
        all_dates = trading_dates(ticker, cfg.start, cfg.end)
        # The last trading day has no next-day close-to-close return inside the
        # configured window, so metrics require decisions through the penultimate
        # trading date. run_agents still caches the final date for completeness.
        metric_dates = all_dates[:-1]
        cached_dates = {date for t, date in decisions if t == ticker}
        missing_for_metrics = [date for date in metric_dates if date not in cached_dates]
        rows.append(
            {
                "ticker": ticker,
                "cached_decisions": len([date for date in all_dates if date in cached_dates]),
                "total_trading_days": len(all_dates),
                "required_for_metrics": len(metric_dates),
                "missing_for_metrics": len(missing_for_metrics),
                "ready_for_metrics": len(missing_for_metrics) == 0,
                "next_missing_date": missing_for_metrics[0] if missing_for_metrics else "",
            }
        )
    return pd.DataFrame(rows)


def metrics_to_markdown(df: pd.DataFrame) -> str:
    headers = ["Model"]
    for ticker in df["ticker"]:
        headers.extend([f"{ticker} CR%", f"{ticker} ARR%", f"{ticker} SR", f"{ticker} MDD%"])
    row = ["TradingAgents"]
    for _, rec in df.iterrows():
        row.extend(
            [
                f"{rec['CR%']:.2f}",
                f"{rec['ARR%']:.2f}",
                f"{rec['SR']:.2f}",
                f"{rec['MDD%']:.2f}",
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(row) + " |",
    ]
    return "\n".join(lines) + "\n"
