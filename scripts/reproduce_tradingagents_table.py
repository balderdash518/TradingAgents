from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.backtesting.tradingagents_repro import (
    compute_metrics,
    decision_status,
    load_config,
    run_agents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the TradingAgents-only row of the experiment table."
    )
    parser.add_argument(
        "--config",
        default="configs/tradingagents_repro.json",
        help="Path to the reproduction config JSON.",
    )
    parser.add_argument(
        "--run-agents",
        action="store_true",
        help="Call LLM agents for missing ticker/date decisions and append to decisions.csv.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip decisions already present in decisions.csv.",
    )
    parser.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="Debug option: only run the first N trading dates per ticker.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cached-decision progress and exit without LLM calls.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.status:
        print(decision_status(cfg).to_string(index=False))
        return

    if args.run_agents:
        cache = run_agents(cfg, resume=not args.no_resume, limit_dates=args.limit_dates)
        print(f"Decision cache written to: {cache}")
    else:
        print("Skipping LLM calls. Reading existing decision cache.")

    df = compute_metrics(cfg)
    print(df.to_string(index=False))
    print(f"Metrics CSV: {cfg.output_dir / 'tradingagents_metrics.csv'}")
    print(f"Markdown table: {cfg.output_dir / 'tradingagents_table.md'}")


if __name__ == "__main__":
    main()
