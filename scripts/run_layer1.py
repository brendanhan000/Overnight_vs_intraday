#!/usr/bin/env python3
"""Run Layer 1 — Table 1 replication + honest costs + Fig. 2 + baselines.

Examples
  # Real data via yfinance (uses the warm cache from Layer 0):
  python3 scripts/run_layer1.py --source yahoo --universe sp500 --start 2010-01-01 --end 2019-12-31

  # Quick synthetic smoke test (no network beyond Fama-French factors):
  python3 scripts/run_layer1.py --source synthetic
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovi.config import load_config  # noqa: E402


def _synthetic_panel():
    from ovi.core.synthetic import make_synthetic_panel

    # Cross-sectional spread in overnight drift so the sort has something to find.
    return make_synthetic_panel(
        n_tickers=120, n_days=2016, seed=11,
        overnight_mu=0.0005, overnight_sd=0.012,
        intraday_mu=0.0003, intraday_sd=0.012, missing_open_frac=0.01,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Layer 1 persistence/reversal backtest")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", choices=["synthetic", "yahoo"], default="yahoo")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--universe", default="sp500", help="yahoo: 'demo' or 'sp500'")
    p.add_argument("--tickers", default=None)
    p.add_argument("--out", default="reports/output")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-fig2", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = load_config(args.config)
    start = args.start or cfg.data.start_date
    end = args.end or cfg.data.end_date
    note = ""

    if args.source == "synthetic":
        print("[synthetic] generating tape...")
        panel = _synthetic_panel()
    else:
        from ovi.data.universe import get_universe
        from ovi.data.yahoo import YahooLoader

        tickers = ([t.strip().upper() for t in args.tickers.split(",")]
                   if args.tickers else get_universe(args.universe))
        print(f"[yahoo] loading {len(tickers)} tickers {start} -> {end or 'today'}...")
        loader = YahooLoader(cache_dir=cfg.data.cache_dir)
        panel = loader.load(tickers, start, end)
        got = panel["ticker"].nunique() if not panel.empty else 0
        note = f"SURVIVORSHIP: {got}/{len(tickers)} requested tickers returned data."
        print(f"[yahoo] {note}")
        if panel.empty:
            print("ERROR: no data.", file=sys.stderr)
            return 3

    from ovi.backtest.engine import run_layer1
    from ovi.reports.layer1 import format_report, write_report

    print("[layer1] building monthly panel, factors, portfolios, costs, Fig.2...")
    res = run_layer1(
        panel, cfg,
        do_baselines=not args.no_baselines, do_fig2=not args.no_fig2,
        survivorship_note=note,
    )
    print(format_report(res))
    paths = write_report(res, args.out)
    print(f"\n[saved] {paths['report']}\n[saved] {paths['fig2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
