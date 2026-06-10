#!/usr/bin/env python3
"""Run Layer 2 — TugOfWar timing + overnight-intraday spread feature.

Examples
  python3 scripts/run_layer2.py --source yahoo --universe sp500 --start 2010-01-01 --end 2019-12-31
  # A 60-month EWMA wants a long sample; pull more history for a real test:
  python3 scripts/run_layer2.py --source yahoo --universe sp500 --start 2004-01-01
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

    return make_synthetic_panel(n_tickers=120, n_days=2520, seed=11,
                                overnight_mu=0.0005, intraday_mu=0.0003,
                                missing_open_frac=0.01)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Layer 2 TugOfWar timing")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", choices=["synthetic", "yahoo", "polygon", "schwab"],
                   default="yahoo")
    p.add_argument("--dividend-adjust", action="store_true",
                   help="schwab: add dividends back into the overnight leg")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--universe", default="sp500")
    p.add_argument("--tickers", default=None)
    p.add_argument("--out", default="reports/output")
    p.add_argument("--features", default="features")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = load_config(args.config)
    start = args.start or cfg.data.start_date
    end = args.end or cfg.data.end_date

    if args.source == "synthetic":
        print("[synthetic] generating tape...")
        panel = _synthetic_panel()
    else:
        from ovi.data.sources import resolve_panel

        print(f"[{args.source}] loading {start} -> {end or 'today'}...")
        panel, note = resolve_panel(
            args.source, cfg=cfg, start=start, end=end, universe=args.universe,
            tickers=args.tickers, dividend_adjust=args.dividend_adjust,
        )
        print(f"[{args.source}] {note}")
        if panel.empty:
            print("ERROR: no data.", file=sys.stderr)
            return 3

    from ovi.backtest.tugofwar import run_layer2
    from ovi.reports.layer2 import format_report, write_report

    print("[layer2] building factors, EWMA TugOfWar, predictive regressions, spread feature...")
    res = run_layer2(panel, cfg, features_dir=args.features)
    print(format_report(res))
    paths = write_report(res, args.out)
    print(f"\n[saved] {paths['report']}\n[saved] {paths['fig']}")
    print(f"[saved] {res.spread_paths['per_name']}\n[saved] {res.spread_paths['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
