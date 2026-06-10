#!/usr/bin/env python3
"""Run Layer 0 — print the overnight/intraday market split before any portfolio code.

Data sources (--source):
  synthetic : built-in tape with an overnight drift; no network/key. Proves the
              pipeline end-to-end.
  yahoo     : yfinance (current-listed universe; survivorship-biased, flagged).
  polygon   : Polygon grouped-daily whole-tape (needs POLYGON_API_KEY).

Examples
  python3 scripts/run_layer0.py --source synthetic
  python3 scripts/run_layer0.py --source yahoo --universe sp500 --start 2010-01-01 --end 2019-12-31
  python3 scripts/run_layer0.py --source yahoo --tickers AAPL,MSFT,XOM --start 2015-01-01
  POLYGON_API_KEY=... python3 scripts/run_layer0.py --source polygon --start 2011-01-01 --end 2013-12-31
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ovi.config import load_config  # noqa: E402
from ovi.reports.layer0 import format_report, run_layer0  # noqa: E402


def _synthetic_panel():
    from ovi.core.synthetic import make_synthetic_panel

    return make_synthetic_panel(
        n_tickers=200, n_days=756, seed=7,
        overnight_mu=0.0006, intraday_mu=0.0002, missing_open_frac=0.02,
        splits={"T005": (300, 2.0), "T050": (500, 3.0)},
    )


def _resolve_tickers(args):
    from ovi.data.universe import get_universe

    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        txt = Path(args.tickers_file).read_text().split()
        return [t.strip().upper() for t in txt if t.strip()]
    return get_universe(args.universe)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Layer 0 market decomposition")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", choices=["synthetic", "yahoo", "polygon", "schwab"],
                   default="synthetic")
    p.add_argument("--demo", action="store_true", help="alias for --source synthetic")
    p.add_argument("--dividend-adjust", action="store_true",
                   help="schwab: add dividends back so they land in the overnight leg")
    p.add_argument("--start", default=None, help="override config start date")
    p.add_argument("--end", default=None, help="override config end date")
    p.add_argument("--universe", default="demo", help="yahoo: 'demo' or 'sp500'")
    p.add_argument("--tickers", default=None, help="yahoo: comma-separated override")
    p.add_argument("--tickers-file", default=None, help="yahoo: whitespace-delimited file")
    p.add_argument("--save", default=None, help="write the report to this path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = load_config(args.config)
    source = "synthetic" if args.demo else args.source
    start = args.start or cfg.data.start_date
    end = args.end or cfg.data.end_date
    extra_notes = []

    if source == "synthetic":
        print("[synthetic] generating tape (no network/API)...")
        panel = _synthetic_panel()
    else:
        from ovi.data.sources import resolve_panel

        tickers = ",".join(_resolve_tickers(args)) if (args.tickers or args.tickers_file) else None
        print(f"[{source}] loading {start} -> {end or 'today'} (cache: {cfg.data.cache_dir})...")
        panel, note = resolve_panel(
            source, cfg=cfg, start=start, end=end, universe=args.universe,
            tickers=tickers, dividend_adjust=args.dividend_adjust,
        )
        print(f"[{source}] {note}")
        extra_notes.append(note)
        if panel.empty:
            print("ERROR: no data returned.", file=sys.stderr)
            return 3

    res = run_layer0(panel, cfg)
    report = format_report(res)
    print(report)
    for note in extra_notes:
        print(f"\n  !! {note}")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        body = report + ("\n\n" + "\n".join(f"  !! {n}" for n in extra_notes) if extra_notes else "")
        Path(args.save).write_text(body + "\n")
        print(f"\n[saved] {args.save}")
    return 0 if res.sanity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
