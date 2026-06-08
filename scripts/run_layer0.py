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
    p.add_argument("--source", choices=["synthetic", "yahoo", "polygon"],
                   default="synthetic")
    p.add_argument("--demo", action="store_true", help="alias for --source synthetic")
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

    elif source == "yahoo":
        from ovi.data.yahoo import YahooLoader

        tickers = _resolve_tickers(args)
        print(f"[yahoo] loading {len(tickers)} tickers {start} -> {end or 'today'} "
              f"(cache: {cfg.data.cache_dir})...")
        loader = YahooLoader(cache_dir=cfg.data.cache_dir)
        panel = loader.load(tickers, start, end)
        got = panel["ticker"].nunique() if not panel.empty else 0
        print(f"[yahoo] {got}/{len(tickers)} tickers returned data "
              f"({len(loader.missing)} missing).")
        extra_notes.append(
            f"SURVIVORSHIP WARNING: Yahoo serves current-listed names only "
            f"({got}/{len(tickers)} returned); delisted names are absent."
        )
        if panel.empty:
            print("ERROR: no data returned.", file=sys.stderr)
            return 3

    else:  # polygon
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            print("ERROR: POLYGON_API_KEY is not set. Use --source yahoo or "
                  "--source synthetic, or set the key.", file=sys.stderr)
            return 2
        from ovi.data.polygon import PolygonClient

        client = PolygonClient(
            api_key=api_key, cache_dir=cfg.data.cache_dir,
            rate_limit_per_min=cfg.data.rate_limit_per_min,
            max_retries=cfg.data.max_retries, base_url=cfg.data.base_url,
        )
        print(f"[polygon] loading grouped-daily {start} -> {end or 'today'}...")
        panel = client.load_range(start, end, progress_every=50)
        if panel.empty:
            print("ERROR: no data returned for the requested range.", file=sys.stderr)
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
