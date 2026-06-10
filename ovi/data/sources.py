"""Unified price-source resolver shared by the run_layer{0,1,2} scripts.

Handles the three REAL sources (yahoo / polygon / schwab) behind one call so each
runner stays thin. Synthetic data is layer-specific and stays in the runners.
Returns ``(panel, note)`` where ``panel`` is the tidy OHLC+adj_close frame and
``note`` is a one-line provenance/caveat string for the report.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ..config import Config

REAL_SOURCES = ("yahoo", "polygon", "schwab")


def _tickers(universe: str, tickers: Optional[str]) -> List[str]:
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    from .universe import get_universe

    return get_universe(universe)


def resolve_panel(
    source: str,
    *,
    cfg: Config,
    start: str,
    end: Optional[str],
    universe: str = "sp500",
    tickers: Optional[str] = None,
    dividend_adjust: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Load the price panel for ``source``; return (panel, provenance_note)."""
    if source == "yahoo":
        from .yahoo import YahooLoader

        names = _tickers(universe, tickers)
        loader = YahooLoader(cache_dir=cfg.data.cache_dir)
        panel = loader.load(names, start, end)
        got = panel["ticker"].nunique() if not panel.empty else 0
        note = (f"yahoo: {got}/{len(names)} tickers returned data. SURVIVORSHIP-biased "
                f"(current-listed only); split+dividend adjusted close.")
        return panel, note

    if source == "schwab":
        from .schwab import SchwabLoader

        names = _tickers(universe, tickers)
        loader = SchwabLoader(cache_dir=cfg.data.cache_dir, dividend_adjust=dividend_adjust)
        panel = loader.load(names, start, end)
        got = panel["ticker"].nunique() if not panel.empty else 0
        adj = ("split+DIVIDEND adjusted (ex-dates added back)" if dividend_adjust
               else "SPLIT-ONLY adjusted — dividends NOT in overnight leg (~0.15%/mo); "
                    "pass --dividend-adjust to correct")
        note = f"schwab: {got}/{len(names)} tickers returned data. {adj}."
        return panel, note

    if source == "polygon":
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise RuntimeError("POLYGON_API_KEY not set for --source polygon.")
        from .polygon import PolygonClient

        client = PolygonClient(
            api_key=api_key, cache_dir=cfg.data.cache_dir,
            rate_limit_per_min=cfg.data.rate_limit_per_min,
            max_retries=cfg.data.max_retries, base_url=cfg.data.base_url,
        )
        panel = client.load_range(start, end, progress_every=50)
        note = "polygon: whole-tape grouped daily; includes delisted names (survivorship-safe)."
        return panel, note

    raise ValueError(f"unknown real source {source!r} (use one of {REAL_SOURCES})")
