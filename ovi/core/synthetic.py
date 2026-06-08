"""Synthetic OHLC panel generator.

Doubles as (a) the fixture for the decomposition unit tests and (b) the data for
the Layer-0 demo so the full pipeline can run without hitting the Polygon API.

Design so the decomposition is exactly recoverable:
  * We simulate a *smooth adjusted* price path from per-day overnight and intraday
    factors.  ``adj_close`` follows that smooth path (corporate actions removed).
  * A split on day ``d`` with factor ``k`` divides the *raw* prices by ``k`` from
    day ``d`` onward (a real split makes the raw quote jump); ``adj_close`` stays
    smooth.  Because the intraday leg uses the *ratio* close/open, the split
    cancels within the day, so the intraday leg is unaffected and the entire split
    lands in the overnight leg — exactly the paper's convention.

This lets the tests assert the (1+intra)(1+night)=(1+cc) identity to machine
precision and check that a known split shows up in the overnight leg only.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .decomposition import (
    COL_ADJ_CLOSE,
    COL_CLOSE,
    COL_DATE,
    COL_OPEN,
    COL_TICKER,
)


def make_synthetic_panel(
    n_tickers: int = 6,
    n_days: int = 90,
    *,
    seed: int = 0,
    overnight_mu: float = 0.0006,
    overnight_sd: float = 0.010,
    intraday_mu: float = 0.0004,
    intraday_sd: float = 0.010,
    start: str = "2018-01-02",
    missing_open_frac: float = 0.0,
    splits: Optional[Dict[str, Tuple[int, float]]] = None,
    deterministic: bool = False,
) -> pd.DataFrame:
    """Return a tidy OHLC panel: one row per (ticker, date).

    Columns: ticker, date, open, high, low, close, volume, adj_close, shares,
    market_cap.  ``open``/``close`` are RAW (carry split jumps); ``adj_close`` is
    the smooth adjusted path.

    Parameters of note:
      overnight_mu/intraday_mu : per-day mean log factors.  Default makes the
        market's overnight leg drift faster than intraday (the paper's result),
        so the Layer-0 demo reproduces the qualitative split.
      missing_open_frac : fraction of stock-days whose open is set to NaN.
      splits : ``{ticker: (day_index, factor)}`` e.g. ``{"AAA": (40, 2.0)}`` for a
        2:1 split at the open of day 40.
      deterministic : if True, use the *constant* factors exp(mu) (sd ignored) so
        tests get exact, reproducible leg values.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    splits = splits or {}

    frames = []
    for ti, tkr in enumerate(tickers):
        if deterministic:
            o_fac = np.full(n_days, np.exp(overnight_mu))
            i_fac = np.full(n_days, np.exp(intraday_mu))
        else:
            o_fac = np.exp(rng.normal(overnight_mu, overnight_sd, n_days))
            i_fac = np.exp(rng.normal(intraday_mu, intraday_sd, n_days))

        # Smooth ADJUSTED path. adj_open_t = adj_close_{t-1} * overnight_factor,
        # adj_close_t = adj_open_t * intraday_factor.
        base = 20.0 + 80.0 * (ti / max(n_tickers - 1, 1))  # spread starting prices
        adj_open = np.empty(n_days)
        adj_close = np.empty(n_days)
        prev_close = base
        for s in range(n_days):
            adj_open[s] = prev_close * o_fac[s]
            adj_close[s] = adj_open[s] * i_fac[s]
            prev_close = adj_close[s]

        # Raw prices = adjusted / cumulative split factor (splits push raw down).
        cum_split = np.ones(n_days)
        if tkr in splits:
            d, k = splits[tkr]
            cum_split[d:] = k
        raw_open = adj_open / cum_split
        raw_close = adj_close / cum_split
        high = np.maximum(raw_open, raw_close) * 1.001
        low = np.minimum(raw_open, raw_close) * 0.999

        shares = float(rng.integers(5, 500)) * 1_000_000.0  # constant per ticker
        frames.append(
            pd.DataFrame(
                {
                    COL_TICKER: tkr,
                    COL_DATE: dates,
                    COL_OPEN: raw_open,
                    "high": high,
                    "low": low,
                    COL_CLOSE: raw_close,
                    "volume": rng.integers(1e5, 1e7, n_days).astype("float64"),
                    COL_ADJ_CLOSE: adj_close,
                    "shares": shares,
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    panel["market_cap"] = panel[COL_CLOSE] * panel["shares"]

    if missing_open_frac > 0:
        mask = rng.random(len(panel)) < missing_open_frac
        # Never drop the very first day per ticker (keeps things simple).
        first = panel.groupby(COL_TICKER, sort=False).head(1).index
        mask[panel.index.isin(first)] = False
        panel.loc[mask, COL_OPEN] = np.nan

    return panel
