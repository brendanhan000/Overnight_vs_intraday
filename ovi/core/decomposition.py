"""The heart of the project: overnight / intraday return decomposition.

Following Lou, Polk & Skouras (2019). For each stock i on day s::

    intraday_ret  = close_s / open_s - 1                      (RAW open & close)
    cc_ret        = adj_close_s / adj_close_{s-1} - 1         (ADJUSTED close-to-close)
    overnight_ret = (1 + cc_ret) / (1 + intraday_ret) - 1

Because corporate actions (splits/dividends) are assumed to occur overnight, the
intraday leg uses *raw* prices and the close-to-close leg uses *adjusted* prices;
the overnight leg is the residual, so **all** corporate-action adjustment lands in
the overnight leg by construction.

The defining identity, which holds exactly day-by-day and therefore also after
monthly compounding (products commute)::

    (1 + intraday_ret) * (1 + overnight_ret) == (1 + cc_ret)
    (1 + intraday_m)   * (1 + overnight_m)   == (1 + cc_m)

Missing-open handling (spec): if the open is unavailable you cannot trade at the
open, so you hold the overnight position from the prior close into the next
available open. We implement that exactly by setting ``intraday_ret = 0`` on a
missing-open day and letting ``overnight_ret`` absorb the whole adjusted move
(``overnight_ret = cc_ret``). Across a run of missing-open days the overnight legs
telescope, so the cumulative overnight return from the last realized open to the
next available open equals (adjusted) ``open_next / close_last`` — i.e. exactly
"held overnight" — while the identity is preserved throughout.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

# --- canonical column names -------------------------------------------------
COL_TICKER = "ticker"
COL_DATE = "date"
COL_OPEN = "open"
COL_CLOSE = "close"
COL_ADJ_CLOSE = "adj_close"

LEG_INTRADAY = "intraday_ret"
LEG_OVERNIGHT = "overnight_ret"
LEG_CC = "cc_ret"
DAILY_LEGS: Tuple[str, str, str] = (LEG_INTRADAY, LEG_OVERNIGHT, LEG_CC)

MONTH_COL = "month"
MONTHLY_LEGS: Tuple[str, str, str] = ("intraday_m", "overnight_m", "cc_m")

# Bad-data bounds (spec): drop any stock-day whose intraday/overnight/cc move is
# below -30% or above +50%.
BAD_LOWER = -0.30
BAD_UPPER = 0.50


def compute_daily_legs(
    df: pd.DataFrame,
    *,
    ticker_col: str = COL_TICKER,
    date_col: str = COL_DATE,
    open_col: str = COL_OPEN,
    close_col: str = COL_CLOSE,
    adj_close_col: str = COL_ADJ_CLOSE,
) -> pd.DataFrame:
    """Add ``intraday_ret``, ``overnight_ret`` and ``cc_ret`` columns.

    Input is a long (tidy) panel with one row per (ticker, date). The frame is
    returned sorted by [ticker, date] with the three leg columns appended; the
    input is not mutated. The previous adjusted close is taken within each ticker,
    so the frame need not be pre-sorted.
    """
    required = {ticker_col, date_col, open_col, close_col, adj_close_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_daily_legs missing columns: {sorted(missing)}")

    out = df.sort_values([ticker_col, date_col], kind="mergesort").copy()

    # Close-to-close on ADJUSTED prices, previous close within each ticker.
    prev_adj = out.groupby(ticker_col, sort=False)[adj_close_col].shift(1)
    adj_close = out[adj_close_col]
    valid_cc = adj_close.gt(0) & prev_adj.gt(0)  # NaN comparisons -> False
    cc = (adj_close / prev_adj - 1.0).where(valid_cc)

    # Intraday on RAW prices. If the open is unavailable we realize no intraday
    # position (intraday = 0) and the overnight leg absorbs the full move.
    open_ = out[open_col]
    close_ = out[close_col]
    valid_open = open_.gt(0) & close_.gt(0)
    intraday = pd.Series(
        np.where(valid_open, close_ / open_ - 1.0, 0.0),
        index=out.index,
        dtype="float64",
    )

    # Overnight is the residual that makes the daily identity hold exactly.
    overnight = (1.0 + cc) / (1.0 + intraday) - 1.0

    # Where the close-to-close leg is undefined (first day per ticker, or bad
    # prices) all three legs are undefined together, so the valid-day set is
    # identical across legs and the identity survives monthly compounding.
    intraday = intraday.where(valid_cc)
    overnight = overnight.where(valid_cc)

    out[LEG_INTRADAY] = intraday
    out[LEG_OVERNIGHT] = overnight
    out[LEG_CC] = cc
    return out


def identity_residual(
    df: pd.DataFrame,
    intraday_col: str = LEG_INTRADAY,
    overnight_col: str = LEG_OVERNIGHT,
    cc_col: str = LEG_CC,
) -> pd.Series:
    """(1+intraday)(1+overnight) - (1+cc); should be ~0 on every valid row."""
    return (1.0 + df[intraday_col]) * (1.0 + df[overnight_col]) - (1.0 + df[cc_col])


def max_identity_error(df: pd.DataFrame, **cols) -> float:
    """Largest absolute identity violation over rows where all legs are present."""
    resid = identity_residual(df, **cols)
    return float(np.nanmax(np.abs(resid.values))) if len(resid) else 0.0


def apply_bad_data_filter(
    df: pd.DataFrame,
    *,
    lower: float = BAD_LOWER,
    upper: float = BAD_UPPER,
    legs: Sequence[str] = DAILY_LEGS,
) -> Tuple[pd.DataFrame, int]:
    """Null out (exclude) stock-days where any leg is < ``lower`` or > ``upper``.

    Returns the filtered frame plus the number of stock-days removed. A removed
    day has all three legs set to NaN, so it is skipped — as a flat day — during
    monthly compounding, and the valid-day set stays identical across legs.
    """
    out = df.copy()
    bad = pd.Series(False, index=out.index)
    for leg in legs:
        bad = bad | out[leg].lt(lower) | out[leg].gt(upper)  # NaN -> False
    n_bad = int(bad.sum())
    if n_bad:
        out.loc[bad, list(legs)] = np.nan
    return out, n_bad


def aggregate_monthly(
    df: pd.DataFrame,
    *,
    ticker_col: str = COL_TICKER,
    date_col: str = COL_DATE,
) -> pd.DataFrame:
    """Compound daily legs to monthly within each (ticker, calendar-month).

        intraday_m  = prod(1 + intraday_ret) - 1
        overnight_m = prod(1 + overnight_ret) - 1
        cc_m        = prod(1 + cc_ret) - 1

    NaN legs (first day, bad data, missing) are skipped — treated as flat days.
    A stock-month with no valid day yields NaN (via ``min_count=1``). Returns one
    row per (ticker, month) with a ``month`` Period[M] column and an ``n_days``
    count of valid trading days.
    """
    out = df.copy()
    out[MONTH_COL] = out[date_col].dt.to_period("M")
    out["_p_intra"] = 1.0 + out[LEG_INTRADAY]
    out["_p_night"] = 1.0 + out[LEG_OVERNIGHT]
    out["_p_cc"] = 1.0 + out[LEG_CC]

    g = out.groupby([ticker_col, MONTH_COL], sort=True)
    monthly = pd.DataFrame(
        {
            "intraday_m": g["_p_intra"].prod(min_count=1) - 1.0,
            "overnight_m": g["_p_night"].prod(min_count=1) - 1.0,
            "cc_m": g["_p_cc"].prod(min_count=1) - 1.0,
            "n_days": g[LEG_CC].count(),
        }
    ).reset_index()
    return monthly


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """NaN-aware weighted mean. Rows with NaN value or weight are dropped."""
    v = values.to_numpy(dtype="float64")
    w = weights.to_numpy(dtype="float64")
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    wsum = w[ok].sum()
    if wsum <= 0:
        return float("nan")
    return float(np.dot(v[ok], w[ok]) / wsum)
