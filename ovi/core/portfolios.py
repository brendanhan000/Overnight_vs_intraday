"""Cross-sectional portfolio formation for Layer 1 (Table 1).

Pipeline: daily OHLC -> daily legs -> bad-data filter -> per-(ticker, month) legs
plus end-of-month price and a dollar-volume cap proxy -> eligibility & lagged
weights (no look-ahead) -> decile sorts on a *lagged* leg -> value-weighted
long(decile 10) - short(decile 1), measured in the holding month's overnight,
intraday, and close-to-close legs.

No look-ahead, by construction:
  * the SIGNAL for holding month h is the leg return from month ``h - lag``
    (``lag >= 1``), obtained by shifting within each ticker;
  * ELIGIBILITY and WEIGHTS use info as of the end of month ``h - 1``
    (lagged price and lagged dollar volume).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .decomposition import (
    COL_CLOSE,
    COL_DATE,
    COL_TICKER,
    aggregate_monthly,
    apply_bad_data_filter,
    compute_daily_legs,
)

HOLDING_LEGS = ("overnight_m", "intraday_m", "cc_m")


@dataclass
class MonthlyPanel:
    panel: pd.DataFrame           # ticker, month, legs, price, weights, eligible
    n_bad_data: int
    n_tickers: int
    months: pd.PeriodIndex        # sorted unique eligible months


def build_monthly_panel(daily: pd.DataFrame, cfg: Config) -> MonthlyPanel:
    """Per-(ticker, month) legs + eligibility + lagged value weights."""
    fcfg = cfg.filters
    legs = compute_daily_legs(daily)
    legs, n_bad = apply_bad_data_filter(
        legs, lower=fcfg.bad_data_lower, upper=fcfg.bad_data_upper
    )
    monthly = aggregate_monthly(legs)  # ticker, month, overnight_m, intraday_m, cc_m, n_days

    legs = legs.copy()
    legs["month"] = legs[COL_DATE].dt.to_period("M")
    legs["_dvol"] = legs[COL_CLOSE] * legs.get("volume", np.nan)
    aux = (
        legs.groupby([COL_TICKER, "month"])
        .agg(price=(COL_CLOSE, "last"), dollar_vol=("_dvol", "mean"))
        .reset_index()
    )
    panel = monthly.merge(aux, on=[COL_TICKER, "month"], how="left")
    panel = panel.sort_values([COL_TICKER, "month"], kind="mergesort").reset_index(drop=True)

    # Lagged (end of prior month) fields for eligibility and weighting.
    grp = panel.groupby(COL_TICKER, sort=False)
    panel["lag_price"] = grp["price"].shift(1)
    panel["lag_weight"] = grp["dollar_vol"].shift(1)

    # Price filter, then drop the bottom market-cap (proxy) quintile per month.
    panel["price_ok"] = panel["lag_price"].ge(fcfg.min_price) & panel["lag_weight"].gt(0)
    q = fcfg.exclude_bottom_mktcap_quantile
    bp = (
        panel[panel["price_ok"]]
        .groupby("month")["lag_weight"]
        .quantile(q)
        .rename("mc_bp")
    )
    panel = panel.merge(bp, on="month", how="left")
    panel["eligible"] = panel["price_ok"] & panel["lag_weight"].ge(panel["mc_bp"])

    months = pd.PeriodIndex(sorted(panel.loc[panel["eligible"], "month"].unique()), freq="M")
    return MonthlyPanel(
        panel=panel, n_bad_data=int(n_bad),
        n_tickers=int(panel[COL_TICKER].nunique()), months=months,
    )


def _assign_deciles(s: pd.Series, n: int) -> pd.Series:
    """Stable decile labels 1..n (ties broken by first occurrence for even bins)."""
    if s.notna().sum() < n:
        return pd.Series(np.nan, index=s.index)
    ranks = s.rank(method="first")
    return pd.qcut(ranks, n, labels=False).astype(float) + 1.0


def form_long_short(
    mp: MonthlyPanel,
    signal_col: str,
    cfg: Config,
    lag: int = 1,
    legs: Sequence[str] = HOLDING_LEGS,
) -> Dict[str, object]:
    """Form the VW decile long-short on a lagged signal.

    Returns:
      ls       : DataFrame indexed by holding month, columns = ``legs`` (decile
                 ``long`` minus decile ``short`` VW return for each leg).
      deciles  : DataFrame [decile x leg] of time-averaged VW decile returns.
      weights  : dict leg-independent {('long'|'short'): {month: Series(ticker->w)}}
                 used for turnover; built from the formed book.
      n_months : number of holding months.
    """
    pcfg = cfg.portfolio
    n = pcfg.n_deciles
    p = mp.panel.sort_values([COL_TICKER, "month"], kind="mergesort").copy()
    p["signal"] = p.groupby(COL_TICKER, sort=False)[signal_col].shift(lag)

    use = p[p["eligible"] & p["signal"].notna() & p["lag_weight"].gt(0)].copy()
    use = use.dropna(subset=list(legs))
    use["decile"] = use.groupby("month")["signal"].transform(lambda s: _assign_deciles(s, n))
    use = use.dropna(subset=["decile"])
    use["decile"] = use["decile"].astype(int)

    # VW each leg within (month, decile).
    ls_cols: Dict[str, pd.Series] = {}
    decile_means: Dict[str, pd.Series] = {}
    for leg in legs:
        use["_wl"] = use["lag_weight"] * use[leg]
        gd = use.groupby(["month", "decile"])
        vw = (gd["_wl"].sum() / gd["lag_weight"].sum()).rename("vw").reset_index()
        wide = vw.pivot(index="month", columns="decile", values="vw")
        if pcfg.long_decile in wide and pcfg.short_decile in wide:
            ls_cols[leg] = wide[pcfg.long_decile] - wide[pcfg.short_decile]
        decile_means[leg] = wide.mean(axis=0)  # time-avg per decile

    ls = pd.DataFrame(ls_cols).sort_index()
    deciles = pd.DataFrame(decile_means)  # index=decile, columns=legs

    # Per-month name weights for the long and short books (for turnover/costs).
    def _book_weights(decile_val: int) -> Dict[pd.Period, pd.Series]:
        sub = use[use["decile"] == decile_val]
        out: Dict[pd.Period, pd.Series] = {}
        for m, g in sub.groupby("month"):
            w = g.groupby(COL_TICKER)["lag_weight"].sum()
            tot = w.sum()
            if tot > 0:
                out[m] = w / tot
        return out

    weights = {"long": _book_weights(pcfg.long_decile),
               "short": _book_weights(pcfg.short_decile)}

    # Trading days per holding month (max names' day count) for daily-harvest costs.
    ndays = use.groupby("month")["n_days"].max()

    return {"ls": ls, "deciles": deciles, "weights": weights,
            "n_months": int(len(ls)), "n_days_by_month": ndays,
            "rep_price": float(use["lag_price"].median())}


def book_turnover(weights_by_month: Dict[pd.Period, pd.Series]) -> pd.Series:
    """Monthly one-sided turnover 0.5*sum|w_t - w_{t-1}| for a single book."""
    months = sorted(weights_by_month.keys())
    out = {}
    prev = None
    for m in months:
        w = weights_by_month[m]
        if prev is None:
            out[m] = 1.0  # initial build = full turnover
        else:
            allnames = w.index.union(prev.index)
            wc = w.reindex(allnames).fillna(0.0)
            pc = prev.reindex(allnames).fillna(0.0)
            out[m] = float(0.5 * (wc - pc).abs().sum())
        prev = w
    return pd.Series(out).sort_index()


def long_short_turnover(weights: Dict[str, Dict[pd.Period, pd.Series]]) -> pd.Series:
    """Combined long+short turnover per month (both sides fully tradeable)."""
    lt = book_turnover(weights["long"])
    st = book_turnover(weights["short"])
    return lt.add(st, fill_value=0.0)
