"""Dumb-baseline bake-off (spec): the strategy must beat these on a risk-adjusted,
post-cost basis to count.

  (a) SPY overnight-only buy-and-hold  -- the classic overnight-drift benchmark.
  (b) random-decile-sort long-short    -- random signal -> VW decile 10 - 1.
  (c) coin-flip signal                 -- random long-half / short-half each month.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from ..core.decomposition import COL_TICKER, aggregate_monthly, compute_daily_legs
from ..core.portfolios import MonthlyPanel, form_long_short
from ..core.stats import describe


def spy_overnight_buyhold(start: str, end: str, cache_dir: str) -> pd.Series:
    """Monthly overnight return of a SPY buy-and-hold (long-only)."""
    from ..data.yahoo import YahooLoader

    df = YahooLoader(cache_dir=cache_dir).load(["SPY"], start, end)
    if df.empty:
        return pd.Series(dtype=float)
    m = aggregate_monthly(compute_daily_legs(df))
    s = m.set_index("month")["overnight_m"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s.sort_index()


def _random_signal_ls(mp: MonthlyPanel, cfg: Config, seed: int, leg: str = "cc_m") -> pd.Series:
    panel = mp.panel.copy()
    rng = np.random.default_rng(seed)
    panel["_rand"] = rng.random(len(panel))
    mp2 = MonthlyPanel(panel, mp.n_bad_data, mp.n_tickers, mp.months)
    res = form_long_short(mp2, "_rand", cfg, lag=1)
    return res["ls"].get(leg, pd.Series(dtype=float))


def _coin_flip_ls(mp: MonthlyPanel, cfg: Config, seed: int, leg: str = "cc_m") -> pd.Series:
    """Random long-half / short-half each month (VW), measured on ``leg``."""
    panel = mp.panel.copy()
    rng = np.random.default_rng(seed)
    use = panel[panel["eligible"] & panel["lag_weight"].gt(0)].dropna(subset=[leg]).copy()
    use["_coin"] = rng.integers(0, 2, len(use))  # 0 short, 1 long
    use["_wl"] = use["lag_weight"] * use[leg]
    g = use.groupby(["month", "_coin"])
    vw = (g["_wl"].sum() / g["lag_weight"].sum()).rename("vw").reset_index()
    wide = vw.pivot(index="month", columns="_coin", values="vw")
    if 1 not in wide or 0 not in wide:
        return pd.Series(dtype=float)
    s = (wide[1] - wide[0]).sort_index()
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def _summarize_random(series_list, lags: int) -> Dict[str, float]:
    stats = [describe(s, lags) for s in series_list if len(s) > 3]
    if not stats:
        return {"mean": float("nan"), "mean_t": float("nan"),
                "sharpe_ann": float("nan"), "frac_sig": float("nan"), "n_iter": 0}
    means = np.array([s["mean"] for s in stats])
    ts = np.array([s["tstat"] for s in stats])
    shp = np.array([s["sharpe_ann"] for s in stats])
    return {
        "mean": float(np.nanmean(means)),
        "mean_t": float(np.nanmean(ts)),
        "sharpe_ann": float(np.nanmean(shp)),
        "frac_sig": float(np.mean(np.abs(ts) > 2.0)),
        "n_iter": len(stats),
    }


def run_baselines(
    mp: MonthlyPanel,
    cfg: Config,
    start: str,
    end: str,
    n_iter: int = 20,
    leg: str = "cc_m",
) -> Dict[str, object]:
    """Return summary stats for all three dumb baselines."""
    lags = cfg.backtest.newey_west_lags
    spy = spy_overnight_buyhold(start, end, cfg.data.cache_dir)
    spy_stats = describe(spy, lags) if len(spy) > 3 else {}

    rand = [_random_signal_ls(mp, cfg, 100 + i, leg) for i in range(n_iter)]
    coin = [_coin_flip_ls(mp, cfg, 500 + i, leg) for i in range(n_iter)]

    return {
        "spy_overnight": {
            "mean": spy_stats.get("mean", float("nan")),
            "mean_t": spy_stats.get("tstat", float("nan")),
            "sharpe_ann": spy_stats.get("sharpe_ann", float("nan")),
            "skew": spy_stats.get("skew", float("nan")),
            "n": spy_stats.get("n", 0),
        },
        "random_decile_ls": _summarize_random(rand, lags),
        "coin_flip_ls": _summarize_random(coin, lags),
        "leg": leg,
    }
