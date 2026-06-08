"""Layer 1 engine — orchestrates the Table 1 replication and the honesty layers.

Produces, for both sorts (Panel A = lagged overnight signal, Panel B = lagged
intraday signal):
  * decile table + long-short for each holding leg (overnight / intraday / cc),
  * mean (NW t), CAPM alpha, FF3 alpha, skewness,
  * the three cost versions (gross / harvest / tradeable),
  * the Fig. 2 lag sweep (four t-stat curves),
  * dumb-baseline bake-off and walk-forward IS/OOS/lockbox segments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..core.portfolios import (
    HOLDING_LEGS,
    MonthlyPanel,
    build_monthly_panel,
    form_long_short,
    long_short_turnover,
)
from ..core.stats import describe, leg_block, nw_mean_tstat
from ..data.factors import load_ff_factors
from .baselines import run_baselines
from .costs import CostModel
from .walkforward import split_series

LEG_LABEL = {"overnight_m": "overnight", "intraday_m": "intraday", "cc_m": "close-close"}


@dataclass
class Layer1Result:
    meta: Dict[str, object]
    panelA: Dict[str, object]  # overnight-sorted
    panelB: Dict[str, object]  # intraday-sorted
    costs: Dict[str, object]
    baselines: Dict[str, object]
    walkforward: Dict[str, object]
    fig2: pd.DataFrame
    factors: pd.DataFrame
    reproduced: Dict[str, bool] = field(default_factory=dict)


def _analyze(mp: MonthlyPanel, signal_col: str, cfg: Config, factors: pd.DataFrame) -> Dict[str, object]:
    form = form_long_short(mp, signal_col, cfg, lag=1)
    ls = form["ls"]
    lags = cfg.backtest.newey_west_lags
    blocks = {leg: leg_block(ls[leg], factors, lags) for leg in ls.columns}
    return {"form": form, "ls": ls, "blocks": blocks, "deciles": form["deciles"]}


def _cost_versions(panelA: Dict[str, object], cfg: Config) -> Dict[str, object]:
    form = panelA["form"]
    ls = panelA["ls"]
    lags = cfg.backtest.newey_west_lags
    cm = CostModel(cfg.costs, rep_price=form["rep_price"])

    # V2 — harvest the overnight leg (daily round trips).
    net_on, cost_on = cm.harvest_overnight_net(ls["overnight_m"], form["n_days_by_month"])
    on_gross = describe(ls["overnight_m"], lags)
    on_net = describe(net_on, lags)
    avg_days = float(form["n_days_by_month"].reindex(ls.index).mean())
    breakeven = cm.breakeven_roundtrip_bps(on_gross["mean"], avg_days)

    # V3 — tradeable as stated (monthly rebalance, hold through).
    turnover = long_short_turnover(form["weights"])
    net_cc, cost_cc = cm.tradeable_net(ls["cc_m"], turnover)
    cc_gross = describe(ls["cc_m"], lags)
    cc_net = describe(net_cc, lags)

    return {
        "rep_price": form["rep_price"],
        "roundtrip_bps": cm.roundtrip_bps,
        "oneway_bps": cm.oneway_bps,
        "commission_bps": cm.commission_bps,
        "avg_trading_days": avg_days,
        "harvest": {
            "gross": on_gross, "net": on_net,
            "monthly_cost_mean": float(cost_on.mean()),
            "breakeven_roundtrip_bps": breakeven,
        },
        "tradeable": {
            "gross": cc_gross, "net": cc_net,
            "turnover_mean": float(turnover.reindex(ls.index).mean()),
            "monthly_cost_mean": float(cost_cc.mean()),
        },
    }


def _walkforward(panelA: Dict[str, object], cfg: Config) -> Dict[str, object]:
    lags = cfg.backtest.newey_west_lags
    out: Dict[str, object] = {}
    for leg in HOLDING_LEGS:
        segs = split_series(panelA["ls"][leg], cfg.backtest.insample_end, cfg.backtest.lockbox_start)
        out[leg] = {seg: describe(s, lags) for seg, s in segs.items()}
    return out


def lag_sweep(
    mp: MonthlyPanel, cfg: Config, lags_list: Optional[List[int]] = None
) -> pd.DataFrame:
    """Fig. 2 analog: NW t-stat of the four persistence/reversal curves vs lag."""
    nwl = cfg.backtest.newey_west_lags
    if lags_list is None:
        hi = min(cfg.backtest.fig2_max_lag_months, max(2, len(mp.months) - 24))
        lags_list = list(range(1, hi + 1))

    def _t(ls: pd.DataFrame, leg: str) -> float:
        if leg not in ls.columns:
            return float("nan")
        return nw_mean_tstat(ls[leg], nwl)["tstat"]

    rows = []
    for L in lags_list:
        fA = form_long_short(mp, "overnight_m", cfg, lag=L)["ls"]
        fB = form_long_short(mp, "intraday_m", cfg, lag=L)["ls"]
        rows.append({
            "lag": L,
            "A_overnight_t": _t(fA, "overnight_m"),   # overnight-sorted -> overnight (persist)
            "A_intraday_t": _t(fA, "intraday_m"),     # overnight-sorted -> intraday (reverse)
            "B_intraday_t": _t(fB, "intraday_m"),     # intraday-sorted  -> intraday (persist)
            "B_overnight_t": _t(fB, "overnight_m"),   # intraday-sorted  -> overnight (reverse)
        })
    return pd.DataFrame(rows).set_index("lag")


def _reproduced(panelA, panelB) -> Dict[str, bool]:
    """Expected US signs: A -> +overnight, -intraday; B -> +intraday, -overnight."""
    a = panelA["blocks"]
    b = panelB["blocks"]
    return {
        "A_overnight_positive": a["overnight_m"]["mean"] > 0 and a["overnight_m"]["mean_t"] > 2,
        "A_intraday_negative": b is not None and a["intraday_m"]["mean"] < 0,
        "B_intraday_positive": b["intraday_m"]["mean"] > 0 and b["intraday_m"]["mean_t"] > 2,
        "B_overnight_negative": b["overnight_m"]["mean"] < 0,
    }


def run_layer1(
    daily: pd.DataFrame,
    cfg: Optional[Config] = None,
    *,
    do_baselines: bool = True,
    do_fig2: bool = True,
    survivorship_note: str = "",
) -> Layer1Result:
    cfg = cfg or Config()
    mp = build_monthly_panel(daily, cfg)
    months = mp.months
    start = months.min().start_time.strftime("%Y-%m-%d")
    end = months.max().end_time.strftime("%Y-%m-%d")
    factors = load_ff_factors(start, end, cfg.data.cache_dir)

    panelA = _analyze(mp, "overnight_m", cfg, factors)
    panelB = _analyze(mp, "intraday_m", cfg, factors)
    costs = _cost_versions(panelA, cfg)
    walkforward = _walkforward(panelA, cfg)
    fig2 = lag_sweep(mp, cfg) if do_fig2 else pd.DataFrame()
    baselines = run_baselines(mp, cfg, start, end) if do_baselines else {}

    meta = {
        "date_min": str(months.min()), "date_max": str(months.max()),
        "n_months": int(len(months)), "n_tickers": mp.n_tickers,
        "n_bad_data": mp.n_bad_data, "n_long_short_months": panelA["form"]["n_months"],
        "weighting": "dollar-volume PROXY (no point-in-time shares outstanding)",
        "survivorship_note": survivorship_note,
        "nw_lags": cfg.backtest.newey_west_lags,
    }
    return Layer1Result(
        meta=meta, panelA=panelA, panelB=panelB, costs=costs, baselines=baselines,
        walkforward=walkforward, fig2=fig2, factors=factors,
        reproduced=_reproduced(panelA, panelB),
    )
