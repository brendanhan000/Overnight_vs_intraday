"""Layer 2 — TugOfWar timing + the overnight-intraday spread feature.

(a) TugOfWar timing (building on the Eq.-1 decomposition):
    * Build anomaly factor long-shorts: price MOMENTUM (12-1) and SHORT-TERM
      REVERSAL (1-month, long losers).
    * For each factor, EWMA its monthly OVERNIGHT and INTRADAY legs (60-month
      half-life), skipping the most recent month.
    * Form the sign-flipped TugOfWar predictor  ToW_t = EWMA(overnight) - EWMA(intraday)
      (the minus sign flips the intraday leg, so a strategy that earns overnight and
      bleeds intraday scores high), aligned so ToW for month m uses data through m-2.
    * Forecast the factor's next-month CLOSE-TO-CLOSE return; Newey-West t-stats.
      Expected sign: positive.

(b) Spread feature: per-name and value-weighted index-level (overnight - intraday)
    monthly series, written to parquet for a downstream regime model.

CAVEAT: a 60-month-half-life EWMA timing signal needs a LONG sample to be
meaningful (the paper uses ~20 years). On a short window the predictor barely
varies and the regression is underpowered — flagged in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import Config
from ..core.decomposition import COL_TICKER
from ..core.portfolios import (
    HOLDING_LEGS,
    MonthlyPanel,
    build_monthly_panel,
    form_long_short,
)
from ..core.stats import describe

FACTORS = ("momentum", "short_term_reversal")


@dataclass
class Layer2Result:
    meta: Dict[str, object]
    factors: Dict[str, object]      # name -> {legs, tow, reg, leg_means}
    pooled_reg: Dict[str, object]
    spread_index: pd.DataFrame
    spread_paths: Dict[str, str]
    reproduced: Dict[str, bool] = field(default_factory=dict)


# --- anomaly factors --------------------------------------------------------
def add_anomaly_signals(mp: MonthlyPanel) -> MonthlyPanel:
    """Add 12-1 momentum and short-term-reversal signals to the monthly panel.

    momentum at month m = cumulative cc return over [m-11, m-1] (12-month window
    skipping the most recent month m); form_long_short's lag=1 then sorts holding
    month h on month h-1's value, i.e. returns over [h-12, h-2]. STR signal is the
    negated lagged 1-month return, so decile 10 = past losers (a reversal long).
    """
    p = mp.panel.sort_values([COL_TICKER, "month"], kind="mergesort").copy()
    p["_l1"] = np.log1p(p["cc_m"])
    g = p.groupby(COL_TICKER, sort=False)["_l1"]
    p["mom"] = np.expm1(g.transform(lambda s: s.shift(1).rolling(11, min_periods=11).sum()))
    p["str_signal"] = -p["cc_m"]
    p = p.drop(columns="_l1")
    return MonthlyPanel(p, mp.n_bad_data, mp.n_tickers, mp.months)


def build_factor_legs(mp: MonthlyPanel, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Monthly overnight/intraday/cc legs for each anomaly factor long-short."""
    mp2 = add_anomaly_signals(mp)
    out: Dict[str, pd.DataFrame] = {}
    out["momentum"] = form_long_short(mp2, "mom", cfg, lag=1)["ls"]
    out["short_term_reversal"] = form_long_short(mp2, "str_signal", cfg, lag=1)["ls"]
    return out


# --- TugOfWar timing --------------------------------------------------------
def tugofwar_predictor(
    legs: pd.DataFrame, halflife: float, skip: int = 1
) -> pd.Series:
    """ToW_m = EWMA(overnight) - EWMA(intraday), using data through month m-1-skip.

    The shift by (1 + skip) makes the predictor for forecast month m depend only on
    legs up to month m-1-skip — i.e. lagged AND skipping the most recent month
    (no look-ahead).
    """
    eo = legs["overnight_m"].ewm(halflife=halflife, min_periods=12).mean()
    ed = legs["intraday_m"].ewm(halflife=halflife, min_periods=12).mean()
    return (eo - ed).shift(1 + skip)


def predict_regression(y: pd.Series, x: pd.Series, lags: int) -> Dict[str, float]:
    """Regress y on x with an intercept and Newey-West SEs; return the slope block."""
    df = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    n = len(df)
    if n < 12:
        return {"coef": float("nan"), "tstat": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n": n}
    X = sm.add_constant(df["x"])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "coef": float(res.params["x"]), "tstat": float(res.tvalues["x"]),
        "intercept": float(res.params["const"]), "r2": float(res.rsquared), "n": n,
    }


# --- spread feature (part b) ------------------------------------------------
def build_spread_features(mp: MonthlyPanel, cfg: Config):
    """Per-name and VW index-level (overnight - intraday) monthly spread series."""
    p = mp.panel
    per = p.loc[p["eligible"], [COL_TICKER, "month", "overnight_m", "intraday_m", "cc_m"]].copy()
    per["spread"] = per["overnight_m"] - per["intraday_m"]

    use = p[p["eligible"] & p["lag_weight"].gt(0)].dropna(subset=list(HOLDING_LEGS)).copy()
    for leg in HOLDING_LEGS:
        use["_wl_" + leg] = use["lag_weight"] * use[leg]
    g = use.groupby("month")
    wsum = g["lag_weight"].sum()
    idx = pd.DataFrame({
        "overnight": g["_wl_overnight_m"].sum() / wsum,
        "intraday": g["_wl_intraday_m"].sum() / wsum,
        "cc": g["_wl_cc_m"].sum() / wsum,
    })
    idx["spread"] = idx["overnight"] - idx["intraday"]
    idx.index.name = "month"
    return per.sort_values([COL_TICKER, "month"]).reset_index(drop=True), idx


def save_features(per: pd.DataFrame, idx: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_p = out / "per_name_overnight_intraday_spread.parquet"
    idx_p = out / "index_overnight_intraday_spread.parquet"
    per.to_parquet(per_p, index=False)
    idx.reset_index().assign(month=lambda d: d["month"].astype(str)).to_parquet(idx_p, index=False)
    return {"per_name": str(per_p), "index": str(idx_p)}


# --- orchestration ----------------------------------------------------------
def run_layer2(
    daily: pd.DataFrame, cfg: Optional[Config] = None, *, features_dir: str = "features"
) -> Layer2Result:
    cfg = cfg or Config()
    mp = build_monthly_panel(daily, cfg)
    hl = cfg.strategy.ewma_half_life_months
    skip = cfg.strategy.skip_recent_months
    lags = cfg.backtest.newey_west_lags

    factor_legs = build_factor_legs(mp, cfg)
    factors: Dict[str, object] = {}
    pooled_y, pooled_x = [], []
    for name, legs in factor_legs.items():
        tow = tugofwar_predictor(legs, hl, skip)
        reg = predict_regression(legs["cc_m"], tow, lags)
        leg_means = {leg: describe(legs[leg], lags) for leg in HOLDING_LEGS}
        factors[name] = {"legs": legs, "tow": tow, "reg": reg, "leg_means": leg_means}
        pair = pd.concat([legs["cc_m"].rename("y"), tow.rename("x")], axis=1).dropna()
        pooled_y.append(pair["y"])
        pooled_x.append(pair["x"])

    pooled_reg = predict_regression(pd.concat(pooled_y), pd.concat(pooled_x), lags)
    per, idx = build_spread_features(mp, cfg)
    paths = save_features(per, idx, features_dir)

    months = mp.months
    meta = {
        "date_min": str(months.min()), "date_max": str(months.max()),
        "n_months": int(len(months)), "n_tickers": mp.n_tickers,
        "halflife": hl, "skip_recent_months": skip, "nw_lags": lags,
        "short_sample_warning": len(months) < 3 * hl,
    }
    reproduced = {
        name: (f["reg"]["coef"] > 0 and f["reg"]["tstat"] > 1.0)
        for name, f in factors.items()
    }
    reproduced["pooled_positive"] = pooled_reg["coef"] > 0 and pooled_reg["tstat"] > 1.0
    return Layer2Result(meta, factors, pooled_reg, idx, paths, reproduced)
