"""Layer 0 — sanity replication.

Decompose the value-weighted *market* (eligible universe) return into overnight
and intraday legs, print the split, and gate on it. Paper ballpark (CRSP,
1993-2013): overnight ~0.55%/mo, intraday ~0.38%/mo. If we can't get a sane
positive split of roughly the right magnitude, STOP and surface a data-quality
report before writing any portfolio code.

The overnight/intraday/cc identity is a *stock-level* property; at the
value-weighted index level it holds only up to a small cross-sectional covariance
term, so we report the index legs without asserting the identity there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..core.decomposition import (
    COL_CLOSE,
    COL_DATE,
    COL_TICKER,
    DAILY_LEGS,
    LEG_CC,
    LEG_INTRADAY,
    LEG_OVERNIGHT,
    apply_bad_data_filter,
    compute_daily_legs,
)


@dataclass
class Layer0Result:
    monthly_vw: pd.DataFrame  # market legs, value(/proxy)-weighted
    monthly_ew: pd.DataFrame  # market legs, equal-weighted (comparison)
    summary: Dict[str, float]
    weighting_kind: str
    weighting_is_proxy: bool
    data_quality: Dict[str, object] = field(default_factory=dict)
    sanity_passed: bool = False
    sanity_notes: str = ""


def _market_daily(daily: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """Value-weighted daily market leg for each of intraday/overnight/cc.

    Operates on rows where all three legs and the weight are present (the legs
    share a valid-day set by construction), so the per-day denominator is common.
    """
    d = daily.dropna(subset=list(DAILY_LEGS) + [weight_col]).copy()
    d = d[d[weight_col] > 0]
    if d.empty:
        return pd.DataFrame(columns=["intraday", "overnight", "cc", "wsum"])
    for leg in DAILY_LEGS:
        d["_wl_" + leg] = d[weight_col] * d[leg]
    g = d.groupby(COL_DATE, sort=True)
    wsum = g[weight_col].sum()
    out = pd.DataFrame(
        {
            "intraday": g["_wl_" + LEG_INTRADAY].sum() / wsum,
            "overnight": g["_wl_" + LEG_OVERNIGHT].sum() / wsum,
            "cc": g["_wl_" + LEG_CC].sum() / wsum,
            "wsum": wsum,
        }
    )
    return out


def _compound_monthly(daily_mkt: pd.DataFrame) -> pd.DataFrame:
    if daily_mkt.empty:
        return pd.DataFrame(columns=["intraday_m", "overnight_m", "cc_m"])
    m = daily_mkt.copy()
    month = m.index.to_period("M")
    out = pd.DataFrame(
        {
            "intraday_m": (1 + m["intraday"]).groupby(month).prod() - 1,
            "overnight_m": (1 + m["overnight"]).groupby(month).prod() - 1,
            "cc_m": (1 + m["cc"]).groupby(month).prod() - 1,
        }
    )
    out.index.name = "month"
    return out


def run_layer0(daily: pd.DataFrame, cfg: Optional[Config] = None) -> Layer0Result:
    """Full Layer-0 pipeline: legs -> filter -> eligibility/weights -> market split."""
    cfg = cfg or Config()
    fcfg = cfg.filters

    legs = compute_daily_legs(daily)
    legs, n_bad = apply_bad_data_filter(
        legs, lower=fcfg.bad_data_lower, upper=fcfg.bad_data_upper
    )
    legs = legs.sort_values([COL_TICKER, COL_DATE], kind="mergesort")
    grp = legs.groupby(COL_TICKER, sort=False)

    # Lagged price (eligibility) and lagged weight (no look-ahead).
    legs["lag_close"] = grp[COL_CLOSE].shift(1)
    has_mktcap = "market_cap" in legs.columns and legs["market_cap"].notna().any()
    if has_mktcap:
        legs["lag_weight"] = grp["market_cap"].shift(1)
        weighting_kind, is_proxy = "value (market cap)", False
    elif "volume" in legs.columns:
        legs["_dollar_vol"] = legs[COL_CLOSE] * legs["volume"]
        legs["lag_weight"] = legs.groupby(COL_TICKER)["_dollar_vol"].shift(1)
        weighting_kind, is_proxy = "dollar-volume PROXY (no shares outstanding)", True
    else:
        legs["lag_weight"] = 1.0
        weighting_kind, is_proxy = "equal (no cap/volume available)", True

    # Universe: price >= min_price AND not in the bottom market-cap quintile,
    # using cross-sectional percentiles of the price-eligible set as the NYSE
    # breakpoint approximation (spec).
    legs["_price_ok"] = legs["lag_close"].ge(fcfg.min_price) & legs["lag_weight"].gt(0)
    pe = legs[legs["_price_ok"]]
    q = fcfg.exclude_bottom_mktcap_quantile
    thr = pe.groupby(COL_DATE)["lag_weight"].quantile(q).rename("mc_breakpoint")
    legs = legs.merge(thr, left_on=COL_DATE, right_index=True, how="left")
    legs["eligible"] = legs["_price_ok"] & legs["lag_weight"].ge(legs["mc_breakpoint"])

    elig = legs[legs["eligible"]].copy()
    monthly_vw = _compound_monthly(_market_daily(elig, "lag_weight"))
    elig["_eq"] = 1.0
    monthly_ew = _compound_monthly(_market_daily(elig, "_eq"))

    # Summary in %/mo.
    def mean_pct(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean() * 100) if len(df) else float("nan")

    summary = {
        "overnight_vw_pct_mo": mean_pct(monthly_vw, "overnight_m"),
        "intraday_vw_pct_mo": mean_pct(monthly_vw, "intraday_m"),
        "cc_vw_pct_mo": mean_pct(monthly_vw, "cc_m"),
        "overnight_ew_pct_mo": mean_pct(monthly_ew, "overnight_m"),
        "intraday_ew_pct_mo": mean_pct(monthly_ew, "intraday_m"),
        "cc_ew_pct_mo": mean_pct(monthly_ew, "cc_m"),
        "n_months": int(len(monthly_vw)),
    }

    valid_days = legs[LEG_CC].notna()
    data_quality = {
        "n_rows": int(len(legs)),
        "n_tickers": int(legs[COL_TICKER].nunique()),
        "n_trading_days": int(legs[COL_DATE].nunique()),
        "date_min": str(pd.Timestamp(legs[COL_DATE].min()).date()),
        "date_max": str(pd.Timestamp(legs[COL_DATE].max()).date()),
        "n_bad_data_removed": int(n_bad),
        "n_missing_open": int((legs["open"].isna() & valid_days).sum()),
        "missing_open_frac": float((legs["open"].isna() & valid_days).mean()),
        "n_eligible_stockdays": int(legs["eligible"].sum()),
        "has_market_cap": bool(has_mktcap),
    }

    # Sanity gate: overnight leg positive and in a loose plausible band (%/mo).
    # The index recombination below is reported but NOT gated: the identity is a
    # stock-level property; at the value-weighted index level it deviates by a
    # cross-sectional covariance term that can legitimately exceed any tight band.
    on = summary["overnight_vw_pct_mo"]
    intr = summary["intraday_vw_pct_mo"]
    cc = summary["cc_vw_pct_mo"]
    summary["recomb_vw_pct_mo"] = ((1 + on / 100) * (1 + intr / 100) - 1) * 100
    passed = bool(np.isfinite(on) and 0.0 < on < 3.0)
    notes = []
    if not (0.0 < on < 3.0):
        notes.append(f"overnight {on:.3f}%/mo outside plausible (0, 3) band")
    if is_proxy:
        notes.append(
            "weighting is a PROXY (no point-in-time shares outstanding); "
            "treat the magnitude as indicative, not exact"
        )

    return Layer0Result(
        monthly_vw=monthly_vw,
        monthly_ew=monthly_ew,
        summary=summary,
        weighting_kind=weighting_kind,
        weighting_is_proxy=is_proxy,
        data_quality=data_quality,
        sanity_passed=passed,
        sanity_notes="; ".join(notes) if notes else "ok",
    )


def format_report(res: Layer0Result) -> str:
    """Human-readable Layer-0 report (printed to console / saved to reports/)."""
    dq = res.data_quality
    s = res.summary
    L = []
    L.append("=" * 70)
    L.append("LAYER 0 — Overnight vs. Intraday market decomposition (sanity check)")
    L.append("=" * 70)
    L.append("")
    L.append(f"Sample: {dq['date_min']} -> {dq['date_max']}  "
             f"({dq['n_trading_days']} trading days, {s['n_months']} months)")
    L.append(f"Tickers: {dq['n_tickers']:,}   Stock-day rows: {dq['n_rows']:,}   "
             f"Eligible stock-days: {dq['n_eligible_stockdays']:,}")
    L.append("")
    L.append("DATA QUALITY")
    L.append(f"  bad-data stock-days removed (<-30% or >+50%): {dq['n_bad_data_removed']:,}")
    L.append(f"  missing-open stock-days (rolled into overnight): "
             f"{dq['n_missing_open']:,} ({dq['missing_open_frac']*100:.2f}%)")
    L.append(f"  shares-outstanding available for true cap weights: {dq['has_market_cap']}")
    L.append("")
    L.append(f"WEIGHTING: {res.weighting_kind}")
    if res.weighting_is_proxy:
        L.append("  !! WARNING: not a point-in-time market-cap weight — see notes.")
    L.append("")
    L.append("MARKET SPLIT (mean monthly return, %/mo)")
    L.append(f"  {'leg':<12}{'value-wt':>12}{'equal-wt':>12}")
    L.append(f"  {'overnight':<12}{s['overnight_vw_pct_mo']:>12.4f}{s['overnight_ew_pct_mo']:>12.4f}")
    L.append(f"  {'intraday':<12}{s['intraday_vw_pct_mo']:>12.4f}{s['intraday_ew_pct_mo']:>12.4f}")
    L.append(f"  {'close-close':<12}{s['cc_vw_pct_mo']:>12.4f}{s['cc_ew_pct_mo']:>12.4f}")
    L.append("")
    L.append(f"  index recombination (info, not gated): (1+on)(1+intra)-1 = "
             f"{s['recomb_vw_pct_mo']:.4f} vs cc {s['cc_vw_pct_mo']:.4f} %/mo")
    L.append("  (gap = value-weighted leg covariance; the identity is exact only per stock)")
    L.append("")
    L.append("PAPER BALLPARK (CRSP 1993-2013): overnight ~0.55 %/mo, intraday ~0.38 %/mo")
    L.append("  (different sample/period here — compare direction & rough magnitude only)")
    L.append("")
    gate = "PASS ✅" if res.sanity_passed else "FAIL ❌"
    L.append(f"SANITY GATE: {gate}   ({res.sanity_notes})")
    if not res.sanity_passed:
        L.append("")
        L.append("  >>> STOP: do not proceed to Layer 1 on this data. Investigate the")
        L.append("  >>> data-quality items above before building portfolio code.")
    L.append("=" * 70)
    return "\n".join(L)
