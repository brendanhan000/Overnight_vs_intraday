"""Statistics: Newey-West t-stats, CAPM / 3-factor alphas, summary stats.

All regressions use HAC (Newey-West) standard errors with a configurable lag
(default 12, per the spec). Returns are decimals; reported figures are scaled to
%/month by the report layer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_NW_LAGS = 12


def nw_mean_tstat(returns: pd.Series, lags: int = DEFAULT_NW_LAGS) -> Dict[str, float]:
    """Mean of a return series with a Newey-West t-stat (regression on a constant)."""
    y = pd.Series(returns).dropna().astype(float)
    n = len(y)
    if n < 3:
        return {"mean": float(y.mean()) if n else float("nan"), "tstat": float("nan"),
                "se": float("nan"), "n": n}
    X = np.ones((n, 1))
    res = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(res.params[0]), "tstat": float(res.tvalues[0]),
            "se": float(res.bse[0]), "n": n}


def regression_alpha(
    y: pd.Series,
    factors: pd.DataFrame,
    factor_cols: Sequence[str],
    lags: int = DEFAULT_NW_LAGS,
) -> Dict[str, object]:
    """Regress ``y`` on factors with an intercept; HAC t-stats.

    ``y`` and ``factors`` are aligned on their (Period[M]) index. Returns the alpha
    (intercept), its NW t-stat, factor betas, and n. Alpha is in the units of y
    (decimal/month).
    """
    df = pd.concat([pd.Series(y, name="_y"), factors[list(factor_cols)]], axis=1).dropna()
    n = len(df)
    if n < len(factor_cols) + 3:
        return {"alpha": float("nan"), "alpha_t": float("nan"),
                "betas": {c: float("nan") for c in factor_cols}, "n": n}
    X = sm.add_constant(df[list(factor_cols)])
    res = sm.OLS(df["_y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha": float(res.params["const"]),
        "alpha_t": float(res.tvalues["const"]),
        "betas": {c: float(res.params[c]) for c in factor_cols},
        "n": n,
    }


def capm_alpha(excess: pd.Series, factors: pd.DataFrame, lags: int = DEFAULT_NW_LAGS):
    """CAPM alpha: regress excess return on Mkt-RF."""
    return regression_alpha(excess, factors, ["Mkt-RF"], lags)


def ff3_alpha(excess: pd.Series, factors: pd.DataFrame, lags: int = DEFAULT_NW_LAGS):
    """Fama-French 3-factor alpha: regress excess return on Mkt-RF, SMB, HML."""
    return regression_alpha(excess, factors, ["Mkt-RF", "SMB", "HML"], lags)


def describe(returns: pd.Series, lags: int = DEFAULT_NW_LAGS) -> Dict[str, float]:
    """Mean (NW t), vol, skewness, annualized Sharpe, min/max for a monthly series."""
    y = pd.Series(returns).dropna().astype(float)
    base = nw_mean_tstat(y, lags)
    std = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
    sharpe = float(base["mean"] / std * np.sqrt(12)) if std and std > 0 else float("nan")
    return {
        **base,
        "std": std,
        "skew": float(y.skew()) if len(y) > 2 else float("nan"),
        "sharpe_ann": sharpe,
        "min": float(y.min()) if len(y) else float("nan"),
        "max": float(y.max()) if len(y) else float("nan"),
    }


def leg_block(
    excess: pd.Series,
    factors: pd.DataFrame,
    lags: int = DEFAULT_NW_LAGS,
) -> Dict[str, object]:
    """One leg's full reporting block: mean+NW t, CAPM alpha, FF3 alpha, skew."""
    d = describe(excess, lags)
    capm = capm_alpha(excess, factors, lags)
    ff3 = ff3_alpha(excess, factors, lags)
    return {
        "mean": d["mean"], "mean_t": d["tstat"], "skew": d["skew"],
        "std": d["std"], "sharpe_ann": d["sharpe_ann"], "n": d["n"],
        "capm_alpha": capm["alpha"], "capm_alpha_t": capm["alpha_t"],
        "ff3_alpha": ff3["alpha"], "ff3_alpha_t": ff3["alpha_t"],
    }
