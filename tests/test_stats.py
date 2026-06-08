"""Tests for Newey-West stats and factor-alpha recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ovi.core.stats import describe, ff3_alpha, nw_mean_tstat, regression_alpha


def _months(n, start="2000-01"):
    return pd.period_range(start, periods=n, freq="M")


def test_nw_mean_positive_series_has_positive_t():
    rng = np.random.default_rng(0)
    y = pd.Series(0.01 + 0.002 * rng.standard_normal(120), index=_months(120))
    out = nw_mean_tstat(y, lags=12)
    assert abs(out["mean"] - 0.01) < 0.002
    assert out["tstat"] > 5  # clearly positive mean


def test_regression_alpha_recovers_known_alpha_beta():
    idx = _months(150)
    rng = np.random.default_rng(1)
    mkt = pd.Series(0.01 * rng.standard_normal(150), index=idx, name="Mkt-RF")
    factors = mkt.to_frame()
    y = 0.005 + 1.2 * mkt  # exact: alpha=0.005, beta=1.2
    out = regression_alpha(y, factors, ["Mkt-RF"], lags=12)
    assert abs(out["alpha"] - 0.005) < 1e-6
    assert abs(out["betas"]["Mkt-RF"] - 1.2) < 1e-6


def test_ff3_alpha_runs_with_three_factors():
    idx = _months(150)
    rng = np.random.default_rng(2)
    factors = pd.DataFrame(
        {
            "Mkt-RF": 0.01 * rng.standard_normal(150),
            "SMB": 0.008 * rng.standard_normal(150),
            "HML": 0.008 * rng.standard_normal(150),
        },
        index=idx,
    )
    y = 0.004 + 0.9 * factors["Mkt-RF"] - 0.3 * factors["HML"] + 0.001 * rng.standard_normal(150)
    out = ff3_alpha(y, factors, lags=12)
    assert abs(out["alpha"] - 0.004) < 0.002


def test_describe_reports_skew_and_sharpe():
    idx = _months(120)
    rng = np.random.default_rng(3)
    y = pd.Series(0.01 + 0.03 * rng.standard_normal(120), index=idx)
    d = describe(y)
    assert "skew" in d and "sharpe_ann" in d
    assert d["sharpe_ann"] > 0


def test_sharpe_annualization_formula():
    # Sharpe must be the monthly mean / monthly std, annualized by sqrt(12).
    idx = _months(120)
    rng = np.random.default_rng(7)
    y = pd.Series(0.01 + 0.04 * rng.standard_normal(120), index=idx)
    d = describe(y)
    expected = y.mean() / y.std(ddof=1) * np.sqrt(12)
    assert abs(d["sharpe_ann"] - expected) < 1e-9
