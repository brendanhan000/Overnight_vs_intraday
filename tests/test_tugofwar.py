"""Tests for Layer 2: TugOfWar predictor alignment (no look-ahead), the
predictive regression, anomaly signals, and the spread feature."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ovi.backtest.tugofwar import (
    add_anomaly_signals,
    build_spread_features,
    predict_regression,
    tugofwar_predictor,
)
from ovi.config import Config
from ovi.core.portfolios import build_monthly_panel
from ovi.core.synthetic import make_synthetic_panel


def _legs(n=80, seed=0):
    idx = pd.period_range("2015-01", periods=n, freq="M")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "overnight_m": 0.01 + 0.02 * rng.standard_normal(n),
            "intraday_m": 0.02 * rng.standard_normal(n),
            "cc_m": 0.03 * rng.standard_normal(n),
        },
        index=idx,
    )


def test_tow_alignment_is_shift_1_plus_skip():
    legs = _legs()
    tow = tugofwar_predictor(legs, halflife=12, skip=1)
    eo = legs["overnight_m"].ewm(halflife=12, min_periods=12).mean()
    ed = legs["intraday_m"].ewm(halflife=12, min_periods=12).mean()
    expected = (eo - ed).shift(2)  # 1 (lag) + 1 (skip)
    pd.testing.assert_series_equal(tow, expected, check_names=False)


def test_tow_does_not_use_most_recent_month():
    # Perturbing the LAST month's legs must not change any existing ToW value
    # (no look-ahead): the most recent month is skipped/lagged out.
    legs = _legs()
    tow = tugofwar_predictor(legs, halflife=12, skip=1)
    bumped = legs.copy()
    bumped.iloc[-1, bumped.columns.get_loc("overnight_m")] += 5.0
    tow_b = tugofwar_predictor(bumped, halflife=12, skip=1)
    pd.testing.assert_series_equal(tow, tow_b, check_names=False)


def test_predict_regression_recovers_slope():
    idx = pd.period_range("2000-01", periods=150, freq="M")
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(150), index=idx)
    y = 0.5 + 2.0 * x + 0.001 * rng.standard_normal(150)
    out = predict_regression(y, x, lags=12)
    assert abs(out["coef"] - 2.0) < 0.01
    assert out["tstat"] > 5
    assert out["n"] == 150


def test_anomaly_signals_added():
    daily = make_synthetic_panel(n_tickers=40, n_days=420, seed=3)
    mp = build_monthly_panel(daily, Config())
    mp2 = add_anomaly_signals(mp)
    assert "mom" in mp2.panel.columns and "str_signal" in mp2.panel.columns
    # STR signal is the negated 1-month return.
    np.testing.assert_allclose(
        mp2.panel["str_signal"].values, -mp2.panel["cc_m"].values, equal_nan=True
    )
    # Momentum needs 11 months of history -> early rows are NaN.
    assert mp2.panel["mom"].isna().any()
    assert mp2.panel["mom"].notna().any()


def test_spread_equals_overnight_minus_intraday():
    daily = make_synthetic_panel(n_tickers=40, n_days=300, seed=4)
    mp = build_monthly_panel(daily, Config())
    per, idx = build_spread_features(mp, Config())
    np.testing.assert_allclose(
        per["spread"].values, (per["overnight_m"] - per["intraday_m"]).values
    )
    np.testing.assert_allclose(
        idx["spread"].values, (idx["overnight"] - idx["intraday"]).values
    )
