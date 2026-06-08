"""Unit tests for the return-decomposition math (the project's heart).

The key invariant is the identity (1+intraday)(1+overnight) = (1+cc), which must
hold to machine precision both daily and after monthly compounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ovi.core.decomposition import (
    LEG_CC,
    LEG_INTRADAY,
    LEG_OVERNIGHT,
    aggregate_monthly,
    apply_bad_data_filter,
    compute_daily_legs,
    identity_residual,
    max_identity_error,
)
from ovi.core.synthetic import make_synthetic_panel

TOL = 1e-12


def test_daily_identity_holds():
    panel = make_synthetic_panel(n_tickers=8, n_days=120, seed=1)
    legs = compute_daily_legs(panel)
    assert max_identity_error(legs) < TOL


def test_monthly_identity_holds():
    panel = make_synthetic_panel(n_tickers=8, n_days=180, seed=2)
    legs = compute_daily_legs(panel)
    monthly = aggregate_monthly(legs)
    resid = (1 + monthly["intraday_m"]) * (1 + monthly["overnight_m"]) - (
        1 + monthly["cc_m"]
    )
    assert float(np.nanmax(np.abs(resid))) < TOL


def test_first_day_per_ticker_is_nan():
    panel = make_synthetic_panel(n_tickers=3, n_days=10, seed=3)
    legs = compute_daily_legs(panel).sort_values(["ticker", "date"])
    first = legs.groupby("ticker").head(1)
    assert first[[LEG_INTRADAY, LEG_OVERNIGHT, LEG_CC]].isna().all().all()


def test_missing_open_rolls_into_overnight():
    # With opens removed, those days must show intraday == 0 and overnight == cc,
    # and the identity must still hold everywhere.
    panel = make_synthetic_panel(
        n_tickers=6, n_days=120, seed=4, missing_open_frac=0.25
    )
    legs = compute_daily_legs(panel)
    missing = legs[legs["open"].isna() & legs[LEG_CC].notna()]
    assert len(missing) > 0  # the fixture actually produced missing opens
    assert np.allclose(missing[LEG_INTRADAY], 0.0, atol=TOL)
    assert np.allclose(missing[LEG_OVERNIGHT], missing[LEG_CC], atol=TOL)
    assert max_identity_error(legs) < TOL


def test_corporate_action_lands_in_overnight_leg():
    # Deterministic constant factors + a 2:1 split on day 40 for one ticker.
    on, intra = 0.01, 0.005
    panel = make_synthetic_panel(
        n_tickers=2,
        n_days=80,
        deterministic=True,
        overnight_mu=np.log(1 + on),
        intraday_mu=np.log(1 + intra),
        splits={"T000": (40, 2.0)},
    )
    legs = compute_daily_legs(panel).sort_values(["ticker", "date"])
    split_tkr = legs[legs["ticker"] == "T000"].reset_index(drop=True)

    # Intraday leg is the within-day ratio, so the split cancels: ~0.5% every day.
    assert np.allclose(split_tkr[LEG_INTRADAY].iloc[1:], intra, atol=1e-12)

    # The overnight leg on the split day recovers the TRUE +1% overnight move,
    # NOT the spurious ~-50% you would get from raw open/prev-raw-close.
    split_day = split_tkr.iloc[40]
    assert split_day[LEG_OVERNIGHT] == pytest.approx(on, abs=1e-12)
    assert max_identity_error(legs) < TOL


def test_bad_data_filter_removes_and_counts():
    panel = make_synthetic_panel(n_tickers=4, n_days=60, seed=5)
    legs = compute_daily_legs(panel)
    # Inject an obviously bad +80% intraday print on a known row.
    idx = legs.index[len(legs) // 2]
    legs.loc[idx, LEG_INTRADAY] = 0.80
    filtered, n_bad = apply_bad_data_filter(legs)
    assert n_bad >= 1
    assert filtered.loc[idx, [LEG_INTRADAY, LEG_OVERNIGHT, LEG_CC]].isna().all()


def test_known_two_stock_hand_calc():
    # Tiny hand-checkable panel: one ticker, three days, no splits.
    df = pd.DataFrame(
        {
            "ticker": ["X", "X", "X"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [10.0, 10.5, 11.0],
            "close": [10.5, 11.0, 10.8],
            "adj_close": [10.5, 11.0, 10.8],  # no corp actions -> adj == raw close
        }
    )
    legs = compute_daily_legs(df).sort_values("date").reset_index(drop=True)
    # Day 2: intraday = 11.0/10.5 - 1; cc = 11.0/10.5 - 1; overnight = 0.
    assert legs[LEG_INTRADAY].iloc[1] == pytest.approx(11.0 / 10.5 - 1)
    assert legs[LEG_CC].iloc[1] == pytest.approx(11.0 / 10.5 - 1)
    assert legs[LEG_OVERNIGHT].iloc[1] == pytest.approx(0.0, abs=1e-15)
    # Day 3: overnight = open/prev_close - 1 = 11.0/11.0 - 1 = 0; intraday = 10.8/11-1.
    assert legs[LEG_OVERNIGHT].iloc[2] == pytest.approx(11.0 / 11.0 - 1, abs=1e-15)
    assert legs[LEG_INTRADAY].iloc[2] == pytest.approx(10.8 / 11.0 - 1)
    assert max_identity_error(legs) < TOL
