"""Tests for the transaction-cost model."""
from __future__ import annotations

import pandas as pd

from ovi.backtest.costs import CostModel
from ovi.config import CostConfig


def test_commission_bps_and_roundtrip_monotonic():
    cm = CostModel(CostConfig(), rep_price=50.0)
    assert abs(cm.commission_bps - (0.005 / 50.0 * 1e4)) < 1e-9  # 1.0 bps
    # default round trip: 5 + 10 + 2*1 + 10 = 27 bps
    assert abs(cm.roundtrip_bps - 27.0) < 1e-9
    wider = CostModel(
        CostConfig(half_spread_bps_open=20.0), rep_price=50.0
    )
    assert wider.roundtrip_bps > cm.roundtrip_bps


def test_harvest_reduces_edge_and_scales_with_days():
    cm = CostModel(CostConfig(), rep_price=50.0)
    idx = pd.period_range("2020-01", periods=6, freq="M")
    gross = pd.Series(0.02, index=idx)
    days = pd.Series(21.0, index=idx)
    net, cost = cm.harvest_overnight_net(gross, days)
    assert (net < gross).all()
    assert (cost > 0).all()
    _, cost2 = cm.harvest_overnight_net(gross, days * 2)
    assert (cost2 > cost).all()  # more trading days -> more cost


def test_tradeable_cost_scales_with_turnover():
    cm = CostModel(CostConfig(), rep_price=50.0)
    idx = pd.period_range("2020-01", periods=4, freq="M")
    gross = pd.Series(0.01, index=idx)
    low = pd.Series(0.1, index=idx)
    high = pd.Series(0.9, index=idx)
    _, c_low = cm.tradeable_net(gross, low)
    _, c_high = cm.tradeable_net(gross, high)
    assert (c_high > c_low).all()


def test_breakeven_positive_for_positive_gross():
    cm = CostModel(CostConfig(), rep_price=50.0)
    be = cm.breakeven_roundtrip_bps(0.01, 21.0)
    assert be > 0
