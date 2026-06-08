"""Tests for portfolio formation — especially NO LOOK-AHEAD in the signal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ovi.config import Config
from ovi.core.portfolios import (
    MonthlyPanel,
    _assign_deciles,
    book_turnover,
    form_long_short,
)


def test_assign_deciles_even_bins():
    s = pd.Series(range(100))
    dec = _assign_deciles(s, 10)
    vc = dec.value_counts()
    assert set(vc.index) == set(range(1, 11))
    assert (vc == 10).all()


def _toy_panel():
    months = pd.period_range("2020-01", periods=3, freq="M")
    data = {"HI": [0.10, -0.10, 0.10], "LO": [-0.10, 0.10, -0.10]}
    rows = []
    for tkr, vals in data.items():
        for mo, v in zip(months, vals):
            rows.append(dict(ticker=tkr, month=mo, overnight_m=v, intraday_m=0.0,
                             cc_m=v, n_days=21, lag_weight=1.0, lag_price=50.0,
                             eligible=True))
    panel = pd.DataFrame(rows)
    return MonthlyPanel(panel, 0, 2, pd.PeriodIndex(months))


def test_signal_is_lagged_not_contemporaneous():
    # HI/LO overnight legs alternate each month. With a 1-month LAGGED signal the
    # long-short must be NEGATIVE (-0.20); a contemporaneous (look-ahead) signal
    # would flip it to +0.20. This pins down no-look-ahead.
    cfg = Config()
    cfg.portfolio.n_deciles = 2
    cfg.portfolio.long_decile = 2
    cfg.portfolio.short_decile = 1
    res = form_long_short(_toy_panel(), "overnight_m", cfg, lag=1)
    ls = res["ls"]["overnight_m"]
    assert res["n_months"] == 2  # first month dropped (signal NaN)
    assert (ls < 0).all()
    assert abs(ls.mean() - (-0.20)) < 1e-12


def test_book_turnover_full_on_init_zero_when_unchanged():
    m = pd.period_range("2020-01", periods=3, freq="M")
    w = {mm: pd.Series({"A": 0.5, "B": 0.5}) for mm in m}
    t = book_turnover(w)
    assert t.iloc[0] == 1.0       # initial build
    assert t.iloc[1] == 0.0       # unchanged
    assert t.iloc[2] == 0.0


def test_form_long_short_runs_on_synthetic_universe():
    from ovi.core.synthetic import make_synthetic_panel
    from ovi.core.portfolios import build_monthly_panel

    daily = make_synthetic_panel(n_tickers=60, n_days=300, seed=9)
    mp = build_monthly_panel(daily, Config())
    res = form_long_short(mp, "overnight_m", Config(), lag=1)
    assert res["n_months"] > 5
    assert set(res["ls"].columns) == {"overnight_m", "intraday_m", "cc_m"}
