"""Network-free tests for the yfinance tidy/reshape logic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ovi.data.yahoo import TIDY_COLUMNS, tidy_yf_download


def _fake_multiindex_download(tickers, n=5):
    """Mimic yf.download(auto_adjust=False) multi-ticker output."""
    dates = pd.bdate_range("2022-01-03", periods=n)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    rng = np.random.default_rng(0)
    data = rng.uniform(10, 100, size=(n, len(cols)))
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date"), columns=cols)


def test_tidy_multiindex_schema_and_tickers():
    raw = _fake_multiindex_download(["AAA", "BBB"])
    tidy = tidy_yf_download(raw, ["AAA", "BBB"])
    assert list(tidy.columns) == TIDY_COLUMNS
    assert set(tidy["ticker"]) == {"AAA", "BBB"}
    assert len(tidy) == 5 * 2
    assert tidy["adj_close"].notna().all()
    assert str(tidy["date"].dtype).startswith("datetime64")


def test_tidy_preserves_raw_vs_adjusted_close():
    raw = _fake_multiindex_download(["AAA"])
    # Force a known split: raw Close = 2x Adj Close for AAA.
    raw[("Close", "AAA")] = 100.0
    raw[("Adj Close", "AAA")] = 50.0
    raw[("Open", "AAA")] = 99.0
    tidy = tidy_yf_download(raw, ["AAA"])
    row = tidy.iloc[0]
    assert row["close"] == 100.0
    assert row["adj_close"] == 50.0
    assert row["open"] == 99.0


def test_tidy_flat_single_ticker():
    dates = pd.bdate_range("2022-01-03", periods=4)
    flat = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13],
            "High": [10, 11, 12, 13],
            "Low": [10, 11, 12, 13],
            "Close": [10.5, 11.5, 12.5, 13.5],
            "Adj Close": [10.0, 11.0, 12.0, 13.0],
            "Volume": [100, 200, 300, 400],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    tidy = tidy_yf_download(flat, ["ZZZ"])
    assert list(tidy.columns) == TIDY_COLUMNS
    assert (tidy["ticker"] == "ZZZ").all()
    assert tidy["close"].tolist() == [10.5, 11.5, 12.5, 13.5]


def test_tidy_empty():
    assert tidy_yf_download(pd.DataFrame(), ["AAA"]).empty
