"""Network-free tests for the Polygon parsing/merge helpers."""
from __future__ import annotations

import pandas as pd

from ovi.data.polygon import MERGED_COLUMNS, merge_raw_adjusted, parse_grouped

RAW_RESULTS = [
    {"T": "AAA", "o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 1000, "vw": 10.2},
    {"T": "BBB", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 2000, "vw": 50.1},
]
# Adjusted call: same tickers, adjusted closes differ where corp actions exist.
ADJ_RESULTS = [
    {"T": "AAA", "o": 5.0, "h": 5.5, "l": 4.75, "c": 5.25, "v": 1000, "vw": 5.1},
    {"T": "BBB", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 2000, "vw": 50.1},
]


def test_parse_grouped_columns_and_date():
    date = pd.Timestamp("2020-03-02")
    df = parse_grouped(RAW_RESULTS, date)
    assert list(df.columns) == [
        "ticker", "date", "open", "high", "low", "close", "volume", "vwap",
    ]
    assert (df["date"] == date).all()
    assert df.loc[df["ticker"] == "AAA", "close"].iloc[0] == 10.5


def test_parse_grouped_empty_is_holiday():
    df = parse_grouped(None, pd.Timestamp("2020-12-25"))
    assert df.empty
    assert "ticker" in df.columns  # schema preserved


def test_merge_keeps_raw_ohlc_and_adjusted_close():
    date = pd.Timestamp("2020-03-02")
    raw = parse_grouped(RAW_RESULTS, date)
    adj = parse_grouped(ADJ_RESULTS, date)
    merged = merge_raw_adjusted(raw, adj)
    assert list(merged.columns) == MERGED_COLUMNS
    aaa = merged[merged["ticker"] == "AAA"].iloc[0]
    # RAW open/close preserved; adj_close comes from the adjusted call.
    assert aaa["open"] == 10.0
    assert aaa["close"] == 10.5
    assert aaa["adj_close"] == 5.25
    # The implied intraday leg is scale-free, identical raw vs adjusted:
    assert abs((10.5 / 10.0) - (5.25 / 5.0)) < 1e-12
