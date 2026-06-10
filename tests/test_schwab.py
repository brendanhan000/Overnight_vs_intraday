"""Network-free tests for the Schwab loader: candle parsing, the dividend
add-back math, and the credentials guardrail."""
from __future__ import annotations

import pandas as pd
import pytest

from ovi.data.schwab import SchwabLoader, parse_candles, total_return_adj_close


def _ms(date: str) -> int:
    return pd.Timestamp(date, tz="UTC").value // 1_000_000


def test_parse_candles_schema_and_dates():
    payload = {
        "symbol": "AAPL", "empty": False,
        "candles": [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1000, "datetime": _ms("2020-01-02")},
            {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5,
             "volume": 1200, "datetime": _ms("2020-01-03")},
        ],
    }
    df = parse_candles(payload, "AAPL")
    assert list(df.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert (df["ticker"] == "AAPL").all()
    assert df["date"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert df["close"].tolist() == [100.5, 101.5]


def test_parse_candles_empty():
    assert parse_candles({"symbol": "X", "empty": True, "candles": []}, "X").empty
    assert parse_candles({}, "X").empty


def test_total_return_adj_close_puts_dividend_in_return():
    # $1 dividend ex-date 01-03: price falls 100->99 but total return is flat.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "close": [100.0, 99.0, 101.0],
        "div": [0.0, 1.0, 0.0],
    })
    adj = total_return_adj_close(df)
    assert adj.iloc[0] == pytest.approx(100.0)
    # (99 + 1)/100 = 1.0  -> total return flat across the ex-date
    assert adj.iloc[1] == pytest.approx(100.0)
    # then (101 + 0)/99
    assert adj.iloc[1] / adj.iloc[0] == pytest.approx(1.0)
    assert adj.iloc[2] / adj.iloc[1] == pytest.approx(101.0 / 99.0)


def test_schwab_loader_requires_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHWAB_APP_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_APP_SECRET", raising=False)
    loader = SchwabLoader(cache_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="SCHWAB_APP_KEY"):
        loader._client()
