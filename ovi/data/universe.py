"""Backtest universes for the yfinance source.

Because yfinance needs an explicit ticker list (unlike Polygon's whole-tape
grouped endpoint), we provide:
  * ``DEMO_TICKERS`` — ~40 liquid, long-lived large caps for fast smoke tests.
  * ``sp500_from_wikipedia()`` — current S&P 500 constituents (needs lxml).

BOTH are **current-listed** sets, so they carry survivorship bias (and, for long
backtests on today's S&P 500, look-ahead "index inclusion" bias). Flag it loudly
in any report built on them.
"""
from __future__ import annotations

from typing import List

# ~40 liquid mega/large caps across sectors; all long-lived (mitigates, but does
# not remove, survivorship bias).
DEMO_TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "C",
    "XOM", "CVX", "COP",
    "JNJ", "PFE", "MRK", "UNH", "ABT", "TMO",
    "HD", "WMT", "COST", "MCD", "NKE", "DIS",
    "PG", "KO", "PEP",
    "INTC", "CSCO", "ORCL", "IBM", "CRM", "ADBE", "QCOM",
    "T", "VZ", "CMCSA",
    "CAT", "GE", "BA", "HON", "MMM",
]


def sp500_from_wikipedia() -> List[str]:
    """Current S&P 500 tickers scraped from Wikipedia (requires lxml).

    Fetched via ``requests`` with a browser User-Agent because Wikipedia returns
    403 to pandas' default urllib agent. Yahoo uses '-' for share-class suffixes
    (e.g. BRK-B, not BRK.B).
    """
    import io

    import pandas as pd
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 (ovi research backtest)"}, timeout=30
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    syms = (
        df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    )
    return sorted(set(syms.tolist()))


def get_universe(name: str = "demo") -> List[str]:
    name = (name or "demo").lower()
    if name == "demo":
        return list(DEMO_TICKERS)
    if name == "sp500":
        return sp500_from_wikipedia()
    raise ValueError(f"unknown universe {name!r} (use 'demo' or 'sp500')")
