"""yfinance loader — backtest data source (drop-in alongside the Polygon loader).

yfinance with ``auto_adjust=False`` returns RAW ``Open/High/Low/Close`` plus a
split-AND-dividend-adjusted ``Adj Close`` — exactly the two price scales the
decomposition needs (raw for the intraday leg, adjusted for close-to-close). It is
produced in the same tidy schema as the Polygon loader so all downstream code is
source-agnostic:

    ticker, date, open, high, low, close, volume, adj_close

CAVEATS (surfaced honestly):
  * SURVIVORSHIP BIAS: Yahoo generally does not serve delisted tickers, so any
    universe pulled here is current-listed only. The loader reports how many of the
    requested tickers actually returned data so the bias is visible.
  * No shares outstanding in the price feed (market-cap weighting uses the
    dollar-volume proxy in Layer 0, same as Polygon).
  * ``auto_adjust=False`` is REQUIRED; newer yfinance defaults to True, which
    overwrites OHLC with adjusted values and drops ``Adj Close``.

Per-ticker parquet cache under ``{cache_dir}/yahoo/{TICKER}.parquet``; cached
ranges are merged so coverage accumulates and nothing is needlessly re-pulled.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

TIDY_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume", "adj_close",
]
_YF_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _drop_tz(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def tidy_yf_download(df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Reshape a ``yf.download`` frame into the tidy schema (pure; no network).

    Handles both the MultiIndex-columns layout (multiple tickers, fields on level
    0 and tickers on level 1) and the flat single-ticker layout.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    if isinstance(df.columns, pd.MultiIndex):
        stacked = df.stack(level=1, future_stack=True)
        stacked.index = stacked.index.set_names(["date", "ticker"])
        tidy = stacked.reset_index()
    else:
        tidy = df.reset_index()
        tidy = tidy.rename(columns={tidy.columns[0]: "date"})
        tidy["ticker"] = tickers[0] if len(tickers) else "UNKNOWN"

    tidy = tidy.rename(columns=_YF_RENAME)
    for c in ("open", "high", "low", "close", "volume", "adj_close"):
        if c not in tidy.columns:
            tidy[c] = pd.NA
    tidy["date"] = _drop_tz(tidy["date"])
    tidy = tidy[TIDY_COLUMNS]
    # Drop rows with no usable price and any empty ticker labels.
    tidy = tidy[tidy["ticker"].notna()]
    tidy = tidy.dropna(subset=["close", "adj_close"], how="all")
    return tidy.sort_values(["ticker", "date"]).reset_index(drop=True)


class YahooLoader:
    """Cached yfinance loader producing the tidy OHLC+adj_close panel."""

    def __init__(self, cache_dir: str = "cache") -> None:
        self.dir = Path(cache_dir) / "yahoo"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.missing: List[str] = []  # requested tickers that returned no data

    def _cache_path(self, ticker: str) -> Path:
        return self.dir / f"{ticker.replace('/', '-')}.parquet"

    def _download(self, tickers: Sequence[str], start: str, end: Optional[str]):
        import yfinance as yf

        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=False,  # REQUIRED: keep raw OHLC + Adj Close
            actions=False,
            progress=False,
            group_by="column",  # fields on level 0, tickers on level 1
            threads=True,
        )
        return tidy_yf_download(raw, list(tickers))

    def load(
        self,
        tickers: Sequence[str],
        start: str,
        end: Optional[str] = None,
        force_refresh: bool = False,
        batch_size: int = 120,
    ) -> pd.DataFrame:
        """Load [start, end] for ``tickers``, pulling+caching only what's missing."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order

        need: List[str] = []
        for t in tickers:
            fp = self._cache_path(t)
            if force_refresh or not fp.exists():
                need.append(t)
                continue
            d = pd.read_parquet(fp)
            covered = (
                not d.empty
                and d["date"].min() <= start_ts
                and d["date"].max() >= end_ts - pd.Timedelta(days=7)
            )
            if not covered:
                need.append(t)

        for i in range(0, len(need), batch_size):
            batch = need[i : i + batch_size]
            pulled = self._download(batch, start, end)
            for t, grp in pulled.groupby("ticker"):
                fp = self._cache_path(str(t))
                if fp.exists():
                    old = pd.read_parquet(fp)
                    grp = (
                        pd.concat([old, grp], ignore_index=True)
                        .drop_duplicates(subset=["ticker", "date"])
                        .sort_values("date")
                    )
                grp.to_parquet(fp, index=False)
            log.info("yahoo: pulled %d/%d needed tickers", min(i + batch_size, len(need)), len(need))

        frames: List[pd.DataFrame] = []
        self.missing = []
        for t in tickers:
            fp = self._cache_path(t)
            if not fp.exists():
                self.missing.append(t)
                continue
            d = pd.read_parquet(fp)
            d = d[(d["date"] >= start_ts) & (d["date"] <= end_ts)]
            if d.empty:
                self.missing.append(t)
            else:
                frames.append(d)
        if self.missing:
            log.warning(
                "yahoo: %d/%d tickers returned no data (delisted/typo): %s",
                len(self.missing), len(tickers), ", ".join(self.missing[:15]),
            )
        if not frames:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)
