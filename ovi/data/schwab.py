"""Schwab Trader (Market Data) API loader — real, tradeable price data.

Uses the ``schwab-py`` wrapper (https://schwab-py.readthedocs.io) for the OAuth 2.0
flow and token refresh, and the per-symbol daily price-history endpoint. Output is
the SAME tidy schema as the yfinance/Polygon loaders, so all downstream code is
source-agnostic:

    ticker, date, open, high, low, close, volume, adj_close

IMPORTANT — price adjustment (verified against Schwab docs):
  Schwab price history is **SPLIT-adjusted but NOT dividend-adjusted**. The
  decomposition needs a split+dividend-adjusted close for the close-to-close leg
  (dividends are assumed to occur overnight, so they belong in the overnight leg).
  Therefore:
    * intraday_ret = close/open is CORRECT as-is (within-day, split cancels).
    * adj_close defaults to the split-adjusted close (``dividend_adjust=False``),
      which OMITS the dividend add-back -> the overnight leg is understated by the
      dividend yield (~0.15%/mo for the S&P 500). This is flagged loudly.
    * With ``dividend_adjust=True`` we add dividends back on their ex-dates to build
      a true total-return adj_close (dividends from a corporate-actions source;
      yfinance by default), so dividends land in the overnight leg exactly per the
      paper's convention.

AUTH (one-time, run by YOU locally — see scripts/schwab_setup.py):
  Register an app on developer.schwab.com with the **Market Data Production**
  product, set the callback to https://127.0.0.1:8182, then set env vars
  SCHWAB_APP_KEY / SCHWAB_APP_SECRET and run the setup script to create the token
  file. Refresh tokens expire after 7 days, so re-run weekly.

CAVEAT: the candle JSON structure (open/high/low/close/volume/datetime epoch-ms)
follows the TD Ameritrade heritage; verify it against a live response with
scripts/schwab_setup.py before trusting a long pull.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

TIDY_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume", "adj_close",
]
DEFAULT_CALLBACK = "https://127.0.0.1:8182"
DEFAULT_TOKEN_PATH = ".schwab_token.json"


def parse_candles(payload: dict, symbol: str) -> pd.DataFrame:
    """Parse a Schwab price-history JSON payload into raw OHLCV rows (pure)."""
    if not payload or payload.get("empty") or not payload.get("candles"):
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(payload["candles"])
    # epoch ms -> calendar date (daily candles are stamped at the session start).
    df["date"] = pd.to_datetime(df["datetime"], unit="ms").dt.normalize()
    df["ticker"] = payload.get("symbol", symbol)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = pd.NA
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def total_return_adj_close(df: pd.DataFrame, div_col: str = "div") -> pd.Series:
    """Build a dividend+split-adjusted close from split-adjusted close + dividends.

    adj_close_t / adj_close_{t-1} = (close_t + D_t) / close_{t-1}, so the ratios are
    total returns (dividends included) while the level is an arbitrary index. Only
    ratios are used downstream, so the level is irrelevant.
    """
    d = df.sort_values("date")
    prev = d["close"].shift(1)
    g = (d["close"] + d[div_col].fillna(0.0)) / prev
    g.iloc[0] = 1.0
    return (d["close"].iloc[0] * g.cumprod()).reindex(df.index)


class SchwabLoader:
    """Cached Schwab daily price loader producing the tidy OHLC+adj_close panel."""

    def __init__(
        self,
        cache_dir: str = "cache",
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        token_path: Optional[str] = None,
        rate_limit_per_min: float = 110.0,  # Schwab cap is 120/min; stay under
        dividend_adjust: bool = False,
    ) -> None:
        self.dir = Path(cache_dir) / "schwab"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.app_key = app_key or os.environ.get("SCHWAB_APP_KEY")
        self.app_secret = app_secret or os.environ.get("SCHWAB_APP_SECRET")
        self.callback_url = callback_url or os.environ.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK)
        self.token_path = token_path or os.environ.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH)
        self.min_interval = 60.0 / rate_limit_per_min if rate_limit_per_min > 0 else 0.0
        self.dividend_adjust = dividend_adjust
        self._client_obj = None
        self._last_call = 0.0
        self.missing: List[str] = []

    # --- auth / client -----------------------------------------------------
    def _client(self):
        if self._client_obj is not None:
            return self._client_obj
        if not (self.app_key and self.app_secret):
            raise RuntimeError(
                "SCHWAB_APP_KEY / SCHWAB_APP_SECRET not set. Register a Market Data "
                "app on developer.schwab.com and export the credentials."
            )
        if not Path(self.token_path).exists():
            raise RuntimeError(
                f"No Schwab token at {self.token_path!r}. Run a one-time login first:\n"
                f"    python3 scripts/schwab_setup.py\n"
                f"(refresh tokens expire after 7 days, so re-run weekly)."
            )
        try:
            from schwab.auth import client_from_token_file
        except ImportError as exc:
            raise RuntimeError(
                "schwab-py is required for --source schwab: pip install schwab-py"
            ) from exc
        self._client_obj = client_from_token_file(
            self.token_path, self.app_key, self.app_secret
        )
        return self._client_obj

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    # --- one symbol --------------------------------------------------------
    def _fetch(self, symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
        client = self._client()
        start_dt = pd.Timestamp(start).to_pydatetime()
        end_dt = (pd.Timestamp(end) if end else pd.Timestamp.today()).to_pydatetime()
        self._throttle()
        resp = client.get_price_history_every_day(
            symbol,
            start_datetime=start_dt,
            end_datetime=end_dt,
            need_extended_hours_data=False,  # regular-session open/close only
        )
        self._last_call = time.monotonic()
        resp.raise_for_status()
        return parse_candles(resp.json(), symbol)

    def _dividends(self, symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
        """Ex-date dividend amounts (split-adjusted) from yfinance, as a fallback
        corporate-actions source for the dividend add-back."""
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not available; cannot dividend-adjust %s", symbol)
            return pd.DataFrame(columns=["date", "div"])
        s = yf.Ticker(symbol).dividends
        if s is None or len(s) == 0:
            return pd.DataFrame(columns=["date", "div"])
        out = s.rename("div").reset_index()
        out.columns = ["date", "div"]
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        return out

    # --- many symbols, cached ---------------------------------------------
    def load(
        self,
        tickers: Sequence[str],
        start: str,
        end: Optional[str] = None,
        force_refresh: bool = False,
        dividend_adjust: Optional[bool] = None,
    ) -> pd.DataFrame:
        div_adj = self.dividend_adjust if dividend_adjust is None else dividend_adjust
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        tickers = list(dict.fromkeys(tickers))
        self.missing = []
        frames: List[pd.DataFrame] = []

        for t in tickers:
            fp = self.dir / f"{t.replace('/', '-')}.parquet"
            if not force_refresh and fp.exists():
                d = pd.read_parquet(fp)
                # Allow the cache to start up to a week after the requested start
                # (the requested start may be a weekend/holiday before the first
                # trading day), so we don't spuriously re-pull covered tickers.
                covered = (not d.empty
                           and d["date"].min() <= start_ts + pd.Timedelta(days=7)
                           and d["date"].max() >= end_ts - pd.Timedelta(days=7))
                if covered:
                    frames.append(d[(d["date"] >= start_ts) & (d["date"] <= end_ts)])
                    continue
            try:
                raw = self._fetch(t, start, end)
            except Exception as exc:  # noqa: BLE001 - surface per-ticker failures
                log.warning("schwab: %s failed (%s)", t, exc)
                self.missing.append(t)
                continue
            if raw.empty:
                self.missing.append(t)
                continue
            raw = raw.sort_values("date").reset_index(drop=True)
            if div_adj:
                divs = self._dividends(t, start, end)
                raw = raw.merge(divs, on="date", how="left")
                raw["adj_close"] = total_return_adj_close(raw)
            else:
                raw["adj_close"] = raw["close"]  # split-only (dividends NOT added)
            raw = raw[TIDY_COLUMNS]
            raw.to_parquet(fp, index=False)
            frames.append(raw[(raw["date"] >= start_ts) & (raw["date"] <= end_ts)])

        if self.missing:
            log.warning("schwab: %d/%d tickers returned no data: %s",
                        len(self.missing), len(tickers), ", ".join(self.missing[:15]))
        if not frames:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)
