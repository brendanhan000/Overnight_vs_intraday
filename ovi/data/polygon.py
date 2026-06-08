"""Polygon.io grouped-daily loader with a local parquet cache.

One call per trading day returns OHLCV for the whole US tape. To get BOTH raw
prices (for the intraday leg) and adjusted prices (for the close-to-close leg) we
make two calls per day:

  * ``adjusted=false`` -> raw open/high/low/close/volume/vwap
  * ``adjusted=true``  -> adjusted close (we keep only its close)

and merge on ticker into one row per (ticker, date) with columns:
``ticker, date, open, high, low, close, volume, vwap, adj_close``.

Caching: one parquet per date under ``{cache_dir}/grouped/{YYYY-MM-DD}.parquet``.
The file's existence means "already pulled" — including market-closed days, which
are cached as an empty frame so they are never re-pulled.

NOTE / known limitations (surfaced honestly):
  * Grouped-daily has **no shares outstanding**, so true market-cap weighting
    needs a separate source (see :meth:`PolygonClient.get_shares_outstanding`,
    which is a current-snapshot stub).
  * Polygon's ``adjusted=true`` adjusts for splits; dividend adjustment coverage
    is provider-dependent. Verify against the Layer-0 split before trusting it.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Column schema of a merged grouped-daily frame.
MERGED_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "adj_close",
]


def parse_grouped(results: Optional[List[dict]], date: pd.Timestamp) -> pd.DataFrame:
    """Parse a grouped-daily ``results`` list into a tidy frame (raw fields)."""
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "vwap"]
    if not results:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            "T": "ticker",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
        }
    )
    df["date"] = pd.Timestamp(date).normalize()
    for c in ("open", "high", "low", "close", "volume", "vwap"):
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def merge_raw_adjusted(raw: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Merge raw OHLCV with the adjusted close (keyed on ticker)."""
    if raw.empty and adj.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS)
    adj_close = adj[["ticker", "close"]].rename(columns={"close": "adj_close"})
    merged = raw.merge(adj_close, on="ticker", how="outer")
    # If date got dropped for outer-only rows, backfill from whichever frame has it.
    if merged["date"].isna().any():
        fill = pd.concat([raw[["ticker", "date"]], adj[["ticker", "date"]]])
        fill = fill.dropna().drop_duplicates("ticker").set_index("ticker")["date"]
        merged["date"] = merged["date"].fillna(merged["ticker"].map(fill))
    return merged[MERGED_COLUMNS]


class PolygonClient:
    """Thin grouped-daily client with throttling, retries, and a parquet cache."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "cache",
        rate_limit_per_min: float = 100.0,
        max_retries: int = 5,
        base_url: str = "https://api.polygon.io",
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY is required to pull data.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.min_interval = 60.0 / rate_limit_per_min if rate_limit_per_min > 0 else 0.0
        self.grouped_dir = Path(cache_dir) / "grouped"
        self.grouped_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self._last_call = 0.0

    # --- low-level HTTP ----------------------------------------------------
    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        params = {**params, "apiKey": self.api_key}
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=60)
                self._last_call = time.monotonic()
            except requests.RequestException as exc:  # network blip -> retry
                log.warning("request error (%s/%s): %s", attempt, self.max_retries, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                log.warning(
                    "HTTP %s (%s/%s); sleeping %.1fs",
                    resp.status_code,
                    attempt,
                    self.max_retries,
                    retry_after,
                )
                time.sleep(retry_after)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"GET {path} failed after {self.max_retries} retries")

    # --- grouped daily -----------------------------------------------------
    def _fetch_grouped(self, date: pd.Timestamp, adjusted: bool) -> pd.DataFrame:
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{ds}"
        payload = self._get(path, {"adjusted": str(adjusted).lower()})
        return parse_grouped(payload.get("results"), date)

    def get_grouped_daily(
        self, date: pd.Timestamp, use_cache: bool = True
    ) -> pd.DataFrame:
        """Merged raw+adjusted grouped daily for one date, cached to parquet.

        Returns an empty frame on market-closed days (also cached, so the closed
        day is never re-pulled).
        """
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        fp = self.grouped_dir / f"{ds}.parquet"
        if use_cache and fp.exists():
            return pd.read_parquet(fp)

        raw = self._fetch_grouped(date, adjusted=False)
        adj = self._fetch_grouped(date, adjusted=True)
        merged = merge_raw_adjusted(raw, adj)
        merged.to_parquet(fp, index=False)  # empty frame on holidays -> cached
        return merged

    def load_range(
        self,
        start: str,
        end: Optional[str] = None,
        use_cache: bool = True,
        progress_every: int = 50,
    ) -> pd.DataFrame:
        """Load (pulling+caching as needed) all trading days in [start, end].

        Weekends are skipped a priori; holidays come back empty from the API and
        are cached as empties. Returns the concatenated tidy panel.
        """
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        days = pd.bdate_range(start=start, end=end)
        frames: List[pd.DataFrame] = []
        for i, day in enumerate(days, 1):
            df = self.get_grouped_daily(day, use_cache=use_cache)
            if not df.empty:
                frames.append(df)
            if progress_every and i % progress_every == 0:
                log.info("loaded %s/%s trading days (last=%s)", i, len(days), day.date())
        if not frames:
            return pd.DataFrame(columns=MERGED_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        return out

    # --- shares outstanding (snapshot stub, not point-in-time) -------------
    def get_shares_outstanding(self, tickers: List[str]) -> Dict[str, float]:
        """CURRENT-snapshot shares outstanding via /v3/reference/tickers.

        WARNING: this is a present-day snapshot, not point-in-time. Using it for
        historical market cap injects look-ahead/restatement bias into the cross
        section; it is adequate only as a rough cap proxy for value-weighting and
        must be flagged. Wire in a point-in-time source for production work.
        """
        out: Dict[str, float] = {}
        for t in tickers:
            payload = self._get(f"/v3/reference/tickers/{t}", {})
            res = payload.get("results", {}) or {}
            shares = res.get("weighted_shares_outstanding") or res.get(
                "share_class_shares_outstanding"
            )
            if shares:
                out[t] = float(shares)
        return out

    # --- open-price method hook -------------------------------------------
    def get_first_30min_vwap(self, date: pd.Timestamp) -> pd.DataFrame:
        """Drop-in hook for the paper's 9:30-10:00 VWAP open (open_price_method).

        Not implemented here because it requires per-ticker minute aggregates
        (``/v2/aggs/ticker/{t}/range/1/minute/...``), which is plan-dependent and
        far more API-intensive than grouped daily. Implement by pulling 09:30-10:00
        minute bars and computing sum(price*vol)/sum(vol), then substitute the
        result for ``open`` before :func:`compute_daily_legs`.
        """
        raise NotImplementedError(
            "first_30min_vwap requires minute aggregates; use open_price_method="
            "'official_open' (default) until the minute-bar pipeline is wired in."
        )
