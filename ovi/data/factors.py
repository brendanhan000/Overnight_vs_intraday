"""Fama-French factors from the Ken French Data Library (free, no API key).

Primary path uses ``pandas_datareader`` (the spec's choice); if that's missing or
fails, we fall back to downloading the public CSV zip from Dartmouth directly with
``requests`` (verified reachable). Either way returns a monthly frame indexed by
``Period[M]`` with columns ``Mkt-RF, SMB, HML, RF`` in **decimals** (Ken French
publishes percent; we divide by 100). Cached to parquet.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

FF3_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RF"]
_UA = {"User-Agent": "Mozilla/5.0 (ovi research backtest)"}


def parse_ff_factors_csv(text: str) -> pd.DataFrame:
    """Parse the monthly block of a Ken French factors CSV (percent -> decimal).

    The file has a preamble, a header row, monthly rows ``YYYYMM,Mkt-RF,SMB,HML,RF``,
    then a blank line and an annual section. We keep only the 6-digit-date monthly
    rows.
    """
    rows = []
    for line in text.splitlines():
        if re.match(r"^\s*\d{6}\s*,", line):
            parts = [p.strip() for p in line.split(",")]
            ym, vals = parts[0], parts[1:5]
            try:
                rows.append([int(ym)] + [float(v) for v in vals])
            except ValueError:
                continue
    if not rows:
        raise ValueError("no monthly factor rows parsed from CSV")
    df = pd.DataFrame(rows, columns=["yyyymm"] + FACTOR_COLS)
    df["month"] = pd.PeriodIndex(df["yyyymm"].astype(str), freq="M")
    df = df.set_index("month").drop(columns="yyyymm")
    return df[FACTOR_COLS] / 100.0  # percent -> decimal


def _download_ff_zip() -> pd.DataFrame:
    resp = requests.get(FF3_MONTHLY_URL, headers=_UA, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("latin-1")
    return parse_ff_factors_csv(text)


def _via_datareader(start: str, end: Optional[str]) -> pd.DataFrame:
    import pandas_datareader.data as web  # optional dependency

    raw = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)[0]
    raw = raw / 100.0  # percent -> decimal
    raw.columns = [c.strip() for c in raw.columns]
    raw.index = raw.index.asfreq("M") if hasattr(raw.index, "asfreq") else raw.index
    return raw[FACTOR_COLS]


def load_ff_factors(
    start: str = "1990-01-01",
    end: Optional[str] = None,
    cache_dir: str = "cache",
    prefer_datareader: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Monthly FF3 + RF as decimals, indexed by Period[M], sliced to [start, end]."""
    cache = Path(cache_dir) / "factors"
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / "ff3_monthly.parquet"

    df: Optional[pd.DataFrame] = None
    if fp.exists() and not force_refresh:
        df = pd.read_parquet(fp)
        df.index = pd.PeriodIndex(df.index, freq="M")

    need_refresh = df is None
    if df is not None:
        want_start = pd.Period(start, "M")
        want_end = pd.Period(end, "M") if end else df.index.max()
        if df.index.min() > want_start or df.index.max() < want_end:
            need_refresh = True

    if need_refresh:
        err = None
        if prefer_datareader:
            try:
                df = _via_datareader(start, end)
            except Exception as exc:  # missing package or network/format issue
                err = exc
        if df is None:
            try:
                df = _download_ff_zip()
            except Exception as exc:
                raise RuntimeError(
                    f"could not load FF factors (datareader err={err}; download err={exc})"
                )
        df.to_parquet(fp)

    out = df.copy()
    out = out[out.index >= pd.Period(start, "M")]
    if end:
        out = out[out.index <= pd.Period(end, "M")]
    return out
