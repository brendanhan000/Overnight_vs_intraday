"""Walk-forward sample splitting: in-sample / out-of-sample / lockbox.

The LOCKBOX is held untouched until the very end and reported once, so OOS choices
can't leak into it.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

SEGMENTS = ("in_sample", "out_of_sample", "lockbox")


def make_segments(insample_end: str, lockbox_start: str) -> Dict[str, "pd.Period"]:
    """Boundary periods (monthly) for the three walk-forward segments."""
    return {
        "insample_end": pd.Period(insample_end, "M"),
        "lockbox_start": pd.Period(lockbox_start, "M"),
    }


def segment_of(month: "pd.Period", bounds: Dict[str, "pd.Period"]) -> str:
    if month <= bounds["insample_end"]:
        return "in_sample"
    if month < bounds["lockbox_start"]:
        return "out_of_sample"
    return "lockbox"


def split_series(series: pd.Series, insample_end: str, lockbox_start: str) -> Dict[str, pd.Series]:
    """Split a monthly (Period[M]-indexed) series into the three segments."""
    bounds = make_segments(insample_end, lockbox_start)
    idx = pd.PeriodIndex(series.index, freq="M")
    s = series.copy()
    s.index = idx
    out: Dict[str, pd.Series] = {}
    out["in_sample"] = s[idx <= bounds["insample_end"]]
    out["out_of_sample"] = s[(idx > bounds["insample_end"]) & (idx < bounds["lockbox_start"])]
    out["lockbox"] = s[idx >= bounds["lockbox_start"]]
    return out
