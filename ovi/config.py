"""Single-source configuration (dataclasses + YAML).

Load with :func:`load_config` (falls back to built-in defaults if the file is
absent). All knobs the spec calls out live here: dates, filters, costs,
open_price_method, EWMA half-life, and lags.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

OPEN_PRICE_METHODS = ("official_open", "first_30min_vwap")
WEIGHTINGS = ("value", "equal", "dollar_volume")
BREAKPOINT_SOURCES = ("universe_pct", "nyse")


@dataclass
class DataConfig:
    start_date: str = "2004-01-01"
    end_date: Optional[str] = None  # None -> today
    cache_dir: str = "cache"
    # Paper uses a 9:30-10:00 VWAP for the open; official_open is the default and
    # the VWAP path is a clean drop-in (see ovi/data/polygon.py).
    open_price_method: str = "official_open"
    rate_limit_per_min: float = 100.0  # free tier is ~5/min; paid is far higher
    max_retries: int = 5
    base_url: str = "https://api.polygon.io"

    def __post_init__(self) -> None:
        if self.open_price_method not in OPEN_PRICE_METHODS:
            raise ValueError(
                f"open_price_method must be one of {OPEN_PRICE_METHODS}, "
                f"got {self.open_price_method!r}"
            )


@dataclass
class FilterConfig:
    min_price: float = 5.0
    exclude_bottom_mktcap_quantile: float = 0.20  # drop bottom quintile by cap
    breakpoint_source: str = "universe_pct"  # 'nyse' needs exchange tags (TODO)
    bad_data_lower: float = -0.30
    bad_data_upper: float = 0.50

    def __post_init__(self) -> None:
        if self.breakpoint_source not in BREAKPOINT_SOURCES:
            raise ValueError(
                f"breakpoint_source must be one of {BREAKPOINT_SOURCES}"
            )


@dataclass
class CostConfig:
    """Honest cost knobs for the 'harvest-the-leg' net strategy."""

    commission_per_share: float = 0.005
    half_spread_bps_close: float = 5.0
    half_spread_bps_open: float = 10.0  # wider for the open window (spec)
    impact_coef_bps: float = 10.0  # turnover/impact penalty coefficient


@dataclass
class PortfolioConfig:
    n_deciles: int = 10
    weighting: str = "value"  # value | equal | dollar_volume
    long_decile: int = 10
    short_decile: int = 1

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")


@dataclass
class StrategyConfig:
    """Layer-2 TugOfWar timing."""

    ewma_half_life_months: float = 60.0
    skip_recent_months: int = 1


@dataclass
class BacktestConfig:
    newey_west_lags: int = 12
    fig2_max_lag_months: int = 60
    # Walk-forward: in-sample <= insample_end; out-of-sample after; lockbox is
    # held untouched until the very end and reported once.
    insample_end: str = "2013-12-31"
    lockbox_start: str = "2019-01-01"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # --- (de)serialization -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = d or {}
        return cls(
            data=DataConfig(**(d.get("data") or {})),
            filters=FilterConfig(**(d.get("filters") or {})),
            costs=CostConfig(**(d.get("costs") or {})),
            portfolio=PortfolioConfig(**(d.get("portfolio") or {})),
            strategy=StrategyConfig(**(d.get("strategy") or {})),
            backtest=BacktestConfig(**(d.get("backtest") or {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def load_config(path: Optional[str] = "config.yaml") -> Config:
    """Load config from YAML, or return defaults if the path is missing/None."""
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return Config.from_dict(raw)
    return Config()
