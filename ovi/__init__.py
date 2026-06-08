"""ovi — Overnight vs. Intraday return decomposition & backtest.

Research package replicating Lou, Polk & Skouras (2019), "A Tug of War",
Journal of Financial Economics 134:192-213.

Package layout (maps to the project spec's module list):
    ovi.data      -- Polygon loader + parquet cache  (spec: "data/")
    ovi.core      -- decomposition, portfolios, stats (spec: "core/")
    ovi.backtest  -- engine, costs, walk-forward      (spec: "backtest/")
    ovi.reports   -- tables + plots                   (spec: "reports/")
The parquet cache lives in a configurable directory (default ``cache/`` at the
repo root, git-ignored), kept separate from code for cleanliness.
"""

__version__ = "0.1.0"
