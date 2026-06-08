"""Transaction-cost model for the three honesty versions.

The headline overnight result is GROSS component attribution. To turn it into a
real P&L you must choose how you'd actually trade it, and that choice dominates:

  V1 gross            : no costs. The monthly-rebalanced long-short's next-month
                        legs (Table 1). Component attribution, not P&L.
  V2 harvest-the-leg  : to ISOLATE the overnight leg you buy at close / sell at
                        open EVERY day across the whole book -> ~21 round trips a
                        month on 2 units of gross notional (long+short). Costs
                        scale with trading days and crush the edge.
  V3 tradeable-as-is  : monthly rebalance, hold through. Costs scale with monthly
                        turnover only (low). Report the held portfolio's realized
                        overnight+intraday decomposition -- a real P&L, but you did
                        NOT separately harvest the overnight leg.

All costs are in bps of traded notional. Per-share commission is converted to bps
via a representative book price.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import CostConfig


@dataclass
class CostModel:
    cfg: CostConfig
    rep_price: float = 50.0  # representative book price for per-share -> bps

    @property
    def commission_bps(self) -> float:
        if self.rep_price <= 0:
            return 0.0
        return self.cfg.commission_per_share / self.rep_price * 1e4

    @property
    def roundtrip_bps(self) -> float:
        """Same-day buy-at-close + sell-at-open round trip (the harvest trade)."""
        return (
            self.cfg.half_spread_bps_close
            + self.cfg.half_spread_bps_open
            + 2.0 * self.commission_bps
            + self.cfg.impact_coef_bps
        )

    @property
    def oneway_bps(self) -> float:
        """One-way rebalance trade at the close."""
        return self.cfg.half_spread_bps_close + self.commission_bps + self.cfg.impact_coef_bps

    def harvest_overnight_net(
        self,
        gross_overnight: pd.Series,
        n_days_by_month: pd.Series,
        gross_units: float = 2.0,
    ):
        """Net the daily-harvest overnight L-S: cost = days * gross_units * rt_bps."""
        days = n_days_by_month.reindex(gross_overnight.index).fillna(0.0)
        cost = days * gross_units * self.roundtrip_bps / 1e4
        return gross_overnight - cost, cost

    def tradeable_net(self, gross_cc: pd.Series, turnover: pd.Series):
        """Net the monthly-rebalanced hold-through L-S: cost = turnover * rt_bps."""
        t = turnover.reindex(gross_cc.index).fillna(0.0)
        cost = t * self.roundtrip_bps / 1e4
        return gross_cc - cost, cost

    def breakeven_roundtrip_bps(
        self, gross_overnight_mean: float, avg_n_days: float, gross_units: float = 2.0
    ) -> float:
        """Round-trip cost (bps) at which the harvested overnight edge hits zero."""
        denom = avg_n_days * gross_units / 1e4
        return gross_overnight_mean / denom if denom > 0 else float("nan")
