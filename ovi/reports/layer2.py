"""Layer 2 reporting: TugOfWar timing table, the tug-of-war cumulative plot,
and the overnight-intraday spread feature summary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..backtest.tugofwar import Layer2Result

PCT = 100.0
LEG_NAME = {"overnight_m": "overnight", "intraday_m": "intraday", "cc_m": "close-close"}


def _legmeans_row(leg_means: Dict[str, dict], leg: str) -> str:
    d = leg_means[leg]
    return f"{d['mean']*PCT:7.3f} (t={d['tstat']:5.2f})"


def format_report(res: Layer2Result) -> str:
    m = res.meta
    L = []
    L.append("=" * 78)
    L.append("LAYER 2 — TugOfWar timing + overnight-intraday spread feature")
    L.append("=" * 78)
    L.append(f"Sample: {m['date_min']} -> {m['date_max']}  ({m['n_months']} months, "
             f"{m['n_tickers']} tickers)")
    L.append(f"EWMA half-life: {m['halflife']:.0f} months | skip recent: "
             f"{m['skip_recent_months']} | NW lags: {m['nw_lags']}")
    if m["short_sample_warning"]:
        L.append("  !! SHORT-SAMPLE WARNING: sample < 3x the EWMA half-life. The 60-month")
        L.append("  !! timing signal barely varies here and the regression is UNDERPOWERED;")
        L.append("  !! treat coefficients as illustrative. Re-run with --start 2004-01-01.")
    L.append("")

    L.append("(a) ANOMALY FACTOR LEG MEANS (%/mo) — the tug of war")
    L.append(f"    {'factor':<20}{'overnight':>18}{'intraday':>18}{'close-close':>18}")
    for name, f in res.factors.items():
        lm = f["leg_means"]
        L.append(f"    {name:<20}{_legmeans_row(lm,'overnight_m'):>18}"
                 f"{_legmeans_row(lm,'intraday_m'):>18}{_legmeans_row(lm,'cc_m'):>18}")
    L.append("    (momentum should earn overnight & bleed intraday; STR the reverse)")
    L.append("")

    L.append("(a) TugOfWar PREDICTIVE REGRESSION:  cc_{t+1} = a + b * ToW_t")
    L.append("    ToW_t = EWMA(overnight) - EWMA(intraday), skip most recent month. Expect b > 0.")
    L.append(f"    {'factor':<20}{'b (coef)':>12}{'NW t':>9}{'R^2':>8}{'n':>6}{'  b>0?':>8}")
    for name, f in res.factors.items():
        r = f["reg"]
        ok = "yes" if (r["coef"] > 0 and r["tstat"] > 1.0) else "no"
        L.append(f"    {name:<20}{r['coef']:12.3f}{r['tstat']:9.2f}{r['r2']:8.3f}{r['n']:6d}{ok:>8}")
    pr = res.pooled_reg
    pok = "yes" if (pr["coef"] > 0 and pr["tstat"] > 1.0) else "no"
    L.append(f"    {'POOLED':<20}{pr['coef']:12.3f}{pr['tstat']:9.2f}{pr['r2']:8.3f}{pr['n']:6d}{pok:>8}")
    L.append("")

    idx = res.spread_index
    L.append("(b) OVERNIGHT - INTRADAY SPREAD FEATURE (value-weighted index, %/mo)")
    L.append(f"    overnight mean {idx['overnight'].mean()*PCT:6.3f} | intraday "
             f"{idx['intraday'].mean()*PCT:6.3f} | spread {idx['spread'].mean()*PCT:6.3f} "
             f"(std {idx['spread'].std()*PCT:5.2f})")
    L.append(f"    per-name series : {res.spread_paths['per_name']}")
    L.append(f"    index series    : {res.spread_paths['index']}")
    L.append("")

    L.append("VERDICT")
    for name, f in res.factors.items():
        rep = res.reproduced[name]
        L.append(f"  [{'PASS' if rep else 'FAIL'}] {name}: TugOfWar predicts next-month cc with b>0 (t>1)")
    L.append(f"  [{'PASS' if res.reproduced['pooled_positive'] else 'FAIL'}] pooled: positive TugOfWar coefficient")
    L.append("  Caveats: dollar-volume VW proxy, survivorship-biased universe, and (above all)")
    L.append("  a short sample for a 60-month EWMA. Signs are the deliverable, not magnitudes.")
    L.append("=" * 78)
    return "\n".join(L)


def plot_tugofwar(res: Layer2Result, path: str) -> None:
    """Cumulative overnight vs intraday vs cc for each factor (the tug of war)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res.factors.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        legs = res.factors[name]["legs"]
        x = legs.index.to_timestamp() if hasattr(legs.index, "to_timestamp") else legs.index
        for leg, lab in (("overnight_m", "overnight"), ("intraday_m", "intraday"), ("cc_m", "close-close")):
            cum = (1 + legs[leg].fillna(0)).cumprod() - 1
            ax.plot(x, cum * 100, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}: cumulative leg returns (the tug of war)")
        ax.set_ylabel("cumulative return (%)")
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("date")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(res: Layer2Result, out_dir: str = "reports/output") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt = out / "layer2_report.md"
    png = out / "layer2_tugofwar.png"
    txt.write_text("```text\n" + format_report(res) + "\n```\n")
    plot_tugofwar(res, str(png))
    return {"report": str(txt), "fig": str(png)}
