"""Layer 1 reporting: Table 1 panels, the three cost versions, baselines,
walk-forward, the Fig. 2 plot, and a full markdown report with paper comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..backtest.engine import Layer1Result

PCT = 100.0
LEGS = ["overnight_m", "intraday_m", "cc_m"]
LEG_NAME = {"overnight_m": "overnight", "intraday_m": "intraday", "cc_m": "close-close"}


def _f(block: Dict, key: str, tkey: str) -> str:
    v = block.get(key, float("nan"))
    t = block.get(tkey, float("nan"))
    return f"{v * PCT:6.3f} (t={t:5.2f})"


def _decile_table(panel: Dict, n_deciles: int) -> str:
    dec = panel["deciles"].reindex(range(1, n_deciles + 1))
    lines = []
    head = "  " + f"{'leg':<11}" + "".join(f"{'D'+str(d):>8}" for d in range(1, n_deciles + 1)) + f"{'L-S':>9}"
    lines.append(head)
    for leg in LEGS:
        if leg not in dec.columns:
            continue
        row = f"  {LEG_NAME[leg]:<11}"
        for d in range(1, n_deciles + 1):
            val = dec.loc[d, leg] if d in dec.index else float("nan")
            row += f"{val * PCT:8.2f}"
        ls_mean = panel["blocks"][leg]["mean"] * PCT
        row += f"{ls_mean:9.2f}"
        lines.append(row)
    return "\n".join(lines)


def _panel_block(panel: Dict, title: str, n_deciles: int) -> str:
    L = [title, ""]
    L.append("  Next-month VW decile-average return (%/mo), and L-S = D10 - D1:")
    L.append(_decile_table(panel, n_deciles))
    L.append("")
    L.append("  Long-short (D10-D1) by leg (%/mo; Sharpe annualized):")
    L.append(f"    {'leg':<12}{'mean (NW t)':>18}{'CAPM alpha':>18}{'FF3 alpha':>18}{'skew':>8}{'Sharpe':>9}")
    for leg in LEGS:
        b = panel["blocks"][leg]
        L.append(
            f"    {LEG_NAME[leg]:<12}{_f(b,'mean','mean_t'):>18}"
            f"{_f(b,'capm_alpha','capm_alpha_t'):>18}"
            f"{_f(b,'ff3_alpha','ff3_alpha_t'):>18}{b['skew']:8.2f}{b['sharpe_ann']:9.2f}"
        )
    return "\n".join(L)


def _costs_block(res: Layer1Result) -> str:
    c = res.costs
    h, t = c["harvest"], c["tradeable"]
    L = ["COST HONESTY  (Panel A, overnight-sorted long-short)", ""]
    L.append(f"  representative price ${c['rep_price']:.2f} | round-trip {c['roundtrip_bps']:.1f} bps "
             f"| one-way {c['oneway_bps']:.1f} bps | commission {c['commission_bps']:.2f} bps")
    L.append(f"  avg trading days/month ~ {c['avg_trading_days']:.1f}")
    L.append("")
    L.append(f"  V1 GROSS overnight L-S        : {_f(h['gross'],'mean','tstat')} %/mo  "
             f"Sharpe {h['gross']['sharpe_ann']:5.2f}   [component attribution]")
    L.append(f"  V2 HARVEST-THE-LEG (daily)    : gross {h['gross']['mean']*PCT:6.3f} -> "
             f"net {_f(h['net'],'mean','tstat')} %/mo  Sharpe {h['net']['sharpe_ann']:5.2f}")
    L.append(f"       avg cost {h['monthly_cost_mean']*PCT:6.2f} %/mo;  break-even round-trip = "
             f"{h['breakeven_roundtrip_bps']:.3f} bps (vs assumed {c['roundtrip_bps']:.1f} bps)")
    L.append(f"  V3 TRADEABLE-AS-STATED (cc)   : gross {t['gross']['mean']*PCT:6.3f} -> "
             f"net {_f(t['net'],'mean','tstat')} %/mo  Sharpe {t['net']['sharpe_ann']:5.2f}")
    L.append(f"       turnover {t['turnover_mean']*PCT:5.1f} %/mo;  cost {t['monthly_cost_mean']*PCT:5.3f} %/mo")
    L.append("")
    L.append("  NOTE: V1 is COMPONENT ALPHA, not P&L. Isolating the overnight leg (V2) means")
    L.append("        trading the entire book daily at the open -> costs dominate the edge.")
    L.append("        V3 is a real, low-turnover P&L but does NOT separately harvest overnight.")
    return "\n".join(L)


def _baselines_block(res: Layer1Result) -> str:
    b = res.baselines
    if not b:
        return ""
    spy, rnd, coin = b["spy_overnight"], b["random_decile_ls"], b["coin_flip_ls"]
    strat = res.panelA["blocks"]["overnight_m"]
    L = ["DUMB-BASELINE BAKE-OFF  (strategy must beat these, risk-adjusted)", ""]
    L.append(f"    {'baseline':<26}{'mean %/mo':>12}{'NW t':>8}{'Sharpe':>9}")
    L.append(f"    {'SPY overnight buy-hold':<26}{spy['mean']*PCT:12.3f}{spy['mean_t']:8.2f}{spy['sharpe_ann']:9.2f}")
    L.append(f"    {'random-decile L-S':<26}{rnd['mean']*PCT:12.3f}{rnd['mean_t']:8.2f}{rnd['sharpe_ann']:9.2f}"
             f"   (frac|t|>2={rnd['frac_sig']:.2f})")
    L.append(f"    {'coin-flip L-S':<26}{coin['mean']*PCT:12.3f}{coin['mean_t']:8.2f}{coin['sharpe_ann']:9.2f}"
             f"   (frac|t|>2={coin['frac_sig']:.2f})")
    L.append(f"    {'>> STRATEGY (A overnight)':<26}{strat['mean']*PCT:12.3f}{strat['mean_t']:8.2f}{strat['sharpe_ann']:9.2f}")
    return "\n".join(L)


def _walkforward_block(res: Layer1Result) -> str:
    wf = res.walkforward
    L = ["WALK-FORWARD  (Panel A long-short, mean %/mo with NW t and n months)", ""]
    L.append(f"    {'leg':<12}{'in-sample':>20}{'out-of-sample':>20}{'LOCKBOX':>20}")
    for leg in LEGS:
        seg = wf[leg]
        def cell(s):
            return f"{s['mean']*PCT:6.2f} (t={s['tstat']:4.1f},n={s['n']})"
        L.append(f"    {LEG_NAME[leg]:<12}{cell(seg['in_sample']):>20}"
                 f"{cell(seg['out_of_sample']):>20}{cell(seg['lockbox']):>20}")
    L.append("    (lockbox held untouched until the end, reported once)")
    return "\n".join(L)


def _reproduced_block(res: Layer1Result) -> str:
    r = res.reproduced
    mark = lambda ok: "PASS" if ok else "FAIL"
    L = ["WHAT REPRODUCED / WHAT DIDN'T", ""]
    L.append(f"  [{mark(r['A_overnight_positive'])}] overnight-sorted -> POSITIVE overnight next month (persistence, t>2)")
    L.append(f"  [{mark(r['A_intraday_negative'])}] overnight-sorted -> NEGATIVE intraday  next month (reversal)")
    L.append(f"  [{mark(r['B_intraday_positive'])}] intraday-sorted  -> POSITIVE intraday  next month (persistence, t>2)")
    L.append(f"  [{mark(r['B_overnight_negative'])}] intraday-sorted  -> NEGATIVE overnight next month (reversal)")
    L.append("")
    L.append("  Caveats: VW uses a dollar-volume PROXY (no point-in-time shares); "
             "yfinance universe is")
    L.append("  current-listed (SURVIVORSHIP-biased); sample/period differ from the paper's CRSP set.")
    if res.meta.get("survivorship_note"):
        L.append(f"  {res.meta['survivorship_note']}")
    return "\n".join(L)


def format_report(res: Layer1Result) -> str:
    m = res.meta
    n_dec = res.panelA["deciles"].index.max()
    n_dec = int(n_dec) if pd.notna(n_dec) else 10
    L = []
    L.append("=" * 78)
    L.append("LAYER 1 — Persistence / reversal backtest (Table 1 replication)")
    L.append("=" * 78)
    L.append(f"Sample: {m['date_min']} -> {m['date_max']}  ({m['n_months']} months, "
             f"{m['n_tickers']} tickers); NW lags={m['nw_lags']}")
    L.append(f"Bad-data stock-days removed: {m['n_bad_data']:,} | weighting: {m['weighting']}")
    L.append("")
    L.append(_panel_block(res.panelA, "PANEL A — sort on LAGGED 1-month OVERNIGHT return", n_dec))
    L.append("")
    L.append(_panel_block(res.panelB, "PANEL B — sort on LAGGED 1-month INTRADAY return", n_dec))
    L.append("")
    L.append("Expected US signs: A -> +overnight / -intraday ; B -> +intraday / -overnight")
    L.append("")
    L.append(_costs_block(res))
    L.append("")
    bl = _baselines_block(res)
    if bl:
        L.append(bl)
        L.append("")
    L.append(_walkforward_block(res))
    L.append("")
    L.append(_reproduced_block(res))
    L.append("=" * 78)
    return "\n".join(L)


def plot_fig2(res: Layer1Result, path: str) -> None:
    """Four t-stat curves vs ranking-to-holding lag (the Fig. 2 analog)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = res.fig2
    if f.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f.index, f["A_overnight_t"], "-o", ms=3, label="overnight-sorted → overnight (persist)")
    ax.plot(f.index, f["A_intraday_t"], "-o", ms=3, label="overnight-sorted → intraday (reverse)")
    ax.plot(f.index, f["B_intraday_t"], "-s", ms=3, label="intraday-sorted → intraday (persist)")
    ax.plot(f.index, f["B_overnight_t"], "-s", ms=3, label="intraday-sorted → overnight (reverse)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(2, color="grey", lw=0.6, ls="--")
    ax.axhline(-2, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("ranking-to-holding lag (months)")
    ax.set_ylabel("Newey-West t-stat of long-short")
    ax.set_title("Fig. 2 analog — persistence/reversal t-stat vs lag")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(res: Layer1Result, out_dir: str = "reports/output") -> Dict[str, str]:
    """Write the text report and the Fig. 2 PNG; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt = out / "layer1_report.md"
    png = out / "layer1_fig2.png"
    body = "```text\n" + format_report(res) + "\n```\n"
    txt.write_text(body)
    plot_fig2(res, str(png))
    return {"report": str(txt), "fig2": str(png)}
