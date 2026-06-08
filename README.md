# Overnight vs. Intraday Return Decomposition & Backtest

Reproduces and stress-tests the core results of **Lou, Polk & Skouras (2019),
"A Tug of War," *Journal of Financial Economics* 134:192–213** — decomposing US
equity returns into **overnight** and **intraday** legs, replicating the
persistence/reversal results, and backtesting the firm-level cross-sectional
strategy with **honest** transaction-cost and execution accounting.

> Status: **Layer 0 scaffold complete & unit-tested.** The decomposition math (the
> heart) is validated to machine precision. Live data pull needs a Polygon key.

## The decomposition (the heart)

For each stock *i*, day *s*:

```
intraday_ret  = close_s / open_s - 1                 # RAW open & close
cc_ret        = adj_close_s / adj_close_{s-1} - 1     # ADJUSTED close-to-close
overnight_ret = (1 + cc_ret) / (1 + intraday_ret) - 1 # residual
```

Corporate actions are assumed to occur overnight, so the intraday leg uses **raw**
prices, close-to-close uses **adjusted** prices, and **all** split/dividend
adjustment lands in the overnight leg by construction. The defining identity holds
exactly, daily and after monthly compounding:

```
(1 + intraday_ret) * (1 + overnight_ret) == (1 + cc_ret)
```

**Missing open:** if the open is unavailable you can't trade at the open, so the
position is held overnight from the prior close to the next available open. We
implement that exactly (`intraday_ret = 0`, `overnight_ret = cc_ret` on a
missing-open day); across a run of such days the overnight legs telescope to
`open_next / close_last`, and the identity is preserved. See
[`ovi/core/decomposition.py`](ovi/core/decomposition.py).

## Layout

```
ovi/
  config.py            # one dataclass/YAML for dates, filters, costs, lags, ...
  data/polygon.py      # Polygon grouped-daily loader + parquet cache  (spec: data/)
  core/decomposition.py# THE HEART: daily legs, bad-data filter, monthly compounding
  core/synthetic.py    # synthetic tape for tests + the Layer-0 demo
  reports/layer0.py    # market VW/EW split + data-quality report + sanity gate
  backtest/            # (Layer 1+) engine, costs, walk-forward
tests/                 # pytest: identity to 1e-12, split-in-overnight, parsing
scripts/run_layer0.py  # entrypoint
config.yaml            # editable configuration
cache/                 # parquet cache (git-ignored)
```

## Setup

Dependencies are standard scientific Python (pandas, numpy, pyarrow, statsmodels,
requests, pyyaml; matplotlib + pandas-datareader for Layer 1):

```bash
python3 -m pip install -r requirements.txt   # or: pip install -e ".[dev]"
```

## Run Layer 0

Layer 0 decomposes the value-weighted market into overnight/intraday and **gates**
on a sane positive split before any portfolio code is written. Three data sources
via `--source`:

```bash
# synthetic — no network/key; proves the pipeline end-to-end:
python3 scripts/run_layer0.py --source synthetic

# yahoo — yfinance (default backtest source). Current-listed universe; flagged
# survivorship bias. Caches per-ticker parquet so re-runs are free:
python3 scripts/run_layer0.py --source yahoo --universe demo  --start 2010-01-01 --end 2019-12-31
python3 scripts/run_layer0.py --source yahoo --universe sp500 --start 2010-01-01 --end 2019-12-31
python3 scripts/run_layer0.py --source yahoo --tickers AAPL,MSFT,XOM --start 2015-01-01

# polygon — whole-tape grouped daily (needs POLYGON_API_KEY):
POLYGON_API_KEY=xxx python3 scripts/run_layer0.py --source polygon --start 2011-01-01 --end 2013-12-31 -v
```

Paper ballpark (CRSP 1993–2013): overnight ≈ 0.55%/mo, intraday ≈ 0.38%/mo. The
gate checks the overnight leg is positive and of plausible magnitude; on failure it
prints a data-quality report and tells you to STOP.

**Validated on real data** (yfinance, 2010–2019): the overnight drift dominates,
reproducing the paper's direction. S&P 500 equal-weight split ≈ **0.77%/mo
overnight vs 0.63%/mo intraday** (value-weight proxy: 1.13 vs 0.14); liquid
mega-caps are even more extreme (overnight ≈ all of the close-to-close return).

## Tests

```bash
python3 -m pytest -q
```

Covers the daily & monthly identity (to 1e-12), corporate-action-in-overnight, the
missing-open roll, the bad-data filter, and the Polygon parse/merge.

## Known data limitations (surfaced honestly)

- **Shares outstanding** is **not** in Polygon grouped-daily, so true point-in-time
  market-cap weighting needs a separate source. Until then Layer 0 falls back to a
  dollar-volume cap **proxy** (or equal weight) and **flags it loudly**;
  `PolygonClient.get_shares_outstanding` is a current-snapshot stub (not
  point-in-time → restatement/look-ahead bias).
- **Adjusted prices**: Polygon's `adjusted=true` covers splits; dividend coverage
  is provider-dependent. The Layer-0 split is a check on this.
- **30-min VWAP open** (`open_price_method="first_30min_vwap"`, the paper's choice)
  needs minute aggregates; it's a documented drop-in hook, not yet wired (default is
  `official_open`).
- **Survivorship**: Polygon grouped-daily is point-in-time per day and includes
  names that later delisted, which mitigates survivorship bias. **yfinance does
  NOT** — it serves current-listed names only, so any yahoo universe is
  survivorship-biased (and, for long backtests on today's S&P 500, carries
  index-inclusion look-ahead). The yahoo loader reports how many requested tickers
  actually returned data so the bias is visible; prefer Polygon for survivorship-
  sensitive conclusions.

## Roadmap

- **Layer 1** — persistence/reversal (Table 1): lagged-leg decile sorts, VW
  long–short, next-month overnight/intraday/cc legs, CAPM & 3-factor alphas,
  Newey-West(12) t-stats, the Fig. 2 lag sweep (1→60 months); plus the
  three honesty versions (gross component, net harvest-the-leg, tradeable-as-stated)
  and the dumb-baseline bake-off.
- **Layer 2** — TugOfWar timing (EWMA legs, 60-mo half-life) and the per-name /
  index overnight−intraday spread feature.

Based on Lou, Polk & Skouras (2019), *JFE* 134:192–213.
