# Overnight vs. Intraday Return Decomposition & Backtest

Reproduces and stress-tests the core results of **Lou, Polk & Skouras (2019),
"A Tug of War," *Journal of Financial Economics* 134:192–213** — decomposing US
equity returns into **overnight** and **intraday** legs, replicating the
persistence/reversal results, and backtesting the firm-level cross-sectional
strategy with **honest** transaction-cost and execution accounting.

> Status: **Layers 0, 1 & 2 complete and validated on real data (yfinance, S&P 500
> 2004–2019).** The decomposition identity is unit-tested to machine precision and
> the "tug of war" reproduces: overnight-sorted L-S = +0.89%/mo overnight (t=3.07,
> FF3 α t=2.34) and −1.36%/mo intraday (t=−5.5); momentum earns overnight and bleeds
> intraday. Harvesting the overnight leg daily nets −10%/mo — component alpha, not
> P&L. 33 unit tests green.

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

## Run Layer 1 (Table 1 + honest costs + Fig. 2)

```bash
python3 scripts/run_layer1.py --source yahoo --universe sp500 --start 2010-01-01 --end 2019-12-31
```

Sorts stocks each month on the **lagged 1-month overnight** (Panel A) and **intraday**
(Panel B) leg, forms VW decile long–shorts, and reports the next month's
overnight/intraday/cc legs with mean, **CAPM** & **FF3** alphas (Fama-French via the
Ken French library — free, no key), **Newey-West(12)** t-stats, and skewness. Writes
a markdown report + the Fig. 2 lag-sweep plot to `reports/output/`. Also runs the
three cost versions, the dumb-baseline bake-off, and the IS/OOS/lockbox walk-forward.

**Result (S&P 500 2010–2019):** the tug of war reproduces — Panel A overnight L-S
**+0.89%/mo (t=3.07)**, intraday **−1.36%/mo (t=−5.5)**; Panel B intraday **+0.91%/mo**
(FF3 α t=2.0), overnight **−0.76%/mo (t=−2.8)**. The overnight edge is **gross
component attribution**: harvesting it requires daily close→open round trips whose
break-even cost is ~2 bps (vs a realistic ~27 bps), so it nets **≈ −10%/mo**.

## Tests

```bash
python3 -m pytest -q
```

33 tests covering: the daily & monthly identity (to 1e-12), corporate-action-in-
overnight, the missing-open roll, the bad-data filter, the Polygon & yfinance
parse/merge, the FF-factor CSV parser, Newey-West stats / alpha / Sharpe, **no-look-
ahead in the decile signal AND the TugOfWar predictor**, the cost-model monotonicity,
and the overnight−intraday spread.

## Live data via Schwab (real broker data)

`--source schwab` pulls real, tradeable prices from the **Schwab Trader (Market
Data) API** via `schwab-py` (OAuth handled by the library). One-time setup, run
locally (it needs your Schwab login — I can't do that step for you):

1. On developer.schwab.com create an app with the **Market Data Production**
   product; set the callback URL to `https://127.0.0.1:8182`; wait for status
   **"Ready For Use"**.
2. `pip install schwab-py`, then export credentials:
   ```bash
   export SCHWAB_APP_KEY=...      # "App Key"
   export SCHWAB_APP_SECRET=...   # "Secret"
   ```
3. `python3 scripts/schwab_setup.py` — opens a browser to log in, writes
   `.schwab_token.json` (git-ignored), test-pulls AAPL, and runs an adjustment
   diagnostic. **Refresh tokens expire after 7 days**, so re-run weekly.
4. Run the pipeline on Schwab data:
   ```bash
   python3 scripts/run_layer1.py --source schwab --universe sp500 \
       --start 2010-01-01 --end 2019-12-31 --dividend-adjust
   ```

**Adjustment caveat (important):** Schwab prices are **split-adjusted but NOT
dividend-adjusted**. The intraday leg (close/open) is unaffected, but the
overnight/close-to-close leg needs dividends. By default `adj_close` is the
split-only close (overnight understated by ~the dividend yield, ≈0.15%/mo);
`--dividend-adjust` adds ex-date dividends back so they land in the overnight leg
per the paper's convention. The loader uses **market-data endpoints only** — it
never touches trading/account endpoints.

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

- **Layer 1** ✅ *done* — persistence/reversal (Table 1): lagged-leg decile sorts, VW
  long–short, next-month overnight/intraday/cc legs, CAPM & 3-factor alphas,
  Newey-West(12) t-stats, the Fig. 2 lag sweep; plus the three honesty versions
  (gross component, net harvest-the-leg, tradeable-as-stated), the dumb-baseline
  bake-off, and the IS/OOS/lockbox walk-forward.
- **Layer 2** ✅ *done* — TugOfWar timing (momentum & short-term-reversal factors,
  EWMA overnight/intraday legs at 60-mo half-life, sign-flipped predictor, NW
  predictive regression) and the per-name / index overnight−intraday spread feature
  (parquet). `python3 scripts/run_layer2.py --source yahoo --universe sp500 --start 2004-01-01`.
  **Result:** the tug of war is clear (momentum 2010s: +1.35%/mo overnight, −1.09%/mo
  intraday); the 60-mo EWMA *timing* coefficient turns the expected positive sign for
  momentum/pooled on the longer 2004–2019 sample but is underpowered (2 factors, one
  survivorship-biased universe — the paper pools many anomalies over CRSP).

## Outputs

`scripts/run_layer{0,1,2}.py` write to `reports/output/` (text reports + Fig. 2 lag
sweep + the tug-of-war plot) and `features/` (the overnight−intraday spread parquet
for a downstream regime model). Both dirs are git-ignored (regenerable).

Based on Lou, Polk & Skouras (2019), *JFE* 134:192–213.
