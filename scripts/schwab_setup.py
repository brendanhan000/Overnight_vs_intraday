#!/usr/bin/env python3
"""One-time Schwab auth + live verification (run this on YOUR machine).

What it does:
  1. Runs the OAuth login flow (opens a browser) and writes the token file.
  2. Pulls a few days of AAPL daily candles and prints them — verifies the live
     response shape matches the loader's parser.
  3. Diagnostic: compares Schwab close vs yfinance Close (split-adj) and Adj Close
     (split+div-adj) around a known dividend ex-date, to confirm Schwab is
     SPLIT-ONLY adjusted (so you know whether to use --dividend-adjust).

Prereqs:
  pip install schwab-py
  export SCHWAB_APP_KEY=...        # 'App Key' from developer.schwab.com
  export SCHWAB_APP_SECRET=...     # 'Secret'
  # App must have the *Market Data Production* product, status "Ready For Use",
  # callback URL https://127.0.0.1:8182
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALLBACK = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
TOKEN_PATH = os.environ.get("SCHWAB_TOKEN_PATH", ".schwab_token.json")


def main() -> int:
    app_key = os.environ.get("SCHWAB_APP_KEY")
    app_secret = os.environ.get("SCHWAB_APP_SECRET")
    if not (app_key and app_secret):
        print("ERROR: set SCHWAB_APP_KEY and SCHWAB_APP_SECRET first.", file=sys.stderr)
        return 2
    try:
        from schwab.auth import easy_client
    except ImportError:
        print("ERROR: pip install schwab-py", file=sys.stderr)
        return 2

    print(f"[auth] launching login flow (callback {CALLBACK}); a browser will open...")
    client = easy_client(
        api_key=app_key, app_secret=app_secret,
        callback_url=CALLBACK, token_path=TOKEN_PATH,
    )
    print(f"[auth] OK — token written to {TOKEN_PATH} (refresh expires in ~7 days)")

    # 1) Test pull + parser check.
    import pandas as pd

    from ovi.data.schwab import parse_candles

    start = pd.Timestamp.today() - pd.Timedelta(days=10)
    resp = client.get_price_history_every_day(
        "AAPL", start_datetime=start.to_pydatetime(), need_extended_hours_data=False
    )
    resp.raise_for_status()
    payload = resp.json()
    print("\n[pull] AAPL last ~10 days (raw candle keys:",
          list(payload.get("candles", [{}])[0].keys()), ")")
    print(parse_candles(payload, "AAPL").tail(5).to_string(index=False))

    # 2) Adjustment diagnostic vs yfinance around a known AAPL ex-dividend date.
    try:
        import yfinance as yf

        win0, win1 = "2023-08-01", "2023-08-31"  # AAPL ex-div 2023-08-11 (~$0.24)
        sresp = client.get_price_history_every_day(
            "AAPL",
            start_datetime=pd.Timestamp(win0).to_pydatetime(),
            end_datetime=pd.Timestamp(win1).to_pydatetime(),
            need_extended_hours_data=False,
        )
        sch = parse_candles(sresp.json(), "AAPL").set_index("date")["close"]
        yf_df = yf.download("AAPL", start=win0, end=win1, auto_adjust=False, progress=False)
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = yf_df.columns.get_level_values(0)
        cmp = pd.DataFrame({
            "schwab_close": sch,
            "yf_close_splitadj": yf_df["Close"],
            "yf_adjclose_splitdivadj": yf_df["Adj Close"],
        }).dropna()
        print("\n[diagnostic] AAPL around 2023-08-11 ex-div (expect schwab_close ~= "
              "yf_close, both differ from yf_adjclose):")
        print(cmp.round(3).to_string())
        print("\n=> If schwab_close tracks yf_close (split-adj) and NOT yf_adjclose,")
        print("   Schwab is split-only -> use --dividend-adjust for a correct overnight leg.")
    except Exception as exc:  # noqa: BLE001
        print(f"[diagnostic] skipped ({exc})")

    print("\n[done] You can now run, e.g.:")
    print("  python3 scripts/run_layer1.py --source schwab --universe sp500 "
          "--start 2010-01-01 --end 2019-12-31 --dividend-adjust")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
