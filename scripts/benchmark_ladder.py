#!/usr/bin/env python3
"""Benchmark ladder: what does each component add over a simple moving average?

Pure description — the frozen live specification is not changed. Builds the
strategy step by step and measures every step on the same data and costs:

  L0  Buy & Hold BTC
  L1  BTC above its 200-day SMA -> long, else cash (the classic simple rule)
  L2  L1 + volatility targeting (40 % target, cap 1.0)
  L3  L2 + short below the 200-day SMA
  L4  L3 with the 7-member ensemble instead of one SMA (= BTC leg of live spec)
  L5  L4 on BTC and ETH, 50/50 (= live specification)
  REF the simple rule (L1) on BTC and ETH, 50/50 — fair same-universe baseline

Pre-registered reading: the live specification has a PROVEN advantage over a
simple moving-average rule only if a paired block bootstrap (30-day blocks,
2000 draws) shows Sharpe(L5) > Sharpe(REF) in >= 95 % of draws. Otherwise the
README must state that the advantage over a plain moving average is not
proven. Signals use the close of day t and act from day t+1 (backtest()
shifts); costs 10 bps per side for every step.

    python scripts/benchmark_ladder.py               # real run (Colab)
    python scripts/benchmark_ladder.py --synthetic   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from btc_predict.daily import backtest, build_positions, load_daily, report  # noqa: E402
from btc_predict.data import make_synthetic, resample  # noqa: E402
from portfolio_strategy import paired_block_bootstrap  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT")
COST = 10.0
SMA = 200
TARGET_VOL = 0.40
BOOT_SHARE = 0.95
PERIODS = [("2018-2020", "2018-01-01", "2020-12-31"),
           ("2021-2022", "2021-01-01", "2022-12-31"),
           ("2023-2024", "2023-01-01", "2024-12-31"),
           ("2025-2026", "2025-01-01", "2026-12-31")]


def simple_frames(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = daily["close"]
    ret = np.log(close).diff()
    above = (close > close.rolling(SMA).mean()).astype(float)
    below = (close < close.rolling(SMA).mean()).astype(float)
    size = (TARGET_VOL / (ret.rolling(20).std() * np.sqrt(365))).clip(upper=1.0).fillna(0.0)
    sma_ready = close.rolling(SMA).mean().notna()

    def frame(pos):
        return pd.DataFrame({"pos": pos.where(sma_ready, 0.0), "ret": ret})

    ens = build_positions(daily, None, allow_short=True, ensemble=True)
    return {
        "L0": pd.DataFrame({"pos": pd.Series(1.0, index=close.index), "ret": ret}),
        "L1": frame(above),
        "L2": frame(above * size),
        "L3": frame((above - below) * size),
        "L4": ens[["pos", "ret"]],
    }


def combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    idx = a.index.intersection(b.index)
    return pd.DataFrame({c: 0.5 * a.loc[idx, c] + 0.5 * b.loc[idx, c]
                         for c in ("pnl", "ret", "pos", "turnover")}, index=idx)


def row(name: str, label: str, bt: pd.DataFrame) -> dict:
    g = report(bt, "gesamt")
    subs = [report(bt.loc[s:e], p) for p, s, e in PERIODS if len(bt.loc[s:e]) > 30]
    years = len(bt) / 365
    return {
        "step": name, "description": label,
        "cagr_%": g["cagr_pct"], "sharpe": g["sharpe"], "max_dd_%": g["max_dd_pct"],
        "worst_phase_cagr_%": min(s["cagr_pct"] for s in subs),
        "bear_2025_26_%": next((s["cagr_pct"] for s in subs if s["period"] == "2025-2026"),
                               float("nan")),
        "time_in_mkt": g["time_in_mkt"],
        "cost_%pa": round(float(bt["turnover"].sum() * COST / 100 / years), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    bts: dict[str, dict[str, pd.DataFrame]] = {}
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            daily = resample(make_synthetic("2019-01-01", "2023-01-01",
                                            seed=101 + i), "1d")
            start, end = "2020-01-01", "2023-01-01"
        else:
            warm = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                       pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warm.strftime("%Y-%m-%d"), args.end, Path("data"))
            start, end = args.start, args.end
        bts[sym] = {k: backtest(f, COST).loc[start:end]
                    for k, f in simple_frames(daily).items()}

    btc, eth = bts["BTCUSDT"], bts["ETHUSDT"]
    live = combine(btc["L4"], eth["L4"])
    ref = combine(btc["L1"], eth["L1"])

    rows = [
        row("L0", "Buy & Hold BTC", btc["L0"]),
        row("L1", "BTC > 200-day SMA: long, else cash", btc["L1"]),
        row("L2", "+ volatility targeting", btc["L2"]),
        row("L3", "+ short below the SMA", btc["L3"]),
        row("L4", "+ 7-filter ensemble (BTC leg of live spec)", btc["L4"]),
        row("L5", "+ ETH, 50/50 (= live specification)", live),
        row("REF", "200-day SMA rule on BTC+ETH, 50/50", ref),
    ]
    pd.set_option("display.width", 250)
    print("== Benchmark ladder (2018-2026, 10 bps per side) ==")
    print(pd.DataFrame(rows).to_string(index=False))

    bs = paired_block_bootstrap(live["pnl"], ref["pnl"], block_days=30,
                                n_draws=2000, seed=42)
    share = float((bs["sharpe_diff"] > 0).mean())
    lo, hi = np.percentile(bs["sharpe_diff"], [5, 95])
    print("\n== Is the advantage over a simple moving average proven? ==")
    print(f"Sharpe live (L5) vs. simple SMA rule on BTC+ETH (REF): "
          f"{rows[5]['sharpe']} vs. {rows[6]['sharpe']}")
    print(f"Bootstrap: L5 better in {share:.1%} of 2000 draws "
          f"(90 % interval of the Sharpe difference: [{lo:+.2f}, {hi:+.2f}])")
    if share >= BOOT_SHARE:
        print("=> PROVEN: the live strategy beats a plain moving-average rule "
              "on the same coins beyond chance.")
    else:
        print("=> NOT PROVEN: the advantage over a plain moving-average rule "
              "on the same coins cannot be separated from chance.")


if __name__ == "__main__":
    main()
