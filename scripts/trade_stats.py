#!/usr/bin/env python3
"""Trade-Statistiken der finalen Spezifikation (Ensemble 50/50).

Zwei Ebenen:
  1. Tagesebene: Anteil positiver Tage, Durchschnitts-Gewinn/-Verlust-Tag.
  2. Episoden-Ebene: eine "Trade-Episode" = zusammenhaengende Long- oder
     Short-Phase eines Beins (Vorzeichen des Ensemble-Signals konstant).
     Win-Rate, Durchschnitts-Gewinn/-Verlust je Episode, Payoff-Ratio,
     Profit-Faktor, schlechteste Episode.

    python scripts/trade_stats.py                 # echter Lauf (Colab)
    python scripts/trade_stats.py --synthetic     # Smoke-Test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict.daily import backtest, build_positions, load_daily  # noqa: E402
from btc_predict.data import make_synthetic, resample  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def episode_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Zerlegt den Backtest in Long-/Short-Episoden (Vorzeichen der
    wirksamen, d.h. um 1 Tag verschobenen Position)."""
    sign = np.sign(bt["pos"].shift(1).fillna(0.0))
    grp = (sign != sign.shift(1)).cumsum()
    rows = []
    for _, seg in bt.groupby(grp):
        s = float(np.sign(bt["pos"].shift(1).fillna(0.0).loc[seg.index[0]]))
        if s == 0:
            continue
        rows.append({
            "start": seg.index[0].date(), "days": len(seg),
            "dir": "long" if s > 0 else "short",
            "pnl_pct": float(seg["pnl"].sum() * 100),
            "avg_pos": float(seg["pos"].shift(1).abs().mean()),
        })
    return pd.DataFrame(rows)


def print_stats(bt: pd.DataFrame, label: str) -> None:
    in_mkt = bt[bt["pos"].shift(1).abs() > 0]
    up = in_mkt["pnl"] > 0
    print(f"\n== {label} ==")
    print(f"  Tagesebene ({len(in_mkt)} Tage im Markt):")
    print(f"    positive Tage : {up.mean():.1%}")
    print(f"    Ø Gewinn-Tag  : +{in_mkt.loc[up, 'pnl'].mean() * 100:.2f}% | "
          f"Ø Verlust-Tag: {in_mkt.loc[~up, 'pnl'].mean() * 100:.2f}%")

    ep = episode_table(bt)
    if ep.empty:
        return
    wins = ep[ep["pnl_pct"] > 0]
    losses = ep[ep["pnl_pct"] <= 0]
    payoff = (wins["pnl_pct"].mean() / abs(losses["pnl_pct"].mean())
              if len(losses) and len(wins) else float("nan"))
    pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())
          if len(losses) and losses["pnl_pct"].sum() != 0 else float("nan"))
    years = len(bt) / 365
    print(f"  Episoden-Ebene ({len(ep)} Episoden, ~{len(ep) / years:.0f}/Jahr, "
          f"Ø {ep['days'].mean():.0f} Tage):")
    print(f"    Win-Rate      : {len(wins) / len(ep):.1%}")
    print(f"    Ø Gewinn-Ep.  : +{wins['pnl_pct'].mean():.1f}% | "
          f"Ø Verlust-Ep.: {losses['pnl_pct'].mean():.1f}%")
    print(f"    Payoff-Ratio  : {payoff:.1f} | Profit-Faktor: {pf:.2f}")
    print(f"    schlechteste Episode: {ep['pnl_pct'].min():.1f}% "
          f"({ep.loc[ep['pnl_pct'].idxmin(), 'start']}, "
          f"{ep.loc[ep['pnl_pct'].idxmin(), 'days']} Tage) | "
          f"beste: +{ep['pnl_pct'].max():.1f}%")
    print(f"    Long-Episoden : {int((ep['dir'] == 'long').sum())} | "
          f"Short: {int((ep['dir'] == 'short').sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    legs = {}
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            daily = resample(make_synthetic("2021-01-01", "2023-01-01",
                                            seed=21 + i), "1d")
        else:
            # Binance-Historie beginnt 2017-08 — frueher gibt es keine Monats-Zips.
            warmup = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                         pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warmup.strftime("%Y-%m-%d"), args.end,
                               Path("data"))
        frame = build_positions(daily, None, allow_short=True, ensemble=True)
        legs[sym] = backtest(frame, 10.0).loc[args.start:args.end]
        print_stats(legs[sym], f"{sym} (Ensemble long/short, volles Bein)")

    idx = legs[SYMBOLS[0]].index.intersection(legs[SYMBOLS[1]].index)
    pf = pd.DataFrame({
        "pos": sum(0.5 * legs[s].loc[idx, "pos"] for s in SYMBOLS),
        "pnl": sum(0.5 * legs[s].loc[idx, "pnl"] for s in SYMBOLS),
    })
    in_mkt = pf[pf["pos"].shift(1).abs() > 0]
    up = in_mkt["pnl"] > 0
    print(f"\n== Portfolio 50/50, Tagesebene ==")
    print(f"    positive Tage : {up.mean():.1%}")
    print(f"    Ø Gewinn-Tag  : +{in_mkt.loc[up, 'pnl'].mean() * 100:.2f}% | "
          f"Ø Verlust-Tag: {in_mkt.loc[~up, 'pnl'].mean() * 100:.2f}%")
    print(f"    Tages-Vol     : {in_mkt['pnl'].std() * 100:.2f}% "
          f"(= typisches Tagesrisiko des Buchs)")


if __name__ == "__main__":
    main()
