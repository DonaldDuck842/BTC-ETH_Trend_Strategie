#!/usr/bin/env python3
"""Studie: "Voll long, wenn die Strategie mehr als 80 % halten wuerde".

Vorab registriert (EIN Lauf, keine Schwellen-Iteration). Pro Bein
(BTC, ETH, je 50 % des Kapitals):

  Variante A — Positionsgroesse:  Ensemble-Zielposition (Signal x Vol-Size)
                                   > 0.8  -> Bein 100 % long, sonst flat.
                                   Greift nur bei starkem Signal UND ruhigem
                                   Markt (Size > 0.8 heisst Jahres-Vol < 50 %)
                                   — wirkt also zugleich als Vol-Filter.
  Variante B — Konsens:           Ensemble-Signal > 0.8, d.h. mind. 6 von 7
                                   Trendfiltern long -> 100 % long, sonst flat.
                                   Unabhaengig von der Volatilitaet.

Signal am Tagesschluss t, Position wirksam ab t+1 (backtest() verschiebt),
10 bps Kosten je Seite. Vergleich: Live-Spezifikation (Ensemble long/short
mit Vol-Targeting) und Buy-and-Hold 50/50.

Einordnung: Beide Varianten sind long-only mit vollem Einsatz OHNE
Vol-Targeting — hoehere Drawdowns sind strukturell zu erwarten. Der Lauf ist
eine Studie auf der bereits verbrauchten Historie 2018-2026; eine Adoption
braeuchte zusaetzlich die Gates aus docs/BETRIEBSREGELN.md.

    python scripts/long_threshold_study.py              # echter Lauf (Colab)
    python scripts/long_threshold_study.py --synthetic  # Smoke-Test
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
from trade_stats import episode_table  # noqa: E402

THRESHOLD = 0.8
COST = 10.0
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PERIODS = [("gesamt", None),
           ("2018-2020", ("2018-01-01", "2020-12-31")),
           ("2021-2022", ("2021-01-01", "2022-12-31")),
           ("2023-2024", ("2023-01-01", "2024-12-31")),
           ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]


def leg_frames(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Positions-Frames je Variante fuer ein Bein (gleiche ret-Spalte)."""
    base = build_positions(daily, None, allow_short=True, ensemble=True)
    var_a = base.copy()
    var_a["pos"] = (base["pos"] > THRESHOLD).astype(float)
    var_b = base.copy()
    var_b["pos"] = (base["sig"] > THRESHOLD).astype(float)
    bh = base.copy()
    bh["pos"] = 1.0
    return {"live": base, "A_pos>0.8": var_a, "B_konsens>0.8": var_b, "buyhold": bh}


def combine(legs: list[pd.DataFrame]) -> pd.DataFrame:
    idx = legs[0].index.intersection(legs[1].index)
    return pd.DataFrame({c: sum(0.5 * l.loc[idx, c] for l in legs)
                         for c in ("pnl", "ret", "pos", "turnover")}, index=idx)


def table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rng in PERIODS:
        seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
        if len(seg) > 30:
            rows.append(report(seg, label))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    per_variant: dict[str, list[pd.DataFrame]] = {}
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            daily = resample(make_synthetic("2021-01-01", "2023-01-01",
                                            seed=51 + i), "1d")
            start, end = "2021-10-01", "2023-01-01"
        else:
            warmup = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                         pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warmup.strftime("%Y-%m-%d"), args.end,
                               Path("data"))
            start, end = args.start, args.end
        for name, frame in leg_frames(daily).items():
            bt = backtest(frame, COST).loc[start:end]
            per_variant.setdefault(name, []).append(bt)

    labels = {
        "live": "LIVE-SPEZ (Ensemble long/short + Vol-Targeting)",
        "A_pos>0.8": "VARIANTE A: 100 % long wenn Zielposition > 0.8",
        "B_konsens>0.8": "VARIANTE B: 100 % long wenn >= 6/7 Filter long",
        "buyhold": "BUY-AND-HOLD 50/50 (Referenz)",
    }
    summary = []
    for name, legs in per_variant.items():
        pf = combine(legs)
        tbl = table(pf)
        print(f"\n== {labels[name]} ==")
        print(tbl.to_string(index=False))
        if name in ("A_pos>0.8", "B_konsens>0.8"):
            for sym, leg in zip(SYMBOLS, legs):
                ep = episode_table(leg)
                if ep.empty:
                    print(f"   {sym}: keine Trades")
                    continue
                wins = ep[ep["pnl_pct"] > 0]
                losses = ep[ep["pnl_pct"] <= 0]
                payoff = (wins["pnl_pct"].mean() / abs(losses["pnl_pct"].mean())
                          if len(wins) and len(losses) else float("nan"))
                print(f"   {sym}: {len(ep)} Trades, Win-Rate "
                      f"{len(wins) / len(ep):.0%}, Ø Dauer {ep['days'].mean():.0f} "
                      f"Tage, Ø Gewinn +{wins['pnl_pct'].mean():.1f}% / Ø Verlust "
                      f"{losses['pnl_pct'].mean():.1f}%, Payoff {payoff:.1f}")
        g = tbl.iloc[0]
        summary.append({"variante": name, "cagr_pct": g["cagr_pct"],
                        "sharpe": g["sharpe"], "max_dd_pct": g["max_dd_pct"],
                        "time_in_mkt": g["time_in_mkt"]})

    print("\n== Ueberblick Gesamtperiode ==")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nLesart: Die Varianten sind long-only mit vollem Einsatz und ohne")
    print("Vol-Targeting — hoeherer MaxDD ist Konstruktion, nicht Zufall.")
    print("Entscheidend ist der Vergleich im Haertetest 2025-26 und der")
    print("MaxDD gegenueber der Live-Spezifikation. Keine Schwellen-")
    print("Iteration (0.7, 0.9 ...) auf denselben Daten.")


if __name__ == "__main__":
    main()
