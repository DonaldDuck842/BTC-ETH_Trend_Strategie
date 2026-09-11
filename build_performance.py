#!/usr/bin/env python3
"""Erzeugt output/performance.csv aus dem Paper-Log (results/paper_log.csv).

Eine Zeile je Handelstag, aufsteigend sortiert:
  datum, equity, benchmark_equity, tagesrendite, w_BTC, w_ETH

Konventionen (siehe auch Spezifikation im Dashboard-Projekt):
- equity: fiktives Portfolio, Start START_CAPITAL am ersten Log-Tag,
  inkl. Handelskosten (COST_BPS_PER_SIDE je gehandelter Seite).
- benchmark_equity: taeglich rebalancter fester 50/50-Mix aus BTC und ETH,
  Buy-and-Hold ohne Kosten, gleiche Startgroesse. (Bewusste Wahl: derselbe
  Asset-Mix wie die Strategie, damit der Vergleich den Strategie-Effekt
  zeigt, nicht den Asset-Mix.)
- tagesrendite: Dezimalzahl (0.0041 = +0.41%), Tag 1 = 0.0.
- w_BTC / w_ETH: Zielgewicht am GESAMTKAPITAL, wie es am Abend des Tages
  t beschlossen wurde (Bein-Anteil x 0.5); negativ = short.

KEIN LOOKAHEAD: Die Rendite an Tag t entsteht aus den Gewichten von t-1
und den Kursen von t. Der Versatz passiert an der mit ">>> SHIFT"
markierten Stelle unten (target_pos.shift(1)). Die w_-Spalten der Zeile t
zeigen dagegen den am Tag t beschlossenen Zielzustand — er wird erst in
der Rendite von t+1 wirksam.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Parameter (bei Bedarf hier korrigieren) -------------------------------
START_CAPITAL = 100_000.0     # USD, fiktiv
COST_BPS_PER_SIDE = 10.0      # Handelskosten je gehandelter Seite (Backtest-Annahme)
LEG_WEIGHTS = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}

LOG_PATH = Path("results/paper_log.csv")
OUT_PATH = Path("output/performance.csv")
PERF_JSON = Path("results/performance.json")  # Konsistenz-Check


def build() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    days = sorted(log["asof"].unique())
    if len(days) < 2:
        sys.exit("Zu wenig Log-Tage fuer eine Performance-Reihe (< 2).")

    pos = log.pivot(index="asof", columns="symbol", values="target_pos").sort_index()
    close = log.pivot(index="asof", columns="symbol", values="ref_close").sort_index()
    logret = np.log(close).diff()

    # Strategie-PnL (log): Gewichte von GESTERN x Kursrendite von HEUTE,
    # minus Kosten auf die heutige Zielaenderung.
    pnl = pd.Series(0.0, index=pos.index)
    for sym, w in LEG_WEIGHTS.items():
        held = pos[sym].shift(1)                      # >>> SHIFT: kein Lookahead
        cost = pos[sym].diff().abs() * COST_BPS_PER_SIDE / 10_000
        pnl += w * (held * logret[sym]).fillna(0.0) - w * cost.fillna(0.0)

    # Benchmark (log): taeglich rebalancter 50/50-Mix, kostenfrei.
    bench = sum(w * logret[sym].fillna(0.0) for sym, w in LEG_WEIGHTS.items())

    equity = START_CAPITAL * np.exp(pnl.cumsum())
    bench_eq = START_CAPITAL * np.exp(bench.cumsum())

    out = pd.DataFrame({
        "datum": pos.index,
        "equity": equity.round(2).values,
        "benchmark_equity": bench_eq.round(2).values,
        "tagesrendite": equity.pct_change().fillna(0.0).round(6).values,
        "w_BTC": (LEG_WEIGHTS["BTCUSDT"] * pos["BTCUSDT"]).round(4).values,
        "w_ETH": (LEG_WEIGHTS["ETHUSDT"] * pos["ETHUSDT"]).round(4).values,
    })
    return out


def main() -> None:
    out = build()
    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"{OUT_PATH}: {len(out)} Tage, "
          f"{out['datum'].iloc[0]} .. {out['datum'].iloc[-1]}, "
          f"Equity {out['equity'].iloc[-1]:,.2f} USD")

    # Konsistenz-Check gegen die bestehende Website-Ausgabe (gleiche Rechnung
    # in scripts/live_signal.py) — Abweichung waere ein Implementierungsfehler.
    if PERF_JSON.exists():
        ref = json.loads(PERF_JSON.read_text())
        if ref.get("as_of") == out["datum"].iloc[-1]:
            diff = abs(float(ref["current_equity_usd"]) - float(out["equity"].iloc[-1]))
            assert diff < 0.05, (
                f"Equity weicht von results/performance.json ab ({diff:.2f} USD)")
            print("Konsistenz-Check gegen results/performance.json: OK")


if __name__ == "__main__":
    main()
