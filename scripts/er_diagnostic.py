#!/usr/bin/env python3
"""Diagnose: Sagt die Efficiency Ratio voraus, wann die Strategie verliert?

Reine Diagnose — aendert nichts an der Strategie und baut keine Regel.
Vorab registriert (EIN Lauf, keine Fenster-Iteration):

  Indikator: Kaufman Efficiency Ratio ueber 20 Tage,
      ER = |Close_t - Close_{t-20}| / Summe |Close_k - Close_{k-1}|,
      gemessen am Schluss von Tag t-1 (Versatz: er.shift(1), markiert).
      Nahe 1 = sauberer Trend, nahe 0 = Seitwaerts-Gezappel.
  Zielgroesse: Netto-PnL der Live-Spezifikation (Ensemble long/short +
      Vol-Targeting, 10 bps) am naechsten Tag und kumuliert ueber die
      naechsten 10 Tage, je Bein (BTC, ETH), Tage gepoolt.
  Einteilung: ER-Fuenftel ueber die gesamte Historie. Hinweis: Die
      Fuenftel-Grenzen nutzen die ganze Stichprobe — fuer eine Diagnose ok,
      eine spaetere Handelsregel braeuchte feste oder rollierende Schwellen.

Kriterium "ein Seitwaerts-Filter hat eine Chance" (alle drei):
  (a) unterstes Fuenftel (Q1, staerkstes Seitwaerts) netto negativ,
  (b) 90-%-Block-Bootstrap-Intervall von (Q1 - Rest) komplett unter 0,
  (c) Q1 < Rest in mindestens 3 von 4 Subperioden.
Sonst: der Indikator erkennt Seitwaertsphasen zu spaet — keine Regel bauen.

    python scripts/er_diagnostic.py               # echter Lauf (Colab)
    python scripts/er_diagnostic.py --synthetic   # Smoke-Test
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

ER_WINDOW = 20
FWD = 10
COST = 10.0
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SUBPERIODS = [("2018-2020", "2018-01-01", "2020-12-31"),
              ("2021-2022", "2021-01-01", "2022-12-31"),
              ("2023-2024", "2023-01-01", "2024-12-31"),
              ("2025-2026", "2025-01-01", "2026-12-31")]
BOOT_DRAWS = 2000
BOOT_BLOCK = 20


def efficiency_ratio(close: pd.Series, n: int = ER_WINDOW) -> pd.Series:
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)


def leg_panel(daily: pd.DataFrame, sym: str) -> pd.DataFrame:
    frame = build_positions(daily, None, allow_short=True, ensemble=True)
    bt = backtest(frame, COST)
    er_prev = efficiency_ratio(daily["close"]).shift(1)   # >>> SHIFT: kein Lookahead
    fwd = bt["pnl"][::-1].rolling(FWD, min_periods=FWD).sum()[::-1]
    return pd.DataFrame({
        "sym": sym, "er": er_prev, "pnl_1d": bt["pnl"], "pnl_10d": fwd,
        "turnover": bt["turnover"], "abs_pos": bt["pos"].shift(1).abs(),
    }).dropna()


def block_boot_diff(q1: np.ndarray, rest: np.ndarray, seed: int = 7) -> tuple:
    """90-%-Intervall der Differenz der Mittelwerte (Q1 - Rest), Blockbootstrap."""
    rng = np.random.default_rng(seed)

    def draw(x):
        n = len(x)
        k = int(np.ceil(n / BOOT_BLOCK))
        starts = rng.integers(0, max(n - BOOT_BLOCK, 1), size=k)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)).reshape(-1)[:n]
        return x[np.clip(idx, 0, n - 1)].mean()

    diffs = np.array([draw(q1) - draw(rest) for _ in range(BOOT_DRAWS)])
    return np.percentile(diffs, [5, 95])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    panels = []
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            daily = resample(make_synthetic("2020-01-01", "2023-01-01",
                                            seed=71 + i), "1d")
        else:
            warm = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                       pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warm.strftime("%Y-%m-%d"), args.end, Path("data"))
            daily = daily.loc[:args.end]
        p = leg_panel(daily, sym)
        panels.append(p if args.synthetic else p.loc[args.start:args.end])
    df = pd.concat(panels).sort_index()
    df["quintil"] = pd.qcut(df["er"], 5, labels=["Q1 seitwaerts", "Q2", "Q3",
                                                  "Q4", "Q5 trend"])

    rows = []
    total = df["pnl_1d"].sum()
    for q, g in df.groupby("quintil", observed=True):
        rows.append({
            "quintil": q,
            "ER_bereich": f"{g['er'].min():.2f}-{g['er'].max():.2f}",
            "tage": len(g),
            "pnl_1d_bps": round(g["pnl_1d"].mean() * 1e4, 2),
            "pnl_p.a._%": round(g["pnl_1d"].mean() * 365 * 100, 1),
            "trefferquote": round(float((g["pnl_1d"] > 0).mean()), 3),
            "pnl_10d_%": round(g["pnl_10d"].mean() * 100, 2),
            "anteil_gesamt_pnl_%": round(g["pnl_1d"].sum() / total * 100, 1),
            "turnover_pro_tag": round(g["turnover"].mean(), 3),
            "avg_|pos|": round(g["abs_pos"].mean(), 2),
        })
    print("== Strategie-Ertrag nach ER-Fuenftel (ER vom Vortag, 20 Tage) ==")
    print(pd.DataFrame(rows).to_string(index=False))

    is_q1 = df["quintil"] == "Q1 seitwaerts"
    q1_1d = df.loc[is_q1, "pnl_1d"].sort_index().to_numpy()
    rest_1d = df.loc[~is_q1, "pnl_1d"].sort_index().to_numpy()
    lo, hi = block_boot_diff(q1_1d, rest_1d)
    crit_a = q1_1d.mean() < 0
    crit_b = hi < 0

    print("\n== Stabilitaet ueber Marktphasen (Q1 vs. Rest, bps/Tag) ==")
    hits = 0
    n_sub = 0
    for label, s, e in SUBPERIODS:
        seg = df.loc[s:e]
        if len(seg) < 100:
            continue
        n_sub += 1
        m1 = seg.loc[seg["quintil"] == "Q1 seitwaerts", "pnl_1d"].mean() * 1e4
        mr = seg.loc[seg["quintil"] != "Q1 seitwaerts", "pnl_1d"].mean() * 1e4
        hits += int(m1 < mr)
        print(f"   {label}: Q1 {m1:+.1f} | Rest {mr:+.1f}  "
              f"{'Q1 schlechter' if m1 < mr else 'Q1 NICHT schlechter'}")
    crit_c = hits >= 3 if n_sub >= 4 else hits == n_sub

    print("\n== Vorab-Kriterium ==")
    print(f"(a) Q1 netto negativ: {q1_1d.mean() * 1e4:+.2f} bps/Tag "
          f"-> {'ja' if crit_a else 'nein'}")
    print(f"(b) 90-%-Intervall (Q1 - Rest): [{lo * 1e4:+.2f}, {hi * 1e4:+.2f}] "
          f"bps/Tag, komplett unter 0 -> {'ja' if crit_b else 'nein'}")
    print(f"(c) Q1 schlechter in {hits} von {n_sub} Phasen (Soll >= 3) "
          f"-> {'ja' if crit_c else 'nein'}")
    if crit_a and crit_b and crit_c:
        print("=> FILTER HAT EINE CHANCE: naechster Schritt waere eine vorab "
              "registrierte Regel mit fester Schwelle und einmaligem Test.")
    else:
        print("=> KEIN FILTER: Die ER erkennt die Verlustphasen der Strategie "
              "nicht rechtzeitig. Keine Regel bauen, keine Fenster-Iteration.")


if __name__ == "__main__":
    main()
