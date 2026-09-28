#!/usr/bin/env python3
"""Letzte Varianten-Runde vor dem Einfrieren der Strategie.

Vorab registriert (EIN Lauf, keine Nachjustierung). Alle Varianten sind
Umbauten der Live-Spezifikation (50/50 BTC/ETH, Ensemble + Vol-Targeting,
10 bps) und werden gegen sie gemessen:

  V1 ohne_shorts     Ensemble long/flat (was bringen die Shorts?)
  V2 min_pos_30      Bein flat, wenn |Zielposition| < 0.30
  V3 min_konsens     Bein flat, wenn |Ensemble-Signal| <= 2/7
  V4 shorts_streng   Short nur, wenn Ensemble-Signal <= -5/7 (Long unveraendert)

Gates fuer eine Uebernahme (alle vier, gegen die Live-Spezifikation):
  (G1) alle Subperioden netto positiv
  (G2) schlechtester Subperioden-Sharpe >= der der Live-Spezifikation
  (G3) Kosten-Drag <= Live + 2 Prozentpunkte p.a.
  (G4) gepaarter Block-Bootstrap (30 Tage, 2000 Draws): Anteil der Draws mit
       Sharpe(Variante) > Sharpe(Live) >= 0.9875 — Bonferroni-Korrektur
       fuer 4 gleichzeitig getestete Varianten (0.95 -> 1 - 0.05/4).
  Bestehen mehrere, gewinnt der hoechste Bootstrap-Anteil. Besteht keine,
  bleibt die Live-Spezifikation und wird eingefroren.

Diagnosen (aendern nichts):
  D1 Short-Beitrag: PnL der Long- und Short-Tage je Bein und Subperiode.
  D2 Live-Spezifikation mit 5-%-Rebalance-Band (reale Ausfuehrungsregel).

    python scripts/final_variants.py               # echter Lauf (Colab)
    python scripts/final_variants.py --synthetic   # Smoke-Test
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
BAND = 0.05
G4_SHARE = 1 - 0.05 / 4
EPS = 1e-9
PERIODS = [("gesamt", None),
           ("2018-2020", ("2018-01-01", "2020-12-31")),
           ("2021-2022", ("2021-01-01", "2022-12-31")),
           ("2023-2024", ("2023-01-01", "2024-12-31")),
           ("2025-2026", ("2025-01-01", "2026-12-31"))]


def variant_frames(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    live = build_positions(daily, None, allow_short=True, ensemble=True)
    out = {"live": live}

    v1 = build_positions(daily, None, allow_short=False, ensemble=True)
    out["V1_ohne_shorts"] = v1

    v2 = live.copy()
    v2["pos"] = live["pos"].where(live["pos"].abs() >= 0.30, 0.0)
    out["V2_min_pos_30"] = v2

    v3 = live.copy()
    v3["pos"] = live["pos"].where(live["sig"].abs() > 2 / 7 + EPS, 0.0)
    out["V3_min_konsens"] = v3

    v4 = live.copy()
    weak_short = (live["sig"] < 0) & (live["sig"] > -5 / 7 + EPS)
    v4["pos"] = live["pos"].where(~weak_short, 0.0)
    out["V4_shorts_streng"] = v4
    return out


def apply_band(frame: pd.DataFrame, band: float) -> pd.DataFrame:
    """Pfadabhaengig: Position wird nur angepasst, wenn die Abweichung vom
    gehaltenen Stand das Band ueberschreitet (reale Live-Ausfuehrungsregel)."""
    held, out = 0.0, []
    for target in frame["pos"].fillna(0.0).to_numpy():
        if abs(target - held) > band:
            held = target
        out.append(held)
    f = frame.copy()
    f["pos"] = out
    return f


def book(legs: list[pd.DataFrame]) -> pd.DataFrame:
    idx = legs[0].index.intersection(legs[1].index)
    return pd.DataFrame({c: sum(0.5 * l.loc[idx, c] for l in legs)
                         for c in ("pnl", "ret", "pos", "turnover")}, index=idx)


def sub_table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rng in PERIODS:
        seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
        if len(seg) > 30:
            rows.append(report(seg, label))
    return pd.DataFrame(rows)


def drag_pa(bt: pd.DataFrame) -> float:
    return float(bt["turnover"].sum() * COST / 100 / (len(bt) / 365))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    per_var: dict[str, list[pd.DataFrame]] = {}
    band_legs, short_rows = [], []
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            daily = resample(make_synthetic("2020-01-01", "2023-01-01",
                                            seed=91 + i), "1d")
            start, end = "2020-10-01", "2023-01-01"
        else:
            warm = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                       pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warm.strftime("%Y-%m-%d"), args.end, Path("data"))
            start, end = args.start, args.end
        frames = variant_frames(daily)
        for name, fr in frames.items():
            per_var.setdefault(name, []).append(backtest(fr, COST).loc[start:end])
        band_legs.append(backtest(apply_band(frames["live"], BAND), COST).loc[start:end])

        live_bt = per_var["live"][-1]
        side = np.sign(live_bt["pos"].shift(1).fillna(0.0))
        for label, rng in PERIODS:
            seg = live_bt if rng is None else live_bt.loc[rng[0]:rng[1]]
            sd = side.loc[seg.index]
            if len(seg) < 30:
                continue
            yrs = len(seg) / 365
            short_rows.append({
                "bein": sym, "periode": label,
                "long_pa_%": round(float(np.expm1(seg["pnl"][sd > 0].sum() / yrs) * 100), 1),
                "short_pa_%": round(float(np.expm1(seg["pnl"][sd < 0].sum() / yrs) * 100), 1),
                "short_tage": f"{(sd < 0).mean():.0%}",
            })

    books = {name: book(legs) for name, legs in per_var.items()}
    live = books["live"]
    tb_live = sub_table(live)
    worst_live = tb_live[tb_live["period"] != "gesamt"]["sharpe"].min()

    print("== D1: Was bringen die Shorts? (Live-Spezifikation, Beitrag p.a. je Seite) ==")
    print(pd.DataFrame(short_rows).to_string(index=False))

    band_bt = book(band_legs)
    tb_band = sub_table(band_bt).iloc[0]
    print("\n== D2: Live-Spezifikation mit 5-%-Handelsband (reale Ausfuehrung) ==")
    print(f"   ohne Band: CAGR {tb_live.iloc[0]['cagr_pct']}% | Sharpe "
          f"{tb_live.iloc[0]['sharpe']} | Kosten {drag_pa(live):.2f}% p.a.")
    print(f"   mit Band : CAGR {tb_band['cagr_pct']}% | Sharpe {tb_band['sharpe']} "
          f"| Kosten {drag_pa(band_bt):.2f}% p.a.")

    print("\n== Rendite p.a. je Marktphase ==")
    matrix = pd.DataFrame({name: sub_table(bt).set_index("period")["cagr_pct"]
                           for name, bt in books.items()})
    print(matrix.to_string())

    summary, passed = [], []
    for name, bt in books.items():
        tb = sub_table(bt)
        subs = tb[tb["period"] != "gesamt"]
        g = tb.iloc[0]
        row = {"variante": name, "cagr_%": g["cagr_pct"], "sharpe": g["sharpe"],
               "maxdd_%": g["max_dd_pct"], "schwaechste_phase_sharpe":
               subs["sharpe"].min(), "kosten_%pa": round(drag_pa(bt), 2)}
        if name != "live":
            bs = paired_block_bootstrap(bt["pnl"], live["pnl"], block_days=30,
                                        n_draws=2000, seed=42)
            share = float((bs["sharpe_diff"] > 0).mean())
            g1 = bool((subs["cagr_pct"] > 0).all())
            g2 = subs["sharpe"].min() >= worst_live
            g3 = drag_pa(bt) <= drag_pa(live) + 2.0
            g4 = share >= G4_SHARE
            row.update({"bootstrap_anteil": round(share, 3),
                        "G1": "ok" if g1 else "X", "G2": "ok" if g2 else "X",
                        "G3": "ok" if g3 else "X", "G4": "ok" if g4 else "X"})
            if g1 and g2 and g3 and g4:
                passed.append((share, name))
        summary.append(row)

    print("\n== Ueberblick und Gates (G4 verlangt Bootstrap-Anteil >= "
          f"{G4_SHARE:.4f}) ==")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n== Urteil ==")
    if passed:
        best = max(passed)[1]
        print(f"=> {best} besteht alle Gates (hoechster Bootstrap-Anteil) und "
              f"ersetzt die Live-Spezifikation. Danach einfrieren.")
    else:
        print("=> Keine Variante besteht alle Gates. Die Live-Spezifikation "
              "bleibt und wird eingefroren.")


if __name__ == "__main__":
    main()
