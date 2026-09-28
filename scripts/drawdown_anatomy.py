#!/usr/bin/env python3
"""Diagnose: Woraus bestehen die groessten Drawdowns der Live-Strategie?

Reine Diagnose, aendert nichts an der Strategie. Fuer die 5 tiefsten
Drawdowns des 50/50-Buchs (Ensemble long/short + Vol-Targeting, 10 bps):
  - Hoch, Tief, Erholung, Tiefe, Dauer Hoch->Tief und bis zur Erholung
  - Crash-Anteil: Anteil der 5 schlechtesten Tage an allen Verlusttagen
    zwischen Hoch und Tief
  - Richtung: an wie vielen Tagen das Buch netto long/short war und welche
    Seite den Verlust gebracht hat
  - Marktumfeld: Kursveraenderung BTC und ETH im selben Zeitraum

Vorab registriertes Kriterium fuer eine Trailing-Stop-Studie:
  Mindestens 2 der 3 tiefsten Drawdowns muessen crash-getrieben sein
  (Crash-Anteil >= 50 %). Sonst entstehen die Verluste schleichend durch
  Richtungswechsel, und ein Stop — der auf schnelle Einbrueche zielt —
  haette nichts zu verhindern.

    python scripts/drawdown_anatomy.py               # echter Lauf (Colab)
    python scripts/drawdown_anatomy.py --synthetic   # Smoke-Test
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
COST = 10.0
TOP_N = 5
CRASH_DAYS = 5
CRASH_SHARE = 0.5


def load_book(start: str, end: str, synthetic: bool) -> tuple[pd.DataFrame, dict]:
    legs = {}
    for i, sym in enumerate(SYMBOLS):
        if synthetic:
            daily = resample(make_synthetic("2020-01-01", "2023-01-01",
                                            seed=81 + i), "1d")
        else:
            warm = max(pd.Timestamp(start) - pd.Timedelta(days=290),
                       pd.Timestamp("2017-08-01"))
            daily = load_daily(sym, warm.strftime("%Y-%m-%d"), end, Path("data"))
        bt = backtest(build_positions(daily, None, allow_short=True,
                                      ensemble=True), COST)
        legs[sym] = bt if synthetic else bt.loc[start:end]
    idx = legs[SYMBOLS[0]].index.intersection(legs[SYMBOLS[1]].index)
    book = pd.DataFrame({
        "pnl": sum(0.5 * legs[s].loc[idx, "pnl"] for s in SYMBOLS),
        # wirksame Netto-Richtung des Buchs (Position von gestern)
        "net_pos": sum(0.5 * legs[s].loc[idx, "pos"].shift(1) for s in SYMBOLS),
    }, index=idx).fillna(0.0)
    for s in SYMBOLS:
        book[f"ret_{s}"] = legs[s].loc[idx, "ret"].fillna(0.0)
        book[f"pnl_{s}"] = 0.5 * legs[s].loc[idx, "pnl"]
        book[f"pos_{s}"] = legs[s].loc[idx, "pos"].shift(1).fillna(0.0)
    return book, legs


def drawdown_episodes(pnl: pd.Series) -> list[dict]:
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    episodes, in_dd, start = [], False, None
    for t, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, t
        elif v >= 0 and in_dd:
            episodes.append((start, t))
            in_dd = False
    if in_dd:
        episodes.append((start, None))
    rows = []
    for s, e in episodes:
        seg = dd.loc[s:e] if e is not None else dd.loc[s:]
        trough = seg.idxmin()
        peak_day = dd.index[dd.index.get_loc(s) - 1] if dd.index.get_loc(s) > 0 else s
        rows.append({"peak": peak_day, "trough": trough, "recovery": e,
                     "depth_log": float(seg.min())})
    return sorted(rows, key=lambda r: r["depth_log"])[:TOP_N]


def describe(book: pd.DataFrame, ep: dict) -> dict:
    fall = book.loc[ep["peak"]:ep["trough"]].iloc[1:]
    # Anteil der 5 schlechtesten Tage an ALLEN Verlusttagen Hoch->Tief
    # (Brutto-Verlust, damit der Wert in [0, 1] bleibt, auch wenn
    # zwischendurch Gewinntage liegen).
    gross_loss = fall["pnl"][fall["pnl"] < 0].sum()
    worst = fall["pnl"].nsmallest(CRASH_DAYS).clip(upper=0).sum()
    crash_share = float(worst / gross_loss) if gross_loss < 0 else float("nan")
    long_days = float((fall["net_pos"] > 0.02).mean())
    short_days = float((fall["net_pos"] < -0.02).mean())
    long_pnl = sum(fall[f"pnl_{s}"][fall[f"pos_{s}"] > 0].sum() for s in SYMBOLS)
    short_pnl = sum(fall[f"pnl_{s}"][fall[f"pos_{s}"] < 0].sum() for s in SYMBOLS)
    rec = ep["recovery"]
    return {
        "hoch": ep["peak"].date(), "tief": ep["trough"].date(),
        "erholt": rec.date() if rec is not None else "offen",
        "tiefe_%": round(float(np.expm1(ep["depth_log"]) * 100), 1),
        "tage_bis_tief": len(fall),
        "tage_bis_erholt": (rec - ep["peak"]).days if rec is not None else None,
        "crash_anteil": round(crash_share, 2),
        "typ": "CRASH" if crash_share >= CRASH_SHARE else "schleichend",
        "schlimmster_tag_%": round(float(np.expm1(fall["pnl"].min()) * 100), 1),
        "long_tage": f"{long_days:.0%}", "short_tage": f"{short_days:.0%}",
        "verlust_long_%": round(float(np.expm1(long_pnl) * 100), 1),
        "verlust_short_%": round(float(np.expm1(short_pnl) * 100), 1),
        "BTC_%": round(float(np.expm1(fall[f"ret_{SYMBOLS[0]}"].sum()) * 100), 1),
        "ETH_%": round(float(np.expm1(fall[f"ret_{SYMBOLS[1]}"].sum()) * 100), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    book, _ = load_book(args.start, args.end, args.synthetic)
    rows = [describe(book, ep) for ep in drawdown_episodes(book["pnl"])]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"== Die {TOP_N} tiefsten Drawdowns (50/50-Buch, Live-Spezifikation) ==")
    print(df[["hoch", "tief", "erholt", "tiefe_%", "tage_bis_tief",
              "tage_bis_erholt", "crash_anteil", "typ",
              "schlimmster_tag_%"]].to_string(index=False))
    print("\n== Richtung und Marktumfeld (Hoch -> Tief) ==")
    print(df[["hoch", "tiefe_%", "long_tage", "short_tage", "verlust_long_%",
              "verlust_short_%", "BTC_%", "ETH_%"]].to_string(index=False))

    top3 = df.head(3)
    n_crash = int((top3["typ"] == "CRASH").sum())
    print("\n== Vorab-Kriterium Trailing-Stop ==")
    print(f"Crash-getrieben unter den 3 tiefsten: {n_crash} von 3 (Soll >= 2)")
    if n_crash >= 2:
        print("=> Stop-Studie sinnvoll: Die groessten Verluste kommen aus "
              "wenigen schnellen Einbruechen.")
    else:
        print("=> Keine Stop-Studie: Die groessten Verluste entstehen "
              "schleichend. Ein Stop gegen schnelle Einbrueche haette dort "
              "wenig zu verhindern.")
    print("\nLesart crash_anteil: Anteil der 5 schlechtesten Tage an allen "
          "Verlusttagen Hoch->Tief. 1.0 = alles an 5 Tagen, 0.2 = verteilt ueber "
          "viele Tage. Drawdowns mit <= 5 Tagen sind automatisch 1.0.")


if __name__ == "__main__":
    main()
