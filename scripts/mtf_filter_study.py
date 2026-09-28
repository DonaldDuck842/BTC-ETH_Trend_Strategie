#!/usr/bin/env python3
"""Studie: 4h-Veto-Filter auf die Tages-Strategie (Multi-Timeframe).

Vorab registriert (EIN Lauf, keine Parameter-Iteration):

  Richtung + Groesse: unveraendert aus der Live-Spezifikation (Tages-
      Ensemble long/short, Vol-Targeting 40 %, Cap 1.0). Tagesposition vom
      Schluss des Tages D gilt fuer alle 4h-Kerzen des Tages D+1.
  4h-Filter: EIN Champion-Member (SMA 50 + Donchian-Mitte 20, Konsens-Regel)
      auf 4h-Kerzen — Parameter uebernommen, nicht gesucht.
  Veto-Regel: Zeigt der 4h-Filter am Schluss von Kerze t GEGEN die
      Tagesrichtung, ist das Bein in Kerze t+1 flat. Gleichgerichtet oder
      neutral -> Tagesposition bleibt. (Versatz: filt.shift(1), markiert.)
  Kosten: 10 bps je Seite auf jede Positionsaenderung (auch intraday).

Vergleichsbasis: Live-Spezifikation auf DEMSELBEN 4h-Raster (Filter immer
an) — muss die Kennzahlen des Tages-Backtests reproduzieren (Sanity-Check).

Gates (alle muessen halten, sonst beerdigt, keine Nachjustierung):
  (M1) alle Subperioden netto positiv
  (M2) MaxDD gesamt mind. 3 Prozentpunkte flacher als die Basis
  (M3) Sharpe gesamt >= Basis - 0.05
  (M4) Kosten-Drag <= Basis + 2 Prozentpunkte p.a.
Informativ (kein Gate): dieselbe Rechnung mit 4 bps (Perp-Taker).

    python scripts/mtf_filter_study.py               # echter Lauf (Colab)
    python scripts/mtf_filter_study.py --synthetic   # Smoke-Test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import data as data_mod  # noqa: E402
from btc_predict.daily import build_positions, momentum_signal  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT")
COST = 10.0
COST_SENS = 4.0
BARS_PER_YEAR = 365 * 6
PERIODS = [("gesamt", None),
           ("2018-2020", ("2018-01-01", "2020-12-31")),
           ("2021-2022", ("2021-01-01", "2022-12-31")),
           ("2023-2024", ("2023-01-01", "2024-12-31")),
           ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]


def leg_positions(daily: pd.DataFrame, h4: pd.DataFrame) -> pd.DataFrame:
    """Effektive 4h-Positionen (Basis und Filter) + 4h-Log-Renditen."""
    dpos = build_positions(daily, None, allow_short=True, ensemble=True)["pos"]
    # Tagesposition vom Schluss D wirkt ab D+1 00:00 UTC.
    eff_daily = dpos.copy()
    eff_daily.index = eff_daily.index + pd.Timedelta(days=1)
    day_of_bar = h4.index.floor("1D")
    base = pd.Series(eff_daily.reindex(day_of_bar).values, index=h4.index).fillna(0.0)

    filt = momentum_signal(h4, 50, 20, allow_short=True)
    prev_filt = filt.shift(1).fillna(0.0)          # >>> SHIFT: kein Lookahead
    veto = (np.sign(base) != 0) & (prev_filt == -np.sign(base))
    filtered = base.where(~veto, 0.0)

    ret = np.log(h4["close"]).diff().fillna(0.0)
    return pd.DataFrame({"base": base, "filtered": filtered, "ret": ret})


def pnl_series(pos: pd.Series, ret: pd.Series, cost_bps: float) -> pd.DataFrame:
    turnover = pos.diff().abs().fillna(pos.abs())
    return pd.DataFrame({"pnl": pos * ret - turnover * cost_bps / 10_000,
                         "turnover": turnover, "pos": pos})


def stats(df: pd.DataFrame, cost_bps: float) -> dict:
    daily = df["pnl"].resample("1D").sum()
    years = len(daily) / 365
    eq = daily.cumsum()
    return {
        "cagr_pct": round((np.exp(daily.sum() / years) - 1) * 100, 1),
        "sharpe": round(daily.mean() / daily.std() * np.sqrt(365), 2)
        if daily.std() > 0 else 0.0,
        "max_dd_pct": round(np.expm1((eq - eq.cummax()).min()) * 100, 1),
        "time_in_mkt": round(float((df["pos"] != 0).mean()), 2),
        "cost_drag_pa": round(df["turnover"].sum() * cost_bps / 100 / years, 2),
    }


def table(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    rows = []
    for label, rng in PERIODS:
        seg = df if rng is None else df.loc[rng[0]:rng[1]]
        if len(seg) > 30 * 6:
            rows.append({"period": label, **stats(seg, cost_bps)})
    return pd.DataFrame(rows)


def load(sym: str, start: str, end: str, synthetic: bool, seed: int):
    if synthetic:
        bars = data_mod.make_synthetic(start, end, seed=seed)
        return data_mod.resample(bars, "1d"), data_mod.resample(bars, "4h")
    warm = max(pd.Timestamp(start) - pd.Timedelta(days=290), pd.Timestamp("2017-08-01"))
    months = data_mod.month_range(warm.strftime("%Y-%m-%d"), end)
    d = data_mod.load_months(sym, "1d", months, Path("data"))
    h = data_mod.load_months(sym, "4h", months, Path("data"))
    return d, h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    start, end = (("2021-10-01", "2022-12-31") if args.synthetic
                  else (args.start, args.end))

    legs = {}
    for i, sym in enumerate(SYMBOLS):
        d, h = load(sym, "2021-01-01" if args.synthetic else args.start,
                    "2023-01-01" if args.synthetic else args.end,
                    args.synthetic, 61 + i)
        legs[sym] = leg_positions(d, h)

    results = {}
    for cost in (COST, COST_SENS):
        for name in ("base", "filtered"):
            parts = [pnl_series(legs[s][name], legs[s]["ret"], cost) for s in SYMBOLS]
            idx = parts[0].index.intersection(parts[1].index)
            pf = pd.DataFrame({c: sum(0.5 * p.loc[idx, c] for p in parts)
                               for c in ("pnl", "turnover")}, index=idx)
            pf["pos"] = sum(0.5 * legs[s].loc[idx, name].abs() for s in SYMBOLS)
            results[(name, cost)] = pf.loc[start:end]

    for name, label in (("base", "BASIS: Live-Spezifikation auf 4h-Raster"),
                        ("filtered", "KANDIDAT: Live-Spez + 4h-Veto-Filter")):
        print(f"\n== {label} | {COST:g} bps/Seite ==")
        print(table(results[(name, COST)], COST).to_string(index=False))

    tb = table(results[("base", COST)], COST)
    tf = table(results[("filtered", COST)], COST)
    sub_f = tf[tf["period"] != "gesamt"]
    gb, gf = tb.iloc[0], tf.iloc[0]
    m1 = bool((sub_f["cagr_pct"] > 0).all())
    m2 = gf["max_dd_pct"] >= gb["max_dd_pct"] + 3.0
    m3 = gf["sharpe"] >= gb["sharpe"] - 0.05
    m4 = gf["cost_drag_pa"] <= gb["cost_drag_pa"] + 2.0
    print("\n== Gates (vorab registriert) ==")
    print(f"(M1) alle Subperioden positiv: {list(sub_f['cagr_pct'])} "
          f"-> {'PASS' if m1 else 'FAIL'}")
    print(f"(M2) MaxDD {gf['max_dd_pct']}% mind. 3pp flacher als "
          f"{gb['max_dd_pct']}% -> {'PASS' if m2 else 'FAIL'}")
    print(f"(M3) Sharpe {gf['sharpe']} >= {gb['sharpe']} - 0.05 "
          f"-> {'PASS' if m3 else 'FAIL'}")
    print(f"(M4) Kosten-Drag {gf['cost_drag_pa']}% <= {gb['cost_drag_pa']}% + 2pp "
          f"-> {'PASS' if m4 else 'FAIL'}")

    s_b = stats(results[("base", COST_SENS)], COST_SENS)
    s_f = stats(results[("filtered", COST_SENS)], COST_SENS)
    print(f"\nInformativ ({COST_SENS:g} bps, Perp-Taker): Basis Sharpe {s_b['sharpe']} "
          f"/ MaxDD {s_b['max_dd_pct']}% | Kandidat Sharpe {s_f['sharpe']} "
          f"/ MaxDD {s_f['max_dd_pct']}%")
    verdict = ("PASS — Kandidat fuer Paper-Test (Betrieb braeuchte 4h-Laeufe)"
               if (m1 and m2 and m3 and m4) else
               "FAIL — beerdigt, Live-Spezifikation bleibt (keine Nachjustierung)")
    print(f"=> {verdict}")
    print("\nSanity: Die BASIS-Zeile 'gesamt' sollte die Tages-Backtest-Werte")
    print("(CAGR ~40 %, Sharpe ~1.03, MaxDD ~-32 %) reproduzieren.")


if __name__ == "__main__":
    main()
