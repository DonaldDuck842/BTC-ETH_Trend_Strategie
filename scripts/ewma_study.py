#!/usr/bin/env python3
"""EWMA-Vol-Sizing-Studie (vorab registriert, EIN Lauf, kein Lambda-Tuning).

Kandidat: identische Live-Spezifikation (Ensemble long/short je Bein, 50/50),
aber Vol-Targeting mit RiskMetrics-EWMA (lambda=0.94, Literatur-Standard)
statt rollierender 20-Tage-Vol. Mechanismus: schnellere Reaktion auf
Vol-Sprvenge (Kaskaden-Tage), traegheitsfreieres De-Risking — genau der
legitime Kern, den die Skeptiker beim beerdigten Intraday-De-Risking als
separaten Kandidaten vorgemerkt hatten.

Gates (gleiche Form wie bei der Ensemble-Adoption; FAIL = beerdigen,
ausdruecklich KEIN zweiter Versuch mit anderem Lambda/Halbwertszeit):
  (V1) alle Subperioden des EWMA-Portfolios netto positiv;
  (V2) schlechtester Subperioden-Sharpe >= Live-Spezifikation;
  (V3) Kosten-Drag <= Live-Spezifikation + 2pp p.a.
  Pflicht-Report (kein Gate): MaxDD-Vergleich, Turnover-Vergleich.

    python scripts/ewma_study.py                 # echter Lauf (Colab)
    python scripts/ewma_study.py --synthetic     # Smoke-Test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict.daily import backtest, build_positions, load_daily, report  # noqa: E402
from btc_predict.data import make_synthetic, resample  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT")
PERIODS = [("gesamt", None),
           ("2018-2020", ("2018-01-01", "2020-12-31")),
           ("2021-2022", ("2021-01-01", "2022-12-31")),
           ("2023-2024", ("2023-01-01", "2024-12-31")),
           ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]
COST = 10.0


def portfolio(vol_method: str, dailies: dict, start: str, end: str) -> pd.DataFrame:
    legs = {}
    for sym, d in dailies.items():
        frame = build_positions(d, None, allow_short=True, ensemble=True,
                                vol_method=vol_method)
        legs[sym] = backtest(frame, COST).loc[start:end]
    idx = legs[SYMBOLS[0]].index.intersection(legs[SYMBOLS[1]].index)
    return pd.DataFrame({
        c: sum(0.5 * legs[s].loc[idx, c] for s in SYMBOLS)
        for c in ("pnl", "ret", "pos", "turnover")}, index=idx)


def table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rng in PERIODS:
        seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
        if len(seg) > 30:
            rows.append(report(seg, label))
    return pd.DataFrame(rows)


def drag_pa(bt: pd.DataFrame) -> float:
    return float(bt["turnover"].sum() * COST / 10_000 / (len(bt) / 365) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    dailies = {}
    for i, sym in enumerate(SYMBOLS):
        if args.synthetic:
            dailies[sym] = resample(make_synthetic("2021-01-01", "2023-01-01",
                                                   seed=31 + i), "1d")
        else:
            warmup = max(pd.Timestamp(args.start) - pd.Timedelta(days=290),
                         pd.Timestamp("2017-08-01"))
            dailies[sym] = load_daily(sym, warmup.strftime("%Y-%m-%d"),
                                      args.end, Path("data"))
    start = args.start if not args.synthetic else "2021-10-01"
    end = args.end if not args.synthetic else "2023-01-01"

    results = {}
    for name, method in (("LIVE-SPEZ (rolling20)", "rolling20"),
                         ("KANDIDAT (ewma94)", "ewma94")):
        bt = portfolio(method, dailies, start, end)
        results[method] = bt
        print(f"\n== {name} | Ensemble 50/50, {COST:g} bps/Seite ==")
        print(table(bt).to_string(index=False))
        print(f"   Kosten-Drag: {drag_pa(bt):.2f}% p.a.")

    base, cand = results["rolling20"], results["ewma94"]
    tb, tc = table(base), table(cand)
    sub_b = tb[tb["period"] != "gesamt"]
    sub_c = tc[tc["period"] != "gesamt"]
    v1 = bool((sub_c["cagr_pct"] > 0).all())
    v2 = float(sub_c["sharpe"].min()) >= float(sub_b["sharpe"].min())
    v3 = drag_pa(cand) <= drag_pa(base) + 2.0
    print("\n== Gates (vorab registriert) ==")
    print(f"(V1) alle Subperioden positiv: {list(sub_c['cagr_pct'])} "
          f"-> {'PASS' if v1 else 'FAIL'}")
    print(f"(V2) schlechtester Subperioden-Sharpe {sub_c['sharpe'].min():.2f} "
          f">= {sub_b['sharpe'].min():.2f} -> {'PASS' if v2 else 'FAIL'}")
    print(f"(V3) Kosten-Drag {drag_pa(cand):.2f}% <= {drag_pa(base):.2f}% + 2pp "
          f"-> {'PASS' if v3 else 'FAIL'}")
    print(f"Report (kein Gate): MaxDD {tc.iloc[0]['max_dd_pct']}% vs "
          f"{tb.iloc[0]['max_dd_pct']}% | Sharpe {tc.iloc[0]['sharpe']} vs "
          f"{tb.iloc[0]['sharpe']}")
    verdict = ("EWMA-Sizing wird Live-Spezifikation"
               if (v1 and v2 and v3) else
               "rolling20 bleibt — EWMA beerdigt (kein Lambda-Tuning)")
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
