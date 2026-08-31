#!/usr/bin/env python3
"""Pivot Phase 1 — Tages-Strategie: Momentum + Vol-Targeting + Funding-Overlay.

Richtung kommt vom Tageschart (robuster, billiger: wenige Trades), das
validierte 5m-Pfadmodell uebernimmt spaeter Einstiegs-Timing und Stops
(Phase 2). Dieser Backtest validiert die Tagesebene fuer sich, mit
Subperioden-Tabelle — der Haertetest ist die Post-2021-Periode.

Regeln (bewusst simpel, keine Optimierung):
  - Momentum:  Close > SMA(50) UND Close > Donchian-Mitte(20) -> long, sonst flat
  - Sizing:    Ziel-Vol 40% p.a. / realisierte 20d-Vol, gedeckelt bei 1.0 (kein Hebel)
  - Funding:   3d-Mittel der Funding-Rate ueber rollierendem 95%-Perzentil
               (1 Jahr) -> flat (ueberhitzte Long-Positionierung)
  - Kosten:    10 bps pro Seite (Spot-Taker inkl. Slippage)

    python scripts/daily_strategy.py                # Standard 2018-01 .. 2026-07
    python scripts/daily_strategy.py --no-funding   # ohne Funding-Overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from btc_predict.daily import (  # noqa: E402
    backtest, build_positions, load_daily, load_funding, report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    data_dir = Path("data")
    daily = load_daily("BTCUSDT", args.start, args.end, data_dir)
    print(f"Tagesdaten: {daily.index[0].date()} .. {daily.index[-1].date()} "
          f"({len(daily)} Tage)")

    funding = None
    if not args.no_funding:
        try:
            funding = load_funding("BTCUSDT", data_dir)
            print(f"Funding-Historie: {funding.index[0].date()} .. "
                  f"{funding.index[-1].date()} ({len(funding)} Punkte)")
        except Exception as exc:  # Funding ist Overlay, kein Muss
            print(f"Funding nicht ladbar ({exc}) — laufe ohne Overlay weiter.")

    periods = [("gesamt", None),
               ("2018-2020", ("2018-01-01", "2020-12-31")),
               ("2021-2022", ("2021-01-01", "2022-12-31")),
               ("2023-2024", ("2023-01-01", "2024-12-31")),
               ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]
    # Vorab deklarierte Varianten — EIN Vergleichslauf, kein Parameter-Sweep:
    variants = [("long/flat", False, None),
                ("long/flat + funding", False, funding),
                ("long/short", True, None),
                ("long/short + funding", True, funding)]
    for name, allow_short, fnd in variants:
        if fnd is None and "funding" in name:
            continue
        frame = build_positions(daily, fnd, allow_short=allow_short)
        bt = backtest(frame, args.cost_bps)
        rows = []
        for label, rng in periods:
            seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
            if len(seg) > 30:
                rows.append(report(seg, label))
        print(f"\n== Variante: {name} | Kosten {args.cost_bps} bps/Seite ==")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nLesart: Massstab ist v.a. 2025-2026 — positiver (oder klar weniger")
    print("negativer) Sharpe und deutlich geringerer MaxDD als Buy-and-Hold.")
    print("Die Varianten sind vorab deklariert; es wird NICHT die beste pro")
    print("Periode gepickt, sondern EINE Variante fuer alle Perioden gewaehlt.")


if __name__ == "__main__":
    main()
