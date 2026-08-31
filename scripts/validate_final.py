#!/usr/bin/env python3
"""Finale Validierung: EINGEFRORENE Konfiguration, mehrere unangetastete Monate.

Kein Sweep, kein Tuning — die Parameter kommen unveraendert aus der Config.
Dieser Lauf ist die letzte Instanz: Danach wird auf diesen Monaten nicht
mehr iteriert. Mehrere Monate (inkl. Abwaerts-Monaten) klaeren zugleich,
ob die Strategie Signal handelt oder nur Markt-Drift vereinnahmt.

    python scripts/validate_final.py --months 2025-01 2025-02 2025-04 2025-06 2025-07
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import dataset as ds_mod  # noqa: E402
from btc_predict import evaluate as ev_mod  # noqa: E402
from btc_predict import model as model_mod  # noqa: E402

DEFAULT_MONTHS = ["2025-01", "2025-02", "2025-04", "2025-06", "2025-07"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    bt = cfg["backtest"]
    tf = cfg["timeframe"]
    emb = cfg.get("embargo_bars", 0)

    for m in args.months:
        t0 = pd.Timestamp(m + "-01")
        ds_mod.assert_no_overlap(t0, t0 + pd.offsets.MonthEnd(1), cfg)

    print(f"== Eingefrorene Konfiguration ==")
    print(f"   Signal: p_up - p_down > {bt['signal_threshold']} | "
          f"Exit-Hysterese {bt['exit_threshold']} | Min-Hold {bt['min_hold_bars']} Bars")
    print(f"   Kosten-Szenarien: Maker 1.5 / Taker 5.0 bps pro Seite\n")

    print("== Training (einmalig, alle Regime) ==")
    train_ds = ds_mod.build_training_set(cfg, synthetic=args.synthetic)
    clf = model_mod.train(train_ds, cfg["model"])

    rows = []
    for m in args.months:
        t0 = pd.Timestamp(m + "-01")
        t1 = t0 + pd.offsets.MonthEnd(1)
        test_ds = ds_mod.load_window(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"),
                                     cfg, synthetic=args.synthetic)
        test_ds = ds_mod.Dataset(test_ds.X.iloc[emb:], test_ds.y.iloc[emb:],
                                 test_ds.fwd_ret.iloc[emb:], test_ds.regime.iloc[emb:])
        proba = model_mod.predict_proba(clf, test_ds.X)
        signal = ev_mod.make_signal(proba, bt["signal_threshold"], bt["allow_short"],
                                    exit_threshold=bt["exit_threshold"],
                                    min_hold_bars=bt["min_hold_bars"])
        bar_ret = test_ds.X["ret_1"]
        row = {"month": m, "buy_hold_pct": round(float(bar_ret.sum() * 100), 2)}
        for label, cost in (("maker", 1.5), ("taker", 5.0)):
            sim = ev_mod.simulate(signal, bar_ret, cost, tf)
            row[f"net_{label}_pct"] = round(sim["total_return_pct"], 2)
            row[f"sharpe_{label}"] = round(sim["sharpe_net"], 2)
        sim_ref = ev_mod.simulate(signal, bar_ret, 1.5, tf)
        row["round_trips"] = sim_ref["round_trips"]
        row["gross_bps_rt"] = round(sim_ref["gross_bps_per_round_trip"], 2)
        row["long_share"] = round(float((signal > 0).sum() / max((signal != 0).sum(), 1)), 2)
        rows.append(row)

        results_stub = {
            "net": {"sharpe_net": row["sharpe_maker"],
                    "total_return_pct": row["net_maker_pct"],
                    "n_trades": int(sim_ref["round_trips"])},
            "classification": {"balanced_accuracy": float("nan")},
        }
        n = ev_mod.log_evaluation(cfg, m, results_stub)
        if n > 1:
            print(f"   ⚠ {m} wurde bereits {n}x bewertet — kein frischer Monat!")

    df = pd.DataFrame(rows)
    print("\n== Ergebnis je Monat ==")
    print(df.to_string(index=False))

    print("\n== Aggregat ==")
    for label in ("maker", "taker"):
        tot = df[f"net_{label}_pct"].sum()
        pos = int((df[f"net_{label}_pct"] > 0).sum())
        print(f"   {label}: Summe {tot:+.2f}% | positive Monate {pos}/{len(df)}")
    corr = float(np.corrcoef(df["buy_hold_pct"], df["net_maker_pct"])[0, 1]) \
        if len(df) > 2 else float("nan")
    print(f"   Korrelation Netto(maker) vs. Buy-and-Hold: {corr:+.2f}")
    print("   (nahe +1 = Strategie reitet nur die Monats-Drift; nahe 0 = eigenes Signal)")
    print("\n   Diese Monate sind jetzt verbraucht: NICHT erneut darauf iterieren.")


if __name__ == "__main__":
    main()
