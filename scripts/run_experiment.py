#!/usr/bin/env python3
"""Ein kompletter Experiment-Lauf: Training auf den Regime-Fenstern,
Test auf einem frei waehlbaren Out-of-Sample-Monat.

Beispiele:
    python scripts/run_experiment.py --test-month 2025-03
    python scripts/run_experiment.py --test-month 2025-03 --config configs/default.yaml
    python scripts/run_experiment.py --synthetic          # Pipeline-Smoke-Test ohne Netz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import dataset as ds_mod  # noqa: E402
from btc_predict import evaluate as ev_mod  # noqa: E402
from btc_predict import model as model_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test-month", default="2025-03",
                    help="Out-of-Sample-Testmonat, Format YYYY-MM")
    ap.add_argument("--synthetic", action="store_true",
                    help="Synthetische Daten statt Binance (Pipeline-Test)")
    ap.add_argument("--save-model", default="results/model.txt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    t0 = pd.Timestamp(args.test_month + "-01")
    t1 = t0 + pd.offsets.MonthEnd(1)
    ds_mod.assert_no_overlap(t0, t1, cfg)

    print(f"== Training: {len(cfg['regimes'])} Regime, Timeframe {cfg['timeframe']} ==")
    train_ds = ds_mod.build_training_set(cfg, synthetic=args.synthetic)
    for name in train_ds.regime.unique():
        n = int((train_ds.regime == name).sum())
        print(f"   {name}: {n:,} Bars")
    print(f"   Klassenverteilung: "
          f"{train_ds.y.value_counts(normalize=True).round(3).to_dict()}")

    clf = model_mod.train(train_ds, cfg["model"])
    print(f"   Best Iteration: {clf.best_iteration_}")
    print("   Top-Features:")
    for feat, imp in model_mod.feature_importance(clf).head(8).items():
        print(f"     {feat:>20}: {imp}")

    print(f"\n== Test: {args.test_month} (out-of-sample) ==")
    test_ds = ds_mod.load_window(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"),
                                 cfg, synthetic=args.synthetic)
    # Embargo: die ersten Bars verwerfen, deren Features in den Vormonat blicken
    emb = cfg.get("embargo_bars", 0)
    test_ds = ds_mod.Dataset(test_ds.X.iloc[emb:], test_ds.y.iloc[emb:],
                             test_ds.fwd_ret.iloc[emb:], test_ds.regime.iloc[emb:])

    proba = model_mod.predict_proba(clf, test_ds.X)
    results = ev_mod.evaluate_month(test_ds, proba, cfg)

    cls, net = results["classification"], results["net"]
    print(f"   Balanced Accuracy : {cls['balanced_accuracy']:.4f}  (Zufall = 0.3333)")
    print(f"   Directional Hits  : {cls['directional_hit_rate']:.4f} "
          f"bei {cls['n_directional_calls']:,} Richtungs-Calls")
    print(f"   Trades            : {net['n_trades']:,} | Time in Market: "
          f"{net['time_in_market']:.1%}")
    print(f"   Ø Haltedauer      : {net['avg_hold_bars']:.1f} Bars | "
          f"Brutto/Round-Trip: {net['gross_bps_per_round_trip']:+.2f} bps")
    print(f"   Brutto-Return     : {net['total_gross_return_pct']:+.2f}%  "
          f"(Sharpe brutto {results['gross_sharpe']:+.2f})")
    print(f"   Kosten bezahlt    : {net['cost_paid_pct']:.2f}%")
    print(f"   NETTO-Return      : {net['total_return_pct']:+.2f}%  "
          f"(Sharpe netto {net['sharpe_net']:+.2f})")

    print("\n== Sweep: Einstiegsschwelle x Kosten (Diagnose, kein Parameter-Shopping) ==")
    sweep = ev_mod.threshold_cost_sweep(test_ds, proba, cfg)
    print(sweep.to_string(index=False))

    n_evals = ev_mod.log_evaluation(cfg, args.test_month, results)
    if n_evals > 1:
        print(f"\n   ⚠ SNOOPING-WARNUNG: Monat {args.test_month} wurde jetzt "
              f"{n_evals}x bewertet. Jede Iteration auf demselben Testmonat "
              f"macht ihn ein Stueck mehr zum Trainingsset. Fuer die finale "
              f"Aussage einen bisher unangetasteten Monat nehmen.")

    if args.save_model:
        model_mod.save(clf, Path(args.save_model))
        print(f"\n   Modell gespeichert: {args.save_model}")


if __name__ == "__main__":
    main()
