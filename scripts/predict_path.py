#!/usr/bin/env python3
"""Multi-Horizont-Pfadprognose: Training auf den Regime-Fenstern, Kalibrierungs-
Test auf einem Out-of-Sample-Monat, plus Beispiel-Fanchart.

    python scripts/predict_path.py --test-month 2025-05
    python scripts/predict_path.py --synthetic
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
from btc_predict import path_model as pm  # noqa: E402
from btc_predict.data import resample  # noqa: E402


def plot_example(bars: pd.DataFrame, preds: dict, horizons: list[int],
                 quantiles: list[float], at: pd.Timestamp, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_lo, q_hi = min(quantiles), max(quantiles)
    hist = bars.loc[:at].tail(40)
    c0 = float(bars.loc[at, "close"])
    tf_min = int((bars.index[1] - bars.index[0]).total_seconds() // 60)
    fut_idx = [at + pd.Timedelta(minutes=tf_min * h) for h in horizons]

    med = [c0 * np.exp(preds[f"cum_ret_{h}"].loc[at, 0.5]) for h in horizons]
    lo = [c0 * np.exp(preds[f"cum_ret_{h}"].loc[at, q_lo]) for h in horizons]
    hi = [c0 * np.exp(preds[f"cum_ret_{h}"].loc[at, q_hi]) for h in horizons]
    wick_hi = c0 * np.exp(preds["max_up"].loc[at, 0.5])
    wick_lo = c0 * np.exp(preds["max_dn"].loc[at, 0.5])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(hist.index, hist["close"], color="#6b7280", lw=1.5, label="Close (real)")
    fx = [at] + fut_idx
    ax.fill_between(fx, [c0] + lo, [c0] + hi, color="#3b82f6", alpha=0.15,
                    label=f"{int(q_lo*100)}–{int(q_hi*100)}%-Kegel")
    ax.plot(fx, [c0] + med, color="#3b82f6", lw=2, marker="o", ms=4,
            label="Median-Pfad (prognostiziert)")
    ax.hlines([wick_hi, wick_lo], xmin=at, xmax=fut_idx[-1], colors="#3b82f6",
              linestyles=":", lw=1.2)
    ax.annotate("erw. Hoch-Exkursion", (fut_idx[-1], wick_hi), fontsize=8,
                color="#374151", ha="right", va="bottom")
    ax.annotate("erw. Tief-Exkursion", (fut_idx[-1], wick_lo), fontsize=8,
                color="#374151", ha="right", va="top")
    ax.axvline(at, color="#d1d5db", lw=1)
    ax.set_title(f"Pfadprognose ab {at:%Y-%m-%d %H:%M} UTC "
                 f"({len(horizons)}x{tf_min}min voraus)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#f3f4f6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"   Beispiel-Fanchart gespeichert: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test-month", default="2025-05")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plot-at", default=None,
                    help="Zeitstempel fuer das Beispiel-Fanchart (Default: Monatsmitte)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["path"]
    horizons, quantiles = pcfg["horizons"], pcfg["quantiles"]

    t0 = pd.Timestamp(args.test_month + "-01")
    t1 = t0 + pd.offsets.MonthEnd(1)
    ds_mod.assert_no_overlap(t0, t1, cfg)

    print(f"== Pfad-Training: {len(cfg['regimes'])} Regime, "
          f"{len(horizons)} Horizonte x {len(quantiles)} Quantile ==")
    X_parts, T_parts = [], []
    for reg in cfg["regimes"]:
        bars = resample(ds_mod.load_bars(reg["start"], reg["end"], cfg,
                                         synthetic=args.synthetic), cfg["timeframe"])
        X, T = pm.make_xy(bars, horizons)
        X_parts.append(X)
        T_parts.append(T)
        print(f"   {reg['name']}: {len(X):,} Bars")
    X_train, T_train = pd.concat(X_parts), pd.concat(T_parts)
    models = pm.train_path_models(X_train, T_train, quantiles, pcfg["model"])
    print(f"   {len(models)} Quantil-Modelle trainiert")

    print(f"\n== Kalibrierungs-Test: {args.test_month} (out-of-sample) ==")
    bars_test = resample(ds_mod.load_bars(t0.strftime("%Y-%m-%d"),
                                          t1.strftime("%Y-%m-%d"), cfg,
                                          synthetic=args.synthetic), cfg["timeframe"])
    X_test, T_test = pm.make_xy(bars_test, horizons)
    emb = cfg.get("embargo_bars", 0)
    X_test, T_test = X_test.iloc[emb:], T_test.iloc[emb:]

    preds = pm.predict_paths(models, X_test, horizons, quantiles)
    report = pm.calibration_report(T_test, preds, T_train, quantiles)
    print(report.to_string(index=False))
    print("\n   Lesart: coverage nahe coverage_soll = Kegel ehrlich kalibriert;")
    print("   pinball_skill > 0 = Modell schlaegt die unbedingte Verteilung;")
    print("   median_dir_hit ~0.5 = Richtung bleibt schwer (erwartet).")

    at = pd.Timestamp(args.plot_at, tz="UTC") if args.plot_at else X_test.index[len(X_test) // 2]
    plot_example(bars_test, preds, horizons, quantiles, at,
                 Path("results/path_example.png"))


if __name__ == "__main__":
    main()
