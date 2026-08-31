#!/usr/bin/env python3
"""Kombinierte Strategie: Richtungsmodell + Pfadmodell als Groessen-Gate.

Idee: Das Richtungssignal (p_up - p_dn) ist schwach (~52-53% Hit), das
Groessensignal (prognostizierte Exkursionen) ist stark. Gehandelt wird nur,
wenn BEIDES stimmt: Richtungs-Edge ueber der Schwelle UND die prognostizierte
Exkursion in Signalrichtung deckt ein Vielfaches der Round-Trip-Kosten.

    python scripts/run_combined.py --test-month 2025-05
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
from btc_predict import path_model as pm  # noqa: E402
from btc_predict.data import resample  # noqa: E402


def skew_edge(preds: dict) -> pd.Series:
    """Kegel-Asymmetrie als Richtungssignal in [-1, 1]: prognostiziert das
    Pfadmodell deutlich mehr Aufwaerts- als Abwaerts-Exkursion, ist das ein
    aktives Richtungsurteil — das Gegenmittel zum passiven Median."""
    up = preds["max_up"][0.5]
    dn = preds["max_dn"][0.5]
    return ((up + dn) / (up - dn).replace(0, np.nan)).fillna(0.0).clip(-1, 1)


def gated_signal(edge: pd.Series, preds: dict, enter_thr: float,
                 k_cost: float, cost_bps_per_side: float, bt: dict) -> pd.Series:
    """Edge (beliebige Quelle), genullt wo die prognostizierte Exkursion die
    Kosten nicht deckt; danach normale Hysterese-Logik."""
    need = k_cost * 2 * cost_bps_per_side / 10_000  # k x Round-Trip, in Log-Return
    up_ok = preds["max_up"][0.5] >= need
    dn_ok = preds["max_dn"][0.5] <= -need
    gated = edge.where((edge > 0) & up_ok | (edge < 0) & dn_ok, 0.0)
    # Proba-Frame so rekonstruieren, dass make_signal exakt die gegatete Edge
    # sieht (p_up - p_down == gated) und die Schwellen-Skala erhalten bleibt.
    gp = pd.DataFrame(index=edge.index)
    gp["p_up"] = gated.clip(lower=0.0)
    gp["p_flat"] = 0.0
    gp["p_down"] = (-gated).clip(lower=0.0)
    return ev_mod.make_signal(gp, enter_thr, bt["allow_short"],
                              exit_threshold=bt.get("exit_threshold_frac", 0.5) * enter_thr,
                              min_hold_bars=bt.get("min_hold_bars", 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test-month", default="2025-05")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["path"]
    bt = cfg["backtest"]
    horizons, quantiles = pcfg["horizons"], pcfg["quantiles"]

    t0 = pd.Timestamp(args.test_month + "-01")
    t1 = t0 + pd.offsets.MonthEnd(1)
    ds_mod.assert_no_overlap(t0, t1, cfg)

    print("== Training Richtungsmodell ==")
    train_ds = ds_mod.build_training_set(cfg, synthetic=args.synthetic)
    clf = model_mod.train(train_ds, cfg["model"])

    print("== Training Pfadmodell ==")
    X_parts, T_parts = [], []
    for reg in cfg["regimes"]:
        bars = resample(ds_mod.load_bars(reg["start"], reg["end"], cfg,
                                         synthetic=args.synthetic), cfg["timeframe"])
        X, T = pm.make_xy(bars, horizons)
        X_parts.append(X)
        T_parts.append(T)
    models = pm.train_path_models(pd.concat(X_parts), pd.concat(T_parts),
                                  quantiles, pcfg["model"])

    print(f"\n== Kombinierter Test: {args.test_month} ==")
    test_ds = ds_mod.load_window(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"),
                                 cfg, synthetic=args.synthetic)
    emb = cfg.get("embargo_bars", 0)
    test_ds = ds_mod.Dataset(test_ds.X.iloc[emb:], test_ds.y.iloc[emb:],
                             test_ds.fwd_ret.iloc[emb:], test_ds.regime.iloc[emb:])
    proba = model_mod.predict_proba(clf, test_ds.X)
    preds = pm.predict_paths(models, test_ds.X, horizons, quantiles)

    bar_ret = test_ds.X["ret_1"]
    bh = float(bar_ret.sum() * 100)
    print(f"   Buy-and-Hold des Monats: {bh:+.2f}%  (Drift-Referenz)")

    dir_edge = proba["p_up"] - proba["p_down"]
    sources = {
        "direction": dir_edge,                       # Klassifikator-Edge
        "skew": skew_edge(preds),                    # Kegel-Asymmetrie (Pfadmodell)
        "blend": (dir_edge + skew_edge(preds)) / 2,  # beide Urteile gemittelt
    }

    rows = []
    for src_name, edge in sources.items():
        for cost in bt.get("cost_scenarios_bps_per_side", [0.0, 1.5, 5.0]):
            for thr in bt.get("sweep_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5]):
                for k in (0.0, 2.0):
                    sig = gated_signal(edge, preds, thr, k, max(cost, 1e-9), bt)
                    sim = ev_mod.simulate(sig, bar_ret, cost, cfg["timeframe"])
                    long_share = float((sig > 0).sum() / max((sig != 0).sum(), 1))
                    rows.append({
                        "source": src_name, "cost_bps": cost, "enter_thr": thr,
                        "k_cost_gate": k,
                        "round_trips": sim["round_trips"],
                        "gross_bps_per_rt": round(sim["gross_bps_per_round_trip"], 2),
                        "net_ret_pct": round(sim["total_return_pct"], 2),
                        "sharpe_net": round(sim["sharpe_net"], 2),
                        "long_share": round(long_share, 2),
                    })
    sweep = pd.DataFrame(rows)
    out_csv = Path("results/combined_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_csv, index=False)
    print(f"   Voller Sweep ({len(sweep)} Zellen) -> {out_csv}")

    print("\n   Beste Zelle je Signalquelle x Kostenszenario:")
    best_rows = sweep.loc[sweep.groupby(["source", "cost_bps"])["sharpe_net"].idxmax()]
    print(best_rows.to_string(index=False))
    print("\n   long_share nahe 1.0 bei positivem Buy-and-Hold = Verdacht auf")
    print("   reine Drift-Vereinnahmung statt Signal — auf anderem Monat gegenpruefen.")


if __name__ == "__main__":
    main()
