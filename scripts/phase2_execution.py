#!/usr/bin/env python3
"""Phase 2: Misst, ob das validierte 5m-Pfadmodell die Ausfuehrung der
Tages-Strategie-Flips verbessert.

Regel (vorab festgelegt, kein Tuning):
  Bei einem Tages-Flip (Richtungswechsel der Champion-Variante long/short)
  wird nicht blind am ersten 5m-Bar des Tages gefuellt (naive Ausfuehrung),
  sondern bis zu 12 Bars (1h) gewartet: Solange der Median der prognosti-
  zierten 15-Minuten-Bewegung GEGEN die Einstiegsrichtung zeigt (erwarteter
  Dip beim Long-Einstieg), wird gewartet; dreht die Prognose, wird zum
  naechsten Bar-Open gefuellt. Spaetestens bei Bar 12 wird zwangsgefuellt.

Metrik: Preisverbesserung vs. naive Ausfuehrung in bps pro Flip.
Ausgewertet wird ueber ALLE Zeitraeume, in denen das Pfadmodell nicht
trainiert wurde (2018-2020, 2023-01..09, 2024-Q4, 2025-01..2026-07),
fuer maximale Stichprobengroesse bei sauberem Out-of-Sample.

    python scripts/phase2_execution.py
    python scripts/phase2_execution.py --windows 2025-01-01:2026-07-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import daily as daily_mod  # noqa: E402
from btc_predict import dataset as ds_mod  # noqa: E402
from btc_predict import path_model as pm  # noqa: E402
from btc_predict.data import resample  # noqa: E402

MAX_WAIT_BARS = 12  # 1 Stunde auf 5m


def timed_fill(day_bars: pd.DataFrame, med_pred: pd.Series, direction: float) -> tuple[float, int]:
    """Gefuellt wird am Open des Bars NACH dem Entscheidungs-Bar."""
    idx = day_bars.index
    for b in range(min(MAX_WAIT_BARS, len(idx) - 1)):
        pred = med_pred.get(idx[b], np.nan)
        if np.isnan(pred) or direction * pred >= 0:
            return float(day_bars["open"].iloc[b + 1]), b + 1
    b = min(MAX_WAIT_BARS, len(idx) - 1)
    return float(day_bars["open"].iloc[b]), b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--windows", nargs="*", default=None,
                    help="Eval-Fenster als start:end-Paare; Default: alle "
                         "Zeitraeume ausserhalb der Pfadmodell-Trainingsregime")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["path"]
    horizons, quantiles = pcfg["horizons"], pcfg["quantiles"]
    data_dir = Path(cfg["data"]["data_dir"])

    # Alle Zeitraeume, in denen das Pfadmodell NICHT trainiert wurde:
    if args.windows:
        windows = [tuple(w.split(":")) for w in args.windows]
    else:
        windows = [("2018-01-01", "2020-12-31"),
                   ("2023-01-01", "2023-09-30"),
                   ("2024-10-01", "2024-12-31"),
                   ("2025-01-01", "2026-07-31")]
    for w0, w1 in windows:
        for reg in cfg["regimes"]:
            r0, r1 = pd.Timestamp(reg["start"]), pd.Timestamp(reg["end"])
            if pd.Timestamp(w0) <= r1 and pd.Timestamp(w1) >= r0:
                raise ValueError(f"Fenster {w0}..{w1} ueberschneidet "
                                 f"Trainings-Regime {reg['name']}")

    # --- Tages-Signal (Champion: long/short, ohne Funding) ---
    warmup = (pd.Timestamp(windows[0][0]) - pd.Timedelta(days=80)).strftime("%Y-%m-%d")
    dbars = daily_mod.load_daily(cfg["data"]["symbol"], warmup, windows[-1][1], data_dir)
    frame = daily_mod.build_positions(dbars, None, allow_short=True)
    direction = np.sign(frame["pos"])
    all_flips = direction[direction != direction.shift(1)]
    all_flips = all_flips[all_flips != 0]  # nur Eintritte in eine Richtung timen

    print("Pfadmodell-Training auf Regime-Fenstern ...")
    X_parts, T_parts = [], []
    for reg in cfg["regimes"]:
        bars = resample(ds_mod.load_bars(reg["start"], reg["end"], cfg), cfg["timeframe"])
        X, T = pm.make_xy(bars, horizons)
        X_parts.append(X)
        T_parts.append(T)
    models = pm.train_path_models(pd.concat(X_parts), pd.concat(T_parts),
                                  quantiles, pcfg["model"])

    key = f"cum_ret_{cfg['label']['horizon_bars']}"
    all_rows = []
    for w0, w1 in windows:
        bars5 = resample(ds_mod.load_bars(w0, w1, cfg), cfg["timeframe"])
        X5 = pm.build_features(bars5)
        valid = X5.notna().all(axis=1)
        med_pred = pd.Series(models[(key, 0.5)].predict(X5[valid]),
                             index=X5.index[valid])
        flips = all_flips.loc[w0:w1]
        for day, dir_ in flips.items():
            # Signal entsteht am Close von Tag D -> fruehester Fill ist D+1 00:00.
            day0 = day.normalize() + pd.Timedelta(days=1)
            day_bars = bars5.loc[day0:day0 + pd.Timedelta(hours=4)]
            if len(day_bars) < MAX_WAIT_BARS + 1:
                continue
            naive = float(day_bars["open"].iloc[0])
            fill, wait = timed_fill(day_bars, med_pred, float(dir_))
            all_rows.append({
                "window": f"{w0[:7]}..{w1[:7]}", "day": day0.date(), "dir": int(dir_),
                "improvement_bps": float(dir_ * (naive - fill) / naive * 10_000),
                "wait_bars": wait,
            })
        print(f"   Fenster {w0}..{w1}: {len(flips)} Flips ausgewertet")
    res = pd.DataFrame(all_rows)
    if res.empty:
        print("Keine auswertbaren Flips — Zeitraum/Daten pruefen.")
        return

    def stats(sub: pd.DataFrame) -> dict:
        n = len(sub)
        mean_imp = sub["improvement_bps"].mean()
        t = mean_imp / (sub["improvement_bps"].std() / np.sqrt(n)) if n > 3 else float("nan")
        return {"n_flips": n,
                "mean_bps": round(mean_imp, 2),
                "median_bps": round(sub["improvement_bps"].median(), 2),
                "hit_rate": round(float((sub["improvement_bps"] > 0).mean()), 3),
                "t_stat": round(t, 2)}

    per_win = pd.DataFrame([{"window": w, **stats(g)}
                            for w, g in res.groupby("window")])
    pooled = stats(res)

    print(f"\n== Ausfuehrungs-Timing vs. naive Ausfuehrung, je Fenster ==")
    print(per_win.to_string(index=False))
    total_years = sum((pd.Timestamp(w1) - pd.Timestamp(w0)).days for w0, w1 in windows) / 365
    annual_bps = pooled["mean_bps"] * pooled["n_flips"] / total_years
    print(f"\n== Gepoolt ({pooled['n_flips']} Flips ueber {total_years:.1f} Jahre) ==")
    print(f"   Verbesserung  : Ø {pooled['mean_bps']:+.2f} bps | "
          f"Median {pooled['median_bps']:+.2f} bps pro Flip")
    print(f"   Trefferquote  : {pooled['hit_rate']:.1%}")
    print(f"   Hochgerechnet : {annual_bps:+.0f} bps ({annual_bps/100:+.2f}%) pro Jahr")
    print(f"   t-Statistik   : {pooled['t_stat']:+.2f}  (|t| > 2 = belastbar)")
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/phase2_flips.csv", index=False)
    print("\n   Einzel-Flips -> results/phase2_flips.csv")


if __name__ == "__main__":
    main()
