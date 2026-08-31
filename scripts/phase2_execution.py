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

    python scripts/phase2_execution.py --start 2025-01-01 --end 2026-07-31
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
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-07-31")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["path"]
    horizons, quantiles = pcfg["horizons"], pcfg["quantiles"]
    data_dir = Path(cfg["data"]["data_dir"])

    # --- Tages-Signal (Champion: long/short, ohne Funding) ---
    warmup = (pd.Timestamp(args.start) - pd.Timedelta(days=80)).strftime("%Y-%m-%d")
    dbars = daily_mod.load_daily(cfg["data"]["symbol"], warmup, args.end, data_dir)
    frame = daily_mod.build_positions(dbars, None, allow_short=True)
    direction = np.sign(frame["pos"])
    flips = direction[direction != direction.shift(1)].loc[args.start:]
    flips = flips[flips != 0]  # nur Eintritte in eine Richtung timen
    print(f"Tages-Flips im Zeitraum: {len(flips)}")

    # --- Pfadmodell trainieren (Regime-Fenster, kein Overlap mit Zeitraum) ---
    for reg in cfg["regimes"]:
        if pd.Timestamp(reg["end"]) >= pd.Timestamp(args.start):
            raise ValueError(f"Regime {reg['name']} ueberschneidet Eval-Zeitraum")
    print("Pfadmodell-Training auf Regime-Fenstern ...")
    X_parts, T_parts = [], []
    for reg in cfg["regimes"]:
        bars = resample(ds_mod.load_bars(reg["start"], reg["end"], cfg), cfg["timeframe"])
        X, T = pm.make_xy(bars, horizons)
        X_parts.append(X)
        T_parts.append(T)
    models = pm.train_path_models(pd.concat(X_parts), pd.concat(T_parts),
                                  quantiles, pcfg["model"])

    # --- 5m-Bars + Prognosen fuer den Eval-Zeitraum ---
    bars5 = resample(ds_mod.load_bars(args.start, args.end, cfg), cfg["timeframe"])
    X5 = pm.build_features(bars5)
    valid = X5.notna().all(axis=1)
    key = f"cum_ret_{cfg['label']['horizon_bars']}"
    med_pred = pd.Series(models[(key, 0.5)].predict(X5[valid]), index=X5.index[valid])

    # --- Flips durchgehen ---
    rows = []
    for day, dir_ in flips.items():
        # Signal entsteht am Close von Tag D -> fruehester Fill ist D+1 00:00.
        day0 = day.normalize() + pd.Timedelta(days=1)
        day_bars = bars5.loc[day0:day0 + pd.Timedelta(hours=4)]
        if len(day_bars) < MAX_WAIT_BARS + 1:
            continue
        naive = float(day_bars["open"].iloc[0])
        fill, wait = timed_fill(day_bars, med_pred, float(dir_))
        rows.append({
            "day": day0.date(), "dir": int(dir_),
            "improvement_bps": float(dir_ * (naive - fill) / naive * 10_000),
            "wait_bars": wait,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        print("Keine auswertbaren Flips — Zeitraum/Daten pruefen.")
        return

    n = len(res)
    mean_imp = res["improvement_bps"].mean()
    med_imp = res["improvement_bps"].median()
    win = (res["improvement_bps"] > 0).mean()
    years = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days / 365
    annual_bps = mean_imp * n / years

    print(f"\n== Ausfuehrungs-Timing vs. naive Ausfuehrung ({n} Flips) ==")
    print(f"   Verbesserung  : Ø {mean_imp:+.2f} bps | Median {med_imp:+.2f} bps "
          f"pro Flip")
    print(f"   Trefferquote  : {win:.1%} der Flips besser als naiv")
    print(f"   Ø Wartezeit   : {res['wait_bars'].mean():.1f} Bars")
    print(f"   Hochgerechnet : {annual_bps:+.0f} bps ({annual_bps/100:+.2f}%) pro Jahr")
    # Signifikanz grob: t-Statistik des Mittelwerts
    t = mean_imp / (res["improvement_bps"].std() / np.sqrt(n)) if n > 3 else float("nan")
    print(f"   t-Statistik   : {t:+.2f}  (|t| > 2 = belastbar)")
    print("\n   Zustandekommen pro Flip -> results/phase2_flips.csv")
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/phase2_flips.csv", index=False)


if __name__ == "__main__":
    main()
