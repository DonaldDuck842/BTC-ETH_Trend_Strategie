"""Multi-Horizont-Pfadprognose per Quantil-Regression.

Statt rekursiv Kerze auf Kerze zu verketten (Fehler potenzieren sich, und
vorhergesagte Kerzen haben kein echtes Volumen/Orderflow als Feature-Input),
sagt fuer jeden Horizont h ein EIGENES Modell direkt vorher:

  - cum_ret_h : kumulative Log-Rendite bis Bar t+h (Quantile 10/50/90 %)
  - max_up    : maximale Hoch-Exkursion ueber alle h Bars ("oberer Docht")
  - max_dn    : maximale Tief-Exkursion ("unterer Docht")

Ergebnis pro Zeitpunkt: ein Median-Pfad mit Unsicherheitskegel plus
erwartete Docht-Spannen — die Richtung bleibt der schwere Teil, die
Groessenordnung (Vol) ist der gut praedizierbare Teil.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features


def build_path_targets(bars: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Zielvariablen je Zeile t, alle als Log-Distanz zum Close von t."""
    logc = np.log(bars["close"])
    t = pd.DataFrame(index=bars.index)
    for h in horizons:
        t[f"cum_ret_{h}"] = logc.shift(-h) - logc
    H = max(horizons)
    fwd_high = pd.concat([bars["high"].shift(-i) for i in range(1, H + 1)], axis=1).max(axis=1)
    fwd_low = pd.concat([bars["low"].shift(-i) for i in range(1, H + 1)], axis=1).min(axis=1)
    t["max_up"] = np.log(fwd_high / bars["close"])
    t["max_dn"] = np.log(fwd_low / bars["close"])
    return t


def make_xy(bars: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = build_features(bars)
    T = build_path_targets(bars, horizons)
    valid = X.notna().all(axis=1) & T.notna().all(axis=1)
    return X[valid], T[valid]


def train_path_models(X: pd.DataFrame, T: pd.DataFrame, quantiles: list[float],
                      model_cfg: dict) -> dict[tuple[str, float], lgb.LGBMRegressor]:
    models: dict[tuple[str, float], lgb.LGBMRegressor] = {}
    for col in T.columns:
        for q in quantiles:
            reg = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                num_leaves=model_cfg["num_leaves"],
                learning_rate=model_cfg["learning_rate"],
                n_estimators=model_cfg["n_estimators"],
                min_child_samples=model_cfg["min_child_samples"],
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                verbose=-1,
            )
            reg.fit(X, T[col])
            models[(col, q)] = reg
    return models


def predict_paths(models: dict, X: pd.DataFrame, horizons: list[int],
                  quantiles: list[float]) -> dict[str, pd.DataFrame]:
    """Prognosen je Ziel: DataFrame (Zeit x Quantil) mit Log-Renditen."""
    out: dict[str, pd.DataFrame] = {}
    targets = [f"cum_ret_{h}" for h in horizons] + ["max_up", "max_dn"]
    for col in targets:
        out[col] = pd.DataFrame(
            {q: models[(col, q)].predict(X) for q in quantiles}, index=X.index)
    return out


def pinball_loss(y: pd.Series, pred: pd.Series, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def calibration_report(T_test: pd.DataFrame, preds: dict[str, pd.DataFrame],
                       T_train: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Pro Ziel: Intervall-Abdeckung und Pinball-Loss-Skill vs. naive Baseline.

    Baseline = konstantes empirisches Quantil aus dem Training (reine
    Unbedingt-Verteilung). skill > 0 heisst: das Modell weiss mehr als
    die historische Verteilung.
    """
    q_lo, q_hi = min(quantiles), max(quantiles)
    rows = []
    for col, pred in preds.items():
        y = T_test[col]
        cover = float(((y >= pred[q_lo]) & (y <= pred[q_hi])).mean())
        skills = []
        for q in quantiles:
            base = pinball_loss(y, pd.Series(T_train[col].quantile(q), index=y.index), q)
            mod = pinball_loss(y, pred[q], q)
            skills.append(1 - mod / base if base > 0 else 0.0)
        med = pred[0.5] if 0.5 in pred.columns else pred[quantiles[len(quantiles) // 2]]
        dir_hit = float((np.sign(med) == np.sign(y))[y != 0].mean())
        rows.append({
            "target": col,
            "coverage": round(cover, 3),
            "coverage_soll": round(q_hi - q_lo, 2),
            "pinball_skill": round(float(np.mean(skills)), 4),
            "median_dir_hit": round(dir_hit, 3),
        })
    return pd.DataFrame(rows)
