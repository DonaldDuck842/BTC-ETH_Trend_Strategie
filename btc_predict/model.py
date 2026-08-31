"""Training des Multiclass-Klassifikators (LightGBM)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import Dataset

# Klassen-Encoding: -1 -> 0 (down), 0 -> 1 (flat), 1 -> 2 (up)
CLASS_ORDER = [-1.0, 0.0, 1.0]


def _split_by_time(ds: Dataset, val_fraction: float) -> tuple[Dataset, Dataset]:
    """Validierungs-Split: das zeitliche Ende JEDES Regimes, kein Random-Split.

    So enthaelt die Validierung alle Regime und liegt strikt nach ihrem
    jeweiligen Trainingsteil — Random-Splits waeren bei ueberlappenden
    Labels Leakage.
    """
    val_mask = pd.Series(False, index=ds.X.index)
    for name in ds.regime.unique():
        idx = ds.regime.index[ds.regime == name]
        n_val = max(1, int(len(idx) * val_fraction))
        val_mask.loc[idx[-n_val:]] = True
    tr = Dataset(ds.X[~val_mask], ds.y[~val_mask], ds.fwd_ret[~val_mask], ds.regime[~val_mask])
    va = Dataset(ds.X[val_mask], ds.y[val_mask], ds.fwd_ret[val_mask], ds.regime[val_mask])
    return tr, va


def _move_weights(fwd_ret: pd.Series) -> pd.Series:
    """Gewichtet Bars proportional zur Groesse der Folgebewegung (gedeckelt).

    Gegenmittel zur Median-Passivitaet: Das Modell soll die Bars richtig
    machen, auf denen Geld verdient/verloren wird, statt die vielen
    ruhigen Bars zu optimieren."""
    scale = fwd_ret.abs().median()
    if scale <= 0:
        return pd.Series(1.0, index=fwd_ret.index)
    return (fwd_ret.abs() / scale).clip(0.25, 4.0)


def train(ds: Dataset, model_cfg: dict) -> lgb.LGBMClassifier:
    tr, va = _split_by_time(ds, model_cfg.get("validation_fraction", 0.1))
    enc = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_tr = tr.y.map(enc).astype(int)
    y_va = va.y.map(enc).astype(int)
    w_tr = w_va = None
    if model_cfg.get("weight_by_move", False):
        w_tr = _move_weights(tr.fwd_ret)
        w_va = _move_weights(va.fwd_ret)

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        num_leaves=model_cfg["num_leaves"],
        learning_rate=model_cfg["learning_rate"],
        n_estimators=model_cfg["n_estimators"],
        min_child_samples=model_cfg["min_child_samples"],
        subsample=model_cfg["subsample"],
        subsample_freq=1,
        colsample_bytree=model_cfg["colsample_bytree"],
        reg_lambda=model_cfg["reg_lambda"],
        verbose=-1,
    )
    clf.fit(
        tr.X, y_tr,
        sample_weight=w_tr,
        eval_set=[(va.X, y_va)],
        eval_sample_weight=[w_va] if w_va is not None else None,
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"], verbose=False)],
    )
    return clf


def predict_proba(clf: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    proba = clf.predict_proba(X)
    return pd.DataFrame(proba, index=X.index, columns=["p_down", "p_flat", "p_up"])


def feature_importance(clf: lgb.LGBMClassifier) -> pd.Series:
    imp = pd.Series(clf.feature_importances_, index=clf.feature_name_)
    return imp.sort_values(ascending=False)


def save(clf: lgb.LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def predict_proba_booster(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    proba = booster.predict(X.values)
    return pd.DataFrame(np.asarray(proba), index=X.index,
                        columns=["p_down", "p_flat", "p_up"])
