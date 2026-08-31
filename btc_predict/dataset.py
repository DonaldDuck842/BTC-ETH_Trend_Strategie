"""Zusammenbau der Trainings- und Testdatensaetze aus den Regime-Fenstern."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data as data_mod
from .features import build_features, build_labels, forward_returns


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    fwd_ret: pd.Series
    regime: pd.Series  # Name des Regimes je Zeile (fuer Diagnose & Val-Split)


def _prepare_window(bars_1m: pd.DataFrame, cfg: dict) -> Dataset:
    tf = cfg["timeframe"]
    bars = data_mod.resample(bars_1m, tf)
    X = build_features(bars)
    y = build_labels(bars, cfg["label"]["horizon_bars"], cfg["label"]["threshold_bps"])
    fwd = forward_returns(bars, cfg["label"]["horizon_bars"])
    valid = X.notna().all(axis=1) & y.notna()
    return Dataset(X[valid], y[valid], fwd[valid], pd.Series("", index=X[valid].index))


def load_bars(start: str, end: str, cfg: dict, synthetic: bool = False,
              download: bool = True) -> pd.DataFrame:
    """1m-Bars fuer ein Zeitfenster (echt oder synthetisch), inkl. Enddatum."""
    if synthetic:
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return data_mod.make_synthetic(start, end_excl,
                                       seed=abs(hash((start, end))) % 2**31)
    months = data_mod.month_range(start, end)
    bars_1m = data_mod.load_months(
        cfg["data"]["symbol"], cfg["data"]["base_interval"], months,
        Path(cfg["data"]["data_dir"]), download=download)
    if cfg["data"].get("use_flow", False):
        flow = data_mod.load_flow_months(cfg["data"]["symbol"], months,
                                         Path(cfg["data"]["data_dir"]),
                                         download=download)
        bars_1m = bars_1m.join(flow, how="left")
        flow_cols = ["tb_vol", "ts_vol", "tb_cnt", "ts_cnt", "big_tb_vol", "big_ts_vol"]
        bars_1m[flow_cols] = bars_1m[flow_cols].fillna(0.0)
    return bars_1m.loc[start:pd.Timestamp(end).strftime("%Y-%m-%d 23:59:59")]


def load_window(start: str, end: str, cfg: dict, synthetic: bool = False,
                download: bool = True) -> Dataset:
    bars_1m = load_bars(start, end, cfg, synthetic=synthetic, download=download)
    return _prepare_window(bars_1m, cfg)


def assert_no_overlap(test_start: pd.Timestamp, test_end: pd.Timestamp, cfg: dict) -> None:
    """Testfenster darf kein Trainings-Regime schneiden — hartes Abbruchkriterium."""
    for reg in cfg["regimes"]:
        r0, r1 = pd.Timestamp(reg["start"]), pd.Timestamp(reg["end"])
        if test_start <= r1 and test_end >= r0:
            raise ValueError(
                f"Testfenster {test_start.date()}..{test_end.date()} ueberschneidet "
                f"Trainings-Regime '{reg['name']}' ({r0.date()}..{r1.date()}). "
                "Anderen Testmonat waehlen oder Regime-Fenster anpassen."
            )


def build_training_set(cfg: dict, synthetic: bool = False, download: bool = True) -> Dataset:
    """Konkateniert alle Regime-Fenster zu einem Trainingsdatensatz."""
    parts: list[Dataset] = []
    for reg in cfg["regimes"]:
        ds = load_window(reg["start"], reg["end"], cfg, synthetic=synthetic,
                         download=download)
        ds.regime[:] = reg["name"]
        parts.append(ds)
    X = pd.concat([p.X for p in parts])
    y = pd.concat([p.y for p in parts])
    fwd = pd.concat([p.fwd_ret for p in parts])
    regime = pd.concat([p.regime for p in parts])
    order = X.index.argsort()
    return Dataset(X.iloc[order], y.iloc[order], fwd.iloc[order], regime.iloc[order])
