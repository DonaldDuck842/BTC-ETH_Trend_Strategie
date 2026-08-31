"""Out-of-Sample-Bewertung: Klassifikationsmetriken + kostenbewusste PnL-Simulation.

Zentraler Grundsatz: Accuracy ist nicht das Kriterium. Ein Signal lebt oder
stirbt an (avg PnL pro Trade in bps) vs. (Round-Trip-Kosten in bps).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .dataset import Dataset

BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120}


def make_signal(proba: pd.DataFrame, signal_threshold: float,
                allow_short: bool = True, exit_threshold: float | None = None,
                min_hold_bars: int = 1) -> pd.Series:
    """Position in {-1, 0, +1} aus den Klassen-Wahrscheinlichkeiten.

    Mit Hysterese gegen Signal-Flackern: Einstieg erst wenn
    |p_up - p_down| > signal_threshold, Ausstieg erst wenn die Edge in
    Positionsrichtung unter exit_threshold faellt UND die Mindesthaltedauer
    erreicht ist. Flip nur bei starkem Gegensignal.
    """
    edge = (proba["p_up"] - proba["p_down"]).to_numpy()
    if exit_threshold is None:
        exit_threshold = signal_threshold  # klassisches Verhalten ohne Hysterese
    pos = np.zeros(len(edge))
    cur, hold = 0.0, 0
    for t in range(len(edge)):
        e = edge[t]
        if cur == 0.0:
            if e > signal_threshold:
                cur, hold = 1.0, 0
            elif allow_short and e < -signal_threshold:
                cur, hold = -1.0, 0
        else:
            hold += 1
            if hold >= min_hold_bars:
                if allow_short and cur * e < -signal_threshold:
                    cur, hold = -cur, 0          # starkes Gegensignal: Flip
                elif cur * e < exit_threshold:
                    cur, hold = 0.0, 0           # Edge weg: raus
        pos[t] = cur
    return pd.Series(pos, index=proba.index)


def simulate(signal: pd.Series, bar_returns: pd.Series, cost_bps_per_side: float,
             timeframe: str) -> dict:
    """Vektorisierte Simulation: Position wird zum naechsten Bar wirksam.

    pnl_t = pos_{t-1} * ret_t - |Δpos| * Kosten_pro_Seite
    """
    pos = signal.shift(1).fillna(0.0)
    ret = bar_returns.reindex(signal.index).fillna(0.0)
    cost = cost_bps_per_side / 10_000
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = pos * ret - turnover * cost
    gross = pos * ret

    n_bars = len(pnl)
    bars_year = BARS_PER_YEAR.get(timeframe, 105_120)
    ann = np.sqrt(bars_year)
    sharpe = float(pnl.mean() / pnl.std() * ann) if pnl.std() > 0 else 0.0
    n_trades = int((turnover > 0).sum())
    time_in_market = float((pos != 0).mean())

    in_pos = pos != 0
    avg_gross_bps = float(gross[in_pos].mean() * 10_000) if in_pos.any() else 0.0
    round_trips = float(turnover.sum() / 2)
    gross_per_rt_bps = float(gross.sum() * 10_000 / round_trips) if round_trips else 0.0
    avg_hold = float(in_pos.sum() / round_trips) if round_trips else 0.0

    return {
        "round_trips": round_trips,
        "gross_bps_per_round_trip": gross_per_rt_bps,
        "avg_hold_bars": avg_hold,
        "total_return_pct": float(pnl.sum() * 100),
        "total_gross_return_pct": float(gross.sum() * 100),
        "sharpe_net": sharpe,
        "n_trades": n_trades,
        "time_in_market": time_in_market,
        "avg_gross_bps_per_bar_in_pos": avg_gross_bps,
        "cost_paid_pct": float((turnover * cost).sum() * 100),
        "n_bars": n_bars,
        "equity_curve": pnl.cumsum(),
    }


def classification_report(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    pred_class = proba[["p_down", "p_flat", "p_up"]].values.argmax(axis=1) - 1
    bacc = balanced_accuracy_score(y_true, pred_class)
    cm = confusion_matrix(y_true, pred_class, labels=[-1, 0, 1])
    # Trefferquote nur dort, wo das Modell eine Richtung ausruft:
    directional = pred_class != 0
    hit = float((np.sign(y_true[directional]) == pred_class[directional]).mean()) \
        if directional.any() else float("nan")
    return {
        "balanced_accuracy": float(bacc),
        "directional_hit_rate": hit,
        "n_directional_calls": int(directional.sum()),
        "confusion_matrix": cm.tolist(),
        "class_share_true": y_true.value_counts(normalize=True).to_dict(),
    }


def evaluate_month(ds_test: Dataset, proba: pd.DataFrame, cfg: dict) -> dict:
    bt = cfg["backtest"]
    tf = cfg["timeframe"]
    cost = bt["fee_bps_per_side"] + bt["slippage_bps_per_side"]
    signal = make_signal(proba, bt["signal_threshold"], bt["allow_short"],
                         exit_threshold=bt.get("exit_threshold"),
                         min_hold_bars=bt.get("min_hold_bars", 1))

    # Simulation auf 1-Bar-Renditen: ret_1 zum Zeitpunkt t ist der Return des
    # Bars t; simulate() verschiebt das Signal um einen Bar, sodass das Signal
    # vom Close t-1 den Return des Bars t verdient — keine Ueberlappung.
    bar_ret = ds_test.X["ret_1"]

    sim = simulate(signal, bar_ret, cost, tf)
    sim_gross = simulate(signal, bar_ret, 0.0, tf)
    cls = classification_report(ds_test.y, proba)

    return {
        "classification": cls,
        "net": {k: v for k, v in sim.items() if k != "equity_curve"},
        "gross_sharpe": sim_gross["sharpe_net"],
        "equity_curve": sim["equity_curve"],
        "signal": signal,
    }


def threshold_cost_sweep(ds_test: Dataset, proba: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Netto-Ergebnis ueber ein Raster aus Einstiegsschwellen und Kosten-Szenarien.

    Zeigt in einem Lauf, ob es IRGENDEINE Kombination aus Selektivitaet und
    Ausfuehrungsqualitaet gibt, bei der das Signal nach Kosten ueberlebt —
    und wo die Gewinnschwelle liegt. Ein Raster auf einem Monat ist bewusst
    Diagnose, keine Parameterwahl: Parameter danach fixieren und auf einem
    FRISCHEN Monat bestaetigen.
    """
    bt = cfg["backtest"]
    tf = cfg["timeframe"]
    bar_ret = ds_test.X["ret_1"]
    scenarios = bt.get("cost_scenarios_bps_per_side",
                       [0.0, 1.5, bt["fee_bps_per_side"] + bt["slippage_bps_per_side"]])
    rows = []
    for thr in bt.get("sweep_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5]):
        sig = make_signal(proba, thr, bt["allow_short"],
                          exit_threshold=bt.get("exit_threshold_frac", 0.5) * thr,
                          min_hold_bars=bt.get("min_hold_bars", 1))
        for cost in scenarios:
            sim = simulate(sig, bar_ret, cost, tf)
            rows.append({
                "enter_thr": thr,
                "cost_bps_side": cost,
                "n_round_trips": sim["round_trips"],
                "avg_hold_bars": round(sim["avg_hold_bars"], 1),
                "gross_bps_per_rt": round(sim["gross_bps_per_round_trip"], 2),
                "net_return_pct": round(sim["total_return_pct"], 2),
                "sharpe_net": round(sim["sharpe_net"], 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Snooping-Log: protokolliert, wie oft ein Testmonat schon angefasst wurde.
# ---------------------------------------------------------------------------

def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def log_evaluation(cfg: dict, test_month: str, results: dict,
                   log_path: Path = Path("results/eval_log.csv")) -> int:
    """Haengt den Lauf ans Log an und gibt zurueck, zum wievielten Mal dieser
    Testmonat insgesamt bewertet wurde (Data-Snooping-Zaehler)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_month": test_month,
        "config_hash": _config_hash(cfg),
        "timeframe": cfg["timeframe"],
        "sharpe_net": results["net"]["sharpe_net"],
        "total_return_pct": results["net"]["total_return_pct"],
        "balanced_accuracy": results["classification"]["balanced_accuracy"],
        "n_trades": results["net"]["n_trades"],
    }
    if log_path.exists():
        log = pd.read_csv(log_path)
    else:
        log = pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(log_path, index=False)
    return int((log["test_month"] == test_month).sum())
