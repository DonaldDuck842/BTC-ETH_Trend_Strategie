#!/usr/bin/env python3
"""Synthetischer Smoke-Test fuer scripts/portfolio_strategy.py (kein Netz).

Prueft die neue Logik mit Assertions:
  - Warmup: Indikatoren am Eval-Start gefuellt (Vergleich mit Eval-only-Lauf)
  - Portfolio-PnL = gewichtete Summe der Bein-PnLs
  - gepaarter Block-Bootstrap: Paarung erhalten, deterministisch, Richtung
  - Shortability-Sensitivitaet: keine Shorts vor Cutoff
  - Skript-Endlauf mit --synthetic (Gates + Korrelation im Output)

    python3 tests/test_portfolio_strategy.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from btc_predict import data as data_mod  # noqa: E402
from btc_predict.daily import backtest, build_positions  # noqa: E402
from portfolio_strategy import (  # noqa: E402
    build_portfolio, paired_block_bootstrap, run_leg, strip_shorts_before,
    subperiod_table)

WARMUP, EVAL_START, EVAL_END = "2021-01-01", "2021-06-01", "2022-06-30"


def make_daily(seed: int, drift_per_day: float = 0.0) -> pd.DataFrame:
    """Synthetische Tagesbars; optionaler Aufwaertsdrift erzwingt Momentum-
    Signale (alle Preisspalten gleich skaliert, OHLC bleibt konsistent)."""
    bars = data_mod.make_synthetic(WARMUP, EVAL_END, seed=seed)
    daily = data_mod.resample(bars, "1d")
    if drift_per_day:
        scale = np.exp(drift_per_day * np.arange(len(daily)))
        for c in ("open", "high", "low", "close"):
            daily[c] = daily[c] * scale
    return daily


def test_warmup_and_leg() -> None:
    daily = make_daily(seed=7, drift_per_day=0.005)
    bt = run_leg(daily, EVAL_START, EVAL_END, 0.40, True, 10.0)
    assert bt.index[0] == pd.Timestamp(EVAL_START, tz="UTC"), bt.index[0]
    assert not bt["pnl"].isna().any(), "NaN-PnL im Eval-Fenster"
    # Warmup wirkt: mit Historie ab WARMUP gibt es in den ersten 30 Eval-Tagen
    # Positionen (Drift -> long); Eval-only hat mechanisch pos=0 in den ersten
    # 19 Tagen (20d-Vol leer -> size=0) und SMA50 erst nach 50 Tagen.
    assert (bt["pos"].iloc[:30].abs() > 0).any(), \
        "Warmup unwirksam: keine Position am Eval-Start trotz Drift"
    frame_short = build_positions(daily.loc[EVAL_START:], None,
                                  target_vol=0.40, allow_short=True)
    assert (frame_short["pos"].iloc[:19] == 0).all(), \
        "Eval-only-Lauf muesste anfangs positionslos sein"
    print("ok  Warmup + Bein-Backtest")


def test_portfolio_aggregation() -> None:
    a = run_leg(make_daily(7, 0.005), EVAL_START, EVAL_END, 0.40, True, 10.0)
    b = run_leg(make_daily(11, -0.003), EVAL_START, EVAL_END, 0.40, True, 10.0)
    legs = {"BTCUSDT": a, "ETHUSDT": b}
    pf = build_portfolio(legs, {"BTCUSDT": 0.5, "ETHUSDT": 0.5})
    assert len(pf) == len(a) == len(b)
    assert np.allclose(pf["pnl"].values,
                       0.5 * a["pnl"].values + 0.5 * b["pnl"].values)
    assert np.allclose(pf["ret"].loc[pf["ret"].notna()],
                       (0.5 * a["ret"] + 0.5 * b["ret"]).dropna())
    corr = a["pnl"].corr(b["pnl"])
    assert -1.0 <= corr <= 1.0
    tbl = subperiod_table(pf)
    assert "gesamt" in set(tbl["period"]) and len(tbl) >= 2
    print(f"ok  Portfolio-Aggregation (PnL-Korrelation {corr:.3f})")


def test_paired_bootstrap() -> None:
    a = run_leg(make_daily(7, 0.005), EVAL_START, EVAL_END, 0.40, True, 10.0)
    b = run_leg(make_daily(11, -0.003), EVAL_START, EVAL_END, 0.40, True, 10.0)
    pf = build_portfolio({"x": a, "y": b}, {"x": 0.5, "y": 0.5})
    # Identische Serien: Paarung erhalten -> Differenz exakt 0 in jedem Draw.
    bs_same = paired_block_bootstrap(a["pnl"], a["pnl"], 30, 200, seed=1)
    assert np.all(bs_same["sharpe_diff"] == 0.0)
    assert np.all(bs_same["mean_diff_pa"] == 0.0)
    assert float((bs_same["sharpe_diff"] > 0).mean()) == 0.0, \
        "Gate (c) darf bei identischen Serien nicht passieren"
    # Echte Draws: Anzahl, Endlichkeit, Determinismus per Seed.
    bs1 = paired_block_bootstrap(pf["pnl"], a["pnl"], 30, 2000, seed=42)
    bs2 = paired_block_bootstrap(pf["pnl"], a["pnl"], 30, 2000, seed=42)
    assert bs1["sharpe_diff"].shape == (2000,)
    assert np.isfinite(bs1["sharpe_diff"]).all()
    assert np.array_equal(bs1["sharpe_diff"], bs2["sharpe_diff"])
    assert bs1["sharpe_diff"].std() > 0, "Draws duerfen nicht degeneriert sein"
    # Zu kurze Serie muss hart fehlschlagen statt Unsinn zu rechnen.
    try:
        paired_block_bootstrap(a["pnl"].iloc[:10], a["pnl"].iloc[:10], 30, 10)
        raise AssertionError("Blocklaenge > n haette fehlschlagen muessen")
    except ValueError:
        pass
    print("ok  gepaarter Block-Bootstrap")


def test_strip_shorts() -> None:
    daily = make_daily(seed=11, drift_per_day=-0.004)  # Abwaertsdrift -> Shorts
    frame = build_positions(daily, None, target_vol=0.40, allow_short=True)
    assert (frame["pos"] < 0).any(), "Testaufbau: es muessen Shorts existieren"
    cutoff = "2021-09-01"
    stripped = strip_shorts_before(frame, cutoff)
    cut = pd.Timestamp(cutoff, tz="UTC")
    assert (stripped.loc[stripped.index < cut, "pos"] >= 0).all()
    after_idx = stripped.index >= cut
    assert stripped.loc[after_idx, "pos"].equals(frame.loc[after_idx, "pos"])
    # Original bleibt unveraendert (Kopie, kein In-Place).
    assert (frame["pos"] < 0).any()
    bt = backtest(stripped, 10.0)
    assert not bt["pnl"].loc[EVAL_START:].isna().any()
    print("ok  Shortability-Sensitivitaet")


def test_script_end_to_end() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "portfolio_strategy.py"),
         "--synthetic", "--config", str(ROOT / "configs" / "portfolio.yaml")],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    out = res.stdout
    for needle in ("Bein BTCUSDT standalone", "Bein ETHUSDT standalone",
                   "Portfolio BTCUSDT/ETHUSDT", "Korrelation Tages-PnL",
                   "Gate", "(c) gepaarter Block-Bootstrap",
                   "Zusatz-Gate Live-Umstieg", "Sensitivitaet"):
        assert needle in out, f"'{needle}' fehlt im Skript-Output"
    print("ok  Skript-Endlauf --synthetic")


if __name__ == "__main__":
    test_warmup_and_leg()
    test_portfolio_aggregation()
    test_paired_bootstrap()
    test_strip_shorts()
    test_script_end_to_end()
    print("alle Smoke-Tests bestanden")
