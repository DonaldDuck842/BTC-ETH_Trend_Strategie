#!/usr/bin/env python3
"""Synthetischer Smoke-Test fuer das Lookback-Ensemble (btc_predict.daily).

Laeuft komplett ohne Netzzugang auf make_synthetic-Daten und prueft:
  1. Rueckwaertskompatibilitaet: ensemble=False (Default) liefert bitidentisch
     das bisherige Champion-Verhalten (SMA50 + Donchian20 + Vol-Targeting).
  2. Ensemble-Signal = Mittel der 7 unabhaengig nachgerechneten Member;
     Position = Signal * Size (ohne Funding-Overlay).
  3. Signalwerte liegen auf dem 1/7-Raster und Zwischenstufen kommen vor
     (d.h. das Ensemble unterscheidet sich wirklich vom Champion).
  4. Turnover-Zerlegung (Signal-Flips vs. Vol-Targeting) ist exakt
     (diskrete Produktregel) und report_diagnostics liefert plausible Werte.
  5. allow_short=False haelt das Ensemble-Signal in [0, 1].

    python scripts/smoke_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import data as data_mod  # noqa: E402
from btc_predict.daily import (  # noqa: E402
    ENSEMBLE_DONCH_FRAC, ENSEMBLE_SMA_LOOKBACKS, backtest, build_positions,
    momentum_signal, report, report_diagnostics)


def main() -> None:
    bars = data_mod.make_synthetic("2021-01-01", "2023-01-01", seed=7)
    daily = data_mod.resample(bars, "1d")
    assert len(daily) > 2 * max(ENSEMBLE_SMA_LOOKBACKS), "zu wenige Tage"
    print(f"Synthetische Tagesdaten: {len(daily)} Tage")

    # 1) Rueckwaertskompatibilitaet: Default == altes Verhalten, inline nachgebaut.
    champ = build_positions(daily, None, allow_short=True)
    close = daily["close"]
    sma50 = close.rolling(50).mean()
    donch = (daily["high"].rolling(20).max() + daily["low"].rolling(20).min()) / 2
    old_mom = (((close > sma50) & (close > donch)).astype(float)
               - ((close < sma50) & (close < donch)).astype(float))
    realized = np.log(close).diff().rolling(20).std() * np.sqrt(365)
    old_pos = old_mom * (0.40 / realized).clip(upper=1.0).fillna(0.0)
    assert np.allclose(champ["pos"].values, old_pos.values), \
        "ensemble=False weicht vom bisherigen Champion-Verhalten ab"
    print("OK  1: ensemble=False bitidentisch zum alten Verhalten")

    # 2) Ensemble == Mittel der 7 Member; pos == sig * size (ohne Funding).
    ens = build_positions(daily, None, allow_short=True, ensemble=True)
    members = [momentum_signal(daily, lb,
                               max(int(round(lb * ENSEMBLE_DONCH_FRAC)), 1), True)
               for lb in ENSEMBLE_SMA_LOOKBACKS]
    assert len(members) == 7, "Spezifikation verlangt 7 Member"
    mean_sig = sum(members) / len(members)
    assert np.allclose(ens["sig"].values, mean_sig.values), \
        "Ensemble-Signal ist nicht das Mittel der Member"
    assert np.allclose(ens["pos"].values, (ens["sig"] * ens["size"]).values), \
        "pos != sig * size"
    print("OK  2: Ensemble-Signal = Mittel der 7 Member, pos = sig * size")

    # 3) 1/7-Raster und echte Zwischenstufen.
    grid = ens["sig"] * len(ENSEMBLE_SMA_LOOKBACKS)
    assert np.allclose(grid, grid.round()), "Signal liegt nicht auf 1/7-Raster"
    frac = ens["sig"].abs()
    n_inter = int(((frac > 1e-9) & (frac < 1 - 1e-9)).sum())
    assert n_inter > 0, "keine Zwischenstufen — Ensemble degeneriert zum Champion"
    assert (ens["pos"] != champ["pos"]).any(), "Ensemble identisch zum Champion"
    print(f"OK  3: 1/7-Raster, {n_inter} Tage mit Zwischenstufen-Signal")

    # 4) Turnover-Zerlegung exakt (Produktregel) + Diagnostik-Report.
    start200 = daily.index[max(ENSEMBLE_SMA_LOOKBACKS) - 1]
    bt = backtest(ens.loc[start200:], 10.0)
    pos = bt["pos"].shift(1).fillna(0.0)
    sig = bt["sig"].shift(1).fillna(0.0)
    size = bt["size"].shift(1).fillna(0.0)
    lhs = pos.diff()
    rhs = size * sig.diff() + sig.shift(1) * size.diff()
    assert np.allclose(lhs.iloc[2:], rhs.iloc[2:]), "Produktregel-Zerlegung verletzt"
    d = report_diagnostics(bt, "smoke", 10.0)
    assert d["turnover_pa"] > 0, "kein Turnover — Backtest tot"
    assert d["sig_turn_pa"] + d["vol_turn_pa"] >= d["turnover_pa"] - 0.2, \
        "Zerlegungs-Anteile kleiner als Gesamt-Turnover"
    assert 0.0 < d["avg_change"] < 1.0, "avg_change unplausibel fuer 1/7-Schritte"
    bt_champ = backtest(champ.loc[start200:], 10.0)
    d_champ = report_diagnostics(bt_champ, "smoke", 10.0)
    print("OK  4: Turnover-Zerlegung exakt; Diagnostik:")
    print(pd.DataFrame([{**d_champ, "variante": "champion"},
                        {**d, "variante": "ensemble"}]).to_string(index=False))
    print(pd.DataFrame([{**report(bt_champ, "smoke"), "variante": "champion"},
                        {**report(bt, "smoke"), "variante": "ensemble"}])
          .to_string(index=False))

    # 5) allow_short=False: Signal in [0, 1].
    ens_lf = build_positions(daily, None, allow_short=False, ensemble=True)
    assert float(ens_lf["sig"].min()) >= 0.0 and float(ens_lf["sig"].max()) <= 1.0
    print("OK  5: allow_short=False haelt Ensemble-Signal in [0, 1]")

    print("\nSmoke-Test bestanden. Hinweis: synthetische Daten haben per")
    print("Konstruktion keine Edge — PnL-Zahlen oben sind bedeutungslos,")
    print("der reale Lauf gehoert nach Colab.")


if __name__ == "__main__":
    main()
