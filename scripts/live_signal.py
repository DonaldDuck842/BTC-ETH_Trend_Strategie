#!/usr/bin/env python3
"""Taegliches Live-Signal der finalen Spezifikation (Paper-Trading-Report).

Spezifikation (eingefroren, siehe README): 50/50 BTCUSDT/ETHUSDT, je Bein
Lookback-Ensemble long/short (7 Member) + Vol-Targeting 40% (Cap 1.0),
Ausfuehrung naiv zum naechsten Tagesbeginn, Kostenannahme 10 bps/Seite.

Der Report zeigt je Bein die Zielposition fuer HEUTE (Signal vom gestrigen
Tagesschluss), die Aenderung gegenueber gestern und einen vol-basierten
Orientierungs-Stop (2 Tages-Sigma). Zusaetzlich wird jeder Lauf in
results/paper_log.csv protokolliert; sobald Historie vorhanden ist, wird
der realisierte Paper-PnL mitgerechnet. In Colab die Datei z.B. nach
Google Drive sichern — die Runtime ist fluechtig.

    python scripts/live_signal.py               # echter Lauf (Colab/lokal)
    python scripts/live_signal.py --synthetic   # Pipeline-Smoke-Test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict.daily import build_positions  # noqa: E402
from btc_predict.data import make_synthetic, resample  # noqa: E402

# data-api.binance.vision ist Binances offizieller Marktdaten-Spiegel ohne
# Geo-Block (noetig fuer GitHub-Actions-Runner in den USA); Fallback Haupt-API.
KLINES_APIS = ("https://data-api.binance.vision/api/v3/klines",
               "https://api.binance.com/api/v3/klines")

# Rebalance-Band (Ausfuehrungsregel, siehe docs/BETRIEBSREGELN.md):
# gehandelt wird nur, wenn die Zielposition um mehr als diesen Anteil des
# Bein-Kapitals von gestern abweicht. Der Backtest bezahlt konservativ
# JEDE taegliche Aenderung — weniger oft handeln kann nur Kosten sparen.
REBALANCE_BAND = 0.05
SYMBOLS = ("BTCUSDT", "ETHUSDT")
WEIGHTS = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
LOG_PATH = Path("results/paper_log.csv")


def fetch_recent_daily(symbol: str, n_days: int = 420) -> pd.DataFrame:
    """Juengste Tageskerzen direkt von der Spot-API (genug Warmup fuer
    SMA200 + Vol-Fenster). Die letzte, noch laufende Kerze wird verworfen."""
    rows, last_err = None, None
    for api in KLINES_APIS:
        try:
            resp = requests.get(api, params={
                "symbol": symbol, "interval": "1d",
                "limit": min(n_days, 1000)}, timeout=60)
            resp.raise_for_status()
            rows = resp.json()
            break
        except Exception as exc:  # naechsten Endpoint probieren
            last_err = exc
    if rows is None:
        raise RuntimeError(f"Keine Kursdaten fuer {symbol}: {last_err}")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qv", "n", "tb", "tq", "ig"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close"]].astype(float)
    return df.iloc[:-1]  # laufende Kerze weg


def synthetic_daily(symbol: str) -> pd.DataFrame:
    bars = make_synthetic("2024-01-01", "2025-06-01",
                          seed=13 if symbol == "BTCUSDT" else 17)
    return resample(bars, "1d")[["open", "high", "low", "close"]].iloc[:-1]


def leg_signal(daily: pd.DataFrame) -> dict:
    frame = build_positions(daily, None, allow_short=True, ensemble=True)
    ret = frame["ret"]
    sigma_d = float(ret.rolling(20).std().iloc[-1])
    last, prev = frame.iloc[-1], frame.iloc[-2]
    close = float(daily["close"].iloc[-1])
    return {
        "close": close,
        "signal": float(last["sig"]),
        "size": float(last["size"]),
        "pos": float(last["pos"]),
        "pos_prev": float(prev["pos"]),
        "sigma_daily_pct": sigma_d * 100,
        "stop_long": close * np.exp(-2 * sigma_d),
        "stop_short": close * np.exp(2 * sigma_d),
        "asof": daily.index[-1].date().isoformat(),
    }


def update_paper_log(signals: dict[str, dict]) -> pd.DataFrame | None:
    """Haengt den heutigen Zielzustand an das Log an (idempotent je asof-Tag)
    und liefert das Log zurueck."""
    rows = [{"asof": s["asof"], "symbol": sym, "target_pos": s["pos"],
             "ref_close": s["close"]} for sym, s in signals.items()]
    new = pd.DataFrame(rows)
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        log = log[~log["asof"].isin(new["asof"])]
        log = pd.concat([log, new], ignore_index=True)
    else:
        log = new
    LOG_PATH.parent.mkdir(exist_ok=True)
    log.sort_values(["asof", "symbol"]).to_csv(LOG_PATH, index=False)
    return log


def paper_pnl(log: pd.DataFrame) -> pd.Series | None:
    """Realisierter Paper-PnL: gestriges Ziel x heutiger Close-Return,
    gewichtet, 10 bps/Seite auf Positionsaenderungen."""
    if log["asof"].nunique() < 3:
        return None
    daily_pnl = {}
    for sym, g in log.groupby("symbol"):
        g = g.sort_values("asof").set_index("asof")
        ret = np.log(g["ref_close"]).diff()
        pos = g["target_pos"].shift(1)
        cost = g["target_pos"].diff().abs() * 10 / 10_000
        daily_pnl[sym] = WEIGHTS.get(sym, 0.5) * (pos * ret - cost)
    return pd.DataFrame(daily_pnl).sum(axis=1).dropna()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    signals = {}
    for sym in SYMBOLS:
        daily = synthetic_daily(sym) if args.synthetic else fetch_recent_daily(sym)
        assert len(daily) >= 220, f"{sym}: zu wenig Historie ({len(daily)} Tage)"
        signals[sym] = leg_signal(daily)

    print("=" * 62)
    print(f"  LIVE-SIGNAL  (Datenstand: Tagesschluss {signals['BTCUSDT']['asof']})")
    print(f"  Spez: Ensemble long/short + Vol-Targeting, 50/50, naive Ausf.")
    print("=" * 62)
    total_gross = 0.0
    for sym, s in signals.items():
        w = WEIGHTS[sym]
        tgt = w * s["pos"]
        chg = w * (s["pos"] - s["pos_prev"])
        total_gross += abs(tgt)
        direction = "LONG" if s["pos"] > 0 else ("SHORT" if s["pos"] < 0 else "FLAT")
        print(f"\n  {sym}  @ {s['close']:,.0f} USD   [{direction}]")
        print(f"    Ensemble-Signal : {s['signal']:+.3f} (von -1 bis +1, "
              f"1/7-Raster)")
        print(f"    Vol-Size        : {s['size']:.2f} "
              f"(Tages-Vol {s['sigma_daily_pct']:.2f}%)")
        print(f"    ZIELPOSITION    : {tgt:+.3f} x Portfolio-Kapital "
              f"(Aenderung ggue. gestern: {chg:+.3f})")
        d_leg = abs(s["pos"] - s["pos_prev"])
        if d_leg > REBALANCE_BAND:
            print(f"    >> ORDER HEUTE  : JA — Bein-Position um {d_leg:+.3f} "
                  f"anpassen (ueber Band {REBALANCE_BAND})")
        else:
            print(f"    >> ORDER HEUTE  : nein (Drift {d_leg:.3f} unter "
                  f"Band {REBALANCE_BAND} — Gebuehren sparen)")
        stop = s["stop_long"] if s["pos"] > 0 else s["stop_short"]
        if s["pos"] != 0:
            print(f"    Orientierungs-Stop (2 Sigma): {stop:,.0f} USD")
    print(f"\n  Brutto-Exposure gesamt: {total_gross:.2f} x Kapital "
          f"(Cap: 1.0)")

    log = update_paper_log(signals)
    pnl = paper_pnl(log)
    n_days = log["asof"].nunique()
    print(f"\n  Paper-Log: {n_days} Tage protokolliert -> {LOG_PATH}")
    if pnl is not None and len(pnl):
        total = float(pnl.sum() * 100)
        sharpe = (float(pnl.mean() / pnl.std() * np.sqrt(365))
                  if pnl.std() > 0 else 0.0)
        print(f"  Paper-PnL bisher: {total:+.2f}% | Sharpe (annualisiert, "
              f"noch rauschig): {sharpe:+.2f}")
    else:
        print("  Paper-PnL: ab 3 protokollierten Tagen")
    print("\n  Hinweis: In Colab results/paper_log.csv nach Drive sichern —")
    print("  die Runtime vergisst alles. Go-Live fruehestens nach der im")
    print("  README registrierten Paper-Phase.")


if __name__ == "__main__":
    main()
