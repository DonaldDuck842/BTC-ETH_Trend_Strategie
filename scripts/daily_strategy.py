#!/usr/bin/env python3
"""Pivot Phase 1 — Tages-Strategie: Momentum + Vol-Targeting + Funding-Overlay.

Richtung kommt vom Tageschart (robuster, billiger: wenige Trades), das
validierte 5m-Pfadmodell uebernimmt spaeter Einstiegs-Timing und Stops
(Phase 2). Dieser Backtest validiert die Tagesebene fuer sich, mit
Subperioden-Tabelle — der Haertetest ist die Post-2021-Periode.

Regeln (bewusst simpel, keine Optimierung):
  - Momentum:  Close > SMA(50) UND Close > Donchian-Mitte(20) -> long, sonst flat
  - Sizing:    Ziel-Vol 40% p.a. / realisierte 20d-Vol, gedeckelt bei 1.0 (kein Hebel)
  - Funding:   3d-Mittel der Funding-Rate ueber rollierendem 95%-Perzentil
               (1 Jahr) -> flat (ueberhitzte Long-Positionierung)
  - Kosten:    10 bps pro Seite (Spot-Taker inkl. Slippage)

    python scripts/daily_strategy.py                # Standard 2018-01 .. 2026-07
    python scripts/daily_strategy.py --no-funding   # ohne Funding-Overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import data as data_mod  # noqa: E402

FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"


def load_daily(symbol: str, start: str, end: str, data_dir: Path) -> pd.DataFrame:
    months = data_mod.month_range(start, end)
    df = data_mod.load_months(symbol, "1d", months, data_dir)
    return df


def load_funding(symbol: str, data_dir: Path) -> pd.Series:
    """Komplette Funding-Historie ueber die Futures-API (paginiert), gecacht."""
    cache = data_dir / symbol / "funding.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["funding_rate"]
    rows, start_ms = [], 0
    while True:
        resp = requests.get(FUNDING_API, params={
            "symbol": symbol, "limit": 1000, "startTime": start_ms}, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    s = pd.Series(df["fundingRate"].astype(float).values, index=ts, name="funding_rate")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cache)
    return s


def build_positions(daily: pd.DataFrame, funding: pd.Series | None,
                    target_vol: float = 0.40) -> pd.DataFrame:
    close = daily["close"]
    ret = np.log(close).diff()

    sma50 = close.rolling(50).mean()
    donch_mid = (daily["high"].rolling(20).max() + daily["low"].rolling(20).min()) / 2
    momentum = ((close > sma50) & (close > donch_mid)).astype(float)

    realized = ret.rolling(20).std() * np.sqrt(365)
    size = (target_vol / realized).clip(upper=1.0).fillna(0.0)

    overheat = pd.Series(False, index=close.index)
    if funding is not None:
        f_daily = funding.resample("1D").mean().reindex(close.index.normalize()).ffill()
        f_daily.index = close.index
        f3 = f_daily.rolling(3).mean()
        p95 = f3.rolling(365, min_periods=180).quantile(0.95)
        overheat = (f3 > p95).fillna(False)

    pos = momentum * size
    pos[overheat] = 0.0
    return pd.DataFrame({"pos": pos, "ret": ret, "overheat": overheat.astype(float)})


def backtest(frame: pd.DataFrame, cost_bps_per_side: float = 10.0) -> pd.DataFrame:
    pos = frame["pos"].shift(1).fillna(0.0)   # Signal von gestern handelt heute
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = pos * frame["ret"] - turnover * cost_bps_per_side / 10_000
    out = frame.copy()
    out["pnl"] = pnl
    out["turnover"] = turnover
    return out


def report(bt: pd.DataFrame, label: str) -> dict:
    pnl, ret = bt["pnl"], bt["ret"]
    years = len(pnl) / 365
    cagr = float(np.exp(pnl.sum() / years) - 1) if years > 0 else float("nan")
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(365)) if pnl.std() > 0 else 0.0
    eq = pnl.cumsum()
    maxdd = float((eq - eq.cummax()).min())
    bh = float(np.exp(ret.sum() / years) - 1) if years > 0 else float("nan")
    return {
        "period": label,
        "cagr_pct": round(cagr * 100, 1),
        "bh_cagr_pct": round(bh * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(maxdd * 100, 1),
        "time_in_mkt": round(float((bt["pos"].shift(1) > 0).mean()), 2),
        "trades_py": round(float((bt["turnover"] > 0.05).sum() / years), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    data_dir = Path("data")
    daily = load_daily("BTCUSDT", args.start, args.end, data_dir)
    print(f"Tagesdaten: {daily.index[0].date()} .. {daily.index[-1].date()} "
          f"({len(daily)} Tage)")

    funding = None
    if not args.no_funding:
        try:
            funding = load_funding("BTCUSDT", data_dir)
            print(f"Funding-Historie: {funding.index[0].date()} .. "
                  f"{funding.index[-1].date()} ({len(funding)} Punkte)")
        except Exception as exc:  # Funding ist Overlay, kein Muss
            print(f"Funding nicht ladbar ({exc}) — laufe ohne Overlay weiter.")

    frame = build_positions(daily, funding)
    bt = backtest(frame, args.cost_bps)

    periods = [("gesamt", None),
               ("2018-2020", ("2018-01-01", "2020-12-31")),
               ("2021-2022", ("2021-01-01", "2022-12-31")),
               ("2023-2024", ("2023-01-01", "2024-12-31")),
               ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]
    rows = []
    for label, rng in periods:
        seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
        if len(seg) > 30:
            rows.append(report(seg, label))
    print("\n== Momentum + Vol-Targeting"
          + ("" if args.no_funding else " + Funding-Overlay")
          + f" | Kosten {args.cost_bps} bps/Seite ==")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nLesart: Strategie muss v.a. in 2025-2026 mit positivem Sharpe und")
    print("deutlich geringerem MaxDD als Buy-and-Hold bestehen — CAGR darf")
    print("unter Buy-and-Hold liegen, das ist der Preis fuer den Drawdown-Schutz.")


if __name__ == "__main__":
    main()
