"""Tages-Strategie-Kern: Momentum long/short + Vol-Targeting (Champion-Variante).

Aus scripts/daily_strategy.py extrahiert, damit Phase 2 (5m-Ausfuehrung)
dieselbe Signal-Logik importieren kann. Funding-Overlay bleibt als Option
erhalten, ist aber im Varianten-Vergleich durchgefallen (kostete in jeder
Periode Rendite) — Champion laeuft ohne.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import data as data_mod

FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"


def load_daily(symbol: str, start: str, end: str, data_dir: Path) -> pd.DataFrame:
    months = data_mod.month_range(start, end)
    return data_mod.load_months(symbol, "1d", months, data_dir)


def load_funding(symbol: str, data_dir: Path) -> pd.Series:
    """Komplette Funding-Historie ueber die Futures-API, in Zeitfenstern
    paginiert (die API ignoriert startTime-only-Pagination), gecacht."""
    cache = data_dir / symbol / "funding.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["funding_rate"]
    rows = []
    start_ms = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    window_ms = 1000 * 8 * 3600 * 1000
    while start_ms < now_ms:
        end_ms = min(start_ms + window_ms, now_ms)
        resp = requests.get(FUNDING_API, params={
            "symbol": symbol, "limit": 1000,
            "startTime": start_ms, "endTime": end_ms}, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        start_ms = (batch[-1]["fundingTime"] + 1) if batch else end_ms + 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    s = pd.Series(df["fundingRate"].astype(float).values, index=ts, name="funding_rate")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cache)
    return s


def build_positions(daily: pd.DataFrame, funding: pd.Series | None,
                    target_vol: float = 0.40, allow_short: bool = False) -> pd.DataFrame:
    close = daily["close"]
    ret = np.log(close).diff()

    sma50 = close.rolling(50).mean()
    donch_mid = (daily["high"].rolling(20).max() + daily["low"].rolling(20).min()) / 2
    long_sig = ((close > sma50) & (close > donch_mid)).astype(float)
    short_sig = ((close < sma50) & (close < donch_mid)).astype(float) if allow_short else 0.0
    momentum = long_sig - short_sig

    realized = ret.rolling(20).std() * np.sqrt(365)
    size = (target_vol / realized).clip(upper=1.0).fillna(0.0)

    overheat = pd.Series(False, index=close.index)
    oversold = pd.Series(False, index=close.index)
    if funding is not None:
        f_daily = funding.resample("1D").mean().reindex(close.index.normalize()).ffill()
        f_daily.index = close.index
        f3 = f_daily.rolling(3).mean()
        p95 = f3.rolling(365, min_periods=180).quantile(0.95)
        p05 = f3.rolling(365, min_periods=180).quantile(0.05)
        overheat = (f3 > p95).fillna(False)
        oversold = (f3 < p05).fillna(False)

    pos = momentum * size
    pos[overheat & (pos > 0)] = 0.0
    pos[oversold & (pos < 0)] = 0.0
    return pd.DataFrame({"pos": pos, "ret": ret, "overheat": overheat.astype(float)})


def backtest(frame: pd.DataFrame, cost_bps_per_side: float = 10.0) -> pd.DataFrame:
    pos = frame["pos"].shift(1).fillna(0.0)
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

    def max_dd_pct(series: pd.Series) -> float:
        eq = series.cumsum()
        return float(np.expm1((eq - eq.cummax()).min()) * 100)

    bh = float(np.exp(ret.sum() / years) - 1) if years > 0 else float("nan")
    bh_sharpe = float(ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0.0
    return {
        "period": label,
        "cagr_pct": round(cagr * 100, 1),
        "bh_cagr_pct": round(bh * 100, 1),
        "sharpe": round(sharpe, 2),
        "bh_sharpe": round(bh_sharpe, 2),
        "max_dd_pct": round(max_dd_pct(pnl), 1),
        "bh_max_dd_pct": round(max_dd_pct(ret.fillna(0)), 1),
        "time_in_mkt": round(float((bt["pos"].shift(1).abs() > 0).mean()), 2),
        "trades_py": round(float((bt["turnover"] > 0.05).sum() / years), 1),
    }
