"""Datenbeschaffung: Binance Vision Monats-Zips (Spot-Klines, 1m) -> Parquet.

Binance Vision ist ein oeffentliches Archiv ohne API-Key:
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def month_range(start: str, end: str) -> list[str]:
    """Liste der Monate 'YYYY-MM' von start bis end (inklusive)."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start=start, end=end, freq="M")]


def _parquet_path(data_dir: Path, symbol: str, interval: str, month: str) -> Path:
    return data_dir / symbol / interval / f"{month}.parquet"


def download_month(symbol: str, interval: str, month: str, data_dir: Path) -> Path:
    """Laedt einen Monat 1m-Klines und legt ihn als Parquet ab (idempotent)."""
    out = _parquet_path(data_dir, symbol, interval, month)
    if out.exists():
        return out
    url = BINANCE_VISION_URL.format(symbol=symbol, interval=interval, month=month)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=KLINE_COLUMNS)
    # Ab 2025 liefert Binance Vision Mikrosekunden-Timestamps, davor Millisekunden.
    unit = "us" if df["open_time"].iloc[0] > 10**14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.drop(columns=["close_time", "ignore"]).set_index("open_time").sort_index()
    df = df.astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def load_months(symbol: str, interval: str, months: list[str], data_dir: Path,
                download: bool = True) -> pd.DataFrame:
    """Laedt Monate aus dem lokalen Parquet-Cache, laedt fehlende bei Bedarf herunter."""
    frames = []
    for m in months:
        path = _parquet_path(data_dir, symbol, interval, m)
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} fehlt und download=False")
            download_month(symbol, interval, m, data_dir)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregiert 1m-Bars auf den Ziel-Timeframe (z.B. '5m')."""
    if timeframe in ("1m", "1min"):
        return df
    rule = timeframe.replace("m", "min")
    agg = df.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        n_trades=("n_trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    )
    return agg.dropna(subset=["open", "close"])


def make_synthetic(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Synthetische 1m-Bars im Binance-Schema — nur fuer Pipeline-Smoke-Tests.

    Random Walk mit Vol-Clustering; enthaelt konstruktionsbedingt kein
    echtes Signal, ein korrekter Backtest muss hier ~0 Edge zeigen.
    """
    idx = pd.date_range(start=start, end=end, freq="1min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    n = len(idx)
    vol_state = np.exp(np.cumsum(rng.normal(0, 0.01, n)) * 0.1)
    rets = rng.normal(0, 0.0006, n) * vol_state
    close = 40_000 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1); open_[0] = close[0]
    spread = np.abs(rng.normal(0, 0.0004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(5, 3, n)) * vol_state
    taker_ratio = np.clip(rng.beta(5, 5, n), 0.05, 0.95)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "n_trades": rng.integers(50, 500, n),
        "taker_buy_base": volume * taker_ratio,
        "taker_buy_quote": volume * taker_ratio * close,
    }, index=idx)
    df.index.name = "open_time"
    return df
