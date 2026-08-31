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


AGGTRADES_URL = (
    "https://data.binance.vision/data/spot/monthly/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{month}.zip"
)

# Notional-Schwelle fuer "grosse" Trades (Whale-Flow) in USD — bewusst absolut
# statt Perzentil, damit die Definition ueber die Jahre stabil bleibt.
BIG_TRADE_NOTIONAL_USD = 100_000


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


def download_aggtrades_month(symbol: str, month: str, data_dir: Path) -> Path:
    """Streamt einen aggTrades-Monat (1-2 GB Zip) und speichert NUR die
    1-Minuten-Flow-Aggregate als Parquet (idempotent, Zip wird geloescht).

    Aggregate je Minute:
      tb_vol/ts_vol   Taker-Buy-/Taker-Sell-Volumen (BTC)
      tb_cnt/ts_cnt   Anzahl aggregierter Buy-/Sell-Trades
      big_tb_vol/...  davon Trades mit Notional >= BIG_TRADE_NOTIONAL_USD
    """
    out = data_dir / symbol / "flow1m" / f"{month}.parquet"
    if out.exists():
        return out
    url = AGGTRADES_URL.format(symbol=symbol, month=month)
    tmp = out.parent / f"{month}.zip.tmp"
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
    parts = []
    cols = ["agg_id", "price", "qty", "first_id", "last_id",
            "time", "is_buyer_maker", "best_match"]
    try:
        with zipfile.ZipFile(tmp) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                # Manche Dateien tragen eine Header-Zeile, aeltere nicht.
                head = fh.readline().decode("utf-8", "replace")
                skip = 1 if not head.split(",")[0].strip().isdigit() else 0
            with zf.open(name) as fh:
                for chunk in pd.read_csv(fh, header=None, names=cols, skiprows=skip,
                                         usecols=["price", "qty", "time", "is_buyer_maker"],
                                         chunksize=5_000_000):
                    unit = "us" if chunk["time"].iloc[0] > 10**14 else "ms"
                    minute = pd.to_datetime(chunk["time"], unit=unit, utc=True).dt.floor("1min")
                    ibm = chunk["is_buyer_maker"]
                    if ibm.dtype != bool:
                        ibm = ibm.astype(str).str.lower().eq("true")
                    taker_buy = ~ibm  # Buyer war Taker -> aggressiver Kauf
                    notional = chunk["price"] * chunk["qty"]
                    big = notional >= BIG_TRADE_NOTIONAL_USD
                    g = pd.DataFrame({
                        "minute": minute,
                        "tb_vol": chunk["qty"].where(taker_buy, 0.0),
                        "ts_vol": chunk["qty"].where(~taker_buy, 0.0),
                        "tb_cnt": taker_buy.astype(int),
                        "ts_cnt": (~taker_buy).astype(int),
                        "big_tb_vol": chunk["qty"].where(taker_buy & big, 0.0),
                        "big_ts_vol": chunk["qty"].where(~taker_buy & big, 0.0),
                    }).groupby("minute").sum()
                    parts.append(g)
    finally:
        tmp.unlink(missing_ok=True)
    flow = pd.concat(parts).groupby(level=0).sum().sort_index()
    flow.index.name = "open_time"
    flow.to_parquet(out)
    return out


def load_flow_months(symbol: str, months: list[str], data_dir: Path,
                     download: bool = True) -> pd.DataFrame:
    frames = []
    for m in months:
        path = data_dir / symbol / "flow1m" / f"{m}.parquet"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} fehlt und download=False")
            download_aggtrades_month(symbol, m, data_dir)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


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
    spec = dict(
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
    flow_cols = ("tb_vol", "ts_vol", "tb_cnt", "ts_cnt", "big_tb_vol", "big_ts_vol")
    for c in flow_cols:
        if c in df.columns:
            spec[c] = (c, "sum")
    agg = df.resample(rule, label="left", closed="left").agg(**spec)
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
    n_trades = rng.integers(50, 500, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "n_trades": n_trades,
        "taker_buy_base": volume * taker_ratio,
        "taker_buy_quote": volume * taker_ratio * close,
        # Flow-Spalten im aggTrades-Aggregat-Schema, damit auch use_flow-Configs
        # einen synthetischen Smoke-Test fahren koennen:
        "tb_vol": volume * taker_ratio,
        "ts_vol": volume * (1 - taker_ratio),
        "tb_cnt": (n_trades * taker_ratio).astype(int),
        "ts_cnt": (n_trades * (1 - taker_ratio)).astype(int),
        "big_tb_vol": volume * taker_ratio * 0.2,
        "big_ts_vol": volume * (1 - taker_ratio) * 0.2,
    }, index=idx)
    df.index.name = "open_time"
    return df
