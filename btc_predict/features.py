"""Feature-Engineering auf OHLCV-Bars.

Jede Zeile t benutzt ausschliesslich Informationen bis einschliesslich Bar t
(kein Look-Ahead). Die Taker-Buy-Ratio aus den Binance-Klines dient als
Order-Flow-Proxy — das ist die "Orderbuch-Light"-Information, die in
historischen Klines steckt (echtes L2-Orderbuch braeuchte Tardis o.ae.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Vortages-Level: Hoch, Tief, VWAP und POC (Preis mit dem meisten
    Volumen = Liquiditaetscluster). These: an solchen Leveln clustern
    Limit-Orders, Distanz dazu traegt Information ueber "Gap-Fill"-Zuege."""
    day = df.index.normalize()
    rows = []
    for d, g in df.groupby(day):
        vol = g["volume"].sum()
        if vol <= 0 or len(g) < 10:
            rows.append((d, np.nan, np.nan, np.nan, np.nan))
            continue
        counts, edges = np.histogram(g["close"], bins=50, weights=g["volume"])
        i = int(np.argmax(counts))
        poc = (edges[i] + edges[i + 1]) / 2
        vwap = g["quote_volume"].sum() / vol
        rows.append((d, g["high"].max(), g["low"].min(), poc, vwap))
    lv = pd.DataFrame(rows, columns=["day", "d_high", "d_low", "d_poc", "d_vwap"])
    lv = lv.set_index("day").shift(1)  # nur der VORTAG ist bekannt -> kein Leakage
    out = lv.reindex(day)
    out.index = df.index
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    logc = np.log(close)
    ret1 = logc.diff()

    # --- Returns ueber mehrere Lookbacks ---
    for k in (1, 2, 3, 5, 10, 20, 40, 60):
        f[f"ret_{k}"] = logc.diff(k)

    # --- Volatilitaet & Regime im Kleinen ---
    f["vol_20"] = ret1.rolling(20).std()
    f["vol_60"] = ret1.rolling(60).std()
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["ret1_z"] = ret1 / f["vol_60"]

    # --- Lage im lokalen Range ---
    roll_hi = df["high"].rolling(60).max()
    roll_lo = df["low"].rolling(60).min()
    rng = (roll_hi - roll_lo).replace(0, np.nan)
    f["pos_in_range_60"] = (close - roll_lo) / rng

    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().replace(0, np.nan)
    f["bb_z"] = (close - sma20) / sd20
    f["bb_width"] = sd20 / sma20

    # --- RSI(14) ---
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # --- VWAP-Distanz (rollierend 60 Bars) ---
    pv = (df["quote_volume"]).rolling(60).sum()
    v = df["volume"].rolling(60).sum().replace(0, np.nan)
    f["vwap_dist_60"] = np.log(close / (pv / v))

    # --- Volumen / Aktivitaet ---
    vol_mean = df["volume"].rolling(60).mean()
    vol_std = df["volume"].rolling(60).std().replace(0, np.nan)
    f["volume_z_60"] = (df["volume"] - vol_mean) / vol_std
    trd_mean = df["n_trades"].rolling(60).mean()
    trd_std = df["n_trades"].rolling(60).std().replace(0, np.nan)
    f["trades_z_60"] = (df["n_trades"] - trd_mean) / trd_std

    # --- Order-Flow-Proxy: Taker-Buy-Anteil (0.5 = ausgeglichen) ---
    tbr = (df["taker_buy_base"] / df["volume"].replace(0, np.nan)).fillna(0.5)
    f["taker_ratio"] = tbr - 0.5
    f["taker_ratio_5"] = tbr.rolling(5).mean() - 0.5
    f["taker_ratio_20"] = tbr.rolling(20).mean() - 0.5

    # --- Candle-Geometrie ---
    bar_rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_frac"] = (close - df["open"]) / bar_rng
    f["upper_wick"] = (df["high"] - np.maximum(df["open"], close)) / bar_rng
    f["lower_wick"] = (np.minimum(df["open"], close) - df["low"]) / bar_rng

    # --- Echte Order-Flow-Features (aus aggTrades, falls vorhanden) ---
    if "tb_vol" in df.columns:
        tot_vol = (df["tb_vol"] + df["ts_vol"]).replace(0, np.nan)
        ofi = ((df["tb_vol"] - df["ts_vol"]) / tot_vol).fillna(0.0)
        f["ofi"] = ofi
        f["ofi_5"] = ofi.rolling(5).mean()
        f["ofi_20"] = ofi.rolling(20).mean()
        big_tot = (df["big_tb_vol"] + df["big_ts_vol"]).replace(0, np.nan)
        f["big_ofi"] = ((df["big_tb_vol"] - df["big_ts_vol"]) / big_tot).fillna(0.0)
        f["big_ofi_5"] = f["big_ofi"].rolling(5).mean()
        f["big_share"] = (big_tot / tot_vol).fillna(0.0)
        tot_cnt = (df["tb_cnt"] + df["ts_cnt"]).replace(0, np.nan)
        f["cnt_imb"] = ((df["tb_cnt"] - df["ts_cnt"]) / tot_cnt).fillna(0.0)
        size = (tot_vol / tot_cnt)
        size_mean = size.rolling(60).mean()
        size_std = size.rolling(60).std().replace(0, np.nan)
        f["trade_size_z"] = ((size - size_mean) / size_std).fillna(0.0)

    # --- Level-/Gap-Features (Orderbuch-These: Liquiditaetscluster) ---
    lv = _daily_levels(df)
    for col in ("d_high", "d_low", "d_poc", "d_vwap"):
        f[f"dist_{col}"] = np.log(close / lv[col])
    roll_hi_288 = df["high"].rolling(288).max()   # ~1 Tag auf 5m
    roll_lo_288 = df["low"].rolling(288).min()
    f["dist_roll_hi_1d"] = np.log(close / roll_hi_288)
    f["dist_roll_lo_1d"] = np.log(close / roll_lo_288)
    # Runde 1000er-Level (psychologische Limit-Order-Cluster bei BTC)
    f["round_1000"] = (close % 1000) / 1000 - 0.5
    f["round_1000_abs"] = f["round_1000"].abs()

    # --- Zeitmerkmale (Intraday-Saisonalitaet) ---
    minutes = df.index.hour * 60 + df.index.minute
    f["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    f["dow"] = df.index.dayofweek.astype(float)

    return f


def build_labels(df: pd.DataFrame, horizon_bars: int, threshold_bps: float) -> pd.Series:
    """3-Klassen-Label aus der Vorwaertsrendite ueber horizon_bars.

    +1 wenn fwd_ret > threshold, -1 wenn < -threshold, sonst 0.
    Die letzten horizon_bars Zeilen haben kein Label (NaN) und werden
    beim Training verworfen.
    """
    logc = np.log(df["close"])
    fwd = logc.shift(-horizon_bars) - logc
    thr = threshold_bps / 10_000
    label = pd.Series(0, index=df.index, dtype=float)
    label[fwd > thr] = 1
    label[fwd < -thr] = -1
    label[fwd.isna()] = np.nan
    return label


def forward_returns(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """Vorwaertsrendite (log) ueber den Label-Horizont, fuer die PnL-Simulation."""
    logc = np.log(df["close"])
    return logc.shift(-horizon_bars) - logc
