#!/usr/bin/env python3
"""Laedt alle 1m-Klines herunter, die die Konfiguration benoetigt
(Regime-Fenster + optionale Testmonate). Idempotent — bereits vorhandene
Monate werden uebersprungen.

    python scripts/download_data.py --test-month 2025-03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import data as data_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test-month", action="append", default=[],
                    help="Zusaetzlich zu ladender Testmonat (mehrfach moeglich)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    months: list[str] = []
    for reg in cfg["regimes"]:
        months += data_mod.month_range(reg["start"], reg["end"])
    months += args.test_month
    months = sorted(set(months))

    symbol = cfg["data"]["symbol"]
    interval = cfg["data"]["base_interval"]
    data_dir = Path(cfg["data"]["data_dir"])
    print(f"{len(months)} Monate {symbol} {interval} -> {data_dir}/")
    for m in months:
        path = data_mod.download_month(symbol, interval, m, data_dir)
        print(f"  ok {m} -> {path}")


if __name__ == "__main__":
    main()
