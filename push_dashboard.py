#!/usr/bin/env python3
"""Sendet output/performance.csv an das externe Dashboard.

POST an {DASHBOARD_URL}/api/public/tagesmarken, volle Historie in Bloecken
von maximal 5000 Tagen. Das Ziel macht ein Upsert auf (Strategie, Datum) —
doppeltes Senden ist unschaedlich und holt ausgefallene Tage nach.

Konfiguration AUSSCHLIESSLICH ueber Umgebungsvariablen (keine Werte im Code):
  DASHBOARD_URL    Basis-URL des Dashboards
  DASHBOARD_TOKEN  Token der Strategie (Typ "tagesmarke")
  DASHBOARD_STRAT  exakter Strategiename aus dem Dashboard

  python push_dashboard.py             # senden
  python push_dashboard.py --dry-run   # erste Zeile zeigen, nichts senden

Verhalten im Workflow: Fehlen die Umgebungsvariablen komplett (Secrets noch
nicht angelegt), wird der Push mit Hinweis UEBERSPRUNGEN (Exit 0), damit der
naechtliche Lauf nicht rot wird. Jeder echte Sendefehler (HTTP-Fehler oder
ok:false) bricht mit Exit-Code != 0 ab und gibt die Antwort aus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("output/performance.csv")
CHUNK = 5000
INSTRUMENT = "BTC 50% / ETH 50% (Trendfolge-Ensemble, Paper)"


def load_and_validate() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} fehlt — vorher build_performance.py ausfuehren.")
    df = pd.read_csv(CSV_PATH)
    # Vorab-Pruefungen laut Spezifikation — bei Verstoss abbrechen:
    if df["equity"].isna().any():
        sys.exit("Abbruch: NaN in equity.")
    if (df["equity"] <= 0).any():
        sys.exit("Abbruch: equity <= 0 gefunden.")
    if df["datum"].duplicated().any():
        sys.exit("Abbruch: doppelte Datumswerte.")
    if not df["datum"].is_monotonic_increasing:
        sys.exit("Abbruch: Datumswerte nicht aufsteigend sortiert.")
    return df


def to_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "datum": str(r["datum"]),
            "equity": float(r["equity"]),
            "benchmark_equity": float(r["benchmark_equity"]),
            "tagesrendite": float(r["tagesrendite"]),
            "instrument": INSTRUMENT,
            "zielgewichte": {"BTC": float(r["w_BTC"]), "ETH": float(r["w_ETH"])},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="erste Zeile ausgeben, nichts senden")
    args = ap.parse_args()

    df = load_and_validate()
    rows = to_rows(df)

    if args.dry_run:
        print(f"DRY-RUN — {len(rows)} Tage validiert, nichts gesendet.")
        print("Erste Zeile:", json.dumps(rows[0], indent=1))
        print("Letzte Zeile:", json.dumps(rows[-1], indent=1))
        return

    url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    token = os.environ.get("DASHBOARD_TOKEN", "")
    strat = os.environ.get("DASHBOARD_STRAT", "")
    if not (url and token and strat):
        print("Dashboard-Secrets nicht gesetzt (DASHBOARD_URL/TOKEN/STRAT) — "
              "Push uebersprungen.")
        return

    endpoint = f"{url}/api/public/tagesmarken"
    total_neu = total_akt = 0
    for i in range(0, len(rows), CHUNK):
        payload = {"token": token, "strategie": strat,
                   "tage": rows[i:i + CHUNK]}
        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            answer = resp.json()
        except Exception as exc:
            print(f"Sendefehler (Block ab Zeile {i}): {exc}")
            body = getattr(getattr(exc, "response", None), "text", "")
            if body:
                print("Antwort:", body[:500])
            sys.exit(1)
        if not answer.get("ok"):
            print("Dashboard meldet ok:false — Antwort:",
                  json.dumps(answer)[:500])
            sys.exit(1)
        total_neu += int(answer.get("neu", 0))
        total_akt += int(answer.get("aktualisiert", 0))
        print(f"Block {i // CHUNK + 1}: empfangen={answer.get('empfangen')}, "
              f"neu={answer.get('neu')}, aktualisiert={answer.get('aktualisiert')}")
    print(f"Fertig: {len(rows)} Tage gesendet ({total_neu} neu, "
          f"{total_akt} aktualisiert).")


if __name__ == "__main__":
    main()
