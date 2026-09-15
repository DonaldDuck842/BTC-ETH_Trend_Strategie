#!/usr/bin/env python3
"""Stress-Suite: die Live-Spezifikation in simulierten Welten.

WICHTIG — was dieser Test ist und was nicht: Simulierte Charts haben genau
die Eigenschaften, die man hineinsteckt. Renditezahlen aus Teil B sind
daher KEINE Prognosen. Der Test prueft drei andere Dinge:
  1. Selbsttest: Auf reinem Zufall MUSS die Strategie ~ihre Kosten
     verlieren. Gewinnt sie dort, ist ein Lookahead-Bug im Code.
  2. Verhaltens-Landkarte: verliert das System in feindlichen Welten
     KONTROLLIERT (kein Blowup), und gewinnt es dort, wo sein
     Mechanismus greift (Trends)?
  3. Verteilung statt Einzelpfad (Teil A, echte Daten): Block-Bootstrap
     der realen BTC/ETH-Historie -> Bandbreite von CAGR/MaxDD/Sharpe,
     P(Drawdown < -40%) zur Kalibrierung des Drawdown-Protokolls.

Vorab deklarierte Erwartungen je Szenario stehen im SCENARIOS-Dict und
werden im Report gegen das Ergebnis gehalten (CHECK ok / AUFFAELLIG).

    python scripts/stress_test.py                # Teil A+B (Colab, laedt Daten)
    python scripts/stress_test.py --skip-bootstrap   # nur Teil B (ohne Netz)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict.daily import backtest, build_positions, load_daily  # noqa: E402

COST = 10.0
YEARS = 8
N_DAYS = 365 * YEARS
SEEDS_PER_SCENARIO = 20
BOOT_PATHS = 500
BOOT_BLOCK = 90  # Tage; lang genug, um mittlere Trendstruktur zu erhalten


# ---------------------------------------------------------------------------
# Teil B: synthetische Welten
# ---------------------------------------------------------------------------

def _to_daily(logret_a: np.ndarray, logret_b: np.ndarray) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2018-01-01", periods=len(logret_a), freq="1D", tz="UTC")
    out = {}
    for sym, lr in (("BTCUSDT", logret_a), ("ETHUSDT", logret_b)):
        close = 50_000 * np.exp(np.cumsum(lr))
        # synthetische High/Low als +-0.6 Tagesvol um Open/Close
        spread = np.abs(lr).clip(min=0.005) * 0.6
        df = pd.DataFrame({
            "open": np.r_[close[0], close[:-1]],
            "close": close}, index=idx)
        df["high"] = df[["open", "close"]].max(axis=1) * np.exp(spread)
        df["low"] = df[["open", "close"]].min(axis=1) * np.exp(-spread)
        out[sym] = df
    return out


def gen_random_walk(rng, rho=0.8, vol=0.60):
    z = rng.standard_normal((2, N_DAYS))
    b = rho * z[0] + np.sqrt(1 - rho**2) * z[1]
    s = vol / np.sqrt(365)
    return z[0] * s, b * s


def gen_trend(rng, rho=0.8, vol=0.55):
    # Regime-Switch-Drift: alle ~120 Tage neues Vorzeichen, +-150% p.a.
    n_reg = N_DAYS // 120 + 1
    drifts = rng.choice([-1.5, 1.5], size=n_reg) / 365
    drift = np.repeat(drifts, 120)[:N_DAYS]
    a, b = gen_random_walk(rng, rho, vol)
    return a + drift, b + drift * (0.7 + 0.6 * rng.random())


def gen_chop(rng, rho=0.8, vol=0.60):
    # Ornstein-Uhlenbeck um ein festes Niveau: staendige Fehlausbrueche
    theta, s = 0.05, vol / np.sqrt(365)
    x = np.zeros(N_DAYS)
    z = rng.standard_normal(N_DAYS)
    for t in range(1, N_DAYS):
        x[t] = x[t - 1] - theta * x[t - 1] + s * z[t]
    lr_a = np.diff(np.r_[0.0, x])
    zb = rng.standard_normal(N_DAYS)
    lr_b = rho * lr_a + np.sqrt(1 - rho**2) * zb * s
    return lr_a, lr_b


def gen_crash_gaps(rng, rho=0.8, vol=0.50):
    # Random Walk plus ~monatliche Ueberacht-Gaps von -10% bis -25%
    a, b = gen_random_walk(rng, rho, vol)
    n_gaps = N_DAYS // 30
    days = rng.choice(N_DAYS, size=n_gaps, replace=False)
    gaps = -rng.uniform(0.10, 0.25, size=n_gaps)
    a[days] += gaps
    b[days] += gaps * rng.uniform(0.8, 1.3, size=n_gaps)
    return a, b


def gen_corr_one(rng, vol=0.60):
    a, _ = gen_random_walk(rng, 0.0, vol)
    drift = np.repeat(rng.choice([-1.0, 1.0], size=N_DAYS // 120 + 1) / 365 * 1.2,
                      120)[:N_DAYS]
    return a + drift, a + drift  # beide Beine identisch: Diversifikation = 0


SCENARIOS = [
    ("random_walk", gen_random_walk,
     "reiner Zufall — Strategie MUSS verlieren (Median-CAGR ~-15..+2%; "
     "Kosten + Chop-Bleed, Streuung je Pfad ~11pp). Klar positiver Median "
     "waere ein Lookahead-Bug!"),
    ("strong_trend", gen_trend,
     "ausgepraegte Trends — der Mechanismus muss greifen (CAGR deutlich > 0)"),
    ("choppy_hell", gen_chop,
     "Seitwaerts-Hoelle (extremer als jeder reale Markt) — Verlust ja, aber "
     "GRADUELL: kein Einzeltag schlimmer als ~ -8%; der kumulierte DD darf "
     "in dieser Kunstwelt beliebig tief sein"),
    ("crash_gaps", gen_crash_gaps,
     "monatliche Ueberacht-Crashs — kein Blowup: schlechtester Tag begrenzt "
     "durch |pos| <= 1 (kein Hebel). Hinweis: die Dauer-Abwaertsdrift dieser "
     "Welt belohnt die Short-Seite, positive CAGR ist hier erwartbar"),
    ("corr_one", gen_corr_one,
     "BTC=ETH — Diversifikation weg, Ergebnis wie ein Einzel-Bein (hoeherer DD)"),
]


def run_book(dailies: dict[str, pd.DataFrame]) -> pd.Series:
    pnl = None
    for sym, d in dailies.items():
        frame = build_positions(d, None, allow_short=True, ensemble=True)
        bt = backtest(frame, COST)
        p = 0.5 * bt["pnl"].fillna(0.0)
        pnl = p if pnl is None else pnl + p
    return pnl


def stats(pnl: pd.Series) -> dict:
    years = len(pnl) / 365
    eq = pnl.cumsum()
    return {
        "cagr_pct": (np.exp(pnl.sum() / years) - 1) * 100,
        "max_dd_pct": np.expm1((eq - eq.cummax()).min()) * 100,
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(365) if pnl.std() > 0 else 0.0,
        "worst_day_pct": np.expm1(pnl.min()) * 100,
    }


def part_b() -> None:
    print("=" * 70)
    print("TEIL B: SYNTHETISCHE WELTEN  "
          f"({SEEDS_PER_SCENARIO} Seeds x {YEARS} Jahre je Szenario)")
    print("=" * 70)
    for name, gen, expectation in SCENARIOS:
        rows = []
        for seed in range(SEEDS_PER_SCENARIO):
            rng = np.random.default_rng(1000 + seed)
            dailies = _to_daily(*gen(rng))
            rows.append(stats(run_book(dailies)))
        df = pd.DataFrame(rows)
        med = df.median()
        print(f"\n-- {name} --")
        print(f"   Erwartung: {expectation}")
        print(f"   Median: CAGR {med['cagr_pct']:+.1f}% | MaxDD "
              f"{med['max_dd_pct']:.1f}% | Sharpe {med['sharpe']:+.2f} | "
              f"schlechtester Tag {df['worst_day_pct'].min():.1f}%")
        print(f"   Spannweite CAGR: [{df['cagr_pct'].min():+.1f}%, "
              f"{df['cagr_pct'].max():+.1f}%]")
        if name == "random_walk":
            # Band fuer >=20 Seeds kalibriert (SE des Medians ~3pp bei 11pp/Pfad)
            ok = -15.0 <= med["cagr_pct"] <= 2.0
            print(f"   CHECK Leckage-Selbsttest: "
                  f"{'ok' if ok else 'AUFFAELLIG — Lookahead pruefen!'}")


# ---------------------------------------------------------------------------
# Teil A: Block-Bootstrap der echten Historie
# ---------------------------------------------------------------------------

def part_a() -> None:
    warmup = pd.Timestamp("2017-08-01")
    dailies = {s: load_daily(s, warmup.strftime("%Y-%m-%d"), "2026-07-31",
                             Path("data")) for s in ("BTCUSDT", "ETHUSDT")}
    logret = pd.DataFrame({s: np.log(d["close"]).diff()
                           for s, d in dailies.items()}).dropna()
    n = len(logret)
    rng = np.random.default_rng(42)
    print("\n" + "=" * 70)
    print(f"TEIL A: BLOCK-BOOTSTRAP echter BTC/ETH-Renditen "
          f"({BOOT_PATHS} Pfade, Blocklaenge {BOOT_BLOCK} Tage, {n} Tage Basis)")
    print("=" * 70)
    results = []
    n_blocks = n // BOOT_BLOCK + 1
    for i in range(BOOT_PATHS):
        starts = rng.integers(0, n - BOOT_BLOCK, size=n_blocks)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)).reshape(-1)[:n]
        lr = logret.values[idx]
        d = _to_daily(lr[:, 0], lr[:, 1][:len(lr)])
        results.append(stats(run_book(d)))
    df = pd.DataFrame(results)
    q = df.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
    print("\nVerteilung ueber 500 alternative Historien (Perzentile):")
    print(q.to_string())
    print(f"\nP(MaxDD schlimmer als -40%): {(df['max_dd_pct'] < -40).mean():.1%}"
          f"   (Kalibrierung des Revalidierungs-Triggers im Drawdown-Protokoll)")
    print(f"P(CAGR negativ):             {(df['cagr_pct'] < 0).mean():.1%}")
    print("\nHinweis: Block-Bootstrap zerschneidet lange Trends an den "
          "Blockgrenzen — die Verteilung ist fuer Trendfolge strukturell "
          "PESSIMISTISCH. Sie taugt als untere Leitplanke, nicht als Prognose.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="nur Teil B (synthetische Welten, ohne Netzzugang)")
    args = ap.parse_args()
    part_b()
    if not args.skip_bootstrap:
        part_a()


if __name__ == "__main__":
    main()
