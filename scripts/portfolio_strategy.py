#!/usr/bin/env python3
"""Multi-Asset-Erweiterung: Champion-Tagesstrategie je Bein + Equal-Vol-Portfolio.

Je Symbol laeuft die UNVERAENDERTE Champion-Variante (long/short Momentum +
Vol-Targeting 40%, Cap 1.0, ohne Funding-Overlay) aus btc_predict/daily.py.
Aggregation: fixer 50/50-Split der Netto-PnL-Stroeme.

Begruendung der Aggregation (Vol-Targeting JE BEIN, nicht auf Portfolioebene):
das validierte Champion-Sizing (40% Ziel-Vol, Cap 1.0) bleibt je Bein exakt
erhalten — build_positions/backtest werden nicht angefasst. Vol-Targeting auf
Portfolioebene wuerde implizit Hebel auf das ruhigere Bein legen und damit die
eingefrorene Spezifikation aendern. Preis dafuer: Cap 1.0 je Bein bei
50/50-Split deckelt das Portfolio-Exposure, erwarteter CAGR-Verzicht
36% -> ~28-32% ohne Hebel.

EINGEFRORENE Pass/Fail-Gates (vorab registriert, EIN Lauf, kein Nachjustieren):
  (a) ETH-Bein standalone: Netto-Sharpe > 0 ueber die Gesamt-Evalperiode
      (10 bps/Seite, identische Champion-Parameter).
  (b) Portfolio vs. BTC-only ueber die Gesamt-Evalperiode: MaxDD betragsmaessig
      kleiner UND Sharpe mindestens gleich gross.
  (c) Gepaarter Block-Bootstrap (Blocklaenge 30d, 2000 Draws) auf den taeglich
      GEPAARTEN PnL-Serien: dieselben Tagesbloecke werden fuer Portfolio und
      BTC-only gezogen (erhaelt die ~0.95-Korrelation; KEIN unabhaengiges
      Resampling beider Equity-Kurven). Statistik je Draw: Sharpe(Portfolio)
      - Sharpe(BTC-only). PASS, wenn der Anteil der Draws mit positiver
      Differenz >= 0.95. Zusatzinfo: CI der mittleren Differenz-PnL p.a.
  Zusatz-Gate fuer einen Live-Umstieg (wegen fehlenden Hebels):
      Portfolio-CAGR >= 0.8 x BTC-only-CAGR ueber die Gesamtperiode.

Vorab festgehalten: der modale Ausgang ist FAIL an Kriterium (c) — ein FAIL
ist kein Anlass fuer Gate-Diskussionen. Und ein PASS ist ausdruecklich KEIN
Signifikanznachweis und KEIN automatischer Champion-Wechsel: Live-Umstellung
nur, wenn der User den CAGR-Verzicht explizit gegen die DD-Reduktion abwaegt;
sonst bleibt BTC-only Champion. Bei Umstieg: ETH-Flips naiv zum Tagesschluss,
OHNE das BTC-kalibrierte Phase-2-Timing.

Rein informative, vorab registrierte Sensitivitaeten (kein Gate):
  - ETH-Bein mit 20 bps/Seite (10 bps fuer ETH 2018 ist grenzwertig optimistisch)
  - ETH ohne Shorts vor 2019-10 (Binance-Margin/Perp existierte 2018 nicht)

    python scripts/portfolio_strategy.py                     # echter Lauf (Colab)
    python scripts/portfolio_strategy.py --synthetic         # Smoke-Test ohne Netz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_predict import data as data_mod  # noqa: E402
from btc_predict.daily import (  # noqa: E402
    backtest, build_positions, load_daily, report)

# Subperioden wie beim BTC-Champion-Lauf — Pflichtoutput fuer ETH standalone
# UND Portfolio; keine nachtraegliche Umdeutung einzelner Subperioden.
PERIODS = [("gesamt", None),
           ("2018-2020", ("2018-01-01", "2020-12-31")),
           ("2021-2022", ("2021-01-01", "2022-12-31")),
           ("2023-2024", ("2023-01-01", "2024-12-31")),
           ("2025-2026 (Haertetest)", ("2025-01-01", "2026-12-31"))]

# Feste Seeds je Symbol fuer den synthetischen Modus (Pipeline-Smoke-Test).
SYNTH_SEEDS = {"BTCUSDT": 7, "ETHUSDT": 11}


def load_leg_daily(symbol: str, warmup_start: str, end: str, data_dir: Path,
                   synthetic: bool = False) -> pd.DataFrame:
    """Tagesdaten eines Symbols ab warmup_start (Indikator-Warmup!)."""
    if synthetic:
        bars = data_mod.make_synthetic(warmup_start, end,
                                       seed=SYNTH_SEEDS.get(symbol, 99))
        return data_mod.resample(bars, "1d")
    return load_daily(symbol, warmup_start, end, data_dir)


def run_leg(daily: pd.DataFrame, eval_start: str, eval_end: str,
            target_vol: float, allow_short: bool,
            cost_bps: float) -> pd.DataFrame:
    """Champion-Positionen + Backtest auf der VOLLEN Historie (inkl. Warmup),
    PnL-Zaehlung erst ab eval_start — so sind SMA50/Donchian20/20d-Vol am
    ersten Eval-Tag bereits gefuellt."""
    frame = build_positions(daily, None, target_vol=target_vol,
                            allow_short=allow_short)
    bt = backtest(frame, cost_bps)
    return bt.loc[eval_start:eval_end]


def strip_shorts_before(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Sensitivitaet Shortability: negative Positionen vor cutoff auf 0
    (Ausweis-Pflicht, kein Parameter-Tuning; build_positions bleibt unberuehrt)."""
    out = frame.copy()
    mask = (out.index < pd.Timestamp(cutoff, tz=out.index.tz)) & (out["pos"] < 0)
    out.loc[mask, "pos"] = 0.0
    return out


def build_portfolio(legs: dict[str, pd.DataFrame],
                    weights: dict[str, float]) -> pd.DataFrame:
    """Fixer 50/50-Split der Netto-PnL-Stroeme (gemeinsame Handelstage)."""
    idx = None
    for bt in legs.values():
        idx = bt.index if idx is None else idx.intersection(bt.index)
    cols = {}
    for c in ("pnl", "ret", "pos", "turnover"):
        cols[c] = sum(weights[s] * legs[s].loc[idx, c] for s in legs)
    return pd.DataFrame(cols, index=idx)


def paired_block_bootstrap(pnl_pf: pd.Series, pnl_base: pd.Series,
                           block_days: int = 30, n_draws: int = 2000,
                           seed: int = 42) -> dict:
    """Gepaarter Block-Bootstrap: zieht Bloecke von TAGES-INDIZES und wertet
    beide Serien auf denselben Tagen aus — die Paarung (und damit die hohe
    Korrelation der Stroeme) bleibt erhalten. Kein unabhaengiges Resampling
    der beiden Equity-Kurven.

    Rueckgabe je Draw: Sharpe-Differenz (Portfolio minus Basis) und mittlere
    Differenz-PnL p.a. (aequivalent zum Bootstrap der Differenz-Serie, da
    Tag fuer Tag gepaart gezogen wird)."""
    joined = pd.concat([pnl_pf.rename("pf"), pnl_base.rename("base")],
                       axis=1, join="inner").dropna()
    a = joined["pf"].to_numpy()
    b = joined["base"].to_numpy()
    n = len(a)
    if n <= block_days:
        raise ValueError(f"Serie zu kurz fuer Blocklaenge {block_days}: n={n}")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    offsets = np.arange(block_days)
    sharpe_diff = np.empty(n_draws)
    mean_diff_pa = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, n - block_days + 1, size=n_blocks)
        idx = (starts[:, None] + offsets).reshape(-1)[:n]
        da, db = a[idx], b[idx]
        sd_a, sd_b = da.std(), db.std()
        sa = da.mean() / sd_a * np.sqrt(365) if sd_a > 0 else 0.0
        sb = db.mean() / sd_b * np.sqrt(365) if sd_b > 0 else 0.0
        sharpe_diff[i] = sa - sb
        mean_diff_pa[i] = (da - db).mean() * 365
    return {"sharpe_diff": sharpe_diff, "mean_diff_pa": mean_diff_pa,
            "n_days": n}


def subperiod_table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rng in PERIODS:
        seg = bt if rng is None else bt.loc[rng[0]:rng[1]]
        if len(seg) > 30:
            rows.append(report(seg, label))
    return pd.DataFrame(rows)


def full_period_stats(bt: pd.DataFrame) -> dict:
    """Gesamtperioden-Kennzahlen als Zahlen (fuer die Gates)."""
    return report(bt, "gesamt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/portfolio.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="Synthetische Daten (make_synthetic) statt Binance — "
                         "reiner Pipeline-Smoke-Test, kuerzeres Fenster")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    symbols: list[str] = cfg["symbols"]
    weights = dict(zip(symbols, cfg["weights"]))
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "Gewichte muessen 1 ergeben"
    warmup_start, eval_start, eval_end = (
        cfg["warmup_start"], cfg["eval_start"], cfg["eval_end"])
    if args.synthetic:
        # Kurzes Fenster, gleiche Codepfade — kein 1m-Datenberg ueber 9 Jahre.
        warmup_start, eval_start, eval_end = (
            "2021-01-01", "2021-06-01", "2022-12-31")
    cost = float(cfg["cost_bps_per_side"])
    tvol = float(cfg["target_vol"])
    allow_short = bool(cfg["allow_short"])
    data_dir = Path(cfg.get("data_dir", "data"))

    # ----- Beine rechnen (Champion unveraendert, Warmup vor eval_start) -----
    dailies, legs = {}, {}
    for sym in symbols:
        d = load_leg_daily(sym, warmup_start, eval_end, data_dir,
                           synthetic=args.synthetic)
        n_warm = int((d.index < pd.Timestamp(eval_start, tz=d.index.tz)).sum())
        assert n_warm >= 50, (
            f"{sym}: nur {n_warm} Warmup-Tage vor {eval_start} — SMA50 waere "
            f"am Eval-Start nicht gefuellt")
        dailies[sym] = d
        legs[sym] = run_leg(d, eval_start, eval_end, tvol, allow_short, cost)
        print(f"{sym}: Daten {d.index[0].date()} .. {d.index[-1].date()}, "
              f"Eval {legs[sym].index[0].date()} .. {legs[sym].index[-1].date()} "
              f"({len(legs[sym])} Tage, Warmup {n_warm} Tage)")

    pf = build_portfolio(legs, weights)
    btc = legs[symbols[0]]

    # ----- Pflicht-Tabellen: je Bein + Portfolio -----
    for sym in symbols:
        print(f"\n== Bein {sym} standalone | Champion long/short, "
              f"{cost:g} bps/Seite ==")
        print(subperiod_table(legs[sym]).to_string(index=False))
    print(f"\n== Portfolio {'/'.join(symbols)} "
          f"({'/'.join(str(weights[s]) for s in symbols)}) ==")
    print(subperiod_table(pf).to_string(index=False))

    # ----- Korrelation der PnL-Stroeme -----
    if len(symbols) >= 2:
        s0, s1 = symbols[0], symbols[1]
        both = pd.concat([legs[s0]["pnl"].rename("a"),
                          legs[s1]["pnl"].rename("b"),
                          legs[s0]["ret"].rename("ra"),
                          legs[s1]["ret"].rename("rb")], axis=1,
                         join="inner").dropna()
        print(f"\nKorrelation Tages-PnL {s0} vs {s1}: "
              f"{both['a'].corr(both['b']):.3f} | "
              f"Underlying-Returns: {both['ra'].corr(both['rb']):.3f}")

    # ----- Informative Sensitivitaeten (vorab registriert, KEIN Gate) -----
    eth = symbols[1] if len(symbols) >= 2 else None
    if eth is not None:
        sens_cost = float(cfg.get("eth_cost_sens_bps", 20.0))
        eth_frame = build_positions(dailies[eth], None, target_vol=tvol,
                                    allow_short=allow_short)
        eth_20 = backtest(eth_frame, sens_cost).loc[eval_start:eval_end]
        print(f"\n== Sensitivitaet (informativ): {eth} standalone mit "
              f"{sens_cost:g} bps/Seite ==")
        print(subperiod_table(eth_20).to_string(index=False))

        cutoff = str(cfg.get("eth_no_short_before", "2019-10-01"))
        eth_ns = backtest(strip_shorts_before(eth_frame, cutoff),
                          cost).loc[eval_start:eval_end]
        print(f"\n== Sensitivitaet (informativ): {eth} ohne Shorts vor "
              f"{cutoff} (Binance-Margin/Perp existierte 2018 nicht) ==")
        print(subperiod_table(eth_ns).to_string(index=False))

    # ----- Gates (eingefroren, siehe Docstring) -----
    st_btc = full_period_stats(btc)
    st_pf = full_period_stats(pf)
    st_eth = full_period_stats(legs[eth]) if eth is not None else None

    bs_cfg = cfg.get("bootstrap", {})
    bs = paired_block_bootstrap(
        pf["pnl"], btc["pnl"],
        block_days=int(bs_cfg.get("block_days", 30)),
        n_draws=int(bs_cfg.get("n_draws", 2000)),
        seed=int(bs_cfg.get("seed", 42)))
    share_pos = float((bs["sharpe_diff"] > 0).mean())
    ci_lo, ci_hi = np.percentile(bs["mean_diff_pa"], [2.5, 97.5])

    gate_a = st_eth is not None and st_eth["sharpe"] > 0
    gate_b = (abs(st_pf["max_dd_pct"]) < abs(st_btc["max_dd_pct"])
              and st_pf["sharpe"] >= st_btc["sharpe"])
    gate_c = share_pos >= 0.95
    gate_live = st_pf["cagr_pct"] >= 0.8 * st_btc["cagr_pct"]

    print("\n== Gates (vorab registriert, EIN Lauf) ==")
    if st_eth is not None:
        print(f"(a) ETH-Bein Sharpe > 0 ueber Gesamtperiode: "
              f"Sharpe={st_eth['sharpe']:.2f} -> {'PASS' if gate_a else 'FAIL'}")
    print(f"(b) Portfolio MaxDD < BTC-only UND Sharpe >= BTC-only: "
          f"DD {st_pf['max_dd_pct']:.1f}% vs {st_btc['max_dd_pct']:.1f}%, "
          f"Sharpe {st_pf['sharpe']:.2f} vs {st_btc['sharpe']:.2f} "
          f"-> {'PASS' if gate_b else 'FAIL'}")
    print(f"(c) gepaarter Block-Bootstrap ({bs_cfg.get('block_days', 30)}d, "
          f"{bs_cfg.get('n_draws', 2000)} Draws, {bs['n_days']} Tage): "
          f"Anteil Draws mit Sharpe(PF) > Sharpe(BTC) = {share_pos:.3f} "
          f"(Soll >= 0.95) -> {'PASS' if gate_c else 'FAIL'}")
    print(f"    Info: Differenz-PnL p.a. (PF - BTC) 95%-CI "
          f"[{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%] (log)")
    print(f"Zusatz-Gate Live-Umstieg: Portfolio-CAGR >= 0.8 x BTC-CAGR: "
          f"{st_pf['cagr_pct']:.1f}% vs 0.8*{st_btc['cagr_pct']:.1f}% "
          f"-> {'PASS' if gate_live else 'FAIL'}")
    print("\nLesart: Ein PASS ist KEIN Signifikanznachweis und KEIN")
    print("automatischer Champion-Wechsel — Live-Umstellung nur, wenn der User")
    print("den CAGR-Verzicht explizit gegen die DD-Reduktion abwaegt; sonst")
    print("bleibt BTC-only Champion. Der modale erwartete Ausgang ist FAIL an")
    print("(c); das ist dann kein Anlass fuer Gate-Diskussionen. Bei Umstieg:")
    print("ETH-Flips naiv zum Tagesschluss, ohne Phase-2-Timing.")


if __name__ == "__main__":
    main()
