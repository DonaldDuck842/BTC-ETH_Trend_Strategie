# BTC-Bot — Trend-Autopilot für Bitcoin & Ethereum

[![Daily Signal](https://github.com/DonaldDuck842/BTC-Bot/actions/workflows/daily-signal.yml/badge.svg)](https://github.com/DonaldDuck842/BTC-Bot/actions/workflows/daily-signal.yml)

Ein systematischer Trendfolge-Handelsansatz für BTC und ETH auf Tagesbasis —
vollständig backtested (2018–2026), mit vorab registrierten Kriterien
validiert und täglich automatisch ausgeführt (aktuell im Paper-Trading).

## 📊 Live-Status

| Was | Wo |
|---|---|
| **Testportfolio-Performance** (fiktive 100.000 USD, für Web-Einbindung) | [`results/performance.json`](results/performance.json) · [`results/equity.csv`](results/equity.csv) |
| **Paper-Trading-Log** (täglich aktualisiert) | [`results/paper_log.csv`](results/paper_log.csv) |
| **Täglicher Signal-Lauf** (00:10 UTC, automatisch) | [Actions → Daily Signal](../../actions/workflows/daily-signal.yml) |
| Tagesreport zum Nachlesen | als Artifact an jedem Action-Lauf |
| Phase | **Paper-Trading** (Ziel: ≥ 60 protokollierte Tage vor Go-Live) |
| Strategie-Status | **Eingefroren seit 2026-09-28** — siehe [Betriebsregeln](docs/BETRIEBSREGELN.md) |

Das Log enthält pro Tag und Coin die Zielposition des Systems und den
Referenzkurs (`ref_close`, Tagesschluss in USD). **Einheit von
`target_pos`: Anteil des Kapitals dieses Beins** (jedes Bein = 50 % des
Gesamtkapitals), zwischen −1 und +1; negativ = short, nie über ±1 (kein
Hebel). Beispiel: `target_pos = 0.8` bei BTC heißt 80 % des BTC-Beins
investiert = 40 % des Gesamtkapitals. Der Tagesreport im Action-Log
zeigt dieselbe Zahl bereits aufs Gesamtkapital umgerechnet. Daraus
rechnet der tägliche Report den bisherigen Paper-PnL.

**Web-Einbindung:** Das Testportfolio (fiktive 100.000 USD, kein echtes
Geld) wird täglich fortgeschrieben. Eine Webseite kann die Daten direkt
laden — CORS ist auf raw.githubusercontent.com offen:

```js
fetch("https://raw.githubusercontent.com/DonaldDuck842/BTC-Bot/claude/btc-price-prediction-model-de6mnf/results/performance.json")
  .then(r => r.json())
  .then(p => console.log(p.current_equity_usd, p.total_return_pct, p.series));
```

## Was macht das System?

Zwei einfache, seit Jahrzehnten dokumentierte Prinzipien — nichts anderes:

1. **Trendfolge:** Pro Coin schauen 7 Trendfilter unterschiedlicher
   Geschwindigkeit (≈1–7 Monate Rückblick) auf den Tageschart. Die Position
   ist der Durchschnitt ihrer Stimmen: einstimmig = volle Richtung,
   gespalten = kleine Position, in Siebtel-Schritten. Long und Short.
2. **Volatilitäts-Steuerung:** Die Positionsgröße wird durch die aktuelle
   Schwankungsbreite geteilt (Ziel ≈ 40 % Jahres-Vol, nie über 1× Kapital,
   kein Hebel). Wird der Markt wild, wird automatisch kleiner gesetzt.

Gehandelt wird **einmal täglich** zum Tagesbeginn (Signal vom Vortages-
Schluss), 50 % des Kapitals je Coin. Das System sagt **keine Kurse
voraus** — alle Kurzfrist-Prognoseansätze wurden getestet und sind an den
Handelskosten gescheitert (vollständig dokumentiert im
[Forschungsprotokoll](docs/RESEARCH_LOG.md)).

## Kennzahlen (Backtest 2018–2026, nach 10 bps Kosten je Seite)

| Kennzahl | System | Buy-and-Hold 50/50 |
|---|---|---|
| Rendite p.a. (CAGR) | **+40.2 %** | +15.4 % |
| Sharpe-Ratio | **1.03** | 0.20 |
| Max. Drawdown | **−32.3 %** (Stress-Test: realistisch −40 %, selten bis −57 %) | −88.3 % |
| Schwächste Subperiode | +20.3 % p.a. (2023–24) | −26.7 % p.a. (2025–26) |
| Bärenmarkt-Härtetest 2025–26 | **+22.9 % p.a.** | −26.7 % p.a. |

Ehrliche Einordnung: Das System ist in **jeder** Marktphase positiv
geblieben — bezahlt wird das damit, dass es in extremen Bullenjahren weit
hinter Buy-and-Hold zurückbleibt (2023–24: +20 % vs. +99 %). Ø ~52 %
positive Tage; verdient wird über wenige große Trends, nicht über eine
hohe Trefferquote.

## Repo-Struktur

**🟢 Betrieb (läuft täglich)**
| Datei | Aufgabe |
|---|---|
| `scripts/live_signal.py` | Täglicher Report: Zielpositionen, Größen, Orientierungs-Stops; schreibt das Paper-Log |
| `.github/workflows/daily-signal.yml` | Automatisierung (00:10 UTC), committet das Log ins Repo |
| `results/paper_log.csv` | **Das Live-Dokument** — ein Eintrag pro Tag und Coin |

**🔵 Strategie-Kern**
| Datei | Aufgabe |
|---|---|
| `btc_predict/daily.py` | Die finale Strategie: 7-Member-Lookback-Ensemble, Vol-Targeting, Backtest, Reports |
| `btc_predict/data.py` | Datenbeschaffung (Binance Vision + API), Parquet-Cache, synthetischer Testdaten-Generator |

**🟡 Analyse & Studien (reproduzierbar, mit eingefrorenen Kriterien)**
| Datei | Aufgabe |
|---|---|
| `scripts/daily_strategy.py` | Varianten-Vergleich der Tagesstrategie + Ensemble-Studie |
| `scripts/portfolio_strategy.py` | BTC/ETH-Portfolio-Studie inkl. finaler Spezifikations-Entscheidung |
| `scripts/trade_stats.py` | Win-Rate, Episoden-Statistik, Payoff-Ratio |
| `scripts/ewma_study.py` | EWMA-Vol-Sizing-Studie (durchgefallen, beerdigt) |
| `scripts/validate_final.py` | Mehrmonats-Validierung mit eingefrorener Config |

**⚪ Forschungs-Archiv (beerdigte Kurzfrist-Ansätze — bewusst erhalten)**
| Datei | Aufgabe |
|---|---|
| `btc_predict/{features,dataset,model,evaluate}.py` | 5m-ML-Pipeline (Richtungsklassifikator — durchgefallen) |
| `btc_predict/path_model.py` | 5m-Pfadmodell (validiert: kalibrierter Unsicherheitskegel; als Anzeige-/Stop-Werkzeug nutzbar) |
| `scripts/{run_experiment,run_combined,predict_path,phase2_execution}.py` | Experimente der Kurzfrist-Phase |
| `configs/*.yaml` | Eingefrorene Experiment-Konfigurationen |

## Selbst nachrechnen

Alle Ergebnisse sind mit freien Daten reproduzierbar (lokal oder Google
Colab, kein API-Key nötig):

```bash
git clone https://github.com/DonaldDuck842/BTC-Bot
cd BTC-Bot && pip install -r requirements.txt

python scripts/daily_strategy.py       # Tagesstrategie + Ensemble-Vergleich
python scripts/portfolio_strategy.py   # BTC/ETH-Portfolio + finale Studie
python scripts/trade_stats.py          # Win-Rate & Episoden-Statistik
python scripts/live_signal.py          # heutiges Signal (wie die Action)
```

Jedes Skript hat einen `--synthetic`-Modus als Selbsttest: Auf zufälligen
Daten muss die Pipeline ≈ null Edge zeigen — tut sie das nicht, wäre ein
Leck im Code.

## Regeln & Protokolle

Dieses Projekt lebt von Disziplin, nicht von Prognosekunst:

- **[Betriebsregeln](docs/BETRIEBSREGELN.md)** — Drawdown-Protokoll,
  Paper-Phase & Go-Live-Kriterien, Wechselregel zwischen den Büchern,
  jährliche Revalidierung, News-Disziplin.
- **[Forschungsprotokoll](docs/RESEARCH_LOG.md)** — jede getestete Idee
  mit vorab registriertem Kriterium und Verdikt: 2 adoptiert,
  21 beerdigt. Beerdigt heißt beerdigt.

Die wichtigste Regel zuletzt: **Am laufenden System wird nicht gebastelt.**
Jede Änderung geht ausschließlich über eine neue, vorab registrierte Studie.
