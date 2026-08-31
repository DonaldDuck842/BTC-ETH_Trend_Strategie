# BTC-Bot — Kurzfrist-Preisprediction (1m/5m)

ML-Pipeline, die aus historischen Binance-Klines die **nächste kurzfristige
Preisbewegung** von BTC klassifiziert (up / flat / down über die nächsten
N Bars) und das Ergebnis **kostenbewusst** out-of-sample bewertet.

Trainiert wird auf **je einem Jahr pro Marktregime** (High-Vol 2021,
Bär/Chop 2022, Bull-Trend Okt 2023–Sep 2024), getestet auf einem frei
wählbaren Monat außerhalb dieser Fenster.

## Quickstart

```bash
pip install -r requirements.txt

# 1) Daten laden (Binance Vision, kein API-Key nötig; ~36 Monate 1m-Klines)
python scripts/download_data.py --test-month 2025-03

# 2) Training + Out-of-Sample-Test
python scripts/run_experiment.py --test-month 2025-03

# Pipeline ohne Netzzugang prüfen (synthetische Daten, muss ~0 Edge zeigen):
python scripts/run_experiment.py --synthetic
```

In **Google Colab**: Repo klonen, `pip install -r requirements.txt`,
dieselben zwei Befehle — Binance Vision ist von Colab aus erreichbar.

## Aufbau

| Modul | Aufgabe |
|---|---|
| `btc_predict/data.py` | Download Binance-Vision-Monats-Zips → Parquet-Cache; 1m→5m-Resampling; synthetischer Datengenerator |
| `btc_predict/features.py` | ~30 leakage-freie Features: Returns, Vol, RSI, Bollinger, VWAP-Distanz, Volumen-Z-Scores, **Taker-Buy-Ratio als Order-Flow-Proxy**, Candle-Geometrie, Tageszeit |
| `btc_predict/dataset.py` | Regime-Fenster laden, Features+Labels bauen, Überschneidung Training/Test hart verbieten |
| `btc_predict/model.py` | LightGBM-Multiclass, Validierung = zeitliches Tail-Split je Regime (kein Random-Split) |
| `btc_predict/evaluate.py` | Klassifikationsmetriken + PnL-Simulation mit Fees/Slippage + **Snooping-Log** |
| `btc_predict/path_model.py` | Multi-Horizont-**Pfadprognose**: Quantilregression je Horizont (Median-Pfad + 10–90%-Kegel) plus erwartete Hoch-/Tief-Exkursion („Dochte") |
| `configs/default.yaml` | Alle Stellschrauben: Timeframe, Regime-Fenster, Label-Horizont/-Schwelle, Kosten, Signal-Schwelle, Pfad-Horizonte |

Zwei Skripte:

- `scripts/run_experiment.py` — Richtungsmodell (up/flat/down) + kostenbewusster
  Backtest + Schwellen/Kosten-Sweep.
- `scripts/predict_path.py` — Pfadmodell: 6 Horizonte × 3 Quantile direkt
  (kein rekursives Kerze-auf-Kerze-Verketten — Fehler wuerden sich potenzieren
  und prognostizierte Kerzen haben kein echtes Volumen als Feature-Input),
  Kalibrierungs-Report (Coverage, Pinball-Skill) + Beispiel-Fanchart. |

## Label-Definition

3 Klassen aus der Vorwärtsrendite über `horizon_bars` (Default: 3 Bars à 5m
= 15 Minuten): **up** wenn > +10 bps, **down** wenn < −10 bps, sonst
**flat**. Die Schwelle entspricht bewusst den Round-Trip-Kosten — „up"
heißt „handelbar aufwärts", nicht nur „ein Tick höher".

## Methodik-Regeln (wichtig!)

1. **Erfolgskriterium ist Netto-PnL, nicht Accuracy.** Ein 1m/5m-Signal
   liefert brutto typischerweise wenige bps pro Trade; Taker-Fees +
   Slippage (Default 5 bps/Seite) entscheiden über Leben und Tod. Die
   Simulation weist Brutto, Kosten und Netto getrennt aus.
2. **Testmonat ≠ Trainingsregime.** `assert_no_overlap` bricht hart ab,
   wenn sich Fenster überschneiden. Zusätzlich Embargo von 50 Bars am
   Monatsanfang gegen Label-/Feature-Leakage an der Naht.
3. **„Testen und dann verbessern" hat einen Preis.** Jede Iteration auf
   demselben Testmonat macht ihn schleichend zum Trainingsset
   (Data-Snooping). Das Snooping-Log (`results/eval_log.csv`) zählt mit
   und warnt ab der zweiten Bewertung desselben Monats. Praxisregel:
   - Entwicklungs-Iterationen auf **rotierenden** Monaten (z. B. 2025-01,
     2025-02, …),
   - **ein finaler, nie angefasster Monat** als letzte Instanz — wird der
     nur einmal getestet, ist das Ergebnis glaubwürdig.
4. **Zufalls-Nullpunkt kennen:** Balanced Accuracy 0.333. Der
   Synthetik-Modus (`--synthetic`) muss immer ≈ Zufall + negativen
   Netto-PnL liefern; tut er das nicht, ist Leakage in der Pipeline.

## Ehrliche Erwartungshaltung

Kurzfrist-Prediction auf 1m/5m ist die härteste Liga: Das akademisch am
besten belegte Signal in diesem Horizont ist Order-Flow-Imbalance, und
selbst das liefert brutto nur ~1–3 bps — weniger als eine Taker-Fee.
Realistische Ziele für dieses Projekt, in dieser Reihenfolge:

1. Zeigen, ob die Features **überhaupt** out-of-sample Prädiktivität haben
   (Balanced Accuracy stabil > 0.34–0.35 über mehrere Testmonate).
2. Prüfen, ob nach Kosten etwas übrig bleibt — vermutlich nur mit
   selektiven Signalen (hohe `signal_threshold`, wenige Trades) und/oder
   auf 5m statt 1m.
3. Erst wenn 1+2 bestehen: Ausführungsseite optimieren (Maker-Orders,
   Perp statt Spot wegen niedrigerer Fees).

Echte L2-Orderbuch-Historie (für Order-Flow-Imbalance im engeren Sinn) ist
nicht frei verfügbar (Tardis.dev u. ä. sind kostenpflichtig); die
Taker-Buy-Ratio aus den Klines ist der beste freie Proxy und ist als
Feature enthalten.
