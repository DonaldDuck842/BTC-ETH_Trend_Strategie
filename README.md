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

## Experiment-Status

- **5m-Richtung aus Klines-Features: DURCHGEFALLEN** (Validierung 2025-01..07,
  eingefrorene Config: 2/5 Monate positiv bei Maker-Kosten, Brutto-Edge
  instabil). Nicht weiter tunen.
- **Pfadmodell (Kegel + Exkursionen): VALIDIERT** (Coverage 0.79–0.81 bei
  Soll 0.80, Pinball-Skill +12–18 %).
- **OFI-Experiment: DURCHGEFALLEN, Kurzfrist-Richtung beerdigt.**
  Dev-Sweep 2025-08 zeigte eine einzelne Zelle (thr 0.4, 11.2 bps/RT) ohne
  Plateau; die eingefrorene Bestaetigung auf 2025-09 lieferte 2.4 bps/RT
  und negatives Netto bei Maker-Kosten. Echte aggTrades-Orderflow-Features
  erschienen nie in den Top-Importances. Damit sind Klines- UND
  Orderflow-Features fuer 1m/5m-Richtung mit vorab fixierten Kriterien
  auf frischen Monaten getestet und verworfen — nicht wieder aufmachen.
- **Champion: Tages-Strategie long/short Momentum + Vol-Targeting**
  (`btc_predict/daily.py`): einzige Variante mit positivem Ergebnis in
  allen Subperioden 2018-2026 (CAGR 36%, Sharpe 0.77, MaxDD -35% vs -81%
  Buy-and-Hold). Funding-Overlay verworfen (kostete ueberall Rendite).
- **Phase 2 (Einstiegs-Timing via Pfadmodell): DURCHGEFALLEN.** Gepoolt
  ueber alle Out-of-Training-Fenster (159 Flips, 5.6 Jahre): -5.6 bps/Flip,
  t=-1.38. Median +0.9 aber fetter linker Tail — beim Momentum-Ausbruch
  auf den Dip zu warten kostet gelegentlich brutal (2018-2020: -13.4 bps).
  Naive Ausfuehrung am Tagesbeginn bleibt. Das Pfadmodell wird fuer
  Groessen-/Exkursionsprognose genutzt (Stops, De-Risking), nicht fuer Timing.

## Verbesserungs-Review (adversarial geprueft)

Sechs Kandidaten, je ein Analyst + zwei unabhaengige Skeptiker. Ergebnis:

**Verworfen (in beiden Panel-Laeufen einstimmig):**
- *Kontinuierliche Positionsgroesse*: Kostenersparnis real, aber CAGR-Verlust
  4-8pp sicher, Sharpe-Gewinn unmessbar; tanh-Skalierung = neuer Freiheitsgrad.
- *Intraday-De-Risking via Pfadmodell*: Ziel-Events (Long + Kaskade) liegen
  fast alle in den Pfadmodell-Trainingsfenstern — Test haette Power ~0 und
  wuerde verbrauchte 5m-Monate erneut antasten; auf der Short-Seite wuerde
  er die Verdiensttage de-risken. Legitimer Kern (schnellere Vol-Reaktion)
  waere ein separater Kandidat: EWMA-Vol im Tages-Sizing.
- *Funding-Carry*: Gegen den ehrlichen Benchmark (4-5% risikoarm auf
  Brachkapital) schrumpft der Zusatznutzen auf ~0.5-1pp CAGR und kauft
  Perp-Liquidations- und Boersen-Gegenparteirisiko ein.
- *Maker-Ausfuehrung*: Fee-Arithmetik traegt auf dem unterstellten Venue
  nicht; Flips sind informationsgetriebener Flow (Adverse Selection),
  realistischer Nutzen +10-40 bps/Jahr — Live-Detail, kein Experiment.

**Studien-Ergebnisse (Colab-Lauf 2026-08-31, eingefrorene Gates):**
- *Lookback-Ensemble: ADOPTIERT als neuer Champion.* Alle drei Gates
  bestanden — (b) alle Subperioden positiv (10.0% im Haertetest 2025-26),
  (c) schlechtester Subperioden-Sharpe 0.29 vs 0.14, (d) Kosten-Drag
  3.70% vs 4.98% p.a. (1/7-Schritte erzeugen WENIGER Turnover als
  Voll-Flips). Adoption ausdruecklich wegen Spezifikations-Robustheit;
  die In-Sample-Deltas (Sharpe 0.89 vs 0.75) sind KEIN Evidenz-Claim.
  Live/Signale ab jetzt mit ensemble=True.
- *ETH-Portfolio: Gate (c) FAIL* (Bootstrap 0.893 < 0.95) — per
  Praeregistrierung kein Champion-Wechsel, keine Nachverhandlung.
  Protokollierte Fakten: ETH-Bein Sharpe 1.02 (robust gegen 20 bps
  Kosten und Shortability-Filter), Portfolio schlaegt BTC-only in
  Punktschaetzung bei Sharpe/DD/CAGR. Ein diskretionaerer 50/50-Split
  ist eine Risiko-Praeferenz-Option des Users, kein Evidenz-Claim;
  "Ensemble beide Beine + Portfolio" waere eine NEUE Studie.

**Beschluss (User, 2026-08-31): 50/50 BTC/ETH wird das Live-Buch.**
Die finale Spezifikations-Studie (Ensemble auf beiden Beinen, Gates E1-E3
in `scripts/portfolio_strategy.py`) entscheidet die letzte Design-Frage.
Alle drei Buecher (BTC-only, ETH-only, 50/50) werden dauerhaft
mitreportet — als Diagnose. **Vorab registrierte Wechselregel:** Ein
Umstieg auf ein Einzel-Bein (z.B. ETH-only) ist NUR zulaessig, wenn auf
mindestens 3 Jahren Paper-/Live-Daten (nicht Backtest!) der gepaarte
Block-Bootstrap der Sharpe-Differenz Einzel-Bein minus 50/50 einen Anteil
positiver Draws >= 0.95 zeigt. "ETH lief zuletzt besser" ist KEIN
Wechselgrund — Fuehrungsrotation zwischen den Beinen ist genau der
Mechanismus, von dem der Split lebt.

**FINALE LIVE-SPEZIFIKATION (Studie E1-E3 bestanden, 2026-08-31):**
50/50 BTCUSDT/ETHUSDT, je Bein Lookback-Ensemble long/short (7 Member,
1/7-Schritte) + Vol-Targeting 40% (Cap 1.0), Ausfuehrung naiv zum
naechsten Tagesbeginn, Kostenannahme 10 bps/Seite. Backtest 2018-2026:
Sharpe 1.03, CAGR 40.2%, MaxDD -32.3%, alle Subperioden positiv
(Haertetest 2025-26: +22.9%). Taeglicher Report: `scripts/live_signal.py`.

**EWMA-Vol-Sizing: DURCHGEFALLEN und beerdigt** (Colab-Lauf 2026-08-31):
Gate V2 knapp verfehlt (schlechtester Subperioden-Sharpe 0.59 vs 0.62),
alle Kennzahlen praktisch aequivalent (Sharpe 1.02 vs 1.03). Bei
Gleichstand bleibt der Amtsinhaber; kein Lambda-Tuning. Damit ist die
Forschungsphase geschlossen — alle offenen Kandidaten sind entschieden.

**News-Filter: NICHT VERFOLGT** (Begruendung protokolliert): Mechanismus
redundant zu Momentum + Vol-Targeting (der Preis weiss von der News
zuerst), keine saubere freie Point-in-Time-News-Historie (Look-Ahead in
kuratierten Archiven), strukturell der Zwilling des beerdigten
Funding-Overlays. News-Disziplin ist eine Betriebsregel, kein Signal.

**Betrieb (automatisiert):** `.github/workflows/daily-signal.yml` fuehrt
den taeglichen Lauf um 00:10 UTC aus und committet results/paper_log.csv
ins Repo (Marktdaten via data-api.binance.vision, geo-block-frei).

**Drawdown-Protokoll (vorab registriert, gilt ab Paper-Phase):**
Eingriffe ins laufende System passieren NUR nach dieser Regel, nie nach
Schlagzeile oder Gefuehl:
- Live-Drawdown bis -32% (historisches Maximum): normaler Betrieb,
  keine Aktion. Das ist der eingepreiste Preis der Strategie.
- Live-Drawdown unter -40%: Revalidierungs-Trigger — die Strategie
  laeuft regelkonform weiter, parallel wird eine ausserplanmaessige
  Out-of-Sample-Revalidierung gestartet; Abschalten erst nach deren
  Ergebnis, niemals am selben Tag.
- Kapital erhoehen/senken: nur zu Quartalsterminen, nie intra-Drawdown.
- Jaehrliche Revalidierung (jeweils Januar): die frischen 12 Monate als
  Out-of-Sample-Test der eingefrorenen Spezifikation; Kill-Kriterium:
  zwei aufeinanderfolgende Jahres-Tests mit negativem Netto-PnL BEI
  gleichzeitig positivem Buy-and-Hold-Umfeld => System gilt als
  zerfallen und wird abgeschaltet, nicht nachgetunt.

**Paper-Trading-Phase (vorab registriert):** Vor jedem Einsatz echten
Kapitals mindestens 60 protokollierte Handelstage via
`scripts/live_signal.py` (Log: results/paper_log.csv, extern sichern).
Go-Live-Kriterium ist OPERATIV, nicht performance-basiert: (1) >= 60
Log-Tage mit hoechstens 5 Luecken, (2) die geloggten Zielpositionen
reproduzieren sich aus den Tagesdaten (kein Implementierungsdrift),
(3) der User legt VOR Go-Live die Kapitalgroesse fest. Paper-PnL
entscheidet ausdruecklich NICHT ueber Go-Live — 60 Tage sind Rauschen;
die Phase prueft den Prozess, nicht die Strategie (die ist backtested).

**Urspruenglich als eingefrorene Backtest-Studien implementiert:**
- *Lookback-Ensemble* (`daily_strategy.py`, laeuft automatisch mit):
  7 Member SMA 25..200 x Donchian(0.4*SMA), Champion-Struktur je Member
  (Flat-Zone bleibt erhalten), Position in 1/7-Schritten. Gates vorab:
  (b) alle Subperioden netto positiv, (c) schlechtester Subperioden-Sharpe
  >= Champion, (d) Kosten-Drag <= Champion +2pp p.a. Bei Ambiguitaet bleibt
  der Champion; ein Fail beerdigt die Idee ohne Member-Iteration.
- *ETH-Portfolio-Studie* (`scripts/portfolio_strategy.py`): 50/50-Split
  BTC/ETH, ETH-Shorts erst ab Perp-Verfuegbarkeit (Shortability-Filter),
  gepaarter Block-Bootstrap der Sharpe-Differenz. WICHTIG: Die beiden
  Skeptiker-Panels widersprachen sich hier (einmal 2:0 dagegen, einmal 2:0
  dafuer) — der Kandidat ist eine Grauzone. Kernpunkte contra: ohne Hebel
  ist ein Sharpe-Gewinn nicht monetarisierbar, und Ziel-Vol senken liefert
  denselben DD-Schutz billiger. Studie ja; Adoption braeuchte zusaetzlich
  eine bewusste Hebel-/Zielvol-Entscheidung.

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
