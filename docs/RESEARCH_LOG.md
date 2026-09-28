# Forschungsprotokoll

Jede getestete Idee mit vorab registriertem Pass/Fail-Kriterium und
Verdikt. **Beerdigt heißt beerdigt** — kein Wiederaufmachen, kein
Nachtunen, auch nicht in Verkleidung. Bilanz: **2 adoptiert, 21 beerdigt. Strategie eingefroren am 2026-09-28.**

## Methodik-Regeln (galten für jedes Experiment)

1. **Erfolgsmaß ist Netto-PnL nach realistischen Kosten, nie Accuracy.**
2. **Vorab registrierte Kriterien:** Pass/Fail wird schriftlich fixiert,
   bevor der erste echte Datenlauf passiert. Kein Nachverhandeln.
3. **Testfenster sind Einwegware:** Ein einmal bewerteter Zeitraum wird
   nie wieder als Test verwendet (Snooping-Log zählt mit).
4. **Eine Variante statt Sweeps:** Es wird nicht die beste Zelle pro
   Periode gepickt; Parameter-Raster sind Diagnose, keine Auswahl.
5. **Synthetik-Nullprobe:** Jede Pipeline muss auf Zufallsdaten ≈ null
   Edge zeigen, sonst gilt sie als leck.

## ✅ Adoptiert

### Lookback-Ensemble (Tages-Momentum)
7 Member (SMA 25–200 × Donchian 0.4×SMA, Konsens-Struktur mit Flat-Zone
je Member), Position in Siebtel-Schritten. **Alle drei Gates bestanden**
(alle Subperioden positiv; schlechtester Subperioden-Sharpe 0.29 vs. 0.14;
Kosten-Drag 3.70 % vs. 4.98 % p.a. — Siebtel-Schritte erzeugen *weniger*
Turnover als Voll-Flips). Adoption ausdrücklich wegen
Spezifikations-Robustheit; die In-Sample-Deltas (Sharpe 0.89 vs. 0.75)
sind kein Evidenz-Claim.

### 50/50-Portfolio BTC/ETH + Ensemble beide Beine
ETH-Bein standalone Sharpe 1.02, in allen Subperioden positiv, robust
gegen doppelte Kosten und Shortability-Filter (keine ETH-Shorts vor
Perp-Verfügbarkeit). Das formale Bootstrap-Gate des Portfolios verfehlte
0.95 knapp (0.893) — der 50/50-Split wurde daher als dokumentierte
**Risiko-Präferenz-Entscheidung des Betreibers** adoptiert, nicht als
Signifikanz-Claim. Die finale Studie (Ensemble auf beiden Beinen,
Gates E1–E3) bestand anschließend alle Gates → Live-Spezifikation.

## ⚰️ Beerdigt (chronologisch)

| # | Idee | Kriterium & Ergebnis |
|---|---|---|
| 1 | **5m-Richtungsprognose aus Klines-Features** (LightGBM, up/flat/down) | Eingefrorene Config auf 5 unberührten Monaten: 2/5 positiv bei Maker-Kosten, Brutto-Edge instabil (+13.5 bis −6.7 bps/Trade). Echtes Signal vorhanden (Balanced Accuracy 0.43), aber nicht kostendeckend monetarisierbar. |
| 2 | **5m-Richtung mit echten Orderflow-Daten** (aggTrades: Taker-Flow, Whale-Trades) | Dev-Sweep zeigte eine einzelne Zelle (11.2 bps/RT) ohne Plateau; eingefrorene Bestätigung auf frischem Monat: 2.4 bps/RT, netto negativ. OFI-Features erschienen nie in den Top-Importances. |
| 3 | **Funding-Overlay als Tages-Veto** (Extrem-Funding ⇒ flat) | Kostete in *jeder* Periode Rendite, in der es eingriff — es kappte genau die Trends, die es schützen sollte. |
| 4 | **Einstiegs-Timing der Tages-Flips via 5m-Pfadmodell** („auf den Dip warten") | Gepoolt über 159 Flips / 5.6 Jahre: −5.6 bps/Flip (t = −1.38). Median +0.9, aber fetter linker Tail: Beim Momentum-Ausbruch läuft der Kurs gelegentlich brutal weg (2018–20: −13.4 bps/Flip). |
| 5 | **Kontinuierliche Positionsgröße** (tanh-Sizing) | Skeptiker-Panel 2:0: sicherer CAGR-Verlust 4–8 pp gegen unmessbaren Sharpe-Gewinn; neuer Freiheitsgrad. |
| 6 | **Intraday-De-Risking via Pfadmodell** | Panel 2:0: Ziel-Events liegen fast alle in den Trainingsfenstern des Pfadmodells — Test hätte Power ≈ 0; Short-Seite würde Verdiensttage de-risken. |
| 7 | **Funding-Carry als zweites Standbein** | Panel 2:0: Gegen den ehrlichen Benchmark (4–5 % risikoarm auf Brachkapital) schrumpft der Nutzen auf ~0.5–1 pp und kauft Perp-Liquidations- und Börsen-Gegenparteirisiko ein. |
| 8 | **Maker-Ausführung als Forschungsprojekt** | Panel 2:0: Fee-Arithmetik trägt auf dem unterstellten Venue nicht; Flips sind informationsgetriebener Flow (Adverse Selection). Realistisch +10–40 bps/Jahr — Live-Detail, kein Experiment. |
| 9 | **EWMA-Vol-Sizing** (RiskMetrics λ=0.94) | Gate V2 hauchdünn verfehlt (0.59 vs. 0.62), alle Kennzahlen äquivalent (Sharpe 1.02 vs. 1.03). Bei Gleichstand bleibt der Amtsinhaber; kein λ-Tuning. |
| 10 | **News-Filter** | Nicht studienwürdig: Mechanismus redundant zu Momentum + Vol-Targeting (der Preis weiß von der News zuerst); keine saubere freie Point-in-Time-News-Historie (Look-Ahead in kuratierten Archiven); strukturell der Zwilling von #3. |
| 11–13 | **Rekursive Kerze-auf-Kerze-Prognose, Long/flat-Variante als Champion, Einzel-Lookback nach Ensemble-Adoption** | Architektur- bzw. Varianten-Entscheidungen mit dokumentierter Begründung (Fehlerverkettung; long/short schlug long/flat in allen Subperioden; Ensemble-Gates). |
| 14 | **Voll long ab 80 % Signalstaerke** (`scripts/long_threshold_study.py`: A = Zielposition > 0.8, B = >= 6/7 Filter long; je 100 % long sonst flat) | Beide schwaecher als die Live-Strategie: Sharpe 0.73 / 0.77 vs. 1.03, CAGR +8.8 % / +29.6 % vs. +40.2 %. Mehr Rendite im Bullenmarkt 2023-24 nur marginal (+24.9 % / +34.3 % vs. +20.3 %, B&H +99 %), dafuer Einbruch im Baerenmarkt 2025-26 (-6.4 % / +1.1 % vs. +22.9 %). Befund: Die Short-Seite ist mehr wert als zusaetzliche Bullenmarkt-Partizipation. Keine Schwellen-Iteration. |
| 15 | **4h-Veto-Filter auf die Tagesstrategie** (`scripts/mtf_filter_study.py`: Champion-Member SMA50/Donchian20 auf 4h-Kerzen, Bein flat wenn gegen Tagesrichtung) | Gates M2-M4 durchgefallen: MaxDD -31.9 % vs. -32.3 % (verlangt 3 pp flacher), Sharpe 0.92 vs. 1.03, Kosten-Drag 9.2 % vs. 3.3 % p.a. Basis reproduzierte den Tages-Backtest exakt. Informativ bei 4 bps: Sharpe 1.11 vs. 1.09, MaxDD -27.6 % vs. -31.1 % — Signal hat kleinen Wert, kleiner als Spot-Gebuehren. Nur als neue Studie bei Perp-Umzug denkbar. |
| 16 | **Seitwaerts-Filter via Efficiency Ratio** (`scripts/er_diagnostic.py`: Strategie-PnL nach ER(20)-Fuenftel vom Vortag) | Vor-Diagnose verneint jeden Filter: Alle Fuenftel netto positiv, auch das seitwaertigste (Q1 ~+32 % p.a.-Aequivalent, Q5 Trend ~+87 %). Kriterium (a)-(c) komplett verfehlt (Q1 positiv, Intervall schliesst 0 ein, nur 2/4 Phasen). Befund: Das Ensemble bremst in unklaren Phasen bereits selbst (Ø Position 0.41 in Q1 vs. 0.60 in Q5). Kein Nachschieben einer Q2/Q3-Regel (Data-Mining). |
| 17 | **Trailing-Stop-Loss gegen Crashs** | Nicht studienwuerdig laut Vor-Diagnose (`scripts/drawdown_anatomy.py`): nur 1 der 3 tiefsten Drawdowns crash-getrieben (Soll >= 2). Die groessten Verluste entstanden schleichend im Seitwaertsmarkt 2024 (-32.3 %, Crash-Anteil 0.17) und durch Short-Squeezes in Baerenmarkt-Rallys (2018/19, 2022). Long-seitige Crashs sind nicht die Schwachstelle. Short-Stops bewusst NICHT nachgeschoben (aus denselben Daten abgeleitet; Short-Seite netto stark positiv). |
| 18 | **Ohne Shorts (long/flat-Ensemble)** (`scripts/final_variants.py` V1) | CAGR 25.7 % vs. 40.2 %, Baerenmarkt 2025-26 -1.2 % vs. +22.9 %, Bootstrap 0.43. Short-Beitrag gesamt: BTC +8.4 %, ETH +17.1 % p.a.; Shorts kosten im Bullenmarkt 2023-24 (BTC -13.3 %, ETH -11.5 %), verdienen in Baerenmaerkten. |
| 19 | **Mindestposition 30 %** (V2) | Praktisch identisch (CAGR 40.0 %, Sharpe 1.07), G2 und G4 verfehlt (Bootstrap 0.78). Das Hin und Her kleiner Positionen kostet kaum. |
| 20 | **Mindest-Konsens > 2/7** (V3) | Praktisch identisch (CAGR 39.3 %, Sharpe 1.02), Bootstrap 0.32. |
| 21 | **Strenge Shorts (nur ab -5/7)** (V4) | Knapp: G1-G3 bestanden, CAGR 42.1 %, Sharpe 1.10, MaxDD -26.3 % vs. -32.3 %, aber Bootstrap 0.895 < 0.9875 (Bonferroni fuer 4 Varianten). Aus der Drawdown-Anatomie abgeleitet, also teilweise in-sample. Nicht adoptiert. |

## Validiert, aber nicht als Handelssignal

**5m-Pfadmodell** (`btc_predict/path_model.py`): Quantilregression je
Horizont, ehrlich kalibrierter Unsicherheitskegel (Coverage 0.79–0.81 bei
Soll 0.80), Exkursions-Prognosen mit +12–18 % Pinball-Skill.
**Erkenntnis der ganzen Kurzfrist-Phase:** *Wie stark* sich der Markt
bewegt, ist prognostizierbar — *wohin* und *wann*, nicht. Das Modell
dient daher als Anzeige- und Stop-Werkzeug, nie als Richtungs- oder
Timing-Quelle.

## Verdikt-Besonderheit, festgehalten

Bei der Multi-Asset-Frage kamen zwei unabhängige Skeptiker-Panels in zwei
Läufen zu entgegengesetzten 2:0-Urteilen — Beleg dafür, dass alle
verbliebenen Verbesserungsideen in der Zone „erwarteter Nutzen ≈ +0.04
Sharpe, prinzipiell nicht nachweisbar" liegen. Die Forschungsphase wurde
daraufhin bewusst geschlossen: Weiteres Signal-Jagen hätte negativen
Erwartungswert (Datenverbrauch + Overfitting-Risiko).

## Einfrierung (2026-09-28)

Die Strategie ist nach der letzten vorab registrierten Varianten-Runde
(`scripts/final_variants.py`) **eingefroren**. Keine weiteren Studien auf der
Historie 2018-2026: Diese Daten sind durch 21 Tests ausgereizt, jeder weitere
Test waere ueberwiegend Rauschen. Neue Evidenz kommt nur noch aus frischen
Daten: dem Paper-Trading seit 2026-08-30 und der jaehrlichen Revalidierung
(erstmals Januar 2027). Einzige Ausnahme: ein dort vorab registrierter Test.
