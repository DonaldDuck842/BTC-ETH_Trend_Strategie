# Betriebsregeln

Alle Regeln sind **vorab registriert** — Eingriffe ins laufende System
passieren nur nach diesen Regeln, nie nach Schlagzeile oder Gefühl.
Änderungen an diesem Dokument nur über eine neue, dokumentierte Studie.

## Status: EINGEFROREN (seit 2026-09-28)

Die Spezifikation unten ist endgültig. Die letzte Varianten-Runde (ohne
Shorts, Mindestposition 30 %, Mindest-Konsens, strenge Shorts) hat keine
Variante hervorgebracht, die alle vorab festgelegten Hürden besteht.
Änderungen sind nur noch über die jährliche Revalidierung mit frischen Daten
möglich, nicht über weitere Tests auf der Historie 2018–2026.

## Finale Live-Spezifikation (eingefroren, Studie E1–E3 bestanden, 2026-08-31)

- **Universum:** BTCUSDT und ETHUSDT, fixer 50/50-Kapitalsplit.
- **Signal je Coin:** Lookback-Ensemble long/short — 7 Member
  (SMA 25/50/75/100/125/150/200, Donchian-Fenster = 0.4 × SMA-Fenster,
  je Member Konsens-Regel Close vs. SMA **und** Donchian-Mitte),
  Position = Mittel der Member-Signale (Siebtel-Schritte).
- **Sizing:** Ziel-Vol 40 % p.a. / realisierte 20-Tage-Vol, Cap 1.0
  je Bein (kein Hebel).
- **Ausführung:** naiv zum nächsten Tagesbeginn (00:00 UTC),
  Kostenannahme 10 bps je Seite.
- Referenz-Backtest 2018–2026: Sharpe 1.03, CAGR 40.2 %, MaxDD −32.3 %,
  alle Subperioden positiv.

## Ausführungsregel: Rebalance-Band

Das Modell ändert die Zielposition fast täglich, meist minimal
(Vol-Drift). Der Backtest bezahlt konservativ **jede** dieser Änderungen
mit 10 bps/Seite (~3.3 % Kosten p.a. sind in allen Kennzahlen enthalten).
Live gilt daher: **gehandelt wird nur, wenn die Ist-Position um mehr als
5 % des Bein-Kapitals von der Zielposition abweicht.** Kleine Drifts
werden ausgesessen (spart Gebühren und kann gegenüber dem Backtest nur
besser sein), die 1/7-Signalschritte (≈ 14 %) werden immer gehandelt —
praktisch ergibt das etwa 1–3 Orders pro Woche und Coin. Der tägliche
Report weist „ORDER HEUTE: JA/nein" direkt aus. Das Band ist eine
Ausführungs-Vereinfachung, keine Strategie-Änderung; es wird nicht
nachträglich vergrößert, um Verlusttage zu vermeiden.

## Paper-Trading-Phase & Go-Live

Vor jedem Einsatz echten Kapitals: **mindestens 60 protokollierte
Handelstage** über den automatischen Täglichen Lauf
(`results/paper_log.csv`).

Go-Live-Kriterium ist **operativ, nicht performance-basiert**:

1. ≥ 60 Log-Tage mit höchstens 5 Lücken.
2. Die geloggten Zielpositionen reproduzieren sich aus den Tagesdaten
   (kein Implementierungsdrift).
3. Die Kapitalgröße wird **vor** Go-Live festgelegt — bemessen daran,
   dass **−50 %** davon aushaltbar sind, ohne eingreifen zu wollen (nicht
   −32 %: der historische Wert war laut Stress-Test Glück, siehe unten).

Der Paper-PnL entscheidet ausdrücklich **nicht** über Go-Live: 60 Tage
sind statistisches Rauschen. Die Phase prüft den Prozess, nicht die
Strategie (die ist backtested).

## Drawdown-Protokoll

| Live-Drawdown | Aktion |
|---|---|
| bis −40 % | Normaler Betrieb, keine Aktion. Das ist der typische Drawdown (Median der Stress-Test-Verteilung). |
| −40 % bis −57 % | Selten, aber erwartbar (jede 2. bis jede 20. Historie). Keine Aktion, keine Kapitaländerung — nur dokumentieren. |
| unter −57 % | **Revalidierungs-Trigger** (schlechter als 95 % aller simulierten Historien): System läuft regelkonform weiter; parallel startet eine außerplanmäßige Out-of-Sample-Revalidierung. Abschalten erst nach deren Ergebnis — niemals am selben Tag. |

**Kalibrierung (Stress-Test, `scripts/stress_test.py`, 2026-09):** Block-
Bootstrap der echten BTC/ETH-Renditen, 500 alternative Historien. MaxDD-
Perzentile: 5 % −57.2 %, Median −39.4 %, 95 % −28.8 %. Der historische
Backtest-Drawdown von −32.3 % liegt im günstigsten Viertel. Der frühere
Trigger bei −40 % hätte in 48 % der Historien ausgelöst (Dauer-Fehlalarm)
und wurde deshalb auf das 5-%-Perzentil verschoben. Das Bootstrap-
Verfahren ist für Trendfolge eher pessimistisch (es zerschneidet lange
Trends); die Schwellen sind damit konservativ. P(Rendite über 8 Jahre
negativ) = 0.2 %.

**Dauer statt Tiefe (Drawdown-Anatomie, `scripts/drawdown_anatomy.py`):**
Die tiefsten historischen Drawdowns waren keine Crashs, sondern
schleichend: Seitwärtsmarkt 2024 (−32.3 %, 240 Tage bis zum Tief, **696
Tage** bis zum neuen Hoch) und Short-Squeezes in Bärenmarkt-Rallys
(2018/19, 2022). Unterwasser-Phasen von fast **zwei Jahren** sind also
normal und kein Abschaltgrund — solange die Tiefe im Protokoll oben
bleibt, gilt: keine Aktion.

Kapital erhöhen oder senken: **nur zu Quartalsterminen**, nie mitten im
Drawdown und nie nach einer guten Woche.

## Wechselregel zwischen den Büchern

Alle drei Bücher (BTC-only, ETH-only, 50/50) werden dauerhaft mitreportet
— als Diagnose. Ein Umstieg auf ein Einzel-Bein ist **nur** zulässig, wenn
auf mindestens **3 Jahren Paper-/Live-Daten** (nicht Backtest!) der
gepaarte Block-Bootstrap der Sharpe-Differenz einen Anteil positiver
Draws ≥ 0.95 zeigt. „Lief zuletzt besser" ist kein Wechselgrund —
Führungsrotation zwischen den Beinen ist genau der Mechanismus, von dem
der Split lebt.

## Jährliche Revalidierung

Jeden **Januar**: die frischen 12 Monate als Out-of-Sample-Test der
eingefrorenen Spezifikation. **Zerfalls-Kriterium:** zwei
aufeinanderfolgende Jahres-Tests mit negativem Netto-PnL bei gleichzeitig
positivem Buy-and-Hold-Umfeld ⇒ das System gilt als zerfallen und wird
**abgeschaltet, nicht nachgetunt**.

## News-Disziplin

News sind kein Signal (Begründung im Forschungsprotokoll): Der Preis weiß
von jeder News zuerst, und auf den Vol-Sprung reagiert das Sizing. Die
gefährlichste News-Reaktion ist die eigene — manuelle Eingriffe nach
Schlagzeilen sind durch dieses Dokument ausgeschlossen; es gilt allein
das Drawdown-Protokoll.

## Sicherheit

Das Repo ist öffentlich. Börsen-API-Keys gehören **niemals** ins Repo —
bei späterer Order-Automation ausschließlich als GitHub-Secrets bzw.
lokale Umgebungsvariablen.
