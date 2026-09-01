# Betriebsregeln

Alle Regeln sind **vorab registriert** — Eingriffe ins laufende System
passieren nur nach diesen Regeln, nie nach Schlagzeile oder Gefühl.
Änderungen an diesem Dokument nur über eine neue, dokumentierte Studie.

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
   dass −32 % davon aushaltbar sind, ohne eingreifen zu wollen.

Der Paper-PnL entscheidet ausdrücklich **nicht** über Go-Live: 60 Tage
sind statistisches Rauschen. Die Phase prüft den Prozess, nicht die
Strategie (die ist backtested).

## Drawdown-Protokoll

| Live-Drawdown | Aktion |
|---|---|
| bis −32 % (historisches Maximum) | Normaler Betrieb, keine Aktion. Das ist der eingepreiste Preis der Strategie. |
| unter −40 % | **Revalidierungs-Trigger:** System läuft regelkonform weiter; parallel startet eine außerplanmäßige Out-of-Sample-Revalidierung. Abschalten erst nach deren Ergebnis — niemals am selben Tag. |

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
