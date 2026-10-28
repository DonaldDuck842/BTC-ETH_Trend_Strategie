# Research Log

Every tested idea with its pre-registered pass/fail criterion and verdict.
**Rejected means rejected** — no reopening, no re-tuning, not even in
disguise. Tally: **2 adopted, 21 rejected. Strategy frozen on 2026-09-28.**

## Method rules (applied to every experiment)

1. **Success is measured as net PnL after realistic costs, never as
   accuracy.**
2. **Pre-registered criteria:** pass/fail is written down before the
   first real data run. No renegotiation afterwards.
3. **Test windows are single-use:** a period that has been evaluated once
   is never used as a test again (a snooping log keeps count).
4. **One variant instead of sweeps:** the best cell per period is never
   picked; parameter grids are diagnostics, not a selection method.
5. **Synthetic null test:** every pipeline must show ≈ zero edge on random
   data; otherwise it is considered leaky.

## ✅ Adopted

### Lookback ensemble (daily momentum)
7 members (SMA 25–200 × Donchian 0.4 × SMA, consensus structure with a
flat zone per member), position in steps of one seventh. **All three
gates passed** (all sub-periods profitable; worst sub-period Sharpe 0.29
vs. 0.14; cost drag 3.70 % vs. 4.98 % p.a. — one-seventh steps create
*less* turnover than full flips). Adopted explicitly for robustness of the
specification; the in-sample deltas (Sharpe 0.89 vs. 0.75) are not an
evidence claim.

### 50/50 BTC/ETH portfolio + ensemble on both legs
ETH leg standalone Sharpe 1.02, profitable in all sub-periods, robust to
doubled costs and to a shortability filter (no ETH shorts before
perpetual futures existed). The portfolio narrowly missed its formal
bootstrap gate (0.893 vs. 0.95) — the 50/50 split was therefore adopted as
a documented **risk-preference decision by the operator**, not as a
significance claim. The final study (ensemble on both legs, gates E1–E3)
then passed all gates → live specification.

## ⚰️ Rejected (chronological)

| # | Idea | Criterion & result |
|---|---|---|
| 1 | **5-minute direction forecast from candle features** (LightGBM, up/flat/down) | Frozen configuration on 5 untouched months: 2/5 profitable at maker costs, unstable gross edge (+13.5 to −6.7 bps per trade). A real signal (balanced accuracy 0.43), but not enough to cover costs. |
| 2 | **5-minute direction with real order-flow data** (aggTrades: taker flow, whale trades) | The development sweep showed a single cell (11.2 bps per round trip) without a plateau; frozen confirmation on a fresh month: 2.4 bps per round trip, net negative. Order-flow features never appeared among the top importances. |
| 3 | **Funding-rate overlay as a daily veto** (extreme funding ⇒ flat) | Cost return in *every* period where it intervened — it cut exactly the trends it was meant to protect. |
| 4 | **Entry timing of daily flips via the 5-minute path model** ("wait for the dip") | Pooled over 159 flips / 5.6 years: −5.6 bps per flip (t = −1.38). Median +0.9, but a fat left tail: on momentum breakouts the price sometimes runs away sharply (2018–20: −13.4 bps per flip). |
| 5 | **Continuous position sizing** (tanh sizing) | Skeptic panel 2:0: a certain CAGR loss of 4–8 pp against an unmeasurable Sharpe gain; a new degree of freedom. |
| 6 | **Intraday de-risking via the path model** | Panel 2:0: the target events lie almost entirely in the path model's training windows — the test would have had power ≈ 0; on the short side it would de-risk the most profitable days. |
| 7 | **Funding carry as a second pillar** | Panel 2:0: against the honest benchmark (4–5 % low-risk yield on idle capital) the benefit shrinks to ~0.5–1 pp while adding perpetual-futures liquidation and exchange counterparty risk. |
| 8 | **Maker execution as a research project** | Panel 2:0: the fee arithmetic does not work on the assumed venue; flips are information-driven flow (adverse selection). Realistically +10–40 bps per year — a live detail, not an experiment. |
| 9 | **EWMA volatility sizing** (RiskMetrics λ = 0.94) | Gate V2 missed by a hair (0.59 vs. 0.62), all metrics equivalent (Sharpe 1.02 vs. 1.03). On a tie the incumbent stays; no λ tuning. |
| 10 | **News filter** | Not worth a study: the mechanism is redundant to momentum + volatility targeting (the price knows about the news first); no clean, free point-in-time news history (look-ahead in curated archives); structurally the twin of #3. |
| 11–13 | **Recursive candle-by-candle forecasting, long/flat variant as champion, single lookback after the ensemble adoption** | Architecture and variant decisions with documented reasoning (error compounding; long/short beat long/flat in all sub-periods; ensemble gates). |
| 14 | **Full long above 80 % signal strength** (`scripts/long_threshold_study.py`: A = target position > 0.8, B = ≥ 6/7 filters long; 100 % long, otherwise flat) | Both weaker than the live strategy: Sharpe 0.73 / 0.77 vs. 1.03, CAGR +8.8 % / +29.6 % vs. +40.2 %. Only marginally more return in the 2023–24 bull market (+24.9 % / +34.3 % vs. +20.3 %; Buy & Hold +99 %), but a collapse in the 2025–26 bear market (−6.4 % / +1.1 % vs. +22.9 %). Finding: the short side is worth more than extra bull-market participation. No threshold iteration. |
| 15 | **4-hour veto filter on the daily strategy** (`scripts/mtf_filter_study.py`: champion member SMA 50 / Donchian 20 on 4h candles; leg flat when it points against the daily direction) | Gates M2–M4 failed: max. drawdown −31.9 % vs. −32.3 % (required: 3 pp shallower), Sharpe 0.92 vs. 1.03, cost drag 9.2 % vs. 3.3 % p.a. The baseline reproduced the daily backtest exactly. Informational at 4 bps: Sharpe 1.11 vs. 1.09, max. drawdown −27.6 % vs. −31.1 % — the signal has a small value, smaller than spot fees. Only conceivable as a new study after a move to perpetual futures. |
| 16 | **Sideways filter via efficiency ratio** (`scripts/er_diagnostic.py`: strategy PnL by quintile of the prior-day 20-day efficiency ratio) | The preliminary diagnostic rules out any filter: all quintiles net profitable, even the most sideways one (Q1 ≈ +32 % p.a. equivalent, Q5 trend ≈ +87 %). Criteria (a)–(c) all missed (Q1 positive, interval includes 0, only 2/4 phases). Finding: the ensemble already brakes on its own in unclear phases (average position 0.41 in Q1 vs. 0.60 in Q5). No Q2/Q3 rule mined afterwards (data mining). |
| 17 | **Trailing stop-loss against crashes** | Not worth a study according to the preliminary diagnostic (`scripts/drawdown_anatomy.py`): only 1 of the 3 deepest drawdowns was crash-driven (required: ≥ 2). The largest losses came slowly in the 2024 sideways market (−32.3 %, crash share 0.17) and from short squeezes in bear-market rallies (2018/19, 2022). Long-side crashes are not the weak point. Short-side stops deliberately NOT added (derived from the same data; the short side is strongly net positive). |
| 18 | **No shorts (long/flat ensemble)** (`scripts/final_variants.py` V1) | CAGR 25.7 % vs. 40.2 %, bear market 2025–26 −1.2 % vs. +22.9 %, bootstrap 0.43. Short contribution overall: BTC +8.4 %, ETH +17.1 % p.a.; shorts cost money in the 2023–24 bull market (BTC −13.3 %, ETH −11.5 %) and earn it in bear markets. |
| 19 | **Minimum position 30 %** (V2) | Practically identical (CAGR 40.0 %, Sharpe 1.07); G2 and G4 missed (bootstrap 0.78). Switching small positions back and forth costs very little. |
| 20 | **Minimum consensus > 2/7** (V3) | Practically identical (CAGR 39.3 %, Sharpe 1.02), bootstrap 0.32. |
| 21 | **Strict shorts (only at ≤ −5/7)** (V4) | Close: G1–G3 passed, CAGR 42.1 %, Sharpe 1.10, max. drawdown −26.3 % vs. −32.3 %, but bootstrap 0.895 < 0.9875 (Bonferroni correction for 4 variants). Derived from the drawdown anatomy, i.e. partly in-sample. Not adopted. |

## Validated, but not as a trading signal

**5-minute path model** (`btc_predict/path_model.py`): quantile regression
per horizon, an honestly calibrated uncertainty cone (coverage 0.79–0.81
vs. a target of 0.80), excursion forecasts with +12–18 % pinball skill.
**The lesson of the entire short-term phase:** *how much* the market moves
is predictable — *where* and *when* is not. The model therefore serves as
a display and stop tool, never as a source of direction or timing.

## A notable verdict

On the multi-asset question, two independent skeptic panels reached
opposite 2:0 verdicts in two runs — evidence that all remaining
improvement ideas lie in the zone of "expected benefit ≈ +0.04 Sharpe,
impossible to prove in principle". The research phase was therefore
closed deliberately: further signal hunting would have had a negative
expected value (data consumption + overfitting risk).

## Freeze (2026-09-28)

After the last pre-registered variant round (`scripts/final_variants.py`),
the strategy is **frozen**. No further studies on the 2018–2026 history:
this data has been exhausted by 21 tests, and any further test would be
mostly noise. New evidence comes only from fresh data: the paper trading
since 2026-08-30 and the yearly revalidation (first in January 2027).
The only exception is a test pre-registered there.
