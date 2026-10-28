# Operating Rules

All rules are **pre-registered**: interventions in the running system
happen only according to these rules, never in response to headlines or
gut feeling. This document changes only through a new, documented study.

## Status: FROZEN (since 2026-09-28)

The specification below is final. The last variant round (no shorts,
minimum position 30 %, minimum consensus, strict shorts) produced no
variant that passed all pre-defined hurdles. Changes are only possible
through the yearly revalidation on fresh data, not through further tests
on the 2018–2026 history.

## Final live specification (frozen; study E1–E3 passed, 2026-08-31)

- **Universe:** BTCUSDT and ETHUSDT, fixed 50/50 capital split.
- **Signal per coin:** long/short lookback ensemble — 7 members
  (SMA 25/50/75/100/125/150/200, Donchian window = 0.4 × SMA window;
  each member requires the close to be on the same side of both its SMA
  **and** its Donchian midpoint), position = average of the member
  signals (steps of one seventh).
- **Sizing:** target volatility 40 % p.a. / realized 20-day volatility,
  capped at 1.0 per leg (no leverage).
- **Execution:** at the start of the next day (00:00 UTC),
  cost assumption 10 bps per side.
- Reference backtest 2018–2026: Sharpe 1.03, CAGR 40.2 %, max. drawdown
  −32.3 %, every sub-period profitable.

## Execution rule: rebalance band

The model changes its target position almost every day, usually only
slightly (volatility drift). The backtest conservatively pays for
**every** one of these changes at 10 bps per side (~3.3 % cost p.a. is
included in all figures). Live, the rule is therefore: **trade only when
the actual position deviates from the target by more than 5 % of the
leg's capital.** Small drifts are ignored (this saves fees and can only
improve on the backtest); the one-seventh signal steps (≈ 14 %) are always
traded — in practice about 1–3 orders per week and coin. The daily report
shows "ORDER HEUTE: JA/nein" (order today: yes/no) directly. The band is
an execution simplification, not a strategy change; it is never widened
after the fact to avoid losing days.

## Paper-trading phase & going live

Before any real capital is used: **at least 60 logged trading days**
through the automated daily run (`results/paper_log.csv`).

The go-live criterion is **operational, not performance-based**:

1. ≥ 60 logged days with at most 5 gaps.
2. The logged target positions can be reproduced from the daily data
   (no implementation drift).
3. The capital amount is fixed **before** going live — sized so that a
   **−50 %** drawdown on it is bearable without wanting to intervene (not
   −32 %: according to the stress test, the historical figure was lucky;
   see below).

Paper PnL explicitly does **not** decide about going live: 60 days are
statistical noise. The phase tests the process, not the strategy (which is
backtested).

## Drawdown protocol

| Live drawdown | Action |
|---|---|
| up to −40 % | Normal operation, no action. This is the typical drawdown (median of the stress-test distribution). |
| −40 % to −57 % | Rare but expected (between every 2nd and every 20th simulated history). No action, no change of capital — document only. |
| below −57 % | **Revalidation trigger** (worse than 95 % of all simulated histories): the system keeps running by the rules; in parallel, an unscheduled out-of-sample revalidation starts. Shutting down only after its result — never on the same day. |

**Calibration (stress test, `scripts/stress_test.py`, 2026-09):** block
bootstrap of real BTC/ETH returns, 500 alternative histories. Max.
drawdown percentiles: 5 % −57.2 %, median −39.4 %, 95 % −28.8 %. The
historical backtest drawdown of −32.3 % lies in the most favorable
quartile. The former trigger at −40 % would have fired in 48 % of the
histories (constant false alarms) and was therefore moved to the 5 %
percentile. The bootstrap is rather pessimistic for trend following (it
cuts long trends apart), so the thresholds are conservative.
P(negative return over 8 years) = 0.2 %.

**Duration, not just depth (drawdown anatomy, `scripts/drawdown_anatomy.py`):**
the deepest historical drawdowns were not crashes but slow bleeds: the
2024 sideways market (−32.3 %, 240 days to the low, **696 days** to a new
high) and short squeezes in bear-market rallies (2018/19, 2022). Periods
under water of almost **two years** are therefore normal and no reason to
shut down — as long as the depth stays within the protocol above: no
action.

Increase or decrease capital **only on quarterly dates** — never in the
middle of a drawdown and never after a good week.

## Switching rule between the books

All three books (BTC only, ETH only, 50/50) are reported permanently — as
a diagnostic. Switching to a single leg is allowed **only** if, on at
least **3 years of paper/live data** (not backtest!), a paired block
bootstrap of the Sharpe difference shows a share of positive draws
≥ 0.95. "It has been doing better lately" is not a reason to switch —
rotating leadership between the legs is exactly the mechanism the split
lives on.

## Yearly revalidation

Every **January**: the fresh 12 months are used as an out-of-sample test
of the frozen specification. **Decay criterion:** two consecutive yearly
tests with negative net PnL while Buy & Hold was positive ⇒ the system is
considered broken and is **shut down, not re-tuned**.

## News discipline

News is not a signal (reasoning in the Research Log): the price knows
about every piece of news first, and the sizing reacts to the volatility
spike. The most dangerous reaction to news is one's own — manual
interventions after headlines are ruled out by this document; only the
drawdown protocol applies.

## Security

The repository is public. Exchange API keys must **never** be stored in
the repository — for later order automation, only as GitHub secrets or
local environment variables.
