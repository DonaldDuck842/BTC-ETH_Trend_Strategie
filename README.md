# BTC/ETH Trend Strategy — Can a systematic strategy beat Buy & Hold on Bitcoin?

[![Daily Signal](https://github.com/DonaldDuck842/BTC-ETH_Trend_Strategie/actions/workflows/daily-signal.yml/badge.svg)](https://github.com/DonaldDuck842/BTC-ETH_Trend_Strategie/actions/workflows/daily-signal.yml)

A systematic daily trend-following strategy for Bitcoin and Ethereum —
fully backtested (2018–2026), validated against pre-registered criteria,
and executed automatically every day (currently in paper trading).

## The question behind this project

Buying Bitcoin and holding it has been one of the best trades of the last
decade — and one of the most painful. Investors who held through 2018 or
2022 sat through drawdowns of **more than 75 %**, and many sold at the
bottom. So the starting question was simple:

> **Is there a systematic way to beat Buy & Hold on Bitcoin — not by
> predicting the future, but by following trends and controlling risk?**

"Beating" here means two things: earning more over a full market cycle,
and doing it with far smaller drawdowns, so that the strategy is
something a real person can actually stick with.

### How the project got here

The project did **not** start where it ended. The first idea, inspired by
AI models that try to predict the next candle on a price chart, was to
forecast Bitcoin's moves on the **1- and 5-minute chart** with machine
learning, using historical prices and order-flow data.

That approach was tested seriously — and it failed honestly. The models
found a *real* signal, but it was worth only 2–5 basis points per trade,
less than the ~10 basis points that trading fees cost. Adding real
order-flow data did not change that. One lesson survived: **how much**
the market will move is predictable; **where** it will move next is
almost not.

So the project pivoted to the daily chart, to a well-documented idea:
trend following with volatility control. Every refinement since then
was tested with criteria fixed *before* looking at the result, and most
ideas were rejected. The full history — 2 adopted ideas, 21 rejected —
is in the [Research Log](docs/RESEARCH_LOG.md).

### The answer so far

Over the full cycle, **yes**: in the 2018–2026 backtest, the strategy
returned about **+40 % per year** versus +15 % for a 50/50 BTC/ETH
Buy & Hold (+19 % for Bitcoin alone), with a maximum drawdown of
**−32 %** instead of −88 %. It stayed profitable in both bear markets.

But not in every phase: in the 2023–24 bull market it made +20 % per year
while Buy & Hold made +99 %. And the backtest starts in January 2018,
close to a market top, which flatters any comparison with Buy & Hold.
The strategy's real edge is not out-earning Bitcoin in bull markets —
it is **staying profitable through bear markets** and avoiding the
deepest crashes.

## 📊 Live status

| What | Where |
|---|---|
| **Test portfolio performance** (fictional USD 100,000, for web embedding) | [`results/performance.json`](results/performance.json) · [`results/equity.csv`](results/equity.csv) |
| **Paper-trading log** (updated daily) | [`results/paper_log.csv`](results/paper_log.csv) |
| **Daily signal run** (00:10 UTC, automated) | [Actions → Daily Signal](../../actions/workflows/daily-signal.yml) |
| Daily report | attached as an artifact to every Action run |
| Phase | **Paper trading** (goal: ≥ 60 logged days before going live) |
| Strategy status | **Frozen since 2026-09-28** — see [Operating Rules](docs/OPERATING_RULES.md) |

The log contains, for each day and coin, the system's target position and
the reference price (`ref_close`, daily close in USD). **Unit of
`target_pos`: fraction of that leg's capital** (each leg = 50 % of total
capital), between −1 and +1; negative = short, never beyond ±1 (no
leverage). Example: `target_pos = 0.8` for BTC means 80 % of the BTC leg
is invested = 40 % of total capital. The daily report in the Action log
shows the same number already converted to total capital.

**Web embedding:** the test portfolio (fictional USD 100,000, no real
money) is updated daily. A website can load it directly — CORS is open on
raw.githubusercontent.com:

```js
fetch("https://raw.githubusercontent.com/DonaldDuck842/BTC-ETH_Trend_Strategie/claude/btc-price-prediction-model-de6mnf/results/performance.json")
  .then(r => r.json())
  .then(p => console.log(p.current_equity_usd, p.total_return_pct, p.series));
```

## How the strategy works

Two simple principles that have been documented for decades — nothing else:

1. **Trend following:** for each coin, 7 trend filters of different speeds
   (≈ 1–7 months of lookback) vote on the daily chart. The position is the
   average of their votes: unanimous = full position, split = small
   position, in steps of one seventh. Long and short.
2. **Volatility targeting:** the position size is divided by the current
   volatility (target ≈ 40 % annualized, never more than 1× capital, no
   leverage). When the market gets wild, positions shrink automatically.

The strategy trades **once a day** at the start of the day (signal from the
previous day's close), with 50 % of capital per coin. It **does not
predict prices** — every short-term forecasting approach was tested and
failed on trading costs (fully documented in the
[Research Log](docs/RESEARCH_LOG.md)).

## Key figures (backtest 2018–2026, after 10 bps cost per side)

| Metric | Strategy | Buy & Hold 50/50 |
|---|---|---|
| Annual return (CAGR) | **+40.2 %** | +15.4 % |
| Sharpe ratio | **1.03** | 0.20 |
| Max. drawdown | **−32.3 %** (stress test: realistically −40 %, rarely down to −57 %) | −88.3 % |
| Weakest sub-period | +20.3 % p.a. (2023–24) | −26.7 % p.a. (2025–26) |
| Bear-market test 2025–26 | **+22.9 % p.a.** | −26.7 % p.a. |

Honest reading: the strategy stayed profitable in **every** market phase.
The price for that is lagging far behind Buy & Hold in extreme bull years
(2023–24: +20 % vs. +99 %). About 52 % of days are positive; the money is
made through a few large trends, not through a high hit rate.

## Repository structure

**🟢 Operations (runs daily)**
| File | Purpose |
|---|---|
| `scripts/live_signal.py` | Daily report: target positions, sizes, orientation stops; writes the paper log and the test portfolio |
| `build_performance.py` / `push_dashboard.py` | Daily performance series and optional push to an external dashboard |
| `.github/workflows/daily-signal.yml` | Automation (00:10 UTC); commits the log to the repo |
| `results/paper_log.csv` | **The live document** — one entry per day and coin |

**🔵 Strategy core**
| File | Purpose |
|---|---|
| `btc_predict/daily.py` | The final strategy: 7-member lookback ensemble, volatility targeting, backtest, reports |
| `btc_predict/data.py` | Data loading (Binance Vision + API), Parquet cache, synthetic test-data generator |

**🟡 Analyses & studies (reproducible, with frozen criteria)**
| File | Purpose |
|---|---|
| `scripts/daily_strategy.py` | Variant comparison of the daily strategy + ensemble study |
| `scripts/portfolio_strategy.py` | BTC/ETH portfolio study incl. the final specification decision |
| `scripts/final_variants.py` | Final variant round before freezing (no shorts, minimum position, strict shorts) |
| `scripts/stress_test.py` | Synthetic market worlds + block bootstrap of the real history |
| `scripts/drawdown_anatomy.py` | Where the largest drawdowns came from |
| `scripts/trade_stats.py` | Win rate, episode statistics, payoff ratio |
| `scripts/{ewma_study,mtf_filter_study,er_diagnostic,long_threshold_study}.py` | Rejected refinement ideas, kept for transparency |

**⚪ Research archive (rejected short-term approaches — kept on purpose)**
| File | Purpose |
|---|---|
| `btc_predict/{features,dataset,model,evaluate}.py` | 5-minute ML pipeline (direction classifier — failed) |
| `btc_predict/path_model.py` | 5-minute path model (validated: calibrated uncertainty cone; usable for display and stops) |
| `scripts/{run_experiment,run_combined,predict_path,phase2_execution,validate_final}.py` | Experiments from the short-term phase |
| `configs/*.yaml` | Frozen experiment configurations |

## Reproduce it yourself

All results can be reproduced with free data (locally or in Google Colab,
no API key needed):

```bash
git clone https://github.com/DonaldDuck842/BTC-ETH_Trend_Strategie
cd BTC-ETH_Trend_Strategie && pip install -r requirements.txt

python scripts/daily_strategy.py       # daily strategy + ensemble comparison
python scripts/portfolio_strategy.py   # BTC/ETH portfolio + final study
python scripts/trade_stats.py          # win rate & episode statistics
python scripts/live_signal.py          # today's signal (same as the Action)
```

Every script has a `--synthetic` mode as a self-test: on random data the
pipeline must show ≈ zero edge — if it does not, there is a leak in the
code.

## Rules & logs

This project is built on discipline, not on forecasting skill:

- **[Operating Rules](docs/OPERATING_RULES.md)** — drawdown protocol,
  paper phase & go-live criteria, switching rule between the books,
  yearly revalidation, news discipline.
- **[Research Log](docs/RESEARCH_LOG.md)** — every tested idea with its
  pre-registered criterion and verdict: 2 adopted, 21 rejected. Rejected
  means rejected.

The most important rule comes last: **nobody tinkers with the running
system.** Every change goes through a new, pre-registered study.
