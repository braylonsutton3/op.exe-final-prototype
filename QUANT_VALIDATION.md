# OP.exe Quant validation report

## Supplied data

- File tested: `Dataset_NQ_1min_2022_2025(2).csv`
- 1,048,575 one-minute rows
- December 26, 2022 through December 11, 2025
- Timestamp ET, OHLC, volume, RTH VWAP, and ETH VWAP
- No duplicate timestamps or missing OHLCV values found
- No depth-of-book, queue, bid/ask, or historical delta fields

## Conservative structural baseline

The baseline evaluated completed candles at 10:00 and 10:30 ET, allowed no
more than one trade per session, rejected lower-timeframe signals strongly
opposed by 30m/1h context, counted the stop first when both barriers touched in
one one-minute bar, and deducted one tick of adverse slippage plus estimated
commission. Settings were three MNQ contracts and a $300 maximum planned risk.

At the stricter quality-76 gate over the combined 2023–December 2025 window:

| Metric | Result |
|---|---:|
| Trades | 52 |
| Net result | +$2,983.50 |
| Win rate | 50.00% |
| Profit factor | 1.838 |
| Maximum drawdown | -$811.50 |
| Positive months with trades | 16 |
| Negative months with trades | 14 |

This threshold was selected after comparing thresholds on the 2024–2025 portion,
so the combined number is descriptive and is not a pristine untouched
out-of-sample result. The first 35 sessions of separately run segments were used
as indicator warm-up.

## Higher-frequency, multiple-trades-per-day experiment

`quant_backtest.py` tested completed 5-minute decisions, 30m/1h alignment,
structure/liquidity events, ATR stops, 1.5R–1.8R targets, cost/slippage, a
Monte Carlo gate, one position at a time, and multiple possible trades per day.
The tested configurations failed:

| Quality / probability gate | Trades | Net result | Profit factor | Max drawdown |
|---|---:|---:|---:|---:|
| 60 / 0.50 | 205 | -$2,578.38 | 0.812 | -$3,526.39 |
| 65 / 0.50 | 126 | -$4,024.58 | 0.575 | -$4,034.55 |
| 70 / 0.50 | 99 | -$3,689.48 | 0.524 | -$3,699.45 |
| 70 / 0.52 | 73 | -$2,427.21 | 0.551 | -$2,565.77 |

That experimental path is included for reproducible research, but it is not the
live model and is not production-qualified. The deployable app retains the
stricter structural gate and uses the Monte Carlo/analogue layer only as an
additional veto. It does not loosen the engine merely to produce more trades.

## What is and is not established

The conservative rules produced a positive aggregate result in this one file.
They did not make money every month. The multi-trade experiment lost money. The
file cannot test order flow because it does not contain historical order-book
data. The results do not establish a guarantee, near-100% accuracy, or future
profitability. A legitimate next test is locked-parameter walk-forward or paper
trading with recorded live bars and depth, including disconnects and real fills.
