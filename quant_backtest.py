"""Fast no-look-ahead multi-trade backtest for OP.exe Quant.

The test works on 5-minute decisions derived from the supplied 1-minute file.
It permits multiple trades per day while enforcing one open trade at a time.
All indicators are shifted or built from completed bars. Stops are counted first
when stop and target touch inside the same 5-minute candle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from op_engine import ET, atr, load_nq_csv, resample_ohlcv, rsi


def adx_fast(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df.High.diff()
    down = -df.Low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, period).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / period, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / period, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    bars = resample_ohlcv(df_1m, "5m")
    close = bars.Close
    bars["ema20"] = close.ewm(span=20, adjust=False).mean()
    bars["ema50"] = close.ewm(span=50, adjust=False).mean()
    bars["ema200"] = close.ewm(span=200, adjust=False).mean()
    bars["atr"] = atr(bars)
    bars["rsi"] = rsi(close)
    bars["adx"] = adx_fast(bars)
    bars["vol_ratio"] = bars.Volume / bars.Volume.rolling(20).median().shift(1).replace(0, np.nan)
    bars["prior_high20"] = bars.High.rolling(20).max().shift(1)
    bars["prior_low20"] = bars.Low.rolling(20).min().shift(1)
    bars["body"] = (bars.Close - bars.Open).abs()
    bars["body_ratio"] = bars.body / bars.body.rolling(20).median().shift(1).replace(0, np.nan)

    local = bars.tz_convert(ET)
    dates = pd.Series(local.index.date, index=bars.index)
    typical = (bars.High + bars.Low + bars.Close) / 3.0
    pv = typical * bars.Volume
    bars["vwap"] = pv.groupby(dates).cumsum() / bars.Volume.groupby(dates).cumsum().replace(0, np.nan)

    for tf, label in (("30m", "30m"), ("1h", "1h")):
        higher = resample_ohlcv(df_1m, tf)
        h20 = higher.Close.ewm(span=20, adjust=False).mean()
        h50 = higher.Close.ewm(span=50, adjust=False).mean()
        trend = pd.Series(np.where(h20 > h50, 1, -1), index=higher.index)
        bars[f"trend_{label}"] = trend.reindex(bars.index, method="ffill")

    bull = pd.DataFrame(index=bars.index)
    bear = pd.DataFrame(index=bars.index)
    bull["price_ema"] = np.where(bars.Close > bars.ema20, 1.0, 0.0)
    bear["price_ema"] = np.where(bars.Close < bars.ema20, 1.0, 0.0)
    bull["ema_fast"] = np.where(bars.ema20 > bars.ema50, 1.5, 0.0)
    bear["ema_fast"] = np.where(bars.ema20 < bars.ema50, 1.5, 0.0)
    bull["ema_regime"] = np.where(bars.ema50 > bars.ema200, 2.0, 0.0)
    bear["ema_regime"] = np.where(bars.ema50 < bars.ema200, 2.0, 0.0)
    bull["rsi"] = np.where(bars.rsi >= 55, 1.0, 0.0)
    bear["rsi"] = np.where(bars.rsi <= 45, 1.0, 0.0)
    bull["vwap"] = np.where(bars.Close > bars.vwap, 1.25, 0.0)
    bear["vwap"] = np.where(bars.Close < bars.vwap, 1.25, 0.0)
    bull["htf30"] = np.where(bars.trend_30m > 0, 1.5, 0.0)
    bear["htf30"] = np.where(bars.trend_30m < 0, 1.5, 0.0)
    bull["htf1h"] = np.where(bars.trend_1h > 0, 1.5, 0.0)
    bear["htf1h"] = np.where(bars.trend_1h < 0, 1.5, 0.0)
    bull["bos"] = np.where(bars.Close > bars.prior_high20, 2.0, 0.0)
    bear["bos"] = np.where(bars.Close < bars.prior_low20, 2.0, 0.0)
    bull_sweep = (bars.Low < bars.prior_low20) & (bars.Close > bars.prior_low20)
    bear_sweep = (bars.High > bars.prior_high20) & (bars.Close < bars.prior_high20)
    bull["sweep"] = np.where(bull_sweep, 1.75, 0.0)
    bear["sweep"] = np.where(bear_sweep, 1.75, 0.0)
    bull_disp = (bars.Close > bars.Open) & (bars.body_ratio >= 1.5)
    bear_disp = (bars.Close < bars.Open) & (bars.body_ratio >= 1.5)
    bull["displacement"] = np.where(bull_disp, 1.25, 0.0)
    bear["displacement"] = np.where(bear_disp, 1.25, 0.0)
    leader_bull = bull.sum(axis=1) > bear.sum(axis=1)
    volume_confirm = bars.vol_ratio >= 1.25
    bull["volume"] = np.where(volume_confirm & leader_bull, 0.75, 0.0)
    bear["volume"] = np.where(volume_confirm & ~leader_bull, 0.75, 0.0)
    strength_confirm = bars.adx >= 22
    bull["adx"] = np.where(strength_confirm & leader_bull, 0.75, 0.0)
    bear["adx"] = np.where(strength_confirm & ~leader_bull, 0.75, 0.0)
    possible = 18.25
    bars["bull"] = bull.sum(axis=1)
    bars["bear"] = bear.sum(axis=1)
    dominant = bars[["bull", "bear"]].max(axis=1)
    separation = (bars.bull - bars.bear).abs() / (bars.bull + bars.bear).replace(0, np.nan)
    bars["quality"] = 100 * (0.72 * dominant / possible + 0.28 * separation).clip(0, 0.99)
    bars["candidate"] = np.where(bars.bull > bars.bear, 1, np.where(bars.bear > bars.bull, -1, 0))
    bars["separation"] = 100 * separation
    bars["required_event"] = bull_sweep | bear_sweep | bull_disp | bear_disp | (bars.Close > bars.prior_high20) | (bars.Close < bars.prior_low20)
    return bars.dropna(subset=["atr", "ema200", "trend_30m", "trend_1h"])


def mc_probability(returns: np.ndarray, direction: int, stop_atr: float, target_atr: float, paths: int, seed: int) -> tuple[float, float]:
    if len(returns) < 250:
        return 0.5, 0.0
    rng = np.random.default_rng(seed)
    horizon = 24
    draws = rng.choice(returns, size=(paths, horizon), replace=True) * direction
    paths_cum = np.cumsum(draws, axis=1)
    target_hit = paths_cum >= target_atr
    stop_hit = paths_cum <= -stop_atr
    outcomes = []
    for trow, srow, final in zip(target_hit, stop_hit, paths_cum[:, -1]):
        ti = int(np.argmax(trow)) if trow.any() else 10**9
        si = int(np.argmax(srow)) if srow.any() else 10**9
        outcomes.append(1 if ti < si else -1 if si < 10**9 else np.clip(final / stop_atr, -1, target_atr / stop_atr))
    arr = np.asarray(outcomes, dtype=float)
    resolved = arr[np.isin(arr, [-1, 1])]
    probability = float((resolved == 1).mean()) if len(resolved) else 0.5
    return probability, float(arr.mean())


def run_backtest(
    df_1m: pd.DataFrame,
    quality_threshold: float = 76,
    min_probability: float = 0.54,
    contracts: int = 3,
    max_risk: float = 300,
    simulations: int = 400,
    cooldown_bars: int = 1,
    max_trades_per_day: int = 0,
) -> tuple[pd.DataFrame, dict]:
    bars = build_features(df_1m)
    local = bars.tz_convert(ET)
    in_session = local.between_time("09:35", "15:30")
    eligible_index = set(in_session.index.tz_convert("UTC"))
    normalized_returns = (bars.Close.diff() / bars.atr.shift(1)).replace([np.inf, -np.inf], np.nan)
    trades = []
    position_until = -1
    daily_count: dict = {}
    i = 1000
    while i < len(bars) - 25:
        ts = bars.index[i]
        if i <= position_until or ts not in eligible_index:
            i += 1
            continue
        row = bars.iloc[i]
        date = str(ts.tz_convert(ET).date())
        if max_trades_per_day and daily_count.get(date, 0) >= max_trades_per_day:
            i += 1
            continue
        if row.quality < quality_threshold or row.separation < 35 or not bool(row.required_event) or row.candidate == 0:
            i += 1
            continue
        direction = int(row.candidate)
        stop_points = min(0.90 * row.atr, max_risk / max(contracts * 2.0, 1e-9))
        target_points = stop_points * (1.8 if row.quality >= 82 else 1.5)
        stop_atr, target_atr = stop_points / row.atr, target_points / row.atr
        sample = normalized_returns.iloc[max(0, i - 2500):i].dropna().clip(-3, 3).to_numpy()
        probability, ev_r = mc_probability(sample, direction, stop_atr, target_atr, simulations, seed=42 + i)
        if probability < min_probability or ev_r < 0.03:
            i += 1
            continue
        entry = float(row.Close)
        stop = entry - stop_points if direction == 1 else entry + stop_points
        target = entry + target_points if direction == 1 else entry - target_points
        exit_price = float(bars.Close.iloc[i + 24])
        outcome = "TIME"
        exit_i = i + 24
        for j in range(i + 1, min(i + 25, len(bars))):
            future = bars.iloc[j]
            stop_hit = future.Low <= stop if direction == 1 else future.High >= stop
            target_hit = future.High >= target if direction == 1 else future.Low <= target
            if stop_hit:  # conservative same-bar assumption
                exit_price, outcome, exit_i = stop, "STOP", j
                break
            if target_hit:
                exit_price, outcome, exit_i = target, "TARGET", j
                break
        points = (exit_price - entry) * direction
        gross = points * 2.0 * contracts
        costs = contracts * 2.0 + contracts * 0.50
        net = gross - costs
        trades.append({
            "date": date, "entry_time": ts, "exit_time": bars.index[exit_i],
            "signal": "LONG" if direction == 1 else "SHORT", "entry": round(entry, 2),
            "stop": round(stop, 2), "target": round(target, 2), "exit": round(exit_price, 2),
            "quality": round(float(row.quality), 2), "mc_probability": round(probability, 4),
            "mc_expected_r": round(ev_r, 4), "outcome": outcome,
            "gross_pnl": round(gross, 2), "costs": round(costs, 2), "net_pnl": round(net, 2),
        })
        daily_count[date] = daily_count.get(date, 0) + 1
        position_until = exit_i + cooldown_bars
        i = position_until + 1

    result = pd.DataFrame(trades)
    if result.empty:
        return result, {"trades": 0, "note": "No trades passed all gates."}
    result["month"] = pd.to_datetime(result.date).dt.to_period("M").astype(str)
    monthly = result.groupby("month").net_pnl.agg(["sum", "count"])
    equity = result.net_pnl.cumsum()
    dd = equity - equity.cummax()
    gp = result.loc[result.net_pnl > 0, "net_pnl"].sum()
    gl = -result.loc[result.net_pnl <= 0, "net_pnl"].sum()
    summary = {
        "trades": int(len(result)), "net_pnl": round(float(result.net_pnl.sum()), 2),
        "win_rate": round(float((result.net_pnl > 0).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl else None,
        "max_drawdown": round(float(dd.min()), 2), "average_trade": round(float(result.net_pnl.mean()), 2),
        "positive_months": int((monthly["sum"] > 0).sum()), "negative_months": int((monthly["sum"] < 0).sum()),
        "months_with_trades": int(len(monthly)),
        "monthly": {m: {"net_pnl": round(float(r["sum"]), 2), "trades": int(r["count"])} for m, r in monthly.iterrows()},
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--quality", type=float, default=76)
    parser.add_argument("--probability", type=float, default=0.54)
    parser.add_argument("--contracts", type=int, default=3)
    parser.add_argument("--risk", type=float, default=300)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--max-trades-day", type=int, default=0, help="0 means no daily limit")
    parser.add_argument("--output", default="quant_trades.csv")
    parser.add_argument("--summary", default="quant_summary.json")
    args = parser.parse_args()
    data = load_nq_csv(args.csv)
    trades, summary = run_backtest(
        data, args.quality, args.probability, args.contracts, args.risk,
        args.simulations, max_trades_per_day=args.max_trades_day,
    )
    trades.to_csv(args.output, index=False)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

