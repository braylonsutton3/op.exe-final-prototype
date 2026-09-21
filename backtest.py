"""Conservative walk-forward-style backtest for OP.exe.

Usage:
    python backtest.py /path/to/Dataset_NQ_1min_2022_2025.csv

The test takes at most one trade per RTH day. It evaluates at 10:00 and 10:30
ET using only bars whose timestamps are available at that checkpoint. If stop
and target are both touched in the same future candle, the stop is counted
first. One tick of adverse slippage plus configurable commission is deducted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from op_engine import ET, TF_RULES, EngineSettings, analyze_frame, dol_levels, load_nq_csv, opening_setup, resample_ohlcv


def evaluate_exit(future: pd.DataFrame, direction: str, stop: float, target: float) -> tuple[float, str, pd.Timestamp]:
    for ts, row in future.iterrows():
        if direction == "LONG":
            stop_hit, target_hit = row["Low"] <= stop, row["High"] >= target
        else:
            stop_hit, target_hit = row["High"] >= stop, row["Low"] <= target
        if stop_hit:
            return stop, "STOP", ts
        if target_hit:
            return target, "TARGET", ts
    if future.empty:
        return np.nan, "NO DATA", pd.NaT
    return float(future["Close"].iloc[-1]), "TIME", future.index[-1]


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min())


def run_backtest(
    df: pd.DataFrame,
    settings: EngineSettings,
    commission_per_contract_round_trip: float = 1.50,
    slippage_ticks: int = 1,
) -> tuple[pd.DataFrame, dict]:
    frames = {tf: resample_ohlcv(df, tf) for tf in TF_RULES}
    local = df.tz_convert(ET)
    dates = sorted(set(local.index.date))
    trades = []

    for n, date in enumerate(dates):
        # Intraday validation needs enough history for the 1-hour EMA200.
        if n < 35:
            continue
        day = local[local.index.date == date].between_time("09:30", "15:55")
        if day.empty:
            continue
        trade = None
        for checkpoint_text in ("10:00", "10:30"):
            candidate_rows = day.between_time(checkpoint_text, checkpoint_text)
            if candidate_rows.empty:
                continue
            checkpoint = candidate_rows.index[-1].tz_convert("UTC")
            df1_context = df.loc[:checkpoint].tail(6000)
            levels = dol_levels(df1_context)
            setup_context = opening_setup(df1_context)
            output = {}
            higher_bias = None
            tested_timeframes = ("1h", "30m", "5m", "1m")
            for tf in tested_timeframes:
                frame = frames[tf].loc[:checkpoint]
                result = analyze_frame(
                    frame, tf, df1_context, settings, orderflow=None,
                    higher_bias=higher_bias, feed_fresh=True,
                    precomputed_levels=levels, precomputed_setup=setup_context,
                )
                output[tf] = result
                if result.signal in ("LONG", "SHORT"):
                    higher_bias = result.signal

            # 1m has first priority, then 5m. 30m/1h may not strongly oppose.
            chosen = output["1m"] if output["1m"].signal in ("LONG", "SHORT") else output["5m"]
            if chosen.signal not in ("LONG", "SHORT"):
                continue
            opponents = sum(output[tf].signal not in ("WAIT", chosen.signal) for tf in ("30m", "1h"))
            if opponents:
                continue
            trade = (checkpoint, chosen, output)
            break

        if n % 75 == 0:
            print(f"Processed {n}/{len(dates)} sessions; trades={len(trades)}", flush=True)

        if trade is None:
            continue
        checkpoint, chosen, output = trade
        future = df.loc[df.index > checkpoint].tz_convert(ET)
        future = future[(future.index.date == date)].between_time("09:31", "15:55").tz_convert("UTC")
        exit_price, outcome, exit_time = evaluate_exit(future, chosen.signal, chosen.stop, chosen.take_profit)
        if not np.isfinite(exit_price):
            continue
        points = (exit_price - chosen.entry) if chosen.signal == "LONG" else (chosen.entry - exit_price)
        gross = points * 2.0 * settings.contracts  # MNQ = $2 per index point
        costs = settings.contracts * commission_per_contract_round_trip + slippage_ticks * 0.50 * settings.contracts
        net = gross - costs
        trades.append({
            "date": str(date), "entry_time": checkpoint, "exit_time": exit_time,
            "signal": chosen.signal, "source_tf": chosen.timeframe, "quality": chosen.quality,
            "entry": chosen.entry, "stop": chosen.stop, "target": chosen.take_profit,
            "exit": exit_price, "outcome": outcome, "gross_pnl": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "rr": chosen.risk_reward, "setup": chosen.setup,
            "30m": output["30m"].signal, "1h": output["1h"].signal,
            "4h": "NOT USED FOR ENTRY TEST", "1d": "NOT USED FOR ENTRY TEST",
        })

    result = pd.DataFrame(trades)
    if result.empty:
        return result, {"trades": 0, "note": "No trades passed the strict gates."}
    result["month"] = pd.to_datetime(result["date"]).dt.to_period("M").astype(str)
    monthly = result.groupby("month")["net_pnl"].agg(["sum", "count"])
    wins, losses = result[result.net_pnl > 0], result[result.net_pnl <= 0]
    gross_profit = float(wins.net_pnl.sum())
    gross_loss = abs(float(losses.net_pnl.sum()))
    summary = {
        "trades": int(len(result)),
        "net_pnl": round(float(result.net_pnl.sum()), 2),
        "win_rate": round(float((result.net_pnl > 0).mean() * 100), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "max_drawdown": round(max_drawdown(result.net_pnl), 2),
        "average_trade": round(float(result.net_pnl.mean()), 2),
        "positive_months": int((monthly["sum"] > 0).sum()),
        "negative_months": int((monthly["sum"] < 0).sum()),
        "flat_months": int((monthly["sum"] == 0).sum()),
        "months_with_trades": int(len(monthly)),
        "monthly": {idx: {"net_pnl": round(float(row["sum"]), 2), "trades": int(row["count"])} for idx, row in monthly.iterrows()},
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", default="backtest_trades.csv")
    parser.add_argument("--summary", default="backtest_summary.json")
    parser.add_argument("--quality", type=float, default=76.0)
    parser.add_argument("--contracts", type=int, default=3)
    parser.add_argument("--risk", type=float, default=300.0)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    args = parser.parse_args()
    data = load_nq_csv(args.csv)
    if args.start:
        data = data.loc[pd.Timestamp(args.start, tz=ET).tz_convert("UTC"):]
    if args.end:
        data = data.loc[:pd.Timestamp(args.end, tz=ET).tz_convert("UTC")]
    settings = EngineSettings(contracts=args.contracts, max_risk_dollars=args.risk, min_quality=args.quality)
    trades, summary = run_backtest(data, settings)
    trades.to_csv(args.output, index=False)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
