"""Probability and Monte Carlo layer for OP.exe Quant.

This module validates an existing structural trade plan. It never creates a
direction from Monte Carlo alone, and it uses only bars available at the time
of analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from op_engine import EngineSettings, SignalResult, TF_RULES, analyze_frames, atr


HORIZONS = {"1m": 90, "5m": 48, "30m": 16, "1h": 12, "4h": 8, "1d": 10}


@dataclass
class QuantSettings:
    simulations: int = 1200
    block_size: int = 5
    min_probability: float = 0.56
    min_probability_lower_bound: float = 0.50
    min_expected_value_r: float = 0.05
    analog_lookback: int = 3000
    minimum_analogs: int = 35
    random_seed: int = 42
    round_trip_cost_dollars: float = 2.00


@dataclass
class QuantResult:
    timeframe: str
    signal: str
    candidate: str
    quality: float
    probability_target_first: Optional[float]
    probability_lower_bound: Optional[float]
    expected_value_r: Optional[float]
    monte_carlo_probability: Optional[float]
    analog_probability: Optional[float]
    analog_count: int
    entry: Optional[float]
    stop: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    projected_risk: Optional[float]
    projected_reward: Optional[float]
    setup: str
    target_name: str
    reasons: list[str]
    warnings: list[str]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _wilson_lower(successes: float, trials: float, z: float = 1.645) -> float:
    """One-sided 95% Wilson lower bound."""
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return float(max(0.0, (center - margin) / denom))


def _first_barrier(path: np.ndarray, target: float, stop: float) -> int:
    target_hits = np.flatnonzero(path >= target)
    stop_hits = np.flatnonzero(path <= -stop)
    target_i = int(target_hits[0]) if target_hits.size else 10**9
    stop_i = int(stop_hits[0]) if stop_hits.size else 10**9
    if target_i < stop_i:
        return 1
    if stop_i <= target_i and stop_i < 10**9:
        return -1
    return 0


def _normalized_returns(df: pd.DataFrame) -> tuple[np.ndarray, float, str]:
    a = atr(df).replace(0, np.nan)
    normalized = (df["Close"].diff() / a.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    current_atr = float(a.iloc[-1])
    recent = normalized.tail(2500).clip(-3.0, 3.0).to_numpy(dtype=float)
    recent_vol = float(normalized.tail(120).std())
    baseline_vol = float(normalized.tail(1000).std())
    ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
    regime = "HIGH" if ratio >= 1.25 else "LOW" if ratio <= 0.80 else "NORMAL"
    return recent, current_atr, regime


def bootstrap_monte_carlo(
    df: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    timeframe: str,
    settings: QuantSettings,
) -> Dict[str, Any]:
    samples, current_atr, regime = _normalized_returns(df)
    if len(samples) < 200 or not np.isfinite(current_atr) or current_atr <= 0:
        return {"probability": np.nan, "lower": np.nan, "ev_r": np.nan, "resolved": 0, "regime": regime}
    sign = 1.0 if direction == "LONG" else -1.0
    stop_distance = abs(entry - stop) / current_atr
    target_distance = abs(target - entry) / current_atr
    if stop_distance <= 0 or target_distance <= 0:
        return {"probability": np.nan, "lower": np.nan, "ev_r": np.nan, "resolved": 0, "regime": regime}

    horizon = HORIZONS[timeframe]
    block = max(1, min(settings.block_size, horizon))
    rng = np.random.default_rng(settings.random_seed + sum(map(ord, timeframe)))
    outcomes = []
    terminal_r = []
    starts_max = max(1, len(samples) - block)
    for _ in range(settings.simulations):
        pieces = []
        while sum(len(x) for x in pieces) < horizon:
            start = int(rng.integers(0, starts_max))
            pieces.append(samples[start:start + block])
        path = np.cumsum(np.concatenate(pieces)[:horizon] * sign)
        outcome = _first_barrier(path, target_distance, stop_distance)
        outcomes.append(outcome)
        if outcome == 1:
            terminal_r.append(target_distance / stop_distance)
        elif outcome == -1:
            terminal_r.append(-1.0)
        else:
            terminal_r.append(float(np.clip(path[-1] / stop_distance, -1.0, target_distance / stop_distance)))
    outcomes = np.asarray(outcomes)
    wins = int((outcomes == 1).sum())
    losses = int((outcomes == -1).sum())
    resolved = wins + losses
    probability = wins / resolved if resolved else 0.5
    return {
        "probability": float(probability),
        "lower": _wilson_lower(wins, resolved),
        "ev_r": float(np.mean(terminal_r)),
        "resolved": int(resolved),
        "regime": regime,
        "p10_r": float(np.quantile(terminal_r, 0.10)),
        "median_r": float(np.median(terminal_r)),
        "p90_r": float(np.quantile(terminal_r, 0.90)),
    }


def historical_barrier_analogs(
    df: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    timeframe: str,
    settings: QuantSettings,
) -> Dict[str, Any]:
    """Evaluate prior, directionally similar states with fixed R barriers."""
    if len(df) < 300:
        return {"probability": np.nan, "lower": np.nan, "count": 0, "ev_r": np.nan}
    data = df.tail(settings.analog_lookback + HORIZONS[timeframe] + 250).copy()
    close = data["Close"]
    atr_s = atr(data)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    momentum = close.diff(5)
    current_atr = float(atr_s.iloc[-1])
    stop_r = abs(entry - stop) / current_atr
    target_r = abs(target - entry) / current_atr
    desired_sign = 1 if direction == "LONG" else -1
    current_vol_ratio = float(atr_s.iloc[-1] / atr_s.tail(200).median())
    outcomes = []
    horizon = HORIZONS[timeframe]
    start = max(220, len(data) - settings.analog_lookback)
    # Step by two to reduce autocorrelation and runtime.
    for i in range(start, len(data) - horizon - 1, 2):
        if not np.isfinite(atr_s.iloc[i]) or atr_s.iloc[i] <= 0:
            continue
        trend_sign = 1 if ema20.iloc[i] > ema50.iloc[i] else -1
        momentum_sign = 1 if momentum.iloc[i] > 0 else -1
        if trend_sign != desired_sign or momentum_sign != desired_sign:
            continue
        local_ratio = float(atr_s.iloc[i] / atr_s.iloc[max(0, i - 200):i + 1].median())
        if not (0.65 * current_vol_ratio <= local_ratio <= 1.55 * current_vol_ratio):
            continue
        future = data.iloc[i + 1:i + 1 + horizon]
        base = float(close.iloc[i])
        if direction == "LONG":
            favorable = (future["High"].to_numpy() - base) / float(atr_s.iloc[i])
            adverse = (base - future["Low"].to_numpy()) / float(atr_s.iloc[i])
        else:
            favorable = (base - future["Low"].to_numpy()) / float(atr_s.iloc[i])
            adverse = (future["High"].to_numpy() - base) / float(atr_s.iloc[i])
        target_hits = np.flatnonzero(favorable >= target_r)
        stop_hits = np.flatnonzero(adverse >= stop_r)
        ti = int(target_hits[0]) if target_hits.size else 10**9
        si = int(stop_hits[0]) if stop_hits.size else 10**9
        outcomes.append(1 if ti < si else -1 if si < 10**9 else 0)
    resolved = [x for x in outcomes if x != 0]
    if not resolved:
        return {"probability": np.nan, "lower": np.nan, "count": 0, "ev_r": np.nan}
    wins = sum(x == 1 for x in resolved)
    probability = wins / len(resolved)
    rr = target_r / stop_r
    ev_r = probability * rr - (1 - probability)
    return {
        "probability": float(probability), "lower": _wilson_lower(wins, len(resolved)),
        "count": int(len(resolved)), "ev_r": float(ev_r),
    }


def validate_plan(
    base: SignalResult,
    df: pd.DataFrame,
    quant: QuantSettings,
    provider_fresh: bool,
) -> QuantResult:
    warnings = list(base.warnings)
    reasons = list(base.reasons)
    candidate = base.signal
    if candidate not in ("LONG", "SHORT"):
        bull = float(base.diagnostics.get("bull_points", 0) or 0)
        bear = float(base.diagnostics.get("bear_points", 0) or 0)
        candidate = "LONG" if bull > bear else "SHORT" if bear > bull else "WAIT"
    required = all(x is not None for x in (base.entry, base.stop, base.take_profit))
    if candidate == "WAIT" or not required:
        return QuantResult(
            base.timeframe, "WAIT", candidate, base.quality, None, None, None, None, None, 0,
            base.entry, base.stop, base.take_profit, base.risk_reward,
            base.projected_risk, base.projected_reward, base.setup, base.target_name,
            reasons, warnings, {"base": base.diagnostics},
        )

    mc = bootstrap_monte_carlo(df, candidate, base.entry, base.stop, base.take_profit, base.timeframe, quant)
    analog = historical_barrier_analogs(df, candidate, base.entry, base.stop, base.take_profit, base.timeframe, quant)
    mc_p = mc.get("probability", np.nan)
    analog_p = analog.get("probability", np.nan)
    if np.isfinite(analog_p) and analog.get("count", 0) >= quant.minimum_analogs:
        probability = 0.55 * mc_p + 0.45 * analog_p
        effective_n = min(quant.simulations, mc.get("resolved", 0)) + analog["count"]
        pseudo_wins = probability * effective_n
        lower = _wilson_lower(pseudo_wins, effective_n)
        ev_r = 0.55 * mc.get("ev_r", 0) + 0.45 * analog.get("ev_r", 0)
    else:
        probability = mc_p
        lower = mc.get("lower", np.nan)
        ev_r = mc.get("ev_r", np.nan)
        warnings.append("Too few historical analogs; probability relies primarily on bootstrap paths")

    failures = []
    if base.signal not in ("LONG", "SHORT"):
        failures.append("structural engine did not pass")
    if not provider_fresh:
        failures.append("market feed is not verified fresh")
    if not np.isfinite(probability) or probability < quant.min_probability:
        failures.append(f"estimated target-first probability below {quant.min_probability:.0%}")
    if not np.isfinite(lower) or lower < quant.min_probability_lower_bound:
        failures.append(f"probability lower bound below {quant.min_probability_lower_bound:.0%}")
    if not np.isfinite(ev_r) or ev_r < quant.min_expected_value_r:
        failures.append(f"expected value below {quant.min_expected_value_r:.2f}R")
    if base.projected_risk and base.projected_reward:
        cost_r = quant.round_trip_cost_dollars / max(base.projected_risk, 1e-9)
        if ev_r - cost_r < quant.min_expected_value_r:
            failures.append("expected value is inadequate after estimated costs")
    signal = "WAIT" if failures else candidate
    if failures:
        reasons.insert(0, "WAIT: " + "; ".join(failures))
    else:
        reasons.insert(0, f"{candidate}: structural and statistical gates passed")
    diagnostics = {
        "base": base.diagnostics, "monte_carlo": mc, "historical_analogs": analog,
        "combined_probability": probability, "combined_lower_bound": lower,
        "combined_expected_value_r": ev_r,
    }
    return QuantResult(
        timeframe=base.timeframe, signal=signal, candidate=candidate, quality=base.quality,
        probability_target_first=round(float(probability), 4) if np.isfinite(probability) else None,
        probability_lower_bound=round(float(lower), 4) if np.isfinite(lower) else None,
        expected_value_r=round(float(ev_r), 4) if np.isfinite(ev_r) else None,
        monte_carlo_probability=round(float(mc_p), 4) if np.isfinite(mc_p) else None,
        analog_probability=round(float(analog_p), 4) if np.isfinite(analog_p) else None,
        analog_count=int(analog.get("count", 0)), entry=base.entry, stop=base.stop,
        take_profit=base.take_profit, risk_reward=base.risk_reward,
        projected_risk=base.projected_risk, projected_reward=base.projected_reward,
        setup=base.setup, target_name=base.target_name, reasons=reasons[:12],
        warnings=warnings, diagnostics=diagnostics,
    )


def quant_analyze_all(
    frames: Dict[str, pd.DataFrame],
    df_1m: pd.DataFrame,
    engine: EngineSettings,
    quant: QuantSettings,
    orderflow: Optional[Dict[str, Any]] = None,
    feed_fresh: bool = True,
) -> Dict[str, QuantResult]:
    base = analyze_frames(frames, df_1m, engine, orderflow=orderflow, feed_fresh=feed_fresh)
    return {
        tf: validate_plan(base[tf], frames[tf].iloc[:-1] if len(frames[tf]) > 1 else frames[tf], quant, feed_fresh)
        for tf in TF_RULES
    }

