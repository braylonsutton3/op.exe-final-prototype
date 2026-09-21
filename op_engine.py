"""Core calculations for OP.exe.

The engine is deliberately deterministic and uses completed candles only.
It does not place orders and it does not represent its quality score as a
probability of profit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:  # lets the offline backtester run before web dependencies are installed
    requests = None


ET = "America/New_York"
TF_RULES = {
    "1m": "1min",
    "5m": "5min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50


@dataclass
class EngineSettings:
    contracts: int = 3
    max_risk_dollars: float = 300.0
    min_rr: float = 1.50
    min_quality: float = 76.0
    require_orderflow: bool = False
    atr_stop_multiplier: float = 0.90
    atr_target_cap: float = 3.00
    fvg_lookback: int = 160
    swing_window: int = 3
    stale_seconds: int = 180
    hard_stale_seconds: int = 900
    tick_size: float = MNQ_TICK_SIZE
    tick_value: float = MNQ_TICK_VALUE


@dataclass
class SignalResult:
    timeframe: str
    signal: str
    quality: float
    entry: Optional[float]
    stop: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    projected_risk: Optional[float]
    projected_reward: Optional[float]
    target_name: str
    stop_name: str
    setup: str
    reasons: list[str]
    warnings: list[str]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def round_tick(value: Optional[float], tick_size: float = MNQ_TICK_SIZE) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(round(float(value) / tick_size) * tick_size, 8)


def normalize_ohlcv(df: pd.DataFrame, assume_et: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = df.copy()
    aliases = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume", "t": "Datetime",
        "timestamp ET": "Datetime", "timestamp": "Datetime", "date": "Datetime",
    }
    out = out.rename(columns={c: aliases.get(str(c), str(c)) for c in out.columns})
    if "Datetime" in out.columns:
        dt = pd.to_datetime(out.pop("Datetime"), errors="coerce")
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(ET if assume_et else "UTC", ambiguous="NaT", nonexistent="shift_forward")
        dt = dt.dt.tz_convert("UTC")
        out.index = dt
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif out.index.tz is None:
        out.index = out.index.tz_localize(ET if assume_et else "UTC").tz_convert("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in out:
            out[col] = 0.0 if col == "Volume" else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[["Open", "High", "Low", "Close", "Volume"]]
    return out.dropna(subset=["Open", "High", "Low", "Close"]).loc[lambda x: ~x.index.isna()].sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


def load_nq_csv(path_or_buffer: Any) -> pd.DataFrame:
    return normalize_ohlcv(pd.read_csv(path_or_buffer), assume_et=True)


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1m":
        return df_1m.copy()
    rule = TF_RULES[timeframe]
    local = df_1m.tz_convert(ET)
    # origin=start_day keeps hour and four-hour candles aligned to midnight ET.
    result = local.resample(rule, origin="start_day", label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "High", "Low", "Close"])
    return result.tz_convert("UTC")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["Close"].shift(1)
    tr = pd.concat(
        [(df["High"] - df["Low"]), (df["High"] - previous).abs(), (df["Low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def session_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    local = df.tz_convert(ET)
    day = local[local.index.date == local.index[-1].date()]
    volume = day["Volume"].clip(lower=0)
    if day.empty or volume.sum() <= 0:
        return np.nan
    typical = (day["High"] + day["Low"] + day["Close"]) / 3.0
    return float((typical * volume).sum() / volume.sum())


def confirmed_swings(df: pd.DataFrame, window: int = 3, lookback: int = 260) -> tuple[list, list]:
    data = df.tail(lookback)
    highs, lows = [], []
    if len(data) < window * 2 + 3:
        return highs, lows
    hv, lv = data["High"].to_numpy(), data["Low"].to_numpy()
    for i in range(window, len(data) - window):
        if hv[i] >= hv[i - window:i + window + 1].max():
            highs.append((data.index[i], float(hv[i])))
        if lv[i] <= lv[i - window:i + window + 1].min():
            lows.append((data.index[i], float(lv[i])))
    return highs, lows


def structure_state(df: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
    highs, lows = confirmed_swings(df.iloc[:-1], window)
    close = float(df["Close"].iloc[-1])
    if not highs or not lows:
        return {"trend": "MIXED", "bos": "NONE", "high": None, "low": None}
    last_high, last_low = highs[-1][1], lows[-1][1]
    trend = "MIXED"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
            trend = "BULL"
        elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
            trend = "BEAR"
    bos = "BULL" if close > last_high else "BEAR" if close < last_low else "NONE"
    return {"trend": trend, "bos": bos, "high": last_high, "low": last_low}


def liquidity_sweep(df: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
    if len(df) < 20:
        return {"side": "NONE", "level": None}
    highs, lows = confirmed_swings(df.iloc[:-1], window)
    bar = df.iloc[-1]
    if highs and bar["High"] > highs[-1][1] and bar["Close"] < highs[-1][1]:
        return {"side": "BEAR", "level": highs[-1][1]}
    if lows and bar["Low"] < lows[-1][1] and bar["Close"] > lows[-1][1]:
        return {"side": "BULL", "level": lows[-1][1]}
    return {"side": "NONE", "level": None}


def displacement(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 30:
        return {"side": "NONE", "ratio": 0.0}
    bodies = (df["Close"] - df["Open"]).abs()
    base = float(bodies.iloc[-21:-1].median())
    bar = df.iloc[-1]
    body = abs(float(bar["Close"] - bar["Open"]))
    ratio = body / base if base > 0 else 0.0
    body_share = body / max(float(bar["High"] - bar["Low"]), 1e-9)
    if ratio >= 1.5 and body_share >= 0.60:
        return {"side": "BULL" if bar["Close"] > bar["Open"] else "BEAR", "ratio": ratio}
    return {"side": "NONE", "ratio": ratio}


def active_fvgs(df: pd.DataFrame, lookback: int = 160) -> list[Dict[str, Any]]:
    gaps: list[Dict[str, Any]] = []
    if len(df) < 5:
        return gaps
    start = max(2, len(df) - lookback)
    for i in range(start, len(df)):
        left, right = df.iloc[i - 2], df.iloc[i]
        side, low, high = None, None, None
        if right["Low"] > left["High"]:
            side, low, high = "BULL", float(left["High"]), float(right["Low"])
        elif right["High"] < left["Low"]:
            side, low, high = "BEAR", float(right["High"]), float(left["Low"])
        if side is None:
            continue
        midpoint = (low + high) / 2.0
        after = df.iloc[i + 1:]
        filled = False
        if not after.empty:
            filled = bool((after["Low"] <= midpoint).any()) if side == "BULL" else bool((after["High"] >= midpoint).any())
        if not filled:
            gaps.append({"side": side, "low": low, "high": high, "mid": midpoint, "time": df.index[i]})
    return gaps


def dol_levels(df_1m: pd.DataFrame, tick_size: float = MNQ_TICK_SIZE) -> list[Dict[str, Any]]:
    """Build objective Draw-on-Liquidity candidates from completed 1m bars."""
    if df_1m.empty:
        return []
    local = df_1m.tz_convert(ET)
    current_date = local.index[-1].date()
    days = sorted(set(local.index.date))
    levels: list[Dict[str, Any]] = []

    def add(name: str, value: Any, strength: float) -> None:
        if value is not None and np.isfinite(value):
            levels.append({"name": name, "price": float(value), "strength": float(strength)})

    if len(days) >= 2:
        previous = local[local.index.date == days[-2]]
        rth = previous.between_time("09:30", "16:00")
        source = rth if not rth.empty else previous
        add("Prior-day high", source["High"].max(), 5.0)
        add("Prior-day low", source["Low"].min(), 5.0)
        add("Prior close", source["Close"].iloc[-1], 4.0)

    day = local[local.index.date == current_date]
    overnight = day.between_time("00:00", "09:29")
    rth = day.between_time("09:30", "16:00")
    opening = day.between_time("09:30", "09:59")
    if not overnight.empty:
        add("Overnight high", overnight["High"].max(), 4.0)
        add("Overnight low", overnight["Low"].min(), 4.0)
    if not rth.empty:
        add("RTH high", rth["High"].max(), 3.0)
        add("RTH low", rth["Low"].min(), 3.0)
        add("9:30 open", rth["Open"].iloc[0], 3.5)
    if len(opening) >= 20:
        add("9:30–9:59 high", opening["High"].max(), 4.5)
        add("9:30–9:59 low", opening["Low"].min(), 4.5)
        add("9:30–9:59 midpoint", (opening["High"].max() + opening["Low"].min()) / 2.0, 3.5)
    current_vwap = session_vwap(df_1m)
    add("Session VWAP", current_vwap, 4.0)

    weekly = local.resample("W-FRI").agg({"High": "max", "Low": "min"}).dropna()
    if len(weekly) >= 2:
        add("Prior-week high", weekly["High"].iloc[-2], 5.0)
        add("Prior-week low", weekly["Low"].iloc[-2], 5.0)
    # Deduplicate levels that are effectively the same price.
    unique: Dict[float, Dict[str, Any]] = {}
    for level in levels:
        key = round(level["price"] / tick_size) * tick_size
        if key not in unique or level["strength"] > unique[key]["strength"]:
            unique[key] = level
    return list(unique.values())


def opening_setup(df_1m: pd.DataFrame) -> tuple[str, str]:
    """Apply the user's Three-Setup Priority System as context, not a standalone trigger."""
    if df_1m.empty:
        return "General confluence", "No opening data"
    local = df_1m.tz_convert(ET)
    day = local[local.index.date == local.index[-1].date()]
    previous_days = sorted(set(local.index.date))
    now = local.index[-1]
    if len(previous_days) >= 2 and not day.empty:
        prev = local[local.index.date == previous_days[-2]].between_time("09:30", "16:00")
        if not prev.empty and now.weekday() in (0, 3):
            gap = float(day["Open"].iloc[0] - prev["Close"].iloc[-1])
            if abs(gap) >= 1.5 * float(atr(df_1m).tail(390).median() or 0):
                return "Priority 1 · Gap reversal", f"Opening gap {gap:+.2f} points"
    opening = day.between_time("09:30", "09:59")
    if len(opening) >= 20:
        width = float(opening["High"].max() - opening["Low"].min())
        if width <= 120:
            return "Priority 2 · Opening-drive continuation", f"Opening range {width:.2f} points"
    return "Priority 3 · ORB fallback", "No higher-priority opening setup qualified"


def orderflow_state(snapshot: Optional[Dict[str, Any]], current_price: float, level: Optional[float], tick_size: float = MNQ_TICK_SIZE) -> Dict[str, Any]:
    """Interpret a MarketBook/Bookmap/Databento bridge snapshot.

    Expected optional fields: timestamp, bid_depth, ask_depth, delta,
    delta_1m, aggressive_buy, aggressive_sell, absorption_bid,
    absorption_ask. Missing fields stay neutral.
    """
    if not snapshot:
        return {"side": "NONE", "score": 0.0, "fresh": False, "reason": "No order-flow snapshot"}
    try:
        ts = pd.Timestamp(snapshot.get("timestamp"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = (pd.Timestamp.now(tz="UTC") - ts.tz_convert("UTC")).total_seconds()
    except Exception:
        age = np.inf
    bid = float(snapshot.get("bid_depth", 0) or 0)
    ask = float(snapshot.get("ask_depth", 0) or 0)
    delta = float(snapshot.get("delta_1m", snapshot.get("delta", 0)) or 0)
    buys = float(snapshot.get("aggressive_buy", 0) or 0)
    sells = float(snapshot.get("aggressive_sell", 0) or 0)
    total_depth = bid + ask
    imbalance = (bid - ask) / total_depth if total_depth > 0 else 0.0
    aggression = (buys - sells) / max(buys + sells, 1.0)
    absorption_bid = bool(snapshot.get("absorption_bid", False))
    absorption_ask = bool(snapshot.get("absorption_ask", False))
    proximity = abs(current_price - level) / tick_size if level is not None else np.inf
    bull = max(imbalance, 0) * 35 + max(aggression, 0) * 30 + (15 if delta > 0 else 0) + (20 if absorption_bid and proximity <= 8 else 0)
    bear = max(-imbalance, 0) * 35 + max(-aggression, 0) * 30 + (15 if delta < 0 else 0) + (20 if absorption_ask and proximity <= 8 else 0)
    side = "BULL" if bull >= 25 and bull > bear else "BEAR" if bear >= 25 and bear > bull else "NONE"
    score = min(100.0, max(bull, bear))
    return {
        "side": side, "score": score, "fresh": age <= 15,
        "reason": f"book imbalance {imbalance:+.2f}, aggression {aggression:+.2f}, delta {delta:+.0f}, age {age:.0f}s",
        "age": age,
    }


def _choose_target(levels: Iterable[Dict[str, Any]], gaps: list[Dict[str, Any]], entry: float, direction: str, atr_value: float) -> tuple[float, str]:
    candidates = []
    for item in levels:
        price = float(item["price"])
        valid = price > entry if direction == "LONG" else price < entry
        if valid:
            distance = abs(price - entry)
            rank = float(item["strength"]) - max(0.0, distance / atr_value - 3.0)
            candidates.append((rank, -distance, price, item["name"]))
    for gap in gaps:
        price = float(gap["mid"])
        valid = price > entry if direction == "LONG" else price < entry
        if valid:
            distance = abs(price - entry)
            candidates.append((3.25, -distance, price, "Unfilled FVG midpoint"))
    if candidates:
        _, _, price, name = max(candidates)
        if abs(price - entry) <= 3.5 * atr_value:
            return price, name
    fallback = entry + 1.8 * atr_value if direction == "LONG" else entry - 1.8 * atr_value
    return fallback, "1.8 ATR extension"


def _choose_stop(df: pd.DataFrame, entry: float, direction: str, atr_value: float, settings: EngineSettings) -> tuple[float, str]:
    structure = structure_state(df, settings.swing_window)
    raw = structure["low"] if direction == "LONG" else structure["high"]
    label = "Confirmed swing invalidation"
    min_distance = 0.55 * atr_value
    max_distance = settings.atr_stop_multiplier * atr_value
    if raw is None or (direction == "LONG" and raw >= entry) or (direction == "SHORT" and raw <= entry):
        raw = entry - max_distance if direction == "LONG" else entry + max_distance
        label = f"{settings.atr_stop_multiplier:.2f} ATR volatility stop"
    distance = abs(entry - raw) + 0.10 * atr_value
    distance = min(max(distance, min_distance), max_distance)
    # Enforce the requested dollar cap across the configured position size.
    point_value = settings.tick_value / max(settings.tick_size, 1e-9)
    dollar_cap_points = settings.max_risk_dollars / max(settings.contracts * point_value, 1e-9)
    distance = min(distance, dollar_cap_points)
    stop = entry - distance if direction == "LONG" else entry + distance
    return stop, label


def analyze_frame(
    df: pd.DataFrame,
    timeframe: str,
    df_1m: pd.DataFrame,
    settings: EngineSettings,
    orderflow: Optional[Dict[str, Any]] = None,
    higher_bias: Optional[str] = None,
    feed_fresh: bool = True,
    precomputed_levels: Optional[list[Dict[str, Any]]] = None,
    precomputed_setup: Optional[tuple[str, str]] = None,
) -> SignalResult:
    warnings: list[str] = []
    if df is None or len(df) < 220:
        return SignalResult(timeframe, "WAIT", 0, None, None, None, None, None, None, "", "", "Insufficient data", ["Need at least 220 completed bars"], warnings, {})

    data = df.copy().iloc[:-1] if len(df) > 221 else df.copy()  # exclude possibly forming bar
    if len(data) < 220:
        data = df.copy()
    close = data["Close"]
    entry = float(close.iloc[-1])
    atr_s = atr(data)
    a = float(atr_s.iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return SignalResult(timeframe, "WAIT", 0, entry, None, None, None, None, None, "", "", "Invalid volatility", ["ATR unavailable"], warnings, {})

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    r = float(rsi(close).iloc[-1])
    ax = float(adx(data).iloc[-1])
    volume_base = float(data["Volume"].iloc[-21:-1].median())
    volume_ratio = float(data["Volume"].iloc[-1] / volume_base) if volume_base > 0 else 1.0
    struct = structure_state(data, settings.swing_window)
    sweep = liquidity_sweep(data, settings.swing_window)
    move = displacement(data)
    gaps = active_fvgs(data, settings.fvg_lookback)
    levels = precomputed_levels if precomputed_levels is not None else dol_levels(df_1m, settings.tick_size)
    vwap = session_vwap(df_1m)
    setup, setup_note = precomputed_setup if precomputed_setup is not None else opening_setup(df_1m)

    bull = bear = possible = 0.0
    reasons: list[str] = []

    def vote(side: str, weight: float, text: str) -> None:
        nonlocal bull, bear, possible
        possible += weight
        if side == "BULL":
            bull += weight
            reasons.append("Bull: " + text)
        elif side == "BEAR":
            bear += weight
            reasons.append("Bear: " + text)

    vote("BULL" if entry > ema20 else "BEAR", 1.0, "price vs EMA20")
    vote("BULL" if ema20 > ema50 else "BEAR", 1.5, "EMA20/50 trend")
    vote("BULL" if ema50 > ema200 else "BEAR", 2.0, "EMA50/200 regime")
    if r >= 55:
        vote("BULL", 0.75, f"RSI {r:.1f}")
    elif r <= 45:
        vote("BEAR", 0.75, f"RSI {r:.1f}")
    else:
        possible += 0.75
    vote(struct["trend"], 1.5, f"swing structure {struct['trend']}")
    vote(struct["bos"], 2.0, f"break of structure {struct['bos']}")
    vote(sweep["side"], 1.75, f"liquidity sweep at {sweep['level']}")
    vote(move["side"], 1.25, f"displacement {move['ratio']:.2f}× median body")
    if timeframe in ("1m", "5m", "30m", "1h") and np.isfinite(vwap):
        vote("BULL" if entry > vwap else "BEAR", 1.25, "session VWAP location")
    else:
        possible += 1.25
    recent_gap = min(gaps, key=lambda g: abs(g["mid"] - entry)) if gaps else None
    if recent_gap:
        # Bullish FVG below price is support; bearish FVG above is resistance.
        if recent_gap["side"] == "BULL" and recent_gap["mid"] <= entry:
            vote("BULL", 1.25, "active bullish FVG support")
        elif recent_gap["side"] == "BEAR" and recent_gap["mid"] >= entry:
            vote("BEAR", 1.25, "active bearish FVG resistance")
        else:
            possible += 1.25
    else:
        possible += 1.25
    if ax >= 22:
        leader = "BULL" if bull > bear else "BEAR" if bear > bull else "NONE"
        vote(leader, 0.75, f"ADX {ax:.1f} confirms directional strength")
    else:
        possible += 0.75
    if volume_ratio >= 1.25:
        leader = "BULL" if bull > bear else "BEAR" if bear > bull else "NONE"
        vote(leader, 0.75, f"volume expansion {volume_ratio:.2f}×")
    else:
        possible += 0.75
    if higher_bias in ("LONG", "SHORT"):
        vote("BULL" if higher_bias == "LONG" else "BEAR", 1.25, f"higher-timeframe bias {higher_bias}")
    else:
        possible += 1.25

    nearest_level = min(levels, key=lambda x: abs(x["price"] - entry))["price"] if levels else None
    flow = orderflow_state(orderflow, entry, nearest_level, settings.tick_size)
    if flow["fresh"] and flow["side"] != "NONE":
        vote(flow["side"], 1.75, "live order flow: " + flow["reason"])
    else:
        possible += 1.75
        if settings.require_orderflow and timeframe in ("1m", "5m"):
            warnings.append("Fresh order-flow confirmation is required but unavailable")

    dominant_side = "LONG" if bull > bear else "SHORT" if bear > bull else "WAIT"
    dominant = max(bull, bear)
    quality = 100.0 * dominant / max(possible, 1e-9)
    separation = 100.0 * abs(bull - bear) / max(bull + bear, 1e-9)
    quality = min(99.0, 0.72 * quality + 0.28 * separation)

    # Strict gates make WAIT the default when evidence is mixed.
    failures = []
    if quality < settings.min_quality:
        failures.append(f"quality {quality:.0f} < {settings.min_quality:.0f}")
    if separation < 35:
        failures.append(f"direction separation {separation:.0f} < 35")
    if not feed_fresh:
        failures.append("live feed is stale or only a delayed fallback")
    if settings.require_orderflow and timeframe in ("1m", "5m") and not (flow["fresh"] and flow["side"] == ("BULL" if dominant_side == "LONG" else "BEAR")):
        failures.append("order flow does not confirm")
    if flow["fresh"] and flow["side"] in ("BULL", "BEAR"):
        opposing = flow["side"] != ("BULL" if dominant_side == "LONG" else "BEAR")
        if opposing and flow["score"] >= 35:
            failures.append("strong order flow opposes candle signal")

    target, target_name = _choose_target(levels, gaps, entry, dominant_side, a) if dominant_side in ("LONG", "SHORT") else (None, "")
    stop, stop_name = _choose_stop(data, entry, dominant_side, a, settings) if dominant_side in ("LONG", "SHORT") else (None, "")
    entry, target, stop = round_tick(entry, settings.tick_size), round_tick(target, settings.tick_size), round_tick(stop, settings.tick_size)
    rr = abs(target - entry) / abs(entry - stop) if target is not None and stop is not None and entry != stop else 0.0
    if rr < settings.min_rr:
        failures.append(f"R:R {rr:.2f} < {settings.min_rr:.2f}")
    if target is not None and abs(target - entry) > settings.atr_target_cap * a:
        failures.append("target is too far for current volatility")
    point_value = settings.tick_value / max(settings.tick_size, 1e-9)
    risk = abs(entry - stop) * point_value * settings.contracts if stop is not None else None
    reward = abs(target - entry) * point_value * settings.contracts if target is not None else None
    if risk is not None and risk > settings.max_risk_dollars + 1:
        failures.append("planned dollar risk exceeds cap")

    signal = "WAIT" if failures or dominant_side == "WAIT" else dominant_side
    if failures:
        reasons.insert(0, "WAIT: " + "; ".join(failures))
    reasons.append(setup_note)
    diagnostics = {
        "bull_points": round(bull, 2), "bear_points": round(bear, 2),
        "separation": round(separation, 1), "atr": round(a, 4), "rsi": round(r, 1),
        "adx": round(ax, 1), "volume_ratio": round(volume_ratio, 2),
        "vwap": round(vwap, 4) if np.isfinite(vwap) else None,
        "structure": struct, "sweep": sweep, "displacement": move,
        "orderflow": flow,
    }
    return SignalResult(
        timeframe=timeframe, signal=signal, quality=round(quality, 1), entry=entry,
        stop=stop, take_profit=target, risk_reward=round(rr, 2),
        projected_risk=round(risk, 2) if risk is not None else None,
        projected_reward=round(reward, 2) if reward is not None else None,
        target_name=target_name, stop_name=stop_name, setup=setup,
        reasons=reasons[:10], warnings=warnings, diagnostics=diagnostics,
    )


def analyze_all_timeframes(
    df_1m: pd.DataFrame,
    settings: EngineSettings,
    orderflow: Optional[Dict[str, Any]] = None,
    feed_fresh: bool = True,
) -> Dict[str, SignalResult]:
    frames = {tf: resample_ohlcv(df_1m, tf) for tf in TF_RULES}
    output: Dict[str, SignalResult] = {}
    higher_bias = None
    for tf in reversed(list(TF_RULES.keys())):
        result = analyze_frame(frames[tf], tf, df_1m, settings, orderflow if tf in ("1m", "5m") else None, higher_bias, feed_fresh)
        output[tf] = result
        if result.signal in ("LONG", "SHORT"):
            higher_bias = result.signal
    return {tf: output[tf] for tf in TF_RULES}


def analyze_frames(
    frames: Dict[str, pd.DataFrame],
    df_1m: pd.DataFrame,
    settings: EngineSettings,
    orderflow: Optional[Dict[str, Any]] = None,
    feed_fresh: bool = True,
) -> Dict[str, SignalResult]:
    """Analyze prebuilt frames, useful when the provider supplies each interval."""
    output: Dict[str, SignalResult] = {}
    higher_bias = None
    for tf in reversed(list(TF_RULES.keys())):
        frame = frames.get(tf, pd.DataFrame())
        result = analyze_frame(frame, tf, df_1m, settings, orderflow if tf in ("1m", "5m") else None, higher_bias, feed_fresh)
        output[tf] = result
        if result.signal in ("LONG", "SHORT"):
            higher_bias = result.signal
    return {tf: output[tf] for tf in TF_RULES}


def fetch_orderflow_bridge(url: str, token: str = "", timeout: int = 4) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    if requests is None:
        raise RuntimeError("requests is not installed; run pip install -r requirements.txt")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("orderflow", payload)


class ProjectXClient:
    """Read-only ProjectX/TopstepX bar client. It never sends order requests."""

    def __init__(self, username: str, api_key: str, base_url: str = "https://api.topstepx.com/api"):
        self.username = username
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None

    def login(self) -> None:
        if requests is None:
            raise RuntimeError("requests is not installed; run pip install -r requirements.txt")
        response = requests.post(
            f"{self.base_url}/Auth/loginKey",
            json={"userName": self.username, "apiKey": self.api_key}, timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success") or not payload.get("token"):
            raise RuntimeError(payload.get("errorMessage") or "ProjectX login failed")
        self.token = payload["token"]

    @property
    def headers(self) -> Dict[str, str]:
        if not self.token:
            self.login()
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "accept": "text/plain"}

    def search_contract(self, text: str = "MNQ", live: bool = True) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/Contract/search", headers=self.headers,
            json={"searchText": text, "live": live}, timeout=12,
        )
        response.raise_for_status()
        contracts = response.json().get("contracts") or []
        active = [c for c in contracts if c.get("activeContract")]
        if not (active or contracts):
            raise RuntimeError(f"No ProjectX contract found for {text}")
        return (active or contracts)[0]

    def bars(self, contract_id: str, days: int = 12, live: bool = True, limit: int = 16000) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        response = requests.post(
            f"{self.base_url}/History/retrieveBars", headers=self.headers,
            json={
                "contractId": contract_id, "live": live,
                "startTime": start.isoformat().replace("+00:00", "Z"),
                "endTime": end.isoformat().replace("+00:00", "Z"),
                "unit": 2, "unitNumber": 1, "limit": min(limit, 20000),
                "includePartialBar": True,
            }, timeout=20,
        )
        response.raise_for_status()
        return normalize_ohlcv(pd.DataFrame(response.json().get("bars") or []))

    def bars_for_timeframe(self, contract_id: str, timeframe: str, live: bool = True) -> pd.DataFrame:
        units = {
            "1m": (2, 1, 12, 18000),
            "5m": (2, 5, 60, 18000),
            "30m": (2, 30, 180, 10000),
            "1h": (3, 1, 420, 10000),
            "4h": (3, 4, 1200, 9000),
            "1d": (4, 1, 1800, 2500),
        }
        unit, unit_number, days, limit = units[timeframe]
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        response = requests.post(
            f"{self.base_url}/History/retrieveBars", headers=self.headers,
            json={
                "contractId": contract_id, "live": live,
                "startTime": start.isoformat().replace("+00:00", "Z"),
                "endTime": end.isoformat().replace("+00:00", "Z"),
                "unit": unit, "unitNumber": unit_number, "limit": min(limit, 20000),
                "includePartialBar": True,
            }, timeout=20,
        )
        response.raise_for_status()
        return normalize_ohlcv(pd.DataFrame(response.json().get("bars") or []))


def feed_age_seconds(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.inf
    return float((pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds())
