"""OP.exe Quant — multi-timeframe market analysis dashboard.

Render start command:
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from op_engine import (
    ET,
    TF_RULES,
    EngineSettings,
    ProjectXClient,
    feed_age_seconds,
    fetch_orderflow_bridge,
    load_nq_csv,
    normalize_ohlcv,
    resample_ohlcv,
)
from quant_engine import QuantSettings, quant_analyze_all


st.set_page_config(page_title="OP.exe", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 1.1rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:10px 12px;}
    .op-card {border:1px solid rgba(128,128,128,.28); border-radius:14px; padding:14px 16px; margin:.35rem 0 1rem;}
    .long {border-left:6px solid #22c55e;} .short {border-left:6px solid #ef4444;} .wait {border-left:6px solid #94a3b8;}
    .op-small {opacity:.75; font-size:.86rem;} .op-title {font-weight:800; font-size:1.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return str(os.getenv(name, default))


FUTURES_YAHOO = {"MNQ":"MNQ=F","NQ":"NQ=F","MES":"MES=F","ES":"ES=F","M2K":"M2K=F","RTY":"RTY=F","MYM":"YM=F","YM":"YM=F","MGC":"MGC=F","GC":"GC=F","MCL":"MCL=F","CL":"CL=F"}


def market_status(symbol: str) -> tuple[bool, str]:
    clean = symbol.upper().strip().replace("!", "")
    if clean.endswith("-USD") or clean.endswith("-USDT"):
        return True, "Crypto trades continuously; venue maintenance can still occur."
    if clean not in FUTURES_YAHOO and not clean.endswith("=F"):
        now_et = pd.Timestamp.now(tz="UTC").tz_convert(ET)
        open_now = now_et.weekday() < 5 and (now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 30)) and now_et.hour < 16
        return open_now, "U.S. regular equity session appears open." if open_now else "U.S. regular equity session is closed."
    now = pd.Timestamp.now(tz="UTC").tz_convert("America/Chicago")
    weekday = now.weekday()
    minute = now.hour * 60 + now.minute
    if weekday == 5:
        return False, "CME equity-index futures are closed Saturday."
    if weekday == 6 and minute < 17 * 60:
        return False, "CME equity-index futures reopen Sunday evening."
    if weekday == 4 and minute >= 16 * 60:
        return False, "The weekly futures session has ended."
    if weekday in (0, 1, 2, 3, 4) and 16 * 60 <= minute < 17 * 60:
        return False, "Daily CME maintenance window."
    return True, "Futures session appears open; holiday schedules can differ."


@st.cache_data(ttl=25, show_spinner=False)
def projectx_frames(username: str, api_key: str, symbol: str) -> tuple[Dict[str, pd.DataFrame], str]:
    client = ProjectXClient(username, api_key)
    contract = client.search_contract(symbol, live=True)
    frames = {tf: client.bars_for_timeframe(contract["id"], tf, live=True) for tf in TF_RULES}
    return frames, str(contract.get("name") or contract.get("id") or symbol)


@st.cache_data(ttl=45, show_spinner=False)
def yahoo_frames(symbol: str = "MNQ=F") -> Dict[str, pd.DataFrame]:
    mapping = {
        "1m": ("7d", "1m"), "5m": ("60d", "5m"), "30m": ("60d", "30m"),
        "1h": ("2y", "1h"), "1d": ("5y", "1d"),
    }
    frames: Dict[str, pd.DataFrame] = {}
    for tf, (period, interval) in mapping.items():
        raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames[tf] = normalize_ohlcv(raw)
    frames["4h"] = resample_ohlcv(frames["1h"], "4h")
    return frames


def merge_history_live(history: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return live
    if live is None or live.empty:
        return history
    return pd.concat([history, live]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()


def render_card(result) -> None:
    css = result.signal.lower() if result.signal in ("LONG", "SHORT") else "wait"
    entry = f"{result.entry:,.2f}" if result.entry is not None else "—"
    stop = f"{result.stop:,.2f}" if result.stop is not None else "—"
    target = f"{result.take_profit:,.2f}" if result.take_profit is not None else "—"
    rr = f"{result.risk_reward:.2f}:1" if result.risk_reward is not None else "—"
    probability = f"{result.probability_target_first:.1%}" if result.probability_target_first is not None else "—"
    lower = f"{result.probability_lower_bound:.1%}" if result.probability_lower_bound is not None else "—"
    ev = f"{result.expected_value_r:+.2f}R" if result.expected_value_r is not None else "—"
    why = result.reasons[0] if result.reasons else "No qualified setup"
    st.markdown(
        f"""
        <div class="op-card {css}">
          <div class="op-title">{result.timeframe} · {result.signal} · Quality {result.quality:.0f}/100</div>
          <div><b>Entry:</b> {entry} &nbsp; <b>Stop:</b> {stop} &nbsp; <b>Take profit:</b> {target} &nbsp; <b>R:R:</b> {rr}</div>
          <div><b>Target-first estimate:</b> {probability} &nbsp; <b>Lower bound:</b> {lower} &nbsp; <b>Expected value:</b> {ev}</div>
          <div><b>Setup:</b> {result.setup} &nbsp; <b>Target:</b> {result.target_name or '—'}</div>
          <div class="op-small">{why}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("OP.exe · Multi-Timeframe Market Analyzer")
st.caption("Independent LONG / SHORT / WAIT decisions using completed candles, structure, DOLs, volatility, FVGs, liquidity, VWAP, order flow, historical analogs, and regime-conditioned Monte Carlo paths.")

with st.form("controls"):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.0, 1.1, 1.0, 1.2])
    symbol = c1.text_input("Asset / contract", value="MNQ", help="Examples: MNQ, MES, AAPL, SPY, BTC-USD")
    contracts = int(c2.number_input("Contracts", min_value=1, max_value=100, value=3, step=1))
    max_risk = float(c3.number_input("Max planned risk ($)", min_value=25.0, max_value=5000.0, value=300.0, step=25.0))
    min_rr = float(c4.number_input("Minimum R:R", min_value=1.0, max_value=5.0, value=1.5, step=0.1))
    min_quality = float(c5.number_input("Minimum quality", min_value=50.0, max_value=95.0, value=76.0, step=1.0))
    analyze = st.form_submit_button("Analyze asset", use_container_width=True)
    s1, s2 = st.columns(2)
    tick_size = float(s1.number_input("Minimum tick / price increment", min_value=0.00000001, value=0.25, format="%.8f"))
    tick_value = float(s2.number_input("Dollar value per tick per contract/share", min_value=0.00000001, value=0.50, format="%.8f"))

with st.expander("Data, order flow, and safety settings", expanded=False):
    d1, d2, d3 = st.columns(3)
    require_live = d1.toggle("Require a fresh live feed for LONG/SHORT", value=True)
    require_orderflow = d2.toggle("Require order-flow confirmation for 1m/5m", value=True)
    auto_refresh = d3.toggle("Refresh every 15 seconds", value=False)
    historical_upload = st.file_uploader("Optional NQ/MNQ 1-minute history CSV", type=["csv"], help="Used to extend DOL and higher-timeframe history. Live bars replace duplicate timestamps.")
    st.caption("ProjectX credentials and any bridge token belong in Render environment variables or Streamlit Secrets—not GitHub.")

    st.markdown("**Optional MarketBook / Bookmap / Databento bridge**")
    bridge_url = st.text_input("Order-flow JSON URL", value=secret("ORDERFLOW_BRIDGE_URL"), type="password")
    use_manual = st.toggle("Use manual order-flow snapshot instead", value=False)
    manual_snapshot: Optional[dict] = None
    if use_manual:
        o1, o2, o3, o4 = st.columns(4)
        bid_depth = o1.number_input("Bid depth", min_value=0.0, value=0.0)
        ask_depth = o2.number_input("Ask depth", min_value=0.0, value=0.0)
        delta = o3.number_input("1-minute delta", value=0.0)
        aggression = o4.number_input("Aggressive buy minus sell", value=0.0)
        manual_snapshot = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "bid_depth": bid_depth, "ask_depth": ask_depth, "delta_1m": delta,
            "aggressive_buy": max(aggression, 0), "aggressive_sell": max(-aggression, 0),
        }

with st.expander("Monte Carlo and probability settings", expanded=False):
    q1, q2, q3, q4 = st.columns(4)
    simulations = int(q1.number_input("Bootstrap paths", min_value=200, max_value=10000, value=1200, step=200))
    min_probability = float(q2.number_input("Minimum target-first probability", min_value=0.50, max_value=0.90, value=0.56, step=0.01, format="%.2f"))
    min_lower_bound = float(q3.number_input("Minimum probability lower bound", min_value=0.40, max_value=0.80, value=0.50, step=0.01, format="%.2f"))
    min_ev = float(q4.number_input("Minimum expected value (R)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f"))

if auto_refresh:
    st_autorefresh(interval=15000, key="op-refresh")

settings = EngineSettings(
    contracts=contracts, max_risk_dollars=max_risk, min_rr=min_rr,
    min_quality=min_quality, require_orderflow=require_orderflow,
    tick_size=tick_size, tick_value=tick_value,
)
quant_settings = QuantSettings(
    simulations=simulations,
    min_probability=min_probability,
    min_probability_lower_bound=min_lower_bound,
    min_expected_value_r=min_ev,
)

username = secret("TOPSTEP_USERNAME")
api_key = secret("TOPSTEP_API_KEY")
frames: Dict[str, pd.DataFrame] = {}
data_source = ""
data_error = ""

with st.spinner("Loading market data and evaluating OP.exe..."):
    if username and api_key:
        try:
            frames, resolved = projectx_frames(username, api_key, symbol)
            data_source = f"ProjectX / TopstepX live bars · {resolved}"
        except Exception as exc:
            data_error = f"ProjectX error: {exc}"
    if not frames:
        try:
            yahoo_symbol = FUTURES_YAHOO.get(symbol.upper().strip(), symbol.upper().strip())
            frames = yahoo_frames(yahoo_symbol)
            data_source = f"Yahoo {yahoo_symbol} fallback (not exchange-grade live depth)"
        except Exception as exc:
            data_error += f" Yahoo error: {exc}"

history = pd.DataFrame()
if historical_upload is not None:
    try:
        history = load_nq_csv(historical_upload)
    except Exception as exc:
        st.error(f"Could not read historical CSV: {exc}")

if frames:
    base_1m = frames.get("1m", pd.DataFrame())
    combined_1m = merge_history_live(history, base_1m)
    if not history.empty:
        # Rebuild all frames from the continuous merged series, ensuring every timeframe uses the same history.
        frames = {tf: resample_ohlcv(combined_1m, tf) for tf in TF_RULES}
    else:
        frames["1m"] = combined_1m
else:
    combined_1m = history
    if not history.empty:
        frames = {tf: resample_ohlcv(history, tf) for tf in TF_RULES}
        data_source = "Uploaded historical CSV only"

orderflow = manual_snapshot
if not use_manual and bridge_url:
    try:
        orderflow = fetch_orderflow_bridge(bridge_url, secret("ORDERFLOW_BRIDGE_TOKEN"))
    except Exception as exc:
        data_error += f" Order-flow bridge error: {exc}"

market_open, market_note = market_status(symbol)
age = feed_age_seconds(base_1m) if 'base_1m' in locals() else np.inf
projectx_live = data_source.startswith("ProjectX")
fresh = bool(projectx_live and np.isfinite(age) and age <= settings.stale_seconds and market_open)
allow_signal = fresh if require_live else bool(np.isfinite(age) and age <= settings.hard_stale_seconds and market_open)

if not frames or frames.get("1m", pd.DataFrame()).empty:
    st.error("No usable market data was returned. Configure ProjectX credentials, enter a supported Yahoo symbol, or upload the NQ 1-minute CSV.")
    st.stop()

results = quant_analyze_all(
    frames, combined_1m, settings, quant_settings,
    orderflow=orderflow, feed_fresh=allow_signal,
)

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Current price", f"{combined_1m['Close'].iloc[-1]:,.2f}")
top2.metric("Data", "LIVE" if fresh else "DELAYED / HISTORICAL")
top3.metric("Market", "OPEN" if market_open else "CLOSED")
top4.metric("1m signal", results["1m"].signal)
top5.metric("5m signal", results["5m"].signal)

if not market_open:
    st.warning(f"MARKET CLOSED — {market_note} OP.exe forces WAIT for actionable signals.")
elif require_live and not fresh:
    st.warning("NO VERIFIED FRESH LIVE FEED — analysis is displayed, but OP.exe forces WAIT. Configure ProjectX credentials for actionable output.")
elif data_source.startswith("Yahoo"):
    st.info("Yahoo is a fallback and can be delayed. Confirm every level against TopstepX before trading.")
if data_error:
    st.caption(data_error)
st.caption(f"Source: {data_source} · newest provider bar age: {age:.0f}s · {market_note}")

for tf in TF_RULES:
    render_card(results[tf])

st.subheader("One-position trade lock")
if "active_quant_trade" not in st.session_state:
    st.session_state.active_quant_trade = None
active = st.session_state.active_quant_trade
current_price = float(combined_1m["Close"].iloc[-1])
if active:
    hit_stop = current_price <= active["stop"] if active["signal"] == "LONG" else current_price >= active["stop"]
    hit_target = current_price >= active["target"] if active["signal"] == "LONG" else current_price <= active["target"]
    if hit_stop or hit_target:
        active["status"] = "TARGET HIT" if hit_target else "STOP HIT"
        st.session_state.active_quant_trade = None
        st.success(f"Tracked trade closed: {active['status']} at current price {current_price:,.2f}.")
    else:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Locked position", f"{active['timeframe']} {active['signal']}")
        t2.metric("Entry", f"{active['entry']:,.2f}")
        t3.metric("Stop", f"{active['stop']:,.2f}")
        t4.metric("Target", f"{active['target']:,.2f}")
        st.warning("New trades are locked until this position closes. OP.exe continues calculating signals for information only.")
        if st.button("Close tracked position manually"):
            st.session_state.active_quant_trade = None
            st.rerun()
else:
    qualified = [tf for tf, result in results.items() if result.signal in ("LONG", "SHORT")]
    if qualified:
        selected_trade_tf = st.selectbox("Qualified timeframe to track", qualified)
        selected_trade = results[selected_trade_tf]
        if st.button("Lock and track this trade", use_container_width=True):
            st.session_state.active_quant_trade = {
                "timeframe": selected_trade_tf, "signal": selected_trade.signal,
                "entry": selected_trade.entry, "stop": selected_trade.stop,
                "target": selected_trade.take_profit,
                "started": pd.Timestamp.now(tz="UTC").isoformat(), "status": "OPEN",
            }
            st.rerun()
    else:
        st.caption("No timeframe currently passes every structural, data, probability, and expected-value gate.")

table = []
for tf, result in results.items():
    table.append({
        "Timeframe": tf, "Signal": result.signal, "Quality": result.quality,
        "Entry": result.entry, "Stop": result.stop, "Take profit": result.take_profit,
        "R:R": result.risk_reward, "Risk $": result.projected_risk,
        "Reward $": result.projected_reward,
        "Target-first %": round(100 * result.probability_target_first, 1) if result.probability_target_first is not None else None,
        "Lower bound %": round(100 * result.probability_lower_bound, 1) if result.probability_lower_bound is not None else None,
        "Expected R": result.expected_value_r, "Setup": result.setup,
    })

st.subheader("All timeframe plans")
st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

with st.expander("Evidence and diagnostics", expanded=False):
    selected_tf = st.selectbox("Timeframe details", list(TF_RULES.keys()))
    selected = results[selected_tf]
    for reason in selected.reasons:
        st.write("•", reason)
    if selected.warnings:
        for warning in selected.warnings:
            st.warning(warning)
    st.json(selected.diagnostics)

with st.expander("Required secrets and order-flow JSON format", expanded=False):
    st.code(
        'TOPSTEP_USERNAME="your TopstepX username"\n'
        'TOPSTEP_API_KEY="your ProjectX API key"\n'
        'ORDERFLOW_BRIDGE_URL="https://your-private-bridge/snapshot"\n'
        'ORDERFLOW_BRIDGE_TOKEN="private token"',
        language="bash",
    )
    st.code(
        '{"timestamp":"2026-09-20T14:30:00Z","bid_depth":1280,"ask_depth":910,'
        '"delta_1m":325,"aggressive_buy":620,"aggressive_sell":390,'
        '"absorption_bid":true,"absorption_ask":false}',
        language="json",
    )

st.divider()
st.caption(
    "OP.exe is a decision-support calculator, not an execution bot or profit guarantee. "
    "The target-first percentage is a bootstrap/analog estimate, not certainty. Stops can slip, "
    "market regimes change, and historical profitability cannot guarantee future profitability."
)
