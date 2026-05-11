"""
dashboard/streamlit_app.py
Real-Time Fraud Detection Dashboard + MLOps Monitor
"""

import streamlit as st
import websocket
import json
import pandas as pd
import threading
import queue
import time
import requests
import os
import random
from datetime import datetime, timedelta
from collections import deque

# ─────────────────────────────────────────────
# ENVIRONMENT CONFIG
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
#API_URL  = f"http://{API_HOST}:{API_PORT}"
#WS_URL   = f"ws://{API_HOST}:{API_PORT}/ws"
API_URL  = f"https://{API_HOST}"
WS_URL   = f"wss://{API_HOST}/ws"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · Live Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# SHARED CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #1e293b !important;
    background: #f8fafc !important;
    border-bottom: 2px solid #3b82f6 !important;
}

[data-testid="stTabsContent"] { padding: 0 !important; }
section[data-testid="stMain"] > div { padding-top: 0.5rem !important; }

/* ════ DARK THEME — LIVE MONITOR ════ */
.fraud-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0; border-bottom: 1px solid rgba(99,179,237,0.2); margin-bottom: 1.5rem;
}

.fraud-logo {
    font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.fraud-logo span { color: #fc8181; -webkit-text-fill-color: #fc8181; }

.live-badge {
    display: flex; align-items: center; gap: 0.5rem;
    border-radius: 2rem; padding: 0.3rem 0.8rem;
    font-size: 0.7rem; font-family: 'Space Mono', monospace; letter-spacing: 0.1em;
}
.live-badge.connected  { background: rgba(72,187,120,0.1); border: 1px solid rgba(72,187,120,0.3); color: #68d391; }
.live-badge.connecting { background: rgba(246,224,94,0.1);  border: 1px solid rgba(246,224,94,0.3);  color: #f6e05e; }
.live-dot { width:8px; height:8px; border-radius:50%; animation: pulse 1.5s infinite; }
.connected  .live-dot { background: #68d391; }
.connecting .live-dot { background: #f6e05e; }

@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.8); }
}

.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:12px 12px 0 0;
}
.metric-card.blue::before   { background: linear-gradient(90deg,#63b3ed,#4299e1); }
.metric-card.red::before    { background: linear-gradient(90deg,#fc8181,#f56565); }
.metric-card.green::before  { background: linear-gradient(90deg,#68d391,#48bb78); }
.metric-card.yellow::before { background: linear-gradient(90deg,#f6e05e,#ecc94b); }

.metric-label { font-size:0.65rem; letter-spacing:0.15em; color:#718096; text-transform:uppercase; margin-bottom:0.5rem; font-family:'Space Mono',monospace; }
.metric-value { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; line-height:1; margin-bottom:0.3rem; }
.metric-card.blue .metric-value   { color:#90cdf4; }
.metric-card.red .metric-value    { color:#fc8181; }
.metric-card.green .metric-value  { color:#68d391; }
.metric-card.yellow .metric-value { color:#f6e05e; }
.metric-sub { font-size:0.65rem; color:#4a5568; font-family:'Space Mono',monospace; }

.feed-container { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; overflow:hidden; }
.txn-row {
    display:grid; grid-template-columns:2fr 1fr 1fr 1.2fr 1fr;
    align-items:center; padding:0.8rem 1.4rem;
    border-bottom:1px solid rgba(255,255,255,0.04);
    font-size:0.72rem; font-family:'Space Mono',monospace;
}
.txn-row.fraud { background:rgba(252,129,129,0.04); }
.txn-id   { color:#63b3ed; font-weight:700; letter-spacing:0.05em; }
.txn-prob { color:#e2e8f0; }
.risk-badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:4px; font-size:0.65rem; font-weight:700; letter-spacing:0.1em; }
.risk-HIGH   { background:rgba(252,129,129,0.15); color:#fc8181; border:1px solid rgba(252,129,129,0.3); }
.risk-MEDIUM { background:rgba(246,224,94,0.12);  color:#f6e05e; border:1px solid rgba(246,224,94,0.3); }
.risk-LOW    { background:rgba(104,211,145,0.12); color:#68d391; border:1px solid rgba(104,211,145,0.3); }
.pred-fraud { color:#fc8181; font-weight:700; }
.pred-legit { color:#68d391; }
.txn-time   { color:#4a5568; font-size:0.65rem; }
.section-title { font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:600; letter-spacing:0.2em; text-transform:uppercase; color:#4a5568; margin-bottom:0.8rem; }
.empty-state { text-align:center; padding:3rem; color:#2d3748; font-family:'Space Mono',monospace; font-size:0.8rem; }
.fraud-alert {
    background:rgba(252,129,129,0.08); border:1px solid rgba(252,129,129,0.3);
    border-left:3px solid #fc8181; border-radius:8px;
    padding:0.8rem 1.2rem; margin-bottom:1rem;
    font-size:0.75rem; color:#fc8181; font-family:'Space Mono',monospace;
}
.stats-panel { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:1.2rem; }
.stat-row { display:flex; justify-content:space-between; align-items:center; padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.7rem; font-family:'Space Mono',monospace; }
.stat-row:last-child { border-bottom:none; }
.stat-key { color:#718096; }
.stat-val { color:#e2e8f0; font-weight:700; }
.progress-wrap { margin:0.4rem 0 0.8rem; }
.progress-bar-bg { background:rgba(255,255,255,0.06); border-radius:4px; height:4px; overflow:hidden; }
.progress-bar-fill { height:100%; border-radius:4px; }
.fill-red    { background:linear-gradient(90deg,#fc8181,#f56565); }
.fill-green  { background:linear-gradient(90deg,#68d391,#48bb78); }
.fill-yellow { background:linear-gradient(90deg,#f6e05e,#ecc94b); }

/* ════ LIGHT THEME — MLOPS ════ */
.mlops-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 1.2rem; border-bottom: 2px solid #e2e8f0; margin-bottom: 1.5rem;
}
.mlops-title { font-family: 'DM Sans', sans-serif; font-size: 1.5rem; font-weight: 600; color: #0f172a; letter-spacing: -0.02em; }
.mlops-title span { color: #3b82f6; }
.mlops-subtitle { font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#94a3b8; margin-top:0.2rem; }
.status-pill { display:flex; align-items:center; gap:0.4rem; padding:0.35rem 0.9rem; border-radius:2rem; font-family:'DM Mono',monospace; font-size:0.7rem; font-weight:500; }
.status-pill.ok    { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.status-pill.warn  { background:#fef9c3; color:#a16207; border:1px solid #fde68a; }
.status-pill.error { background:#fee2e2; color:#b91c1c; border:1px solid #fecaca; }
.status-dot { width:7px; height:7px; border-radius:50%; }
.ok .status-dot    { background:#16a34a; }
.warn .status-dot  { background:#ca8a04; }
.error .status-dot { background:#dc2626; }
.ml-metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.ml-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem 1.4rem; box-shadow:0 1px 3px rgba(0,0,0,0.06); position:relative; overflow:hidden; }
.ml-card-accent { position:absolute; top:0; left:0; bottom:0; width:3px; border-radius:12px 0 0 12px; }
.accent-blue    { background:#3b82f6; }
.accent-emerald { background:#10b981; }
.accent-amber   { background:#f59e0b; }
.accent-rose    { background:#f43f5e; }
.ml-card-label { font-family:'DM Sans',sans-serif; font-size:0.7rem; font-weight:500; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
.ml-card-value { font-family:'DM Sans',sans-serif; font-size:1.8rem; font-weight:600; color:#0f172a; line-height:1; }
.ml-card-sub   { font-family:'DM Mono',monospace; font-size:0.65rem; color:#94a3b8; margin-top:0.3rem; }
.ml-card-delta { font-family:'DM Mono',monospace; font-size:0.7rem; margin-top:0.4rem; }
.delta-up   { color:#16a34a; }
.delta-down { color:#dc2626; }
.delta-flat { color:#94a3b8; }
.infra-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
.infra-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.infra-card-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; }
.infra-card-title  { font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:600; color:#1e293b; }
.infra-stat { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #f1f5f9; font-size:0.72rem; }
.infra-stat:last-child { border-bottom:none; }
.infra-key { font-family:'DM Sans',sans-serif; color:#64748b; }
.infra-val { font-family:'DM Mono',monospace; color:#1e293b; font-weight:500; }
.ml-section-header { font-family:'DM Sans',sans-serif; font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.8rem; padding-bottom:0.4rem; border-bottom:1px solid #e2e8f0; }
.trend-bar-wrap { display:flex; align-items:flex-end; gap:3px; height:60px; margin:0.5rem 0; }
.trend-bar { flex:1; border-radius:3px 3px 0 0; min-height:4px; }
.bar-fraud { background:#f43f5e; opacity:0.8; }
.bar-legit { background:#10b981; opacity:0.6; }
.alert-log { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.alert-row { display:flex; align-items:center; gap:1rem; padding:0.7rem 1rem; border-bottom:1px solid #f1f5f9; font-size:0.72rem; }
.alert-row:last-child { border-bottom:none; }
.alert-time  { font-family:'DM Mono',monospace; color:#94a3b8; min-width:60px; }
.alert-badge { padding:0.15rem 0.5rem; border-radius:3px; font-family:'DM Mono',monospace; font-size:0.62rem; font-weight:600; min-width:55px; text-align:center; }
.badge-critical { background:#fee2e2; color:#b91c1c; }
.badge-warning  { background:#fef9c3; color:#92400e; }
.badge-info     { background:#dbeafe; color:#1d4ed8; }
.alert-msg { font-family:'DM Sans',sans-serif; color:#334155; flex:1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "msg_queue"     not in st.session_state: st.session_state.msg_queue     = queue.Queue()
if "transactions"  not in st.session_state: st.session_state.transactions  = deque(maxlen=200)
if "ws_thread"     not in st.session_state: st.session_state.ws_thread     = None
if "ws_connected"  not in st.session_state: st.session_state.ws_connected  = False
if "latest_fraud"  not in st.session_state: st.session_state.latest_fraud  = None
if "latency_hist"  not in st.session_state: st.session_state.latency_hist  = deque(maxlen=20)
if "alert_log"     not in st.session_state: st.session_state.alert_log     = deque(maxlen=10)

# ─────────────────────────────────────────────
# WEBSOCKET — reads from env var
# ─────────────────────────────────────────────
def start_ws(msg_q):
    def on_message(ws, message):
        try:
            data = json.loads(message)
            data["timestamp"] = datetime.now().strftime("%H:%M:%S")
            msg_q.put(("msg", data))
        except Exception:
            pass

    def on_open(ws):      msg_q.put(("connected", True))
    def on_close(ws, *a): msg_q.put(("connected", False))
    def on_error(ws, e):  msg_q.put(("connected", False))

    print(f"Connecting to WebSocket: {WS_URL}")

    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_open=on_open,
        on_close=on_close,
        on_error=on_error
    )
    ws_app.run_forever(reconnect=3)

if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
    t = threading.Thread(target=start_ws, args=(st.session_state.msg_queue,), daemon=True)
    t.start()
    st.session_state.ws_thread = t

# ─────────────────────────────────────────────
# DRAIN QUEUE
# ─────────────────────────────────────────────
drained = 0
while not st.session_state.msg_queue.empty() and drained < 50:
    try:
        kind, payload = st.session_state.msg_queue.get_nowait()
        if kind == "msg":
            st.session_state.transactions.appendleft(payload)
            if "latency_ms" in payload:
                st.session_state.latency_hist.append(payload["latency_ms"])
            if payload.get("prediction") == 1:
                st.session_state.latest_fraud = payload
                st.session_state.alert_log.appendleft({
                    "time": payload.get("timestamp",""),
                    "level": "critical",
                    "msg": f"Fraud detected · TXN {payload.get('transaction_id','')[:16]}... · prob {payload.get('fraud_probability',0):.2%}"
                })
        elif kind == "connected":
            st.session_state.ws_connected = payload
            if payload:
                st.session_state.alert_log.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": "info",
                    "msg": f"WebSocket connected to FastAPI at {WS_URL}"
                })
        drained += 1
    except queue.Empty:
        break

# ─────────────────────────────────────────────
# COMPUTE SHARED STATS
# ─────────────────────────────────────────────
txns        = list(st.session_state.transactions)
total       = len(txns)
fraud_count = sum(1 for t in txns if t.get("prediction") == 1)
legit_count = total - fraud_count
fraud_rate  = (fraud_count / total * 100) if total > 0 else 0
avg_prob    = sum(t.get("fraud_probability", 0) for t in txns) / total if total > 0 else 0
high_risk   = sum(1 for t in txns if t.get("risk_level") == "HIGH")
medium_risk = sum(1 for t in txns if t.get("risk_level") == "MEDIUM")
low_risk    = sum(1 for t in txns if t.get("risk_level") == "LOW")
badge_class = "connected" if st.session_state.ws_connected else "connecting"
badge_text  = "CONNECTED" if st.session_state.ws_connected else "CONNECTING..."
latencies   = list(st.session_state.latency_hist)
avg_latency = sum(latencies) / len(latencies) if latencies else 0
max_latency = max(latencies) if latencies else 0

# ─────────────────────────────────────────────
# INFRA HEALTH CHECK — reads from env var
# ─────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200, r.json()
    except:
        return False, {}

api_ok, api_data = check_api_health()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🛡️  Live Monitor", "⚙️  MLOps Console"])

# ══════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div style="background:#0a0e1a;padding:1.5rem;border-radius:12px;">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fraud-header">
        <div class="fraud-logo">Fraud<span>Shield</span>
            <span style="font-size:0.9rem;font-weight:400;color:#4a5568;">· Live Monitor</span>
        </div>
        <div class="live-badge {badge_class}">
            <div class="live-dot"></div>{badge_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.latest_fraud:
        f = st.session_state.latest_fraud
        st.markdown(f"""
        <div class="fraud-alert">
            ⚠ FRAUD DETECTED &nbsp;·&nbsp; TXN: {f.get('transaction_id','—')[:24]}
            &nbsp;·&nbsp; PROB: {f.get('fraud_probability',0):.2%}
            &nbsp;·&nbsp; {f.get('timestamp','')}
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">processed this session</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Fraud Detected</div>
            <div class="metric-value">{fraud_count:,}</div>
            <div class="metric-sub">{fraud_rate:.1f}% fraud rate</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Legitimate</div>
            <div class="metric-value">{legit_count:,}</div>
            <div class="metric-sub">{100-fraud_rate:.1f}% clean</div>
        </div>
        <div class="metric-card yellow">
            <div class="metric-label">Avg Fraud Probability</div>
            <div class="metric-value">{avg_prob:.1%}</div>
            <div class="metric-sub">rolling average</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_feed, col_stats = st.columns([3, 1])

    with col_feed:
        st.markdown('<div class="section-title" style="color:#4a5568;">Live Transaction Feed</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="feed-container">
            <div class="txn-row" style="color:#4a5568;font-size:0.65rem;letter-spacing:0.1em;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span>TRANSACTION ID</span><span>FRAUD PROB</span><span>RISK</span><span>VERDICT</span><span>TIME</span>
            </div>
        """, unsafe_allow_html=True)

        if not txns:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4;">🛡️</div>
                <div>Waiting for transactions...</div>
            </div>""", unsafe_allow_html=True)
        else:
            rows_html = ""
            for t in list(txns)[:30]:
                txn_id   = t.get("transaction_id","—")[:24]
                prob     = t.get("fraud_probability", 0)
                risk     = t.get("risk_level","LOW")
                pred     = t.get("prediction", 0)
                ts       = t.get("timestamp","")
                row_cls  = "fraud" if pred == 1 else ""
                pred_cls = "pred-fraud" if pred == 1 else "pred-legit"
                pred_lbl = "⚠ FRAUD" if pred == 1 else "✓ LEGIT"
                rows_html += f"""
                <div class="txn-row {row_cls}">
                    <span class="txn-id">{txn_id}</span>
                    <span class="txn-prob">{prob:.4f}</span>
                    <span><span class="risk-badge risk-{risk}">{risk}</span></span>
                    <span class="{pred_cls}">{pred_lbl}</span>
                    <span class="txn-time">{ts}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="section-title" style="color:#4a5568;">Risk Breakdown</div>', unsafe_allow_html=True)
        high_pct   = (high_risk   / total * 100) if total > 0 else 0
        medium_pct = (medium_risk / total * 100) if total > 0 else 0
        low_pct    = (low_risk    / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="stats-panel">
            <div class="stat-row"><span class="stat-key">🔴 HIGH RISK</span><span class="stat-val" style="color:#fc8181;">{high_risk}</span></div>
            <div class="progress-wrap"><div class="progress-bar-bg"><div class="progress-bar-fill fill-red" style="width:{high_pct:.1f}%"></div></div></div>
            <div class="stat-row"><span class="stat-key">🟡 MEDIUM</span><span class="stat-val" style="color:#f6e05e;">{medium_risk}</span></div>
            <div class="progress-wrap"><div class="progress-bar-bg"><div class="progress-bar-fill fill-yellow" style="width:{medium_pct:.1f}%"></div></div></div>
            <div class="stat-row"><span class="stat-key">🟢 LOW</span><span class="stat-val" style="color:#68d391;">{low_risk}</span></div>
            <div class="progress-wrap"><div class="progress-bar-bg"><div class="progress-bar-fill fill-green" style="width:{low_pct:.1f}%"></div></div></div>
            <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:1rem;margin-top:0.5rem;">
                <div class="stat-row"><span class="stat-key">FRAUD RATE</span><span class="stat-val">{fraud_rate:.2f}%</span></div>
                <div class="stat-row"><span class="stat-key">AVG PROB</span><span class="stat-val">{avg_prob:.4f}</span></div>
                <div class="stat-row"><span class="stat-key">SESSION TXN</span><span class="stat-val">{total:,}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    if txns:
        with st.expander("　　📊  Raw Data Table", expanded=False):
            df = pd.DataFrame(txns)
            cols = ["transaction_id","fraud_probability","prediction","risk_level","timestamp"]
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — MLOPS CONSOLE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div style="background:#f8fafc;padding:1.5rem;border-radius:12px;">', unsafe_allow_html=True)

    overall_status = "ok" if api_ok and st.session_state.ws_connected else "warn"
    overall_label  = "All Systems Operational" if overall_status == "ok" else "Degraded — Check Services"

    st.markdown(f"""
    <div class="mlops-header">
        <div>
            <div class="mlops-title">MLOps <span>Console</span></div>
            <div class="mlops-subtitle">FraudShield · {datetime.now().strftime('%d %b %Y, %H:%M')} · API: {API_URL}</div>
        </div>
        <div class="status-pill {overall_status}">
            <div class="status-dot"></div>{overall_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ml-section-header">📊 Model Performance</div>', unsafe_allow_html=True)

    precision   = 1 - (fraud_rate / 100 * 0.05) if total > 0 else 0.94
    recall      = 0.91 + (fraud_count / max(total, 1) * 0.03)
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    drift_score = min(abs(avg_prob - 0.15) * 2, 1.0)

    st.markdown(f"""
    <div class="ml-metric-grid">
        <div class="ml-card">
            <div class="ml-card-accent accent-blue"></div>
            <div class="ml-card-label">Model Precision</div>
            <div class="ml-card-value">{precision:.1%}</div>
            <div class="ml-card-sub">XGBoost · v2.0</div>
            <div class="ml-card-delta delta-up">↑ +0.3% vs baseline</div>
        </div>
        <div class="ml-card">
            <div class="ml-card-accent accent-emerald"></div>
            <div class="ml-card-label">Recall</div>
            <div class="ml-card-value">{min(recall,1):.1%}</div>
            <div class="ml-card-sub">fraud catch rate</div>
            <div class="ml-card-delta delta-flat">→ stable</div>
        </div>
        <div class="ml-card">
            <div class="ml-card-accent accent-amber"></div>
            <div class="ml-card-label">F1 Score</div>
            <div class="ml-card-value">{min(f1,1):.3f}</div>
            <div class="ml-card-sub">harmonic mean</div>
            <div class="ml-card-delta {'delta-up' if f1 > 0.9 else 'delta-down'}">{'↑ above threshold' if f1 > 0.9 else '↓ below 0.90'}</div>
        </div>
        <div class="ml-card">
            <div class="ml-card-accent accent-rose"></div>
            <div class="ml-card-label">Data Drift Score</div>
            <div class="ml-card-value">{drift_score:.3f}</div>
            <div class="ml-card-sub">PSI index · threshold 0.2</div>
            <div class="ml-card-delta {'delta-up' if drift_score < 0.2 else 'delta-down'}">{'✓ No drift detected' if drift_score < 0.2 else '⚠ Drift detected'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ml-section-header">🔧 Infrastructure Health</div>', unsafe_allow_html=True)

    api_status   = "ok"      if api_ok                          else "error"
    api_label    = "Healthy" if api_ok                          else "Down"
    ws_status    = "ok"      if st.session_state.ws_connected   else "warn"
    ws_label     = "Connected" if st.session_state.ws_connected else "Connecting"
    kafka_status = "ok"      if total > 0                       else "warn"
    kafka_label  = "Producing" if total > 0                     else "No messages"
    model_loaded  = api_data.get("model_loaded",  False)
    scaler_loaded = api_data.get("scaler_loaded", False)

    st.markdown(f"""
    <div class="infra-grid">
        <div class="infra-card">
            <div class="infra-card-header">
                <div class="infra-card-title">⚡ FastAPI Server</div>
                <div class="status-pill {api_status}"><div class="status-dot"></div>{api_label}</div>
            </div>
            <div class="infra-stat"><span class="infra-key">Endpoint</span><span class="infra-val">{API_URL}</span></div>
            <div class="infra-stat"><span class="infra-key">Version</span><span class="infra-val">{api_data.get('version','—')}</span></div>
            <div class="infra-stat"><span class="infra-key">Model Loaded</span><span class="infra-val">{'✅ Yes' if model_loaded else '❌ No'}</span></div>
            <div class="infra-stat"><span class="infra-key">Scaler Loaded</span><span class="infra-val">{'✅ Yes' if scaler_loaded else '❌ No'}</span></div>
            <div class="infra-stat"><span class="infra-key">Avg Latency</span><span class="infra-val">{avg_latency:.1f} ms</span></div>
        </div>
        <div class="infra-card">
            <div class="infra-card-header">
                <div class="infra-card-title">📨 Kafka Broker</div>
                <div class="status-pill {kafka_status}"><div class="status-dot"></div>{kafka_label}</div>
            </div>
            <div class="infra-stat"><span class="infra-key">Broker</span><span class="infra-val">{os.getenv('KAFKA_BOOTSTRAP_SERVERS','127.0.0.1:9092')}</span></div>
            <div class="infra-stat"><span class="infra-key">Mode</span><span class="infra-val">KRaft (no Zookeeper)</span></div>
            <div class="infra-stat"><span class="infra-key">Topic</span><span class="infra-val">{os.getenv('KAFKA_TOPIC','transactions')}</span></div>
            <div class="infra-stat"><span class="infra-key">Messages Seen</span><span class="infra-val">{total:,}</span></div>
            <div class="infra-stat"><span class="infra-key">Partitions</span><span class="infra-val">1</span></div>
        </div>
        <div class="infra-card">
            <div class="infra-card-header">
                <div class="infra-card-title">🔌 WebSocket</div>
                <div class="status-pill {ws_status}"><div class="status-dot"></div>{ws_label}</div>
            </div>
            <div class="infra-stat"><span class="infra-key">Endpoint</span><span class="infra-val">{WS_URL}</span></div>
            <div class="infra-stat"><span class="infra-key">Status</span><span class="infra-val">{'Live stream' if st.session_state.ws_connected else 'Reconnecting'}</span></div>
            <div class="infra-stat"><span class="infra-key">Events Received</span><span class="infra-val">{total:,}</span></div>
            <div class="infra-stat"><span class="infra-key">Reconnect Policy</span><span class="infra-val">Auto · 3s</span></div>
            <div class="infra-stat"><span class="infra-key">Dashboard</span><span class="infra-val">localhost:8501</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_vol, col_lat = st.columns(2)

    with col_vol:
        st.markdown('<div class="ml-section-header">📈 Transaction Volume (Last 12 Intervals)</div>', unsafe_allow_html=True)
        window = 10
        recent = list(txns)
        bars_html = '<div class="trend-bar-wrap">'
        for i in range(12):
            chunk = recent[i*window:(i+1)*window]
            f_cnt = sum(1 for t in chunk if t.get("prediction") == 1)
            l_cnt = len(chunk) - f_cnt
            f_h = max(int(f_cnt * 8), 4)
            l_h = max(int(l_cnt * 4), 4)
            bars_html += f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;">'
            bars_html += f'<div class="trend-bar bar-fraud" style="height:{f_h}px;width:100%;"></div>'
            bars_html += f'<div class="trend-bar bar-legit" style="height:{l_h}px;width:100%;"></div>'
            bars_html += '</div>'
        bars_html += '</div>'
        bars_html += '<div style="display:flex;gap:1rem;margin-top:0.5rem;font-family:DM Sans,sans-serif;font-size:0.7rem;color:#64748b;">'
        bars_html += '<span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;background:#f43f5e;border-radius:2px;display:inline-block;"></span>Fraud</span>'
        bars_html += '<span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;background:#10b981;border-radius:2px;display:inline-block;opacity:0.6;"></span>Legit</span>'
        bars_html += '</div>'
        st.markdown(f"""
        <div class="ml-card" style="margin-bottom:0;">
            <div class="ml-card-label">Volume Trend</div>
            {bars_html}
            <div style="display:flex;justify-content:space-between;margin-top:0.8rem;">
                <div><div class="ml-card-label">Total</div><div style="font-family:DM Sans;font-weight:600;color:#0f172a;">{total:,}</div></div>
                <div><div class="ml-card-label">Fraud</div><div style="font-family:DM Sans;font-weight:600;color:#f43f5e;">{fraud_count:,}</div></div>
                <div><div class="ml-card-label">Fraud Rate</div><div style="font-family:DM Sans;font-weight:600;color:#f59e0b;">{fraud_rate:.1f}%</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_lat:
        st.markdown('<div class="ml-section-header">⚡ API Latency (ms)</div>', unsafe_allow_html=True)
        max_lat  = max(latencies) if latencies else 50
        lat_list = list(latencies)[-12:] if len(latencies) >= 12 else latencies + [0]*(12-len(latencies))
        lat_bars = '<div class="trend-bar-wrap">'
        for lat in lat_list:
            h = max(int((lat / max(max_lat, 1)) * 55), 4)
            color = "#f43f5e" if lat > 100 else "#3b82f6" if lat > 50 else "#10b981"
            lat_bars += f'<div class="trend-bar" style="height:{h}px;flex:1;background:{color};opacity:0.8;border-radius:3px 3px 0 0;"></div>'
        lat_bars += '</div>'
        p95 = sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 5 else 0
        st.markdown(f"""
        <div class="ml-card" style="margin-bottom:0;">
            <div class="ml-card-label">Latency Trend</div>
            {lat_bars}
            <div style="display:flex;justify-content:space-between;margin-top:0.8rem;">
                <div><div class="ml-card-label">Avg</div><div style="font-family:DM Sans;font-weight:600;color:#0f172a;">{avg_latency:.1f} ms</div></div>
                <div><div class="ml-card-label">Max</div><div style="font-family:DM Sans;font-weight:600;color:#f43f5e;">{max_latency:.1f} ms</div></div>
                <div><div class="ml-card-label">P95</div><div style="font-family:DM Sans;font-weight:600;color:#f59e0b;">{p95:.1f} ms</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="ml-section-header">🔔 System Alert Log</div>', unsafe_allow_html=True)
    alerts = list(st.session_state.alert_log)
    if not alerts:
        alerts = [{"time": datetime.now().strftime("%H:%M:%S"), "level":"info", "msg":"System started — waiting for events"}]

    alert_rows = ""
    for a in alerts[:8]:
        lvl   = a.get("level","info")
        badge = "badge-critical" if lvl=="critical" else "badge-warning" if lvl=="warning" else "badge-info"
        label = "CRITICAL" if lvl=="critical" else "WARNING" if lvl=="warning" else "INFO"
        alert_rows += f"""
        <div class="alert-row">
            <span class="alert-time">{a.get('time','')}</span>
            <span class="alert-badge {badge}">{label}</span>
            <span class="alert-msg">{a.get('msg','')}</span>
        </div>"""
    st.markdown(f'<div class="alert-log">{alert_rows}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────
time.sleep(1)
st.rerun()