"""
CENTINA OMNI-QUANT v5.0 — Dashboard Unifié
Dark theme Binance · Trading live · Analytics · Satellites · LLM Council
Lancer : streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CENTINA v5.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Binance theme ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp{background:#0B0E11!important;color:#EAECEF!important}
section[data-testid="stSidebar"]{background:#161A1E!important}
.stApp *{font-family:'Inter',sans-serif}

/* ── Metrics ── */
div[data-testid="stMetric"]{background:#1E2329;border-radius:8px;padding:12px;border:0.5px solid #2B3139}
div[data-testid="stMetric"] label{color:#848E9C!important;font-size:11px!important}
div[data-testid="stMetricValue"]{color:#F0B90B!important;font-size:20px!important;font-weight:700!important}
div[data-testid="stMetricDelta"]{font-size:12px!important}

/* ── Tables ── */
.stDataFrame{background:#1E2329!important}
thead tr th{background:#2B3139!important;color:#848E9C!important;font-size:11px!important}
tbody tr td{background:#1E2329!important;color:#EAECEF!important;font-size:12px!important}
tbody tr:hover td{background:#2B3139!important}

/* ── Buttons ── */
div.stButton>button{background:#1E2329;color:#EAECEF;border:0.5px solid #2B3139;border-radius:4px;font-weight:500}
div.stButton>button:hover{background:#2B3139;border-color:#F0B90B}
div.stButton>button[kind="primary"]{background:#F0B90B;color:#0B0E11;border:none}

/* ── Inputs ── */
div[data-testid="stTextInput"] input{background:#1E2329!important;color:#EAECEF!important;border-color:#2B3139!important}
div[data-testid="stSelectbox"] *{background:#1E2329!important;color:#EAECEF!important}

/* ── Progress ── */
div[data-testid="stProgress"]>div>div{background:#F0B90B!important}
div[data-testid="stProgress"]{background:#2B3139!important;border-radius:4px}

/* ── Alerts / info ── */
div[data-testid="stInfo"]{background:#1E2329;border-color:#2B3139}
div[data-testid="stWarning"]{background:#2D1F0E;border-color:#F0B90B}
div[data-testid="stSuccess"]{background:#0E2D1A;border-color:#02C076}

/* ── Radio / Selectbox labels ── */
label{color:#EAECEF!important}
p{color:#EAECEF}

/* ── Cards custom ── */
.card{background:#1E2329;border-radius:8px;padding:14px;border:0.5px solid #2B3139;margin-bottom:8px}
.prime-alert{background:#2D1F0E;border:1px solid #F0B90B;border-radius:6px;padding:10px;margin:3px 0}
.agent-msg{background:#1E2329;border-left:3px solid #F0B90B;padding:8px 10px;border-radius:0 4px 4px 0;margin:3px 0;font-size:12px}
.user-msg{background:#2B3139;border-left:3px solid #02C076;padding:8px 10px;border-radius:0 4px 4px 0;margin:3px 0;font-size:12px}
.pos-card{background:#1E2329;border-radius:8px;padding:12px;border:0.5px solid #2B3139;margin-bottom:6px}
.sat-card{background:#1E2329;border-radius:6px;padding:10px;border:0.5px solid #2B3139;margin-bottom:4px}
.green{color:#02C076!important;font-weight:600}
.red{color:#F6465D!important;font-weight:600}
.gold{color:#F0B90B!important;font-weight:600}
.badge{background:#2B3139;padding:2px 8px;border-radius:3px;font-size:11px;margin:2px;display:inline-block}
.badge-green{background:rgba(2,192,118,0.15);color:#02C076;padding:2px 8px;border-radius:3px;font-size:11px;margin:2px;display:inline-block}
.badge-red{background:rgba(246,70,93,0.15);color:#F6465D;padding:2px 8px;border-radius:3px;font-size:11px;margin:2px;display:inline-block}
.badge-gold{background:rgba(240,185,11,0.15);color:#F0B90B;padding:2px 8px;border-radius:3px;font-size:11px;margin:2px;display:inline-block}

/* ── Tabs ── */
div[data-testid="stTabs"] button{color:#848E9C!important;background:transparent!important}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#F0B90B!important;border-bottom:2px solid #F0B90B!important}

/* ── Divider ── */
hr{border-color:#2B3139!important}

/* ── Expander ── */
div[data-testid="stExpander"]{background:#1E2329!important;border:0.5px solid #2B3139!important;border-radius:6px}
div[data-testid="stExpander"] summary{color:#EAECEF!important}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).parent.parent
DB_PATH       = ROOT / "data" / "cortex.db"
BRIDGE_FILE   = ROOT / "data" / "bridge_state.json"
CMD_QUEUE     = ROOT / "data" / "cmd_queue.json"
AUDIT_LOG     = ROOT / "data" / "logs" / "audit.jsonl"
WEIGHTS_FILE  = ROOT / "data" / "adaptive_weights.json"
OPTIMISED_FILE= ROOT / "data" / "optimised_params.json"
CB_STATE_FILE = ROOT / "data" / "circuit_breaker_state.json"
ACTIVATION_FILE = ROOT / "data" / ".activation_date"

FNG_LABELS = {
    (0,   25): ("Peur extreme", "#F6465D"),
    (25,  45): ("Peur",         "#FF8C00"),
    (45,  55): ("Neutre",       "#F0B90B"),
    (55,  75): ("Avidite",      "#02C076"),
    (75, 101): ("Avidite extreme","#00E676"),
}

# ── DB helpers ────────────────────────────────────────────────────────────
def _q(sql: str, params=()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_bridge() -> dict:
    if BRIDGE_FILE.exists():
        try:
            data = json.loads(BRIDGE_FILE.read_text(encoding="utf-8"))
            # Normalise les deux formats (ancien et nouveau) — ajoute les alias manquants
            data.setdefault("capital_usdt",   data.get("capital", 200))
            data.setdefault("capital",        data.get("capital_usdt", 200))
            data.setdefault("pnl_day_eur",    data.get("daily_pnl", 0))
            data.setdefault("daily_pnl",      data.get("pnl_day_eur", 0))
            data.setdefault("open_positions", len(data.get("positions", [])))
            data.setdefault("events",         data.get("status_log", []))
            data.setdefault("status_log",     data.get("events", []))
            # cb_level peut être int (0=NORMAL,1=ALERT,…) ou string — normalise en string
            cb_raw = data.get("cb_level", data.get("cb_name", "NORMAL"))
            cb_map = {0: "NORMAL", 1: "ALERT", 2: "RESTRICTED", 3: "STOPPED", 4: "TERMINATED"}
            cb_str = cb_map.get(cb_raw, str(cb_raw)) if isinstance(cb_raw, int) else cb_raw
            data["cb_level"] = cb_str
            data.setdefault("cb_name",        cb_str)
            data.setdefault("voice_active",   False)
            data.setdefault("paper_days_left", 7)
            data.setdefault("ts", "")
            data.setdefault("mode", "PAPER")
            data.setdefault("positions", [])
            data.setdefault("opportunities", [])
            return data
        except Exception as e:
            pass
    return {
        "mode": "PAPER", "cb_level": "NORMAL", "cb_name": "NORMAL",
        "capital_usdt": 200, "capital": 200,
        "pnl_day_eur": 0, "daily_pnl": 0,
        "open_positions": 0, "paper_days_left": 7,
        "positions": [], "opportunities": [],
        "events": [], "status_log": [],
        "voice_active": False, "ts": "",
    }


def write_cmd(cmd: dict) -> None:
    cmd["ts"] = datetime.utcnow().isoformat()
    CMD_QUEUE.parent.mkdir(exist_ok=True)
    queue: list = []
    if CMD_QUEUE.exists():
        try:
            queue = json.loads(CMD_QUEUE.read_text(encoding="utf-8"))
        except Exception:
            queue = []
    queue.append(cmd)
    CMD_QUEUE.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def get_fear_greed_api() -> dict:
    """Fetch Fear & Greed from alternative.me (cached 5min via session_state)."""
    now = time.time()
    cache = st.session_state.get("_fng_cache", {})
    if cache.get("ts", 0) + 300 > now:
        return cache.get("data", {"value": 50, "label": "Neutral"})
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=3)
        d = r.json()["data"][0]
        result = {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        result = {"value": 50, "label": "Neutral"}
    st.session_state["_fng_cache"] = {"ts": now, "data": result}
    return result


def fng_display(value: int) -> tuple[str, str]:
    for (lo, hi), (label, color) in FNG_LABELS.items():
        if lo <= value < hi:
            return label, color
    return "❓ Inconnu", "#848E9C"


def get_klines(symbol: str, interval: str = "1h", limit: int = 120) -> list[dict]:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5,
        )
        return [
            {"t": d[0], "o": float(d[1]), "h": float(d[2]),
             "l": float(d[3]), "c": float(d[4]), "v": float(d[5])}
            for d in r.json()
        ]
    except Exception:
        return []


def ema_calc(closes: list[float], n: int) -> list[float]:
    k = 2 / (n + 1)
    e = closes[0]
    result = [e]
    for c in closes[1:]:
        e = c * k + e * (1 - k)
        result.append(e)
    return result


def paper_days_left() -> int:
    if not ACTIVATION_FILE.exists():
        return 7
    try:
        dt = datetime.fromisoformat(ACTIVATION_FILE.read_text().strip())
        return max(0, 7 - (datetime.utcnow() - dt).days)
    except Exception:
        return 7


def cb_label(level) -> str:
    labels = {0: "🟢 NORMAL", 1: "🟡 ALERT", 2: "🟠 RESTRICTED", 3: "🔴 STOPPED", 4: "⛔ TERMINATED"}
    if isinstance(level, str):
        icons = {"NORMAL": "🟢", "ALERT": "🟡", "RESTRICTED": "🟠", "STOPPED": "🔴", "TERMINATED": "⛔"}
        return f"{icons.get(level, '❓')} {level}"
    return labels.get(int(level), f"❓ {level}")


def signal_badge(signal: str, neutral_ok: bool = False) -> str:
    s = str(signal).upper()
    if s in ("BUY", "BULLISH", "POSITIVE", "LONG", "INFLOW"):
        return f'<span class="badge-green">{signal}</span>'
    if s in ("SELL", "BEARISH", "NEGATIVE", "SHORT", "OUTFLOW", "DUMP"):
        return f'<span class="badge-red">{signal}</span>'
    if s in ("NEUTRAL", "?", "—", ""):
        if neutral_ok:
            return f'<span class="badge">{signal}</span>'
        return f'<span class="badge">{signal}</span>'
    return f'<span class="badge-gold">{signal}</span>'


# ── Load live state ───────────────────────────────────────────────────────
bridge  = load_bridge()
mode    = bridge.get("mode", "PAPER")
cap     = float(bridge.get("capital_usdt", bridge.get("capital", 200)))
pnl_day = float(bridge.get("pnl_day_eur",  bridge.get("daily_pnl", 0)))
n_pos   = int(bridge.get("open_positions", len(bridge.get("positions", []))))
positions = bridge.get("positions", [])
opps      = bridge.get("opportunities", [])
events    = bridge.get("events", bridge.get("status_log", []))
cb_raw    = bridge.get("cb_level", bridge.get("cb_name", "NORMAL"))
agent_alive = BRIDGE_FILE.exists()

# Fear & Greed: prefer from bridge satellite data (no network call at top level)
fng_from_bridge = next((float(o.get("fear_greed", 0)) for o in opps if o.get("fear_greed")), 0)
fng_value = int(fng_from_bridge) if fng_from_bridge else 50
fng_label, fng_color = fng_display(fng_value)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 CENTINA v5.0")
    st.caption("OMNI-QUANT · 6 LLMs · 10 Satellites")
    st.divider()

    # ── Agent status ──────────────────────────────────────────────────
    if agent_alive:
        ts = bridge.get("ts", "")
        st.success(f"🟢 Agent actif · {ts[11:19]} UTC")
    else:
        st.warning("⚫ Agent non démarré")

    # ── Voice listener status ─────────────────────────────────────────
    voice_active = bridge.get("voice_active", False)
    if voice_active:
        st.success("🎙 Écoute vocale active")
    else:
        st.info("🔇 Voix désactivée")

    col_a, col_b = st.columns(2)
    mode_icon = {"PAPER": "🟡", "ADVISOR": "🔵", "SEMI": "🟢", "AUTO": "🟢"}.get(mode, "❓")
    col_a.metric("Mode", f"{mode_icon} {mode}")
    col_b.metric("Positions", f"{n_pos}/3")
    col_a2, col_b2 = st.columns(2)
    col_a2.metric("Capital", f"{cap:.0f}€")
    col_b2.metric("PnL jour", f"{pnl_day:+.2f}€")

    target = 30.0
    prog   = min(1.0, max(0.0, pnl_day / target)) if target > 0 else 0
    st.progress(prog, text=f"Objectif : {pnl_day:.1f} / {target:.0f} EUR")

    days_left = paper_days_left()
    if days_left > 0:
        st.warning(f"⏳ Paper gate : {days_left}j restants")
    else:
        st.success("✅ Paper gate débloqué")

    st.divider()

    # ── Fear & Greed Index ────────────────────────────────────────────
    # Appel API uniquement si pas de données dans le bridge
    if not fng_from_bridge:
        try:
            fng_api = get_fear_greed_api()
            fng_value = fng_api.get("value", 50)
            fng_label, fng_color = fng_display(fng_value)
        except Exception:
            pass
    st.markdown("**Fear & Greed Index**")
    st.markdown(
        f'<div style="background:#1E2329;border-radius:8px;padding:10px;border:0.5px solid #2B3139;text-align:center">'
        f'<span style="font-size:28px;font-weight:700;color:{fng_color}">{fng_value}</span><br>'
        f'<span style="color:{fng_color};font-size:13px">{fng_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(fng_value / 100, text=f"Score : {fng_value}/100")

    st.divider()

    # ── Mode trading ──────────────────────────────────────────────────
    st.markdown("**Mode trading**")
    trading_mode = st.radio(
        "mode_sel",
        ["🟡 PAPER", "🔵 ADVISOR", "⚡ SEMI", "🤖 AUTO"],
        index=["PAPER","ADVISOR","SEMI","AUTO"].index(mode) if mode in ["PAPER","ADVISOR","SEMI","AUTO"] else 0,
        label_visibility="collapsed",
    )
    if st.button("APPLIQUER MODE", use_container_width=True):
        val = trading_mode.split(" ")[-1]
        write_cmd({"cmd": "SET_MODE", "value": val})
        st.success(f"✓ Mode {val}")

    st.divider()

    # ── Circuit Breaker ───────────────────────────────────────────────
    st.markdown(f"**CB** : {cb_label(cb_raw)}")
    if str(cb_raw) in ("ALERT", "RESTRICTED", 1, 2):
        if st.button("🔄 RESET CB", use_container_width=True):
            write_cmd({"cmd": "CB_RESET"})

    st.divider()

    # ── Paramètres ────────────────────────────────────────────────────
    score_min = st.slider("Score minimum", 70, 95, 78)
    pos_pct   = st.slider("Mise / trade (%)", 1, 10, 3)
    if st.button("APPLIQUER PARAMS", use_container_width=True):
        write_cmd({"cmd": "SET_PARAMS", "score_min": score_min, "position_pct": pos_pct / 100})
        st.success("✓ Params envoyés")

    st.divider()

    # ── Contrôles urgents ─────────────────────────────────────────────
    if st.button("🛑 FERME TOUT", type="primary", use_container_width=True):
        write_cmd({"cmd": "CLOSE_ALL"})
    c_p, c_r = st.columns(2)
    if c_p.button("⏸ PAUSE"):
        write_cmd({"cmd": "PAUSE"})
    if c_r.button("▶ REPRISE"):
        write_cmd({"cmd": "RESUME"})

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────
    page = st.radio("Navigation", [
        "🚀 Trading Live",
        "📊 Dashboard",
        "🌐 Intelligence Macro",
        "🎯 Opportunités",
        "📋 Trades",
        "📈 Performance",
        "🧠 Apprentissage",
        "🔬 Simulation",
        "🤖 Conseil LLM",
    ], label_visibility="collapsed")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh 5s", value=True)
    st.caption(f"v5.0 · {datetime.utcnow().strftime('%H:%M:%S')} UTC")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — TRADING LIVE
# ══════════════════════════════════════════════════════════════════════════
if page == "🚀 Trading Live":
    st.markdown("## 🚀 Trading Live")

    col1, col2, col3 = st.columns([2.2, 1.5, 1.3])

    # ── Colonne 1 : Graphique ─────────────────────────────────────────
    with col1:
        syms = [o.get("symbol", o.get("sym", "")) for o in opps if o.get("symbol", o.get("sym", ""))]
        for pos in positions:
            s = pos.get("symbol", "")
            if s and s not in syms:
                syms.insert(0, s)
        if not syms:
            syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

        sel_sym = st.selectbox("Paire", syms, key="sym_sel")
        tf      = st.radio("Timeframe", ["5m", "15m", "1h", "4h"], horizontal=True, key="tf_sel", index=2)

        klines = get_klines(sel_sym, tf, 120)
        if klines:
            opens  = [k["o"] for k in klines]
            highs  = [k["h"] for k in klines]
            lows   = [k["l"] for k in klines]
            closes = [k["c"] for k in klines]
            vols   = [k["v"] for k in klines]
            times  = [datetime.utcfromtimestamp(k["t"] / 1000) for k in klines]

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=times, open=opens, high=highs, low=lows, close=closes,
                name=sel_sym,
                increasing_line_color="#02C076", increasing_fillcolor="#02C076",
                decreasing_line_color="#F6465D", decreasing_fillcolor="#F6465D",
            ))

            ema_colors = {7: "#F0B90B", 25: "#1E90FF", 99: "#FF8C00", 200: "#FF4444"}
            for period, color in ema_colors.items():
                if len(closes) >= period:
                    ema_vals = ema_calc(closes, period)
                    fig.add_trace(go.Scatter(
                        x=times, y=ema_vals, name=f"EMA{period}",
                        line=dict(color=color, width=1), opacity=0.85,
                    ))

            for pos in positions:
                if pos.get("symbol") == sel_sym:
                    for field, color, label in [
                        ("sl",  "#F6465D", "SL"),
                        ("tp1", "#02C076", "TP1"),
                        ("tp2", "#1E90FF", "TP2"),
                    ]:
                        val = pos.get(field) or pos.get(f"{field}_price")
                        if val:
                            fig.add_hline(
                                y=float(val), line_color=color, line_dash="dot",
                                line_width=1,
                                annotation_text=f"{label} {float(val):.4f}",
                                annotation_position="right",
                                annotation_font=dict(color=color, size=10),
                            )

            fig.update_layout(
                height=420,
                paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139", showgrid=True),
                yaxis=dict(gridcolor="#2B3139", showgrid=True, side="right"),
                margin=dict(l=0, r=60, t=30, b=0),
                legend=dict(bgcolor="#1E2329", font=dict(color="#EAECEF", size=10),
                            orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_v = go.Figure()
            vol_colors = ["#02C076" if closes[i] >= opens[i] else "#F6465D" for i in range(len(closes))]
            fig_v.add_trace(go.Bar(x=times, y=vols, marker_color=vol_colors, name="Volume", opacity=0.7))
            fig_v.update_layout(
                height=80, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139", showticklabels=False),
                yaxis=dict(gridcolor="#2B3139", showticklabels=False),
                margin=dict(l=0, r=60, t=0, b=0), showlegend=False,
            )
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("Connexion Binance en cours…")

        # Score de la paire sélectionnée
        cur_opp = next((o for o in opps if o.get("symbol", o.get("sym")) == sel_sym), None)
        if cur_opp:
            sc    = float(cur_opp.get("score", 0))
            strat = cur_opp.get("strategy", cur_opp.get("suggested_strategy", "?"))
            gain  = float(cur_opp.get("estimated_gain_eur", 0))
            dur   = cur_opp.get("estimated_hours", "?")
            deep_p = float(cur_opp.get("deep_proba", 0))
            council_agr = cur_opp.get("council_agreement", 0)
            council_m   = cur_opp.get("council_models", 0)
            col   = "#F0B90B" if sc >= 93 else ("#02C076" if sc >= 78 else "#F6465D")
            risk_method = cur_opp.get("risk_method", "Kelly")
            st.markdown(
                f"**Score :** <span style='color:{col};font-weight:700'>{sc:.1f}/100</span> · "
                f"**Stratégie {strat}** · Gain est. **+{gain:.1f} EUR** · Durée **{dur}h** · "
                f"**ML :** {deep_p:.0%} · **Sizing :** {risk_method}",
                unsafe_allow_html=True,
            )
            if council_agr:
                st.markdown(
                    f"**LLM Council :** {council_agr:.0f}% accord · {council_m} modèles",
                    unsafe_allow_html=True,
                )
            council_txt = cur_opp.get("council_text", "")
            if council_txt:
                st.markdown(f'<div class="agent-msg">🤖 {council_txt}</div>', unsafe_allow_html=True)

            # ── Analyse Satellite Complète ─────────────────────────────
            with st.expander("🔬 Analyse Satellite Complète", expanded=False):
                tab_micro, tab_of, tab_liq, tab_cross, tab_social, tab_whale, tab_ml = st.tabs([
                    "📊 Microstructure",
                    "📈 Orderflow",
                    "💧 Liquidités",
                    "🌍 Cross-Asset",
                    "💬 Social & Sentiment",
                    "🐳 Smart Money",
                    "🤖 Patterns & ML",
                ])

                with tab_micro:
                    vwap_sig = cur_opp.get("vwap_signal", "NEUTRAL")
                    cvd_sig  = cur_opp.get("cvd_signal", "NEUTRAL")
                    buy_dp   = float(cur_opp.get("buy_delta_pct", 50.0))
                    micro_sig = cur_opp.get("microstructure_signal", "—")
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("VWAP Signal", vwap_sig)
                    cm2.metric("CVD Signal", cvd_sig)
                    cm3.metric("Buy Delta %", f"{buy_dp:.1f}%")
                    cm4.metric("Micro Score", micro_sig)
                    st.markdown(
                        f"VWAP : {signal_badge(vwap_sig)} &nbsp; "
                        f"CVD : {signal_badge(cvd_sig)} &nbsp; "
                        f"Buy Delta : <span class='{'badge-green' if buy_dp > 55 else ('badge-red' if buy_dp < 45 else 'badge')}'>{buy_dp:.1f}%</span>",
                        unsafe_allow_html=True,
                    )

                with tab_of:
                    bai  = float(cur_opp.get("bid_ask_imbalance", 1.0))
                    bp   = float(cur_opp.get("buy_pressure", 50.0))
                    abs_ = bool(cur_opp.get("absorption", False))
                    sh   = bool(cur_opp.get("stop_hunt", False))
                    ice  = bool(cur_opp.get("iceberg", False))
                    of_sig = cur_opp.get("orderflow_signal", "—")
                    co1, co2, co3 = st.columns(3)
                    co1.metric("Bid/Ask Imbalance", f"{bai:.2f}x")
                    co2.metric("Buy Pressure", f"{bp:.1f}%")
                    co3.metric("Signal Orderflow", of_sig)
                    st.markdown(
                        f"Absorption : {'<span class=\"badge-green\">✅ Détectée</span>' if abs_ else '<span class=\"badge\">❌ Non</span>'} &nbsp;"
                        f"Stop Hunt : {'<span class=\"badge-gold\">⚠ Oui</span>' if sh else '<span class=\"badge\">❌ Non</span>'} &nbsp;"
                        f"Iceberg : {'<span class=\"badge-gold\">🧊 Oui</span>' if ice else '<span class=\"badge\">❌ Non</span>'}",
                        unsafe_allow_html=True,
                    )
                    st.progress(bp / 100, text=f"Pression acheteurs : {bp:.1f}%")

                with tab_liq:
                    la_pct  = float(cur_opp.get("liq_above_pct", 0))
                    lb_pct  = float(cur_opp.get("liq_below_pct", 0))
                    la_price = float(cur_opp.get("liq_above_price", 0))
                    lb_price = float(cur_opp.get("liq_below_price", 0))
                    la_size  = float(cur_opp.get("liq_above_size", 0))
                    lb_size  = float(cur_opp.get("liq_below_size", 0))
                    cl1, cl2 = st.columns(2)
                    cl1.markdown(
                        f'<div class="sat-card">'
                        f'<div style="color:#02C076;font-weight:700">🟢 Mur de résistance (au-dessus)</div>'
                        f'<div>Prix : <strong>{la_price:,.4f}</strong></div>'
                        f'<div>Distance : <strong>+{la_pct:.2f}%</strong></div>'
                        f'<div>Taille : <strong>{la_size:,.0f} $</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    cl2.markdown(
                        f'<div class="sat-card">'
                        f'<div style="color:#F6465D;font-weight:700">🔴 Mur de support (en-dessous)</div>'
                        f'<div>Prix : <strong>{lb_price:,.4f}</strong></div>'
                        f'<div>Distance : <strong>-{lb_pct:.2f}%</strong></div>'
                        f'<div>Taille : <strong>{lb_size:,.0f} $</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with tab_cross:
                    btc_dom  = float(cur_opp.get("btc_dominance", 50.0))
                    eth_btc  = float(cur_opp.get("eth_btc_ratio", 0.0))
                    eth_trend = cur_opp.get("eth_btc_trend", "NEUTRAL")
                    fund_r   = float(cur_opp.get("funding_rate", 0.0))
                    fund_sig = cur_opp.get("funding_signal", "NEUTRAL")
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("BTC Dominance", f"{btc_dom:.1f}%",
                               delta="↑ Altcoins sous pression" if btc_dom > 50 else "↓ Season altcoins")
                    cc2.metric("ETH/BTC", f"{eth_btc:.5f}")
                    cc3.metric("ETH/BTC Trend", eth_trend)
                    cc4.metric("Funding Rate",  f"{fund_r:.4f}%", delta=fund_sig)
                    st.markdown(
                        f"Dominance BTC : <span class='{'badge-red' if btc_dom > 55 else 'badge-green'}'>{btc_dom:.1f}%</span> &nbsp;"
                        f"ETH/BTC : {signal_badge(eth_trend)} &nbsp;"
                        f"Funding : {signal_badge(fund_sig)}",
                        unsafe_allow_html=True,
                    )

                with tab_social:
                    reddit_t  = cur_opp.get("reddit_trend", "NEUTRAL")
                    fg_val    = float(cur_opp.get("fear_greed", 50.0))
                    cp_score  = float(cur_opp.get("cryptopanic_score", 50.0))
                    social_sc = float(cur_opp.get("social_score", 50.0))
                    fg_lbl, fg_c = fng_display(int(fg_val))
                    cs1, cs2, cs3, cs4 = st.columns(4)
                    cs1.metric("Reddit Trend", reddit_t)
                    cs2.metric("Fear & Greed", f"{fg_val:.0f}", delta=fg_lbl)
                    cs3.metric("CryptoPanic", f"{cp_score:.0f}/100")
                    cs4.metric("Score Social", f"{social_sc:.0f}/100")
                    st.progress(fg_val / 100, text=f"Fear & Greed : {fg_val:.0f}/100 · {fg_lbl}")
                    st.progress(cp_score / 100, text=f"CryptoPanic Score : {cp_score:.0f}/100")
                    st.markdown(
                        f"Reddit : {signal_badge(reddit_t)} &nbsp; "
                        f"Sentiment global : <span style='color:{fg_c}'>{fg_lbl}</span>",
                        unsafe_allow_html=True,
                    )

                with tab_whale:
                    w_buys  = int(cur_opp.get("whale_buys", 0))
                    w_sells = int(cur_opp.get("whale_sells", 0))
                    net_f   = float(cur_opp.get("net_flow", 0.0))
                    lbr     = float(cur_opp.get("large_buy_ratio", 0.5))
                    sm_sig  = cur_opp.get("smart_money", "NEUTRAL")
                    cw1, cw2, cw3, cw4 = st.columns(4)
                    cw1.metric("Whale Achats", str(w_buys))
                    cw2.metric("Whale Ventes", str(w_sells), delta="⚠ VETO si ≥5" if w_sells >= 5 else None,
                               delta_color="inverse" if w_sells >= 5 else "normal")
                    cw3.metric("Net Flow", f"{net_f:+,.0f} $")
                    cw4.metric("Large Buy Ratio", f"{lbr:.0%}")
                    st.markdown(
                        f"Smart Money : {signal_badge(sm_sig)} &nbsp; "
                        f"Baleines : <span class='badge-green'>+{w_buys} achats</span> "
                        f"<span class='{'badge-red' if w_sells >= 5 else 'badge'}'>{w_sells} ventes</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(lbr, text=f"Ratio gros achats : {lbr:.0%}")

                with tab_ml:
                    pat_sig   = cur_opp.get("pattern_signal", "NONE")
                    pat_conf  = float(cur_opp.get("pattern_confidence", 0.0))
                    pat_tgt   = float(cur_opp.get("pattern_target", 0.0))
                    patterns  = cur_opp.get("patterns_found", [])
                    atr_ratio = float(cur_opp.get("atr_ratio", 1.0))
                    atr_pred  = float(cur_opp.get("atr_predicted", 0.0))
                    d_proba   = float(cur_opp.get("deep_proba", 0.0))
                    risk_m    = cur_opp.get("risk_method", "Kelly")
                    sat_b     = cur_opp.get("satellite_bonus", 0)
                    vol_lbl   = cur_opp.get("volatility_label", "—")
                    cml1, cml2, cml3, cml4 = st.columns(4)
                    cml1.metric("DeepPredictor", f"{d_proba:.0%}", delta="✅ Bullish" if d_proba > 0.6 else ("⚠ Neutre" if d_proba > 0.4 else "❌ Bearish"))
                    cml2.metric("Pattern", pat_sig)
                    cml3.metric("Confiance Pattern", f"{pat_conf:.0%}")
                    cml4.metric("ATR Ratio", f"{atr_ratio:.2f}x")
                    if patterns:
                        st.markdown(
                            "Patterns détectés : " + " ".join([f'<span class="badge-gold">{p}</span>' for p in patterns]),
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"Cible pattern : <strong>{pat_tgt:,.4f}</strong> · "
                        f"ATR prédit : <strong>{atr_pred:.4f}</strong> · "
                        f"Volatilité : <span class='badge'>{vol_lbl}</span> · "
                        f"Sizing : <span class='badge-gold'>{risk_m}</span> · "
                        f"Bonus satellites : <span class='{'badge-green' if int(sat_b) > 0 else 'badge'}'>{sat_b:+d} pts</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(d_proba, text=f"Probabilité ML (GradientBoosting) : {d_proba:.0%}")

    # ── Colonne 2 : Advisor ───────────────────────────────────────────
    with col2:
        st.markdown("### 💬 CENTINA Advisor")
        for ev in list(reversed(events))[:20]:
            if isinstance(ev, dict):
                ts_ev  = ev.get("ts", "")[-8:-3] if ev.get("ts") else ""
                txt    = ev.get("text", "")
                ev_type = ev.get("type", "")
            else:
                ts_ev, txt, ev_type = "", str(ev), ""

            if ev_type == "PRIME":
                st.markdown(
                    f'<div class="prime-alert">🚨 <strong>{ts_ev}</strong> {txt}</div>',
                    unsafe_allow_html=True,
                )
            elif ev_type in ("USER_CMD", "USER"):
                st.markdown(
                    f'<div class="user-msg">👤 {ts_ev} {txt}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="agent-msg">🤖 {ts_ev} {txt}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        cmd_text = st.text_input(
            "cmd_input",
            placeholder="GO SOLUSDT · PASSE · STATUT · SOLDE · RAPPORT",
            label_visibility="collapsed",
        )
        if st.button("ENVOYER ↗", use_container_width=True) and cmd_text.strip():
            write_cmd({"cmd": "USER_INPUT", "text": cmd_text.strip()})
            st.rerun()

        cols_btn = st.columns(3)
        for i, (lbl, cmd) in enumerate([("GO", "GO"), ("PASSE", "PASSE"), ("STATUT", "STATUT"),
                                          ("SOLDE", "SOLDE"), ("RAPPORT", "RAPPORT"), ("OPPS", "OPPS")]):
            if cols_btn[i % 3].button(lbl, key=f"btn_{cmd}"):
                write_cmd({"cmd": "USER_INPUT", "text": cmd})
                st.rerun()

    # ── Colonne 3 : Positions + Top Signaux ──────────────────────────
    with col3:
        st.markdown("### 📊 Positions")
        if not positions:
            st.markdown('<p style="color:#848E9C;text-align:center;margin-top:20px">Aucune position</p>', unsafe_allow_html=True)
        for pos in positions:
            pnl_p   = float(pos.get("pnl_eur", 0))
            dur_h   = float(pos.get("elapsed_h", pos.get("duration_h", 0)))
            sym_p   = pos.get("symbol", "?")
            entry_p = pos.get("entry", pos.get("entry_price", 0))
            score_p = pos.get("score", 0)
            tp1_ok  = "✅" if pos.get("tp1_done", pos.get("tp1_hit")) else "⏳"
            tp2_ok  = "✅" if pos.get("tp2_done", pos.get("tp2_hit")) else "⏳"
            pnl_col = "green" if pnl_p >= 0 else "red"
            dur_col = "red" if dur_h > 4 else ("gold" if dur_h > 2.5 else "green")
            st.markdown(
                f'<div class="pos-card">'
                f'<strong>{sym_p}</strong> '
                f'<span style="background:#2B3139;padding:2px 5px;border-radius:3px;font-size:10px">'
                f'Strat {pos.get("strategy","?")} · {score_p:.0f}pts</span><br>'
                f'<span style="font-size:11px;color:#848E9C">Entrée : {entry_p}</span><br>'
                f'<span class="{pnl_col}">PnL : {pnl_p:+.2f}€</span> · '
                f'<span class="{dur_col}">{dur_h:.1f}h</span><br>'
                f'TP1 {tp1_ok} · TP2 {tp2_ok}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, dur_h / 5), text=f"{dur_h:.1f}h / 5h max")
            if st.button(f"✖ FERMER {sym_p}", key=f"close_{sym_p}"):
                write_cmd({"cmd": "CLOSE_POSITION", "symbol": sym_p})

        st.markdown("### 🏆 Top Signaux")
        if not opps:
            st.markdown('<p style="color:#848E9C;text-align:center">Aucun signal actif</p>', unsafe_allow_html=True)
        for opp in opps[:6]:
            sc_o    = float(opp.get("score", 0))
            sym_o   = opp.get("symbol", opp.get("sym", "?"))
            strat_o = opp.get("strategy", opp.get("suggested_strategy", "?"))
            gain_o  = float(opp.get("estimated_gain_eur", 0))
            col_o   = "#F0B90B" if sc_o >= 93 else "#02C076"
            sat_b   = opp.get("satellite_bonus", 0)
            deep_po = float(opp.get("deep_proba", 0))
            c_l, c_b = st.columns([3, 1])
            c_l.markdown(
                f"**{sym_o}** · <span style='color:{col_o}'>{sc_o:.0f}/100</span><br>"
                f"<span style='font-size:11px;color:#848E9C'>"
                f"Strat {strat_o} · +{gain_o:.1f}€ · sat {sat_b:+d}pts · ML {deep_po:.0%}"
                f"</span>",
                unsafe_allow_html=True,
            )
            if c_b.button("GO", key=f"go_{sym_o}"):
                write_cmd({"cmd": "EXECUTE", "symbol": sym_o})


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard CENTINA")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_n  = _q("SELECT COUNT(*) AS n FROM trades WHERE closed_at IS NOT NULL")
    win_n    = _q("SELECT COUNT(*) AS n FROM trades WHERE pnl_usdt>=0 AND closed_at IS NOT NULL")
    pnl_db   = _q("SELECT SUM(pnl_usdt) AS s FROM trades WHERE date(closed_at)=date('now')")
    pnl_tot  = _q("SELECT SUM(pnl_usdt) AS s FROM trades WHERE closed_at IS NOT NULL")

    total_trades = int(total_n["n"].iloc[0]) if not total_n.empty else 0
    wins         = int(win_n["n"].iloc[0]) if not win_n.empty else 0
    win_rate     = wins / total_trades if total_trades > 0 else 0
    pnl_d_db     = float(pnl_db["s"].iloc[0] or 0) if not pnl_db.empty else 0
    pnl_t_db     = float(pnl_tot["s"].iloc[0] or 0) if not pnl_tot.empty else 0

    c1.metric("Capital", f"{cap:.0f} €")
    c2.metric("PnL Jour", f"{pnl_d_db:+.2f} €", delta=f"{pnl_d_db/cap*100:+.2f}%" if cap else None)
    c3.metric("PnL Total", f"{pnl_t_db:+.2f} €")
    c4.metric("Trades", str(total_trades))
    c5.metric("Win Rate", f"{win_rate:.1%}")
    c6.metric("CB", cb_label(cb_raw))

    target = 30.0
    pct    = min(1.0, max(0, pnl_d_db) / target) if target > 0 else 0
    st.progress(pct, text=f"Objectif jour : {pnl_d_db:.2f} / {target:.0f} EUR ({pct*100:.0f}%)")
    st.divider()

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### Courbe d'équité")
        pnl_df = _q("SELECT pnl_usdt, closed_at FROM trades WHERE closed_at IS NOT NULL ORDER BY closed_at")
        if not pnl_df.empty:
            pnl_df["equity"]    = cap + pnl_df["pnl_usdt"].cumsum()
            pnl_df["closed_at"] = pd.to_datetime(pnl_df["closed_at"])
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=pnl_df["closed_at"], y=pnl_df["equity"],
                mode="lines", line=dict(color="#02C076", width=2),
                fill="tozeroy", fillcolor="rgba(2,192,118,0.07)",
            ))
            fig_eq.add_hline(y=cap, line_dash="dash", line_color="#848E9C", line_width=1)
            fig_eq.update_layout(
                height=280, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139"), yaxis=dict(gridcolor="#2B3139"),
                margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Pas encore de trades clôturés.")

    with col_r:
        st.markdown("#### Positions ouvertes")
        open_df = _q(
            "SELECT symbol, strategy, entry_price, tp1_price, sl_price, score_at_entry "
            "FROM trades WHERE closed_at IS NULL"
        )
        if not open_df.empty:
            st.dataframe(open_df, use_container_width=True, height=280, hide_index=True)
        else:
            st.info("Aucune position.")

    st.markdown("#### Derniers 15 trades")
    recent = _q(
        "SELECT symbol, strategy, entry_price, exit_price, pnl_usdt, pnl_pct, "
        "exit_reason, score_at_entry, tp1_hit, opened_at "
        "FROM trades ORDER BY id DESC LIMIT 15"
    )
    if not recent.empty:
        def _clr(v):
            if pd.isna(v): return ""
            return "color: #02C076" if float(v) >= 0 else "color: #F6465D"
        st.dataframe(
            recent.style.applymap(_clr, subset=["pnl_usdt"] if "pnl_usdt" in recent.columns else []),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Aucun trade.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — INTELLIGENCE MACRO
# ══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Intelligence Macro":
    st.markdown("## 🌐 Intelligence Macro — Données Cross-Asset en Temps Réel")

    # ── Fear & Greed central display ──────────────────────────────────
    st.markdown("### 😨 Fear & Greed Index")
    try:
        fg_api = get_fear_greed_api()
    except Exception:
        fg_api = {}
    fg_v = fg_api.get("value", fng_value)
    fg_l, fg_c2 = fng_display(int(fg_v))
    cf1, cf2, cf3 = st.columns([1, 2, 1])
    with cf2:
        st.markdown(
            f'<div style="text-align:center;background:#1E2329;border-radius:12px;padding:20px;border:1px solid #2B3139">'
            f'<div style="font-size:64px;font-weight:900;color:{fg_c2}">{fg_v}</div>'
            f'<div style="font-size:20px;color:{fg_c2};margin-top:4px">{fg_l}</div>'
            f'<div style="color:#848E9C;font-size:12px;margin-top:6px">Source : Alternative.me · Actualisé toutes les 5min</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.progress(fg_v / 100, text=f"{fg_v}/100")

    st.divider()

    # ── Cross-asset data from opportunities ───────────────────────────
    st.markdown("### 📊 Données Cross-Asset (depuis analyse satellites)")
    if opps:
        # Build cross-asset table from all opportunities
        macro_rows = []
        for o in opps:
            btc_dom_o  = o.get("btc_dominance", 0)
            eth_btc_o  = o.get("eth_btc_ratio", 0)
            fund_r_o   = o.get("funding_rate", 0)
            fund_s_o   = o.get("funding_signal", "—")
            eth_tr_o   = o.get("eth_btc_trend", "—")
            if btc_dom_o or eth_btc_o or fund_r_o:
                macro_rows.append({
                    "Paire":            o.get("symbol", "?"),
                    "BTC Dom %":        f"{btc_dom_o:.1f}",
                    "ETH/BTC":          f"{eth_btc_o:.5f}",
                    "ETH/BTC Trend":    eth_tr_o,
                    "Funding Rate %":   f"{fund_r_o:.4f}",
                    "Funding Signal":   fund_s_o,
                    "Fear&Greed":       f"{o.get('fear_greed', 0):.0f}",
                    "Reddit":           o.get("reddit_trend", "—"),
                    "CryptoPanic":      f"{o.get('cryptopanic_score', 0):.0f}/100",
                })
        if macro_rows:
            macro_df = pd.DataFrame(macro_rows)
            st.dataframe(macro_df, use_container_width=True, hide_index=True)

        # ── BTC Dominance gauge from first opp with data ──────────────
        btc_dom_val = next((float(o.get("btc_dominance", 0)) for o in opps if o.get("btc_dominance")), 0)
        funding_val = next((float(o.get("funding_rate", 0)) for o in opps if o.get("funding_rate") is not None), 0)
        eth_btc_val = next((float(o.get("eth_btc_ratio", 0)) for o in opps if o.get("eth_btc_ratio")), 0)

        if btc_dom_val or funding_val or eth_btc_val:
            st.divider()
            st.markdown("### 🔢 Indicateurs Macro Clés")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric(
                "BTC Dominance",
                f"{btc_dom_val:.1f}%",
                delta="Altseason ↑" if btc_dom_val < 45 else ("BTC Focus ↑" if btc_dom_val > 55 else None),
            )
            cm2.metric(
                "ETH/BTC Ratio",
                f"{eth_btc_val:.5f}" if eth_btc_val else "—",
            )
            cm3.metric(
                "Funding Rate",
                f"{funding_val:.4f}%",
                delta="⚠ Suracheté" if funding_val > 0.01 else ("💡 Neutre" if abs(funding_val) < 0.005 else "👀 Short pressure"),
                delta_color="inverse" if funding_val > 0.01 else "normal",
            )
            cm4.metric("Fear & Greed", f"{fg_v}", delta=fg_l)

            # BTC Dom jauge
            fig_dom = go.Figure(go.Indicator(
                mode="gauge+number",
                value=btc_dom_val,
                title={"text": "BTC Dominance %", "font": {"color": "#EAECEF", "size": 14}},
                gauge={
                    "axis":     {"range": [30, 75], "tickcolor": "#848E9C"},
                    "bar":      {"color": "#F0B90B"},
                    "bgcolor":  "#1E2329",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [30, 45], "color": "rgba(2,192,118,0.2)"},
                        {"range": [45, 55], "color": "rgba(240,185,11,0.1)"},
                        {"range": [55, 75], "color": "rgba(246,70,93,0.2)"},
                    ],
                    "threshold": {"value": 50, "line": {"color": "#848E9C", "width": 2}},
                },
                number={"suffix": "%", "font": {"color": "#F0B90B", "size": 28}},
            ))
            fig_dom.update_layout(
                height=220, paper_bgcolor="#0B0E11",
                margin=dict(l=20, r=20, t=40, b=10),
                font={"color": "#EAECEF"},
            )
            c_dom, c_fund = st.columns(2)
            c_dom.plotly_chart(fig_dom, use_container_width=True)

            # Funding rate gauge
            fig_fund = go.Figure(go.Indicator(
                mode="gauge+number",
                value=funding_val * 100,
                title={"text": "Funding Rate (×100)", "font": {"color": "#EAECEF", "size": 14}},
                gauge={
                    "axis":     {"range": [-5, 5], "tickcolor": "#848E9C"},
                    "bar":      {"color": "#1E90FF"},
                    "bgcolor":  "#1E2329",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [-5, -1], "color": "rgba(246,70,93,0.2)"},
                        {"range": [-1,  1], "color": "rgba(240,185,11,0.1)"},
                        {"range": [ 1,  5], "color": "rgba(2,192,118,0.2)"},
                    ],
                    "threshold": {"value": 0, "line": {"color": "#848E9C", "width": 2}},
                },
                number={"font": {"color": "#1E90FF", "size": 28}},
            ))
            fig_fund.update_layout(
                height=220, paper_bgcolor="#0B0E11",
                margin=dict(l=20, r=20, t=40, b=10),
                font={"color": "#EAECEF"},
            )
            c_fund.plotly_chart(fig_fund, use_container_width=True)
    else:
        st.info("Aucun signal actif. Les données macro apparaissent une fois l'agent démarré et les analyses satellites lancées.")

    st.divider()

    # ── Smart Money global ────────────────────────────────────────────
    st.markdown("### 🐳 Smart Money — Vue Globale")
    if opps:
        sm_rows = []
        for o in opps:
            wb = o.get("whale_buys", 0)
            ws = o.get("whale_sells", 0)
            nf = float(o.get("net_flow", 0))
            lbr = float(o.get("large_buy_ratio", 0.5))
            sm_sig = o.get("smart_money", "—")
            if wb or ws or nf:
                sm_rows.append({
                    "Paire":         o.get("symbol", "?"),
                    "Whale Achats":  wb,
                    "Whale Ventes":  ws,
                    "Net Flow ($)":  f"{nf:+,.0f}",
                    "Ratio gros achats": f"{lbr:.0%}",
                    "Signal":        sm_sig,
                    "VETO?":         "⚠ OUI" if ws >= 5 else "—",
                })
        if sm_rows:
            sm_df = pd.DataFrame(sm_rows)
            def _sm_color(v):
                if "BULL" in str(v) or "BUY" in str(v): return "color: #02C076"
                if "BEAR" in str(v) or "DUMP" in str(v): return "color: #F6465D"
                return ""
            st.dataframe(
                sm_df.style.applymap(_sm_color, subset=["Signal"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Données Smart Money disponibles après analyse satellite.")
    else:
        st.info("Lance l'agent pour voir les données Smart Money.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — OPPORTUNITÉS (avec données satellites complètes)
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Opportunités":
    st.markdown("## 🎯 Opportunités & Analyse Complète 10 Satellites")

    if opps:
        st.markdown(f"### {len(opps)} signaux actifs — 30+ métriques par paire")

        # ── Tableau de synthèse ────────────────────────────────────────
        rows = []
        for o in opps:
            rows.append({
                "Paire":         o.get("symbol", o.get("sym", "?")),
                "Score":         f"{float(o.get('score', 0)):.1f}",
                "Strat":         o.get("strategy", "?"),
                "Bonus Sat.":    f"{o.get('satellite_bonus', 0):+d}",
                "ML Proba":      f"{float(o.get('deep_proba', 0)):.0%}",
                "LLM Verdict":   o.get("council_verdict", "—"),
                "LLM Accord":    f"{o.get('council_agreement', 0):.0f}%",
                "Pattern":       o.get("pattern_signal", "—"),
                "VWAP":          o.get("vwap_signal", "—"),
                "CVD":           o.get("cvd_signal", "—"),
                "Orderflow":     o.get("orderflow_signal", "—"),
                "Smart Money":   o.get("smart_money", "—"),
                "Social /100":   f"{float(o.get('social_score', 50)):.0f}",
                "Funding":       o.get("funding_signal", "—"),
                "Volatilité":    o.get("volatility_label", "—"),
                "Gain Est.":     f"+{float(o.get('estimated_gain_eur', 0)):.1f}€",
                "Sizing":        o.get("risk_method", "Kelly"),
            })
        sat_df = pd.DataFrame(rows)

        def _color_score(val):
            try:
                v = float(val)
                if v >= 93:  return "color: #F0B90B"
                if v >= 78:  return "color: #02C076"
                return "color: #F6465D"
            except Exception: return ""

        def _color_verdict(val):
            if val in ("PRIME", "BUY"):  return "color: #02C076"
            if val in ("PASS", "RISKY"): return "color: #F6465D"
            return "color: #F0B90B"

        st.dataframe(
            sat_df.style
                .applymap(_color_score, subset=["Score"])
                .applymap(_color_verdict, subset=["LLM Verdict"]),
            use_container_width=True, hide_index=True,
        )
        st.divider()

        # ── Détail par opportunité ────────────────────────────────────
        st.markdown("### Analyse détaillée par paire")
        for o in opps:
            sym_o    = o.get("symbol", "?")
            sc_o     = float(o.get("score", 0))
            col_sc   = "#F0B90B" if sc_o >= 93 else ("#02C076" if sc_o >= 78 else "#F6465D")
            verdict  = o.get("council_verdict", "—")
            strat_o  = o.get("strategy", "?")
            gain_o   = float(o.get("estimated_gain_eur", 0))
            deep_o   = float(o.get("deep_proba", 0))

            with st.expander(
                f"**{sym_o}** · Score {sc_o:.1f}/100 · Strat {strat_o} · "
                f"Verdict {verdict} · +{gain_o:.1f}€ · ML {deep_o:.0%}",
                expanded=False,
            ):
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("Score Total", f"{sc_o:.1f}/100")
                d2.metric("Bonus Sat.", f"{o.get('satellite_bonus', 0):+d}")
                d3.metric("ML Proba", f"{deep_o:.0%}")
                d4.metric("LLM Verdict", verdict)
                d5.metric("Accord LLM", f"{o.get('council_agreement', 0):.0f}%")
                d6.metric("Modèles LLM", f"{o.get('council_models', 0)}/6")

                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📊 Micro", "📈 Orderflow", "💧 Liquidités",
                    "🌍 Cross-Asset", "💬 Social", "🐳 Whales", "🤖 ML & Patterns",
                ])

                with tab1:
                    r1, r2, r3 = st.columns(3)
                    r1.metric("VWAP Signal",  o.get("vwap_signal", "—"))
                    r2.metric("CVD Signal",   o.get("cvd_signal", "—"))
                    r3.metric("Buy Delta %",  f"{float(o.get('buy_delta_pct', 50)):.1f}%")

                with tab2:
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Signal",          o.get("orderflow_signal", "—"))
                    r2.metric("Bid/Ask Imbal.",  f"{float(o.get('bid_ask_imbalance', 1)):.2f}x")
                    r3.metric("Buy Pressure",    f"{float(o.get('buy_pressure', 50)):.1f}%")
                    r4.metric("Absorption",      "✅" if o.get("absorption") else "❌")
                    r5.metric("Stop Hunt / Ice", f"{'⚠' if o.get('stop_hunt') else '❌'} / {'🧊' if o.get('iceberg') else '❌'}")

                with tab3:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Mur haut (prix)", f"{float(o.get('liq_above_price', 0)):,.4f}")
                    r2.metric("Distance haut",   f"+{float(o.get('liq_above_pct', 0)):.2f}%")
                    r3.metric("Mur bas (prix)",  f"{float(o.get('liq_below_price', 0)):,.4f}")
                    r4.metric("Distance bas",    f"-{float(o.get('liq_below_pct', 0)):.2f}%")
                    st.caption(
                        f"Taille mur haut : {float(o.get('liq_above_size', 0)):,.0f}$ · "
                        f"Taille mur bas : {float(o.get('liq_below_size', 0)):,.0f}$"
                    )

                with tab4:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("BTC Dom.",      f"{float(o.get('btc_dominance', 0)):.1f}%")
                    r2.metric("ETH/BTC",       f"{float(o.get('eth_btc_ratio', 0)):.5f}")
                    r3.metric("ETH/BTC Trend", o.get("eth_btc_trend", "—"))
                    r4.metric("Funding",       f"{float(o.get('funding_rate', 0)):.4f}%")
                    st.caption(f"Signal funding : {o.get('funding_signal', '—')}")

                with tab5:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Reddit Trend",  o.get("reddit_trend", "—"))
                    r2.metric("Fear & Greed",  f"{float(o.get('fear_greed', 50)):.0f}/100")
                    r3.metric("CryptoPanic",   f"{float(o.get('cryptopanic_score', 50)):.0f}/100")
                    r4.metric("Social Score",  f"{float(o.get('social_score', 50)):.0f}/100")

                with tab6:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Whale Achats",  o.get("whale_buys", 0))
                    r2.metric("Whale Ventes",  o.get("whale_sells", 0))
                    r3.metric("Net Flow",      f"{float(o.get('net_flow', 0)):+,.0f}$")
                    r4.metric("Large Buy %",   f"{float(o.get('large_buy_ratio', 0.5)):.0%}")
                    if o.get("whale_sells", 0) >= 5:
                        st.error("⛔ VETO Smart Money : 5+ ventes baleine détectées")

                with tab7:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Pattern",        o.get("pattern_signal", "—"))
                    r2.metric("Conf. Pattern",  f"{float(o.get('pattern_confidence', 0)):.0%}")
                    r3.metric("Cible Pattern",  f"{float(o.get('pattern_target', 0)):,.4f}")
                    r4.metric("DeepPredictor",  f"{float(o.get('deep_proba', 0)):.0%}")
                    st.caption(
                        f"ATR ratio : {float(o.get('atr_ratio', 1)):.2f}x · "
                        f"ATR prédit : {float(o.get('atr_predicted', 0)):.4f} · "
                        f"Volatilité : {o.get('volatility_label', '—')} · "
                        f"Sizing : {o.get('risk_method', 'Kelly')} · "
                        f"LLM texte : {o.get('council_text', '—')}"
                    )
                    patterns = o.get("patterns_found", [])
                    if patterns:
                        st.markdown(
                            "Patterns : " + " ".join([f'<span class="badge-gold">{p}</span>' for p in patterns]),
                            unsafe_allow_html=True,
                        )
    else:
        st.info("Aucun signal actif. Lance l'agent pour alimenter cette vue en temps réel.")

    # Stratégies OCO
    st.divider()
    st.markdown("### Stratégies OCO v5.0")
    c1, c2, c3 = st.columns(3)
    for col, letter, name, desc, tp1, tp2, sl in [
        (c1, "A", "Breakout Momentum", "Breakout 48h + vol×2.5", "+2.5%", "+5.5%", "1.2×ATR"),
        (c2, "B", "Golden Cross",      "EMA7/EMA25 crossover",  "+2.0%", "+5.2%", "1.0×ATR"),
        (c3, "C", "News Momentum",     "News bullish + RSI 55-70","+3.0%","+6.0%","0.8×ATR"),
    ]:
        col.markdown(
            f'<div class="card">'
            f'<div style="color:#848E9C;font-size:11px">Stratégie {letter}</div>'
            f'<div style="font-size:16px;font-weight:700;color:#EAECEF">{name}</div>'
            f'<div style="font-size:12px;color:#848E9C;margin:4px 0">{desc}</div>'
            f'<span class="green">TP1 {tp1}</span> · <span class="green">TP2 {tp2}</span> · <span class="red">SL {sl}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Activité récente (audit)")
    if AUDIT_LOG.exists():
        lines = AUDIT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        records = []
        for line in reversed(lines[-300:]):
            try:
                r = json.loads(line)
                if r.get("event") in ("TRADE_EXECUTED", "POSITION_OPENED", "SIGNAL_SCORED"):
                    records.append({"ts": r.get("ts","")[:16], "event": r.get("event"), "payload": str(r.get("payload",""))[:80]})
                    if len(records) >= 10:
                        break
            except Exception:
                pass
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        else:
            st.info("Aucune activité d'exécution.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — TRADES
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Trades":
    st.markdown("## 📋 Historique des Trades")

    c1, c2, c3 = st.columns(3)
    sym_f    = c1.text_input("Filtrer symbole", "")
    strat_f  = c2.selectbox("Stratégie", ["Toutes","A","B","C","D","E"])
    status_f = c3.selectbox("Statut", ["Tous","Clôturés","Ouverts"])

    sql = "SELECT * FROM trades"
    conds = []
    if sym_f:
        conds.append(f"symbol LIKE '%{sym_f.upper()}%'")
    if strat_f != "Toutes":
        conds.append(f"strategy='{strat_f}'")
    if status_f == "Clôturés":
        conds.append("closed_at IS NOT NULL")
    elif status_f == "Ouverts":
        conds.append("closed_at IS NULL")
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY id DESC LIMIT 300"

    df_t = _q(sql)
    if not df_t.empty:
        def _c(v):
            if pd.isna(v): return ""
            return "color: #02C076" if float(v) >= 0 else "color: #F6465D"
        st.dataframe(
            df_t.style.applymap(_c, subset=["pnl_usdt"] if "pnl_usdt" in df_t.columns else []),
            use_container_width=True, height=420, hide_index=True,
        )
        ca, cb2 = st.columns(2)
        ca.download_button("📥 Export CSV", df_t.to_csv(index=False), "trades.csv", "text/csv")

        closed_t = df_t[df_t["closed_at"].notna()] if "closed_at" in df_t.columns else pd.DataFrame()
        if not closed_t.empty and "pnl_usdt" in closed_t.columns:
            wins_t = closed_t[closed_t["pnl_usdt"] >= 0]
            cb2.metric("Win rate filtré", f"{len(wins_t)/len(closed_t):.1%}")

        if "strategy" in df_t.columns and "pnl_usdt" in df_t.columns and not closed_t.empty:
            st.markdown("#### PnL par stratégie")
            by_strat = closed_t.groupby("strategy")["pnl_usdt"].agg(["sum","count","mean"])
            by_strat.columns = ["PnL total","Trades","Moy/trade"]
            by_strat = by_strat.reset_index()
            c_tab, c_chart = st.columns([1, 2])
            c_tab.dataframe(by_strat, use_container_width=True, hide_index=True)
            fig_strat = go.Figure(go.Bar(
                x=by_strat["strategy"], y=by_strat["PnL total"],
                marker_color=["#02C076" if v >= 0 else "#F6465D" for v in by_strat["PnL total"]],
            ))
            fig_strat.update_layout(
                height=220, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139"), yaxis=dict(gridcolor="#2B3139"),
                margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
            )
            c_chart.plotly_chart(fig_strat, use_container_width=True)
    else:
        st.info("Aucun trade correspondant.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.markdown("## 📈 Performance & Objectifs")

    st.markdown("### 🎯 Objectif 10–30 EUR/jour")
    daily_df = _q(
        "SELECT date(closed_at) AS day, SUM(pnl_usdt) AS daily_pnl, COUNT(*) AS trades "
        "FROM trades WHERE closed_at IS NOT NULL "
        "GROUP BY day ORDER BY day DESC LIMIT 30"
    )
    if not daily_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        avg_pnl   = daily_df["daily_pnl"].mean()
        best_day  = daily_df["daily_pnl"].max()
        worst_day = daily_df["daily_pnl"].min()
        days_in   = (daily_df["daily_pnl"].between(10, 30)).sum()
        c1.metric("PnL moy/jour (30j)", f"{avg_pnl:+.2f}€",
                  delta="✅ cible" if 10 <= avg_pnl <= 30 else "⚠ hors cible",
                  delta_color="normal" if 10 <= avg_pnl <= 30 else "inverse")
        c2.metric("Meilleur jour", f"{best_day:+.2f}€")
        c3.metric("Pire jour", f"{worst_day:+.2f}€")
        c4.metric("Jours dans cible", f"{days_in}/30")

        daily_df2 = daily_df.sort_values("day")
        colors = ["#02C076" if v >= 0 else "#F6465D" for v in daily_df2["daily_pnl"]]
        fig_d = go.Figure(go.Bar(x=daily_df2["day"], y=daily_df2["daily_pnl"], marker_color=colors))
        fig_d.add_hrect(y0=10, y1=30, fillcolor="rgba(2,192,118,0.07)", line_width=0)
        fig_d.update_layout(
            height=250, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
            xaxis=dict(gridcolor="#2B3139"), yaxis=dict(gridcolor="#2B3139"),
            margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
        )
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("Pas assez de données journalières.")

    st.divider()
    st.markdown("### Métriques globales")
    perf = _q("SELECT * FROM performance ORDER BY id DESC LIMIT 1")
    if not perf.empty:
        row = perf.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate",      f"{float(row.get('win_rate',0)):.1%}")
        c1.metric("Total Trades",  int(row.get("total_trades",0)))
        c2.metric("Profit Factor", f"{float(row.get('profit_factor',0)):.2f}")
        c2.metric("Avg Win",       f"{float(row.get('avg_win',0)):+.2f}€")
        c3.metric("Max Drawdown",  f"{float(row.get('max_drawdown',0)):.1%}")
        c3.metric("Avg Loss",      f"{float(row.get('avg_loss',0)):+.2f}€")
        c4.metric("PnL Total",     f"{float(row.get('total_pnl',0)):+.2f}€")
        c4.metric("Capital",       f"{float(row.get('capital_usdt',200)):.0f}€")
    else:
        st.info("Aucun snapshot de performance. Données disponibles après les premiers trades.")

    st.divider()
    st.markdown("### Projection vers 10 000€")
    try:
        import math
        if not perf.empty and float(perf.iloc[0].get("win_rate", 0)) > 0:
            wr       = float(perf.iloc[0]["win_rate"])
            avg_w    = float(perf.iloc[0].get("avg_win", 5.0))
            avg_l    = abs(float(perf.iloc[0].get("avg_loss", -2.0)))
            edge     = wr * avg_w - (1 - wr) * avg_l
            if edge > 0 and cap > 0:
                days_to = math.ceil(math.log(10000 / cap) / math.log(1 + edge / cap))
                st.metric("Jours estimés pour 10 000€", f"~{days_to} jours",
                          delta=f"Edge : +{edge:.2f}€/trade")
            else:
                st.warning("Edge négatif — ajuste les paramètres.")
        else:
            st.info("Lance des trades pour calculer la projection.")
    except Exception:
        st.info("Données insuffisantes.")

    if OPTIMISED_FILE.exists():
        st.divider()
        st.markdown("### Paramètres optimisés (Optuna)")
        data = json.loads(OPTIMISED_FILE.read_text())
        st.caption(f"Mis à jour : {data.get('updated_at','—')[:16]}")
        st.json(data.get("params", {}))


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — APPRENTISSAGE
# ══════════════════════════════════════════════════════════════════════════
elif page == "🧠 Apprentissage":
    st.markdown("## 🧠 Apprentissage Automatique")

    tab1, tab2, tab3, tab4 = st.tabs(["Win Rate", "AdaptiveScorer", "Régimes", "Erreurs"])

    with tab1:
        st.markdown("#### Win rate par symbole (30 jours)")
        wr_q = _q(
            "SELECT symbol, COUNT(*) AS trades, "
            "SUM(CASE WHEN pnl_usdt>=0 THEN 1 ELSE 0 END) AS wins, "
            "AVG(pnl_usdt) AS avg_pnl "
            "FROM trades WHERE closed_at IS NOT NULL "
            "GROUP BY symbol ORDER BY trades DESC"
        )
        if not wr_q.empty:
            wr_q["win_rate"] = wr_q["wins"] / wr_q["trades"]
            st.dataframe(wr_q, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée.")

    with tab2:
        st.markdown("#### Poids adaptatifs (AdaptiveScorer)")
        if WEIGHTS_FILE.exists():
            w_data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
            st.caption(f"Mis à jour : {w_data.get('updated_at','—')[:16]} · {w_data.get('n_trades',0)} trades")
            changes = w_data.get("changes", [])
            if changes:
                st.markdown("**Derniers ajustements :**")
                for ch in changes:
                    st.markdown(f"- {ch}")
            weights = w_data.get("weights", {})
            if weights:
                w_df = pd.DataFrame([{"Critère": k, "Poids": v} for k, v in sorted(weights.items(), key=lambda x: -x[1])])
                fig_w = go.Figure(go.Bar(
                    x=w_df["Critère"], y=w_df["Poids"],
                    marker_color="#F0B90B", opacity=0.85,
                ))
                fig_w.update_layout(
                    height=250, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                    xaxis=dict(gridcolor="#2B3139", tickangle=-30),
                    yaxis=dict(gridcolor="#2B3139"),
                    margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
                )
                st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Poids adaptatifs disponibles après le premier run nocturne (03h00 UTC).")

    with tab3:
        st.markdown("#### Performance par régime de marché")
        regime_q = _q(
            "SELECT regime, COUNT(*) AS trades, "
            "SUM(CASE WHEN pnl_usdt>=0 THEN 1 ELSE 0 END) AS wins, "
            "AVG(pnl_usdt) AS avg_pnl "
            "FROM trades WHERE closed_at IS NOT NULL AND regime IS NOT NULL "
            "GROUP BY regime ORDER BY trades DESC"
        )
        if not regime_q.empty:
            regime_q["win_rate"] = regime_q["wins"] / regime_q["trades"]
            st.dataframe(regime_q, use_container_width=True, hide_index=True)
        else:
            st.info("Données régime indisponibles.")

    with tab4:
        st.markdown("#### Analyse des erreurs (pertes)")
        mistakes_q = _q(
            "SELECT strategy, exit_reason, COUNT(*) AS n, AVG(pnl_usdt) AS avg_pnl "
            "FROM trades WHERE pnl_usdt<0 AND closed_at IS NOT NULL "
            "GROUP BY strategy, exit_reason ORDER BY n DESC LIMIT 20"
        )
        if not mistakes_q.empty:
            st.dataframe(mistakes_q, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune perte analysée pour l'instant.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 8 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Simulation":
    st.markdown("## 🔬 Simulation Historique & Backtests")

    sim_df = _q(
        "SELECT symbol, strategy, timeframe, start_date, end_date, "
        "total_return, cagr, sharpe, max_drawdown, win_rate, nb_trades, "
        "quality_score, created_at FROM historical_simulations ORDER BY id DESC LIMIT 100"
    )
    if not sim_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulations",    len(sim_df))
        c2.metric("Sharpe moyen",   f"{sim_df['sharpe'].mean():.2f}")
        c3.metric("Win rate moyen", f"{sim_df['win_rate'].mean():.1%}")
        c4.metric("Qualité moy.",   f"{sim_df['quality_score'].mean():.0f}/100")
        st.divider()

        def _qual(v):
            if v >= 70: return "color: #02C076"
            if v >= 50: return "color: #F0B90B"
            return "color: #F6465D"

        st.dataframe(
            sim_df.style.applymap(_qual, subset=["quality_score"]),
            use_container_width=True, height=350, hide_index=True,
        )

        if len(sim_df) > 3:
            st.markdown("#### Sharpe vs Rendement")
            fig_sc = go.Figure(go.Scatter(
                x=sim_df["sharpe"], y=sim_df["total_return"],
                mode="markers",
                marker=dict(color="#F0B90B", size=8, opacity=0.8),
                text=sim_df.get("symbol", ""),
            ))
            fig_sc.update_layout(
                height=280, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139", title="Sharpe"),
                yaxis=dict(gridcolor="#2B3139", title="Rendement %"),
                margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Aucune simulation. Lance : `python scripts/run_historical_simulation.py`")
        st.markdown("""
```bash
# Backtest rapide
python scripts/quick_backtest.py --strategy B --days 365

# Simulation complète
python scripts/run_historical_simulation.py --symbol BTCUSDT --strategy all --years 5

# Via Makefile
make backtest   # synthétique (rapide)
make sim        # historique réel
```""")

    st.divider()
    st.markdown("### Paper gate")
    days_l = paper_days_left()
    if ACTIVATION_FILE.exists():
        dt_str = ACTIVATION_FILE.read_text().strip()
        st.metric("Démarré le", dt_str[:10])
        st.progress(min(1.0, (7 - days_l) / 7), text=f"{7 - days_l}/7 jours écoulés · {days_l}j restants")
    else:
        st.info("Paper gate non initialisé.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 9 — CONSEIL LLM
# ══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Conseil LLM":
    st.markdown("## 🤖 Conseil LLM — 6 Modèles en Parallèle")

    # ── Statut clés API ───────────────────────────────────────────────
    st.markdown("### Statut des clés API")
    api_keys = {
        "Grok (xAI)":      "GROK_API_KEY",
        "Gemini":          "GEMINI_API_KEY",
        "OpenAI":          "OPENAI_API_KEY",
        "DeepSeek":        "DEEPSEEK_API_KEY",
        "Kimi":            "KIMI_API_KEY",
        "OpenRouter":      "OPENROUTER_API_KEY",
        "Tavily":          "TAVILY_API_KEY",
        "CoinGlass":       "COINGLASS_API_KEY",
        "WhaleAlert":      "WHALE_ALERT_API_KEY",
    }
    cols_api = st.columns(3)
    for i, (name, env_var) in enumerate(api_keys.items()):
        val = os.getenv(env_var, "")
        icon = "🟢" if val else "⚫"
        label = "Actif" if val else "Non configuré"
        cols_api[i % 3].markdown(
            f'<div class="card" style="padding:8px">'
            f'{icon} <strong>{name}</strong><br>'
            f'<span style="color:#848E9C;font-size:11px">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Derniers verdicts depuis les opportunités en cours ────────────
    if opps:
        has_council = any(o.get("council_verdict") for o in opps)
        if has_council:
            st.markdown("### Verdicts LLM en temps réel (signaux actifs)")
            live_rows = []
            for o in opps:
                if o.get("council_verdict"):
                    live_rows.append({
                        "Paire":        o.get("symbol", "?"),
                        "Score":        f"{float(o.get('score', 0)):.1f}",
                        "Verdict":      o.get("council_verdict", "—"),
                        "Accord %":     f"{o.get('council_agreement', 0):.0f}%",
                        "Modèles":      f"{o.get('council_models', 0)}/6",
                        "Texte":        o.get("council_text", "—")[:60],
                        "Gain Est.":    f"+{float(o.get('estimated_gain_eur', 0)):.1f}€",
                        "ML Proba":     f"{float(o.get('deep_proba', 0)):.0%}",
                    })
            if live_rows:
                ldf = pd.DataFrame(live_rows)
                def _cv2(v):
                    if v in ("PRIME","BUY"):  return "color: #02C076"
                    if v in ("PASS","RISKY"): return "color: #F6465D"
                    return "color: #F0B90B"
                st.dataframe(
                    ldf.style.applymap(_cv2, subset=["Verdict"]),
                    use_container_width=True, hide_index=True,
                )
            st.divider()

    # ── Poids adaptatifs ──────────────────────────────────────────────
    st.markdown("### Poids AdaptiveScorer")
    if WEIGHTS_FILE.exists():
        w_data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        c_inf, c_chart = st.columns([1, 2])
        c_inf.metric("Trades analysés", w_data.get("n_trades", 0))
        c_inf.caption(f"Mis à jour : {w_data.get('updated_at','—')[:16]}")
        changes = w_data.get("changes", [])
        if changes:
            c_inf.markdown("**Derniers ajustements :**")
            for ch in changes[-5:]:
                c_inf.markdown(f"- {ch}")
        weights = w_data.get("weights", {})
        if weights:
            w_df = pd.DataFrame([{"Critère": k, "Poids": v} for k, v in sorted(weights.items(), key=lambda x: -x[1])])
            fig_ww = go.Figure(go.Bar(x=w_df["Critère"], y=w_df["Poids"], marker_color="#F0B90B", opacity=0.85))
            fig_ww.update_layout(
                height=200, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139", tickangle=-25, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#2B3139"),
                margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
            )
            c_chart.plotly_chart(fig_ww, use_container_width=True)
    else:
        st.info("Les poids adaptatifs apparaîtront après le premier run nocturne (03h00 UTC).")

    st.divider()

    # ── Verdicts récents depuis audit ─────────────────────────────────
    st.markdown("### Historique verdicts LLM (audit log)")
    if AUDIT_LOG.exists():
        lines   = AUDIT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        records = []
        for line in reversed(lines[-500:]):
            try:
                r = json.loads(line)
                p = r.get("payload", {})
                if p.get("council_verdict"):
                    records.append({
                        "Heure":       r.get("ts", "")[:16],
                        "Paire":       p.get("symbol", "?"),
                        "Score":       p.get("score", 0),
                        "Verdict":     p.get("council_verdict", "?"),
                        "Accord %":    f"{p.get('council_agreement', 0):.0f}%",
                        "Modèles":     f"{p.get('council_models', 0)}/6",
                        "ML Proba":    f"{float(p.get('deep_proba', 0)):.0%}",
                        "Texte":       p.get("council_text", "—")[:40],
                        "PnL réel":    p.get("pnl_usdt", "—"),
                    })
                    if len(records) >= 25:
                        break
            except Exception:
                pass

        if records:
            vdf = pd.DataFrame(records)

            def _cv(v):
                if v in ("PRIME","BUY"):  return "color: #02C076"
                if v in ("PASS","RISKY"): return "color: #F6465D"
                return "color: #F0B90B"

            st.dataframe(
                vdf.style.applymap(_cv, subset=["Verdict"]),
                use_container_width=True, hide_index=True, height=350,
            )

            st.markdown("#### Distribution des verdicts")
            vc = vdf["Verdict"].value_counts().reset_index()
            vc.columns = ["Verdict", "N"]
            colors_v = {"PRIME": "#F0B90B", "BUY": "#02C076", "PASS": "#848E9C",
                        "RISKY": "#F6465D", "UNCERTAIN": "#2B3139"}
            fig_vc = go.Figure(go.Bar(
                x=vc["Verdict"], y=vc["N"],
                marker_color=[colors_v.get(v, "#2B3139") for v in vc["Verdict"]],
            ))
            fig_vc.update_layout(
                height=200, paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
                xaxis=dict(gridcolor="#2B3139"), yaxis=dict(gridcolor="#2B3139"),
                margin=dict(l=0,r=0,t=10,b=0), font_color="#EAECEF",
            )
            st.plotly_chart(fig_vc, use_container_width=True)

            # Accord moyen par verdict
            try:
                if "Accord %" in vdf.columns:
                    vdf["accord_num"] = vdf["Accord %"].str.replace("%", "").astype(float)
                    accord_by_v = vdf.groupby("Verdict")["accord_num"].mean().reset_index()
                    accord_by_v.columns = ["Verdict", "Accord moyen %"]
                    st.markdown("#### Accord moyen par verdict")
                    st.dataframe(accord_by_v, use_container_width=True, hide_index=True)
            except Exception:
                pass
        else:
            st.info("Aucun verdict LLM dans l'audit log pour l'instant.")
    else:
        st.info("Audit log non disponible. Démarre l'agent pour générer des logs.")

    st.divider()

    # ── Description des 6 modèles ─────────────────────────────────────
    st.markdown("### Les 6 modèles du Conseil")
    models_info = [
        ("Grok (xAI)",    "GROK_API_KEY",      "grok-3-fast",       1.2, "Excellent en analyse sentiment crypto"),
        ("Gemini",        "GEMINI_API_KEY",     "gemini-2.0-flash",  1.2, "Forte analyse cross-asset"),
        ("OpenAI",        "OPENAI_API_KEY",     "gpt-4o-mini",       1.3, "Meilleure précision — poids le plus élevé"),
        ("DeepSeek",      "DEEPSEEK_API_KEY",   "deepseek-chat",     1.0, "Modèle économique, bonne analyse technique"),
        ("Kimi",          "KIMI_API_KEY",       "moonshot-v1-8k",    1.0, "Spécialisé marchés asiatiques"),
        ("OpenRouter",    "OPENROUTER_API_KEY", "mistral-7b",        0.8, "Modèle de secours"),
    ]
    m_cols = st.columns(3)
    for i, (name, key, model_id, weight, desc) in enumerate(models_info):
        val = os.getenv(key, "")
        icon = "🟢" if val else "⚫"
        m_cols[i % 3].markdown(
            f'<div class="card">'
            f'{icon} <strong>{name}</strong> <span style="color:#848E9C;font-size:10px">(poids {weight}x)</span><br>'
            f'<span style="color:#F0B90B;font-size:11px">{model_id}</span><br>'
            f'<span style="color:#848E9C;font-size:11px">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Auto-refresh ──────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
