"""
CENTINA Pro — Interface Web Binance Pro (dark theme)
Dashboard temps réel connecté à l'agent via data/bridge_state.json
"""
import datetime
import json
import pathlib
import time

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="CENTINA Pro", page_icon="🤖", layout="wide")
st.markdown("""<style>
.stApp{background:#0B0E11;color:#EAECEF}
section[data-testid="stSidebar"]{background:#161A1E}
.stMetric{background:#1E2329;border-radius:8px;padding:12px}
.stMetric label{color:#848E9C!important;font-size:12px}
[data-testid="stMetricValue"]{color:#F0B90B;font-size:22px!important}
.agent-msg{background:#1E2329;border-left:3px solid #F0B90B;
           padding:10px;border-radius:4px;margin:4px 0;font-size:12px}
.user-msg{background:#2B3139;border-left:3px solid #02C076;
          padding:10px;border-radius:4px;margin:4px 0;font-size:12px}
.prime-alert{background:#2D1F0E;border:1px solid #F0B90B;
             border-radius:6px;padding:10px;margin:4px 0}
.pos-card{background:#1E2329;border-radius:8px;padding:12px;
          border:0.5px solid #2B3139;margin-bottom:8px}
.green{color:#02C076!important;font-weight:600}
.red{color:#F6465D!important;font-weight:600}
.yellow{color:#F0B90B!important;font-weight:600}
div.stButton>button{width:100%;border-radius:4px;font-weight:500;
                    border:0.5px solid #2B3139;background:#1E2329;color:#EAECEF}
div.stButton>button:hover{background:#2B3139}
</style>""", unsafe_allow_html=True)

BRIDGE_FILE = pathlib.Path("data/bridge_state.json")
CMD_QUEUE   = pathlib.Path("data/cmd_queue.json")


# ── Helpers ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    if BRIDGE_FILE.exists():
        try:
            return json.loads(BRIDGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "mode": "PAPER", "cb_level": "NORMAL", "capital_usdt": 200,
        "pnl_day_eur": 0, "open_positions": 0,
        "positions": [], "opportunities": [], "events": [], "paper_days_left": 7,
    }


def write_cmd(cmd: dict) -> None:
    cmd["ts"] = datetime.datetime.utcnow().isoformat()
    CMD_QUEUE.parent.mkdir(exist_ok=True)
    queue: list = []
    if CMD_QUEUE.exists():
        try:
            queue = json.loads(CMD_QUEUE.read_text(encoding="utf-8"))
        except Exception:
            queue = []
    queue.append(cmd)
    CMD_QUEUE.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def get_klines(symbol: str, interval: str = "1h", limit: int = 100) -> list[dict]:
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


# ── Load state ────────────────────────────────────────────────────────────

state = load_state()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CENTINA Pro")
    st.caption("OMNI-QUANT v5.0")
    st.divider()

    mode     = state.get("mode", "PAPER")
    cb_name  = state.get("cb_level", state.get("cb_name", "NORMAL"))
    cap      = float(state.get("capital_usdt", state.get("capital", 200)))
    pnl      = float(state.get("pnl_day_eur",  state.get("daily_pnl", 0)))
    n_pos    = int(state.get("open_positions", len(state.get("positions", []))))

    mode_icon = {"PAPER": "🟡", "ADVISOR": "🟢", "SEMI": "🟢", "AUTO": "🟢"}.get(mode, "🔴")
    st.markdown(f"{mode_icon} **{mode}**")

    col_a, col_b = st.columns(2)
    col_a.metric("Capital", f"{cap:.0f} €")
    pnl_delta = f"{pnl:+.2f}€"
    col_b.metric("PnL jour", pnl_delta)
    st.metric("Positions", f"{n_pos} / 3")

    target = 30.0
    prog   = min(1.0, max(0.0, pnl / target))
    st.progress(prog, text=f"Objectif : {pnl:.1f} / {target:.0f} EUR")

    paper_left = int(state.get("paper_days_left", 0))
    if paper_left > 0:
        st.warning(f"⏳ Paper gate : {paper_left}j restants")

    st.divider()
    st.markdown("**Mode trading**")
    trading_mode = st.radio(
        "mode_sel",
        ["🧠 ADVISOR", "⚡ SEMI", "🤖 AUTO"],
        index=0,
        label_visibility="collapsed",
    )
    if st.button("APPLIQUER MODE"):
        mode_val = trading_mode.split(" ")[-1]
        write_cmd({"cmd": "SET_MODE", "value": mode_val})
        st.success(f"Mode {mode_val} envoyé")

    st.divider()
    cb_icons = {
        "NORMAL": "🟢", "ALERT": "🟠", "RESTRICTED": "🔴",
        "STOPPED": "⛔", "TERMINATED": "💀",
    }
    st.markdown(f"**Circuit Breaker** : {cb_icons.get(str(cb_name), '❓')} {cb_name}")
    if str(cb_name) in ("ALERT", "RESTRICTED"):
        if st.button("RESET CB"):
            write_cmd({"cmd": "CB_RESET"})

    st.divider()
    pos_pct   = st.slider("Mise / trade (%)", 1, 5, 3)
    score_min = st.slider("Score min",        70, 95, 78)
    if st.button("APPLIQUER PARAMS"):
        write_cmd({"cmd": "SET_PARAMS", "position_pct": pos_pct / 100, "score_min": score_min})

    st.divider()
    if st.button("🛑 FERME TOUT", type="primary"):
        write_cmd({"cmd": "CLOSE_ALL"})
    c_p, c_r = st.columns(2)
    if c_p.button("⏸ PAUSE"):
        write_cmd({"cmd": "PAUSE"})
    if c_r.button("▶ REPRISE"):
        write_cmd({"cmd": "RESUME"})

    auto_refresh = st.checkbox("Auto-refresh (3s)", value=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([2.2, 1.5, 1.3])

with col1:
    opps    = state.get("opportunities", [])
    symbols = [o.get("symbol", o.get("sym", "")) for o in opps] if opps else ["BTCUSDT", "ETHUSDT"]
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT"]
    sel_sym = st.selectbox("Paire", symbols, key="sym_sel")
    tf = st.radio("Timeframe", ["15m", "1h", "4h"], horizontal=True, key="tf_sel")

    klines = get_klines(sel_sym, tf, 100)
    if klines:
        opens  = [k["o"] for k in klines]
        highs  = [k["h"] for k in klines]
        lows   = [k["l"] for k in klines]
        closes = [k["c"] for k in klines]
        times  = [datetime.datetime.utcfromtimestamp(k["t"] / 1000) for k in klines]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=times, open=opens, high=highs, low=lows, close=closes,
            name=sel_sym,
            increasing_line_color="#02C076",
            decreasing_line_color="#F6465D",
        ))
        for period, color in [(7, "#F0B90B"), (25, "#1E90FF"), (99, "#FF8C00"), (200, "#FF4444")]:
            if len(closes) >= period:
                ema_vals = ema_calc(closes, period)
                fig.add_trace(go.Scatter(
                    x=times, y=ema_vals,
                    name=f"EMA{period}",
                    line=dict(color=color, width=1),
                    opacity=0.8,
                ))

        # Niveaux des positions ouvertes
        for pos in state.get("positions", []):
            if pos.get("symbol") == sel_sym:
                for field, color, label in [
                    ("sl", "#F6465D", "SL"),
                    ("tp1", "#02C076", "TP1"),
                    ("tp2", "#1E90FF", "TP2"),
                ]:
                    if field in pos:
                        fig.add_hline(
                            y=pos[field], line_color=color, line_dash="dot",
                            annotation_text=label, annotation_position="right",
                        )

        fig.update_layout(
            height=380,
            paper_bgcolor="#0B0E11", plot_bgcolor="#1E2329",
            xaxis=dict(gridcolor="#2B3139"),
            yaxis=dict(gridcolor="#2B3139"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="#1E2329", font=dict(color="#EAECEF", size=10)),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Connexion Binance en cours…")

    # Score & conseil LLM pour la paire sélectionnée
    cur_opp = next(
        (o for o in opps if o.get("symbol", o.get("sym", "")) == sel_sym), None
    )
    if cur_opp:
        sc      = float(cur_opp.get("score", 0))
        verdict = cur_opp.get("verdict", cur_opp.get("strategy", "?"))
        strat   = cur_opp.get("strategy", cur_opp.get("suggested_strategy", "?"))
        gain    = float(cur_opp.get("estimated_gain_eur", 0))
        dur     = cur_opp.get("estimated_hours", "?")
        col     = "yellow" if sc >= 93 else ("green" if sc >= 78 else "red")
        st.markdown(
            f"**Score CENTINA :** <span class='{col}'>{sc:.1f}/100</span> — "
            f"**{verdict}** | Stratégie **{strat}** | "
            f"Gain NET est. **+{gain:.1f} EUR** | Durée **{dur}h**",
            unsafe_allow_html=True,
        )
        council = cur_opp.get("council_text", "")
        if council:
            st.caption(f"🤖 Conseil LLM : {council}")

with col2:
    st.markdown("### 💬 CENTINA Advisor")

    events = state.get("events", state.get("status_log", []))
    # events peut être liste de dicts ou liste de strings
    for ev in list(reversed(events))[:15]:
        if isinstance(ev, dict):
            ts      = ev.get("ts", "")
            txt     = ev.get("text", "")
            ev_type = ev.get("type", "")
        else:
            ts, txt, ev_type = "", str(ev), ""

        if ev_type == "PRIME":
            st.markdown(
                f'<div class="prime-alert">🚨 <strong>{ts}</strong><br>{txt}</div>',
                unsafe_allow_html=True,
            )
        elif ev_type in ("USER_CMD", "USER"):
            st.markdown(
                f'<div class="user-msg">👤 <strong>{ts}</strong><br>{txt}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="agent-msg">🤖 <strong>{ts}</strong><br>{txt}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    cmd_text = st.text_input(
        "cmd_input",
        placeholder="GO SOLUSDT · PASSE · STATUT · RAPPORT",
        label_visibility="collapsed",
    )
    if st.button("ENVOYER ↗") and cmd_text.strip():
        write_cmd({"cmd": "USER_INPUT", "text": cmd_text.strip()})
        st.rerun()

    btn_cols = st.columns(5)
    for i, (label, cmd) in enumerate([
        ("GO", "GO"), ("PASSE", "PASSE"), ("STATUT", "STATUT"),
        ("RAPPORT", "RAPPORT"), ("OPPS", "OPPS"),
    ]):
        if btn_cols[i].button(label, key=f"btn_{cmd}"):
            write_cmd({"cmd": "USER_INPUT", "text": cmd})
            st.rerun()

with col3:
    st.markdown("### 📊 Positions")
    positions = state.get("positions", [])
    if not positions:
        st.markdown(
            '<p style="color:#848E9C;text-align:center;margin-top:20px">Aucune position</p>',
            unsafe_allow_html=True,
        )
    for pos in positions:
        pnl_pos   = float(pos.get("pnl_eur", 0))
        dur       = float(pos.get("elapsed_h", pos.get("duration_h", 0)))
        pnl_color = "green" if pnl_pos >= 0 else "red"
        dur_color = "red" if dur > 3 else ("yellow" if dur > 2 else "green")
        tp1_icon  = "✅" if pos.get("tp1_done", pos.get("tp1_hit")) else "⏳"
        tp2_icon  = "✅" if pos.get("tp2_done", pos.get("tp2_hit")) else "⏳"
        sym       = pos.get("symbol", "?")
        entry_p   = pos.get("entry", pos.get("entry_price", 0))
        st.markdown(
            f'<div class="pos-card">'
            f'<strong>{sym}</strong> '
            f'<span style="background:#2B3139;padding:2px 6px;border-radius:3px;font-size:11px">'
            f'Strat {pos.get("strategy", "?")}</span><br>'
            f'<span style="font-size:12px;color:#848E9C">Entrée : {entry_p:.4f}</span><br>'
            f'<span class="{pnl_color}">PnL : {pnl_pos:+.2f} EUR</span> | '
            f'<span class="{dur_color}">{dur:.1f}h</span><br>'
            f'TP1 {tp1_icon} | TP2 {tp2_icon}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, dur / 5), text=f"{dur:.1f}h / 5h max")
        if st.button(f"FERMER {sym}", key=f"close_{sym}"):
            write_cmd({"cmd": "CLOSE_POSITION", "symbol": sym})

    st.markdown("### 🏆 Top Signaux")
    for opp in opps[:5]:
        sc_o   = float(opp.get("score", 0))
        gain_o = float(opp.get("estimated_gain_eur", 0))
        sym_o  = opp.get("symbol", opp.get("sym", "?"))
        strat_o = opp.get("strategy", opp.get("suggested_strategy", "?"))
        col_o  = "#F0B90B" if sc_o >= 93 else "#02C076"
        c_lbl, c_btn = st.columns([3, 1])
        c_lbl.markdown(
            f"**{sym_o}** | "
            f"<span style='color:{col_o}'>{sc_o:.0f}/100</span> | "
            f"Strat {strat_o} | +{gain_o:.1f}€",
            unsafe_allow_html=True,
        )
        if c_btn.button("GO", key=f"go_{sym_o}"):
            write_cmd({"cmd": "EXECUTE", "symbol": sym_o})

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(3)
    st.rerun()
