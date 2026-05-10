"""
MODULE 3 — Interface web Binance-style (Streamlit dark theme).
Lancer : streamlit run dashboard/web_interface.py
"""
from __future__ import annotations

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CENTINA OMNI-QUANT v5.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme Binance-style ─────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0B0E11; color: #E8E8E8; }
  section[data-testid="stSidebar"] { background-color: #0D1117; }
  .metric-card {
    background: #161A1E; border-radius: 8px; padding: 16px;
    border: 1px solid #2B3139; margin-bottom: 8px;
  }
  .metric-label { color: #848E9C; font-size: 12px; }
  .metric-value { font-size: 24px; font-weight: 700; }
  .green { color: #0ECB81; }
  .red   { color: #F6465D; }
  .yellow{ color: #F0B90B; }
  div[data-testid="stMetric"] { background: #161A1E; border-radius: 8px; padding: 12px; }
  .stDataFrame { background: #161A1E; }
  button[kind="primary"] { background: #F0B90B; color: #0B0E11; border: none; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ─────────────────────────────────────────────────────────

def _load_db():
    try:
        from src.database.cortex_db import CortexDB
        return CortexDB()
    except Exception:
        return None


def _safe_trades(db) -> pd.DataFrame:
    if db is None:
        return pd.DataFrame()
    try:
        trades = db.get_all_trades()
        if not trades:
            return pd.DataFrame()
        cols = ["id", "symbol", "entry_price", "exit_price", "qty",
                "pnl_usdt", "pnl_pct", "strategy", "exit_reason",
                "tp1_hit", "tp2_hit", "paper", "opened_at", "closed_at"]
        rows = []
        for t in trades:
            rows.append({c: getattr(t, c, None) for c in cols})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Sidebar navigation ───────────────────────────────────────────────────

st.sidebar.markdown("# 📈 CENTINA v5.0")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🎯 Opportunités", "📋 Trades", "📈 Performance", "🧠 Apprentissage", "🔬 Simulation"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mode** : `{os.getenv('DEFAULT_MODE','PAPER')}`")
st.sidebar.markdown(f"**Capital** : `{os.getenv('CAPITAL_USDT','200')} USDT`")
if st.sidebar.button("🔄 Actualiser"):
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color:#848E9C'>Auto-refresh toutes les 5s</small>", unsafe_allow_html=True)


# ── Auto-refresh ─────────────────────────────────────────────────────────
# Streamlit native auto-rerun (5s)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="main_refresh")
except ImportError:
    pass


db = _load_db()
df = _safe_trades(db)


# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard CENTINA")

    # KPIs row
    capital = float(os.getenv("CAPITAL_USDT", "200"))
    closed   = df[df["closed_at"].notna()] if not df.empty else pd.DataFrame()
    pnl_day  = closed["pnl_usdt"].sum() if not closed.empty else 0.0
    pnl_total = closed["pnl_usdt"].sum() if not closed.empty else 0.0
    n_trades  = len(closed)
    win_rate  = (closed["pnl_usdt"] > 0).mean() * 100 if not closed.empty else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Capital", f"{capital:.0f} USDT")
    col2.metric("PnL Jour", f"{pnl_day:+.2f} €", delta=f"{pnl_day/capital*100:+.2f}%" if capital else None)
    col3.metric("PnL Total", f"{pnl_total:+.2f} €")
    col4.metric("Trades", str(n_trades))
    col5.metric("Win Rate", f"{win_rate:.1f}%")
    col6.metric("Objectif", "10-30 €/j")

    st.markdown("---")

    # Equity curve
    if not closed.empty and "closed_at" in closed.columns:
        try:
            import plotly.graph_objects as go
            eq = closed.sort_values("closed_at").copy()
            eq["equity"] = capital + eq["pnl_usdt"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq["closed_at"], y=eq["equity"],
                mode="lines+markers",
                line=dict(color="#0ECB81", width=2),
                fill="tozeroy", fillcolor="rgba(14,203,129,0.08)",
                name="Equity",
            ))
            fig.update_layout(
                paper_bgcolor="#0B0E11", plot_bgcolor="#0D1117",
                font_color="#E8E8E8", title="Courbe d'équité",
                xaxis=dict(gridcolor="#2B3139"), yaxis=dict(gridcolor="#2B3139"),
                height=300, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Courbe indisponible : {e}")
    else:
        st.info("Aucun trade clôturé — courbe d'équité vide.")

    # Progress bar (10-30€ target)
    target = 30.0
    pct = min(1.0, pnl_day / target) if target > 0 else 0
    col_p, col_v = st.columns([3, 1])
    col_p.progress(pct)
    col_v.markdown(f"**{pnl_day:.2f} / {target:.0f} EUR** ({pct*100:.0f}%)")

    # Open positions
    st.markdown("### Positions ouvertes")
    try:
        from src.database.cortex_db import CortexDB
        open_trades = df[df["closed_at"].isna()] if not df.empty else pd.DataFrame()
        if not open_trades.empty:
            st.dataframe(open_trades[["symbol","entry_price","qty","strategy","paper"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Aucune position ouverte.")
    except Exception:
        st.info("Aucune position ouverte.")


# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — OPPORTUNITÉS
# ════════════════════════════════════════════════════════════════════════
elif page == "🎯 Opportunités":
    st.markdown("## 🎯 Opportunités & Stratégies")

    col1, col2, col3 = st.columns(3)
    for col, (letter, name, desc, tp1, tp2, sl) in zip(
        [col1, col2, col3],
        [
            ("A", "Breakout Momentum", "Breakout 48h + vol×2.5", "2.5%", "5.5%", "1.2×ATR"),
            ("B", "Golden Cross",      "EMA7/EMA25 croisement",  "2.0%", "5.2%", "1.0×ATR"),
            ("C", "News Momentum",     "News bullish + RSI 55-70","3.0%", "6.0%", "0.8×ATR"),
        ],
    ):
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Stratégie {letter}</div>
  <div class="metric-value">{name}</div>
  <small style="color:#848E9C">{desc}</small><br>
  <span class="green">TP1 {tp1}</span> · <span class="green">TP2 {tp2}</span> · <span class="red">SL {sl}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dernières opportunités scannées")
    st.info("Lance `python -m src.main` pour alimenter le scanner en temps réel.")

    # Audit log
    st.markdown("### Journal d'audit")
    try:
        audit_path = pathlib.Path("data/logs/audit.jsonl")
        if audit_path.exists():
            lines = audit_path.read_text(encoding="utf-8").strip().split("\n")[-20:]
            import json
            rows = []
            for l in lines:
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun log d'audit.")
    except Exception as e:
        st.error(str(e))


# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRADES
# ════════════════════════════════════════════════════════════════════════
elif page == "📋 Trades":
    st.markdown("## 📋 Historique des trades")

    if df.empty:
        st.info("Aucun trade enregistré.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        syms = ["Tous"] + sorted(df["symbol"].dropna().unique().tolist())
        strats = ["Tous"] + sorted(df["strategy"].dropna().unique().tolist())
        sym_filter   = col1.selectbox("Symbole", syms)
        strat_filter = col2.selectbox("Stratégie", strats)
        status_filter = col3.selectbox("Statut", ["Tous", "Clôturés", "Ouverts"])

        fdf = df.copy()
        if sym_filter != "Tous":
            fdf = fdf[fdf["symbol"] == sym_filter]
        if strat_filter != "Tous":
            fdf = fdf[fdf["strategy"] == strat_filter]
        if status_filter == "Clôturés":
            fdf = fdf[fdf["closed_at"].notna()]
        elif status_filter == "Ouverts":
            fdf = fdf[fdf["closed_at"].isna()]

        def color_pnl(val):
            if isinstance(val, (int, float)):
                return "color: #0ECB81" if val > 0 else "color: #F6465D" if val < 0 else ""
            return ""

        display_cols = [c for c in ["symbol","strategy","entry_price","exit_price","pnl_usdt","pnl_pct","exit_reason","tp1_hit","tp2_hit","paper"] if c in fdf.columns]
        styled = fdf[display_cols].style.applymap(color_pnl, subset=["pnl_usdt","pnl_pct"] if "pnl_usdt" in display_cols else [])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        col_dl, col_m = st.columns([2, 1])
        csv = fdf.to_csv(index=False).encode("utf-8")
        col_dl.download_button("⬇️ Exporter CSV", csv, "trades.csv", "text/csv")

        if not fdf.empty and "pnl_usdt" in fdf.columns and "strategy" in fdf.columns:
            st.markdown("#### PnL par stratégie")
            by_strat = fdf.groupby("strategy")["pnl_usdt"].sum().reset_index()
            try:
                import plotly.express as px
                fig = px.bar(by_strat, x="strategy", y="pnl_usdt",
                             color="pnl_usdt", color_continuous_scale=["#F6465D","#0ECB81"],
                             labels={"pnl_usdt": "PnL USDT", "strategy": "Stratégie"})
                fig.update_layout(paper_bgcolor="#0B0E11", plot_bgcolor="#0D1117",
                                  font_color="#E8E8E8", height=250,
                                  margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(by_strat)


# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.markdown("## 📈 Performance")

    if df.empty:
        st.info("Aucun trade pour calculer la performance.")
    else:
        closed = df[df["closed_at"].notna()].copy()
        if closed.empty:
            st.info("Aucun trade clôturé.")
        else:
            try:
                closed["date"] = pd.to_datetime(closed["closed_at"]).dt.date
                daily = closed.groupby("date")["pnl_usdt"].sum().reset_index()
                daily.columns = ["Date", "PnL"]

                import plotly.express as px
                fig = px.bar(daily, x="Date", y="PnL",
                             color="PnL", color_continuous_scale=["#F6465D","#0ECB81"],
                             title="PnL quotidien (30 derniers jours)")
                fig.update_layout(paper_bgcolor="#0B0E11", plot_bgcolor="#0D1117",
                                  font_color="#E8E8E8", height=300,
                                  margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))

            # Global metrics
            st.markdown("### Métriques globales")
            col1, col2, col3, col4 = st.columns(4)
            total_pnl   = closed["pnl_usdt"].sum()
            n_wins      = (closed["pnl_usdt"] > 0).sum()
            n_total     = len(closed)
            avg_win     = closed[closed["pnl_usdt"]>0]["pnl_usdt"].mean() if n_wins else 0
            avg_loss    = closed[closed["pnl_usdt"]<0]["pnl_usdt"].mean() if (n_total-n_wins)>0 else 0
            col1.metric("Total PnL", f"{total_pnl:+.2f} €")
            col2.metric("Trades gagnants", f"{n_wins}/{n_total}")
            col3.metric("Gain moyen", f"{avg_win:+.2f} €")
            col4.metric("Perte moyenne", f"{avg_loss:+.2f} €")

            # Projection
            capital = float(os.getenv("CAPITAL_USDT", "200"))
            target  = 10_000
            daily_avg = total_pnl / max(1, n_total)
            days_left = (target - capital - total_pnl) / max(0.01, daily_avg * 0.5) if daily_avg > 0 else 9999
            st.markdown(f"**Projection :** à {daily_avg*0.5:.2f} €/jour moyen → **{days_left:.0f} jours** pour atteindre {target:,} USDT")


# ════════════════════════════════════════════════════════════════════════
# PAGE 5 — APPRENTISSAGE
# ════════════════════════════════════════════════════════════════════════
elif page == "🧠 Apprentissage":
    st.markdown("## 🧠 Apprentissage automatique")

    col1, col2 = st.columns(2)

    # Win rate par symbole
    if not df.empty and "pnl_usdt" in df.columns:
        closed = df[df["closed_at"].notna()].copy()
        if not closed.empty:
            wr_by_sym = (
                closed.groupby("symbol")
                .agg(trades=("pnl_usdt","count"), wins=("pnl_usdt", lambda x: (x>0).sum()))
                .reset_index()
            )
            wr_by_sym["win_rate"] = wr_by_sym["wins"] / wr_by_sym["trades"] * 100
            col1.markdown("#### Win rate par symbole")
            col1.dataframe(wr_by_sym.sort_values("win_rate", ascending=False),
                           use_container_width=True, hide_index=True)

    # Regime analysis
    col2.markdown("#### Analyse par régime")
    try:
        from src.brain.regime_detector import RegimeDetector
        col2.info("RegimeDetector disponible. Lance le scanner pour les données.")
    except Exception as e:
        col2.error(str(e))

    st.markdown("---")
    st.markdown("### Analyse des erreurs (Mistake Analyzer)")
    try:
        from src.learning.mistake_analyzer import MistakeAnalyzer
        ma = MistakeAnalyzer()
        st.info("MistakeAnalyzer disponible. Les analyses apparaissent après 10+ trades.")
    except Exception as e:
        st.error(str(e))

    st.markdown("### Optimisation Optuna")
    st.info("Lancer : `python scripts/optimize.py` pour déclencher l'optimisation bayésienne des paramètres.")


# ════════════════════════════════════════════════════════════════════════
# PAGE 6 — SIMULATION
# ════════════════════════════════════════════════════════════════════════
elif page == "🔬 Simulation":
    st.markdown("## 🔬 Simulation & Backtest")

    st.markdown("### Résultats de backtest")
    backtest_dir = pathlib.Path("data/backtest")
    if backtest_dir.exists():
        files = sorted(backtest_dir.glob("*.json"), reverse=True)
        if files:
            selected = st.selectbox("Rapport", [f.name for f in files])
            if selected:
                import json
                data = json.loads((backtest_dir / selected).read_text(encoding="utf-8"))
                st.json(data)
        else:
            st.info("Aucun rapport. Lancer : `python scripts/backtest.py`")
    else:
        st.info("Répertoire data/backtest absent. Lancer un backtest d'abord.")

    st.markdown("---")
    st.markdown("### Paper trading en cours")
    gate = pathlib.Path("data/.activation_date")
    if gate.exists():
        from datetime import date
        start = date.fromisoformat(gate.read_text(encoding="utf-8").strip())
        elapsed = (date.today() - start).days
        remaining = max(0, 7 - elapsed)
        col1, col2 = st.columns(2)
        col1.metric("Démarré le", str(start))
        col2.metric("Jours restants (paper gate)", str(remaining))
        st.progress(min(1.0, elapsed / 7))
    else:
        st.info("Paper gate non initialisé.")
