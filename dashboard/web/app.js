const state = {
  page: "live",
  bridge: {},
  trades: [],
  performance: {},
  simulations: [],
  binance: { ok: false, usdt: 0, assets: {}, testnet: true },
  orders: [],
};

const titles = {
  live:         ["Live Dashboard",        "Etat temps reel de l'agent et du bridge."],
  opportunities:["Opportunites",          "Top signaux, filtres et execution manuelle."],
  trades:       ["Trades",               "Historique complet et resultats."],
  performance:  ["Performance",          "Capital, drawdown, win rate et projection."],
  learning:     ["Apprentissage",        "Historian, erreurs recurrentes et regime learner."],
  simulation:   ["Simulation Historique","Backtests et rapports de qualite."],
  council:      ["Conseil LLM",          "Verdicts et votes des modeles LLM par opportunite."],
  macro:        ["Intelligence Macro",   "Donnees cross-asset, funding, social et Fear & Greed."],
};

function qs(selector) {
  return document.querySelector(selector);
}

function fmtUsdt(value) {
  const n = Number(value || 0);
  return `${n >= 0 ? "" : "-"}${Math.abs(n).toFixed(2)} USDT`;
}

function fmtMoney(value) {
  return fmtUsdt(value);
}

function fmtPct(value) {
  const n = Number(value || 0);
  return `${(n * 100).toFixed(1)}%`;
}

function clsNum(value) {
  return Number(value || 0) >= 0 ? "positive" : "negative";
}

function signal(val) {
  if (!val || val === "NEUTRAL" || val === "") return `<span class="signal neu">NEU</span>`;
  const up = String(val).toUpperCase();
  if (up === "BUY" || up === "BULLISH" || up === "LONG") return `<span class="signal bull">${val}</span>`;
  if (up === "SELL" || up === "BEARISH" || up === "SHORT") return `<span class="signal bear">${val}</span>`;
  return `<span class="signal neu">${val}</span>`;
}

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} ${res.status}`);
  return res.json();
}

async function sendCommand(command) {
  const res = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(command),
  });
  if (!res.ok) throw new Error("Command failed");
  await refresh();
}

function table(el, columns, rows, empty = "En attente de donnees.") {
  if (!rows || rows.length === 0) {
    el.innerHTML = `<tbody><tr><td class="muted">${empty}</td></tr></tbody>`;
    return;
  }
  const head = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows.map((row) => {
    const cells = columns.map((c) => `<td>${c.render ? c.render(row) : (row[c.key] ?? "")}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  el.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}

function renderBars(el, rows, key = "pnl") {
  if (!rows || rows.length === 0) {
    el.innerHTML = `<div class="empty">En attente de donnees.</div>`;
    return;
  }
  const values = rows.map((r) => Number(r[key] || 0));
  const max = Math.max(...values.map((v) => Math.abs(v)), 1);
  el.innerHTML = rows.slice().reverse().map((row) => {
    const value = Number(row[key] || 0);
    const height = Math.max(6, Math.round((Math.abs(value) / max) * 220));
    return `<span class="bar ${value < 0 ? "loss" : ""}" style="height:${height}px" title="${row.day || ""}: ${value.toFixed(2)}"></span>`;
  }).join("");
}

// ── Metrics bar ───────────────────────────────────────────────────────────────

function renderMetrics() {
  const b  = state.bridge;
  const bn = state.binance;

  const binanceEl = qs("#metric-binance");
  if (bn.ok) {
    binanceEl.textContent = `${Number(bn.usdt).toFixed(2)} USDT`;
    binanceEl.className = "positive";
  } else {
    binanceEl.textContent = "- USDT";
    binanceEl.className = "muted";
  }

  qs("#metric-capital").textContent = fmtUsdt(b.capital_usdt ?? b.capital);
  qs("#metric-pnl").textContent     = fmtUsdt(b.pnl_day_eur ?? b.daily_pnl);
  qs("#metric-pnl").className       = clsNum(b.pnl_day_eur ?? b.daily_pnl);
  qs("#metric-positions").textContent = String((b.positions || []).length);
  qs("#metric-cb").textContent        = String(b.cb_name || b.cb_level || "NORMAL");

  const target = Number(b.daily_target || 30);
  const pnl    = Number(b.pnl_day_eur ?? b.daily_pnl ?? 0);
  const pct    = Math.max(0, Math.min(100, target ? (pnl / target) * 100 : 0));
  qs("#target-label").textContent     = `${pnl.toFixed(2)} / ${target.toFixed(0)} USDT`;
  qs("#target-progress").style.width  = `${pct}%`;
}

// ── Live page ─────────────────────────────────────────────────────────────────

function renderLive() {
  const b         = state.bridge;
  const positions = b.positions || [];

  table(qs("#positions-table"), [
    { label: "Symbole",   key: "symbol" },
    { label: "Strategie", key: "strategy" },
    { label: "Entree",    render: (r) => Number(r.entry_price || r.entry || 0).toFixed(6) },
    { label: "TP1",       render: (r) => Number(r.tp1 || r.tp1_price || 0).toFixed(6) },
    { label: "TP2",       render: (r) => Number(r.tp2 || r.tp2_price || 0).toFixed(6) },
    { label: "PnL",       render: (r) => `<span class="${clsNum(r.pnl_usdt)}">${fmtUsdt(r.pnl_usdt)}</span>` },
    { label: "Clore",     render: (r) => `<button class="cancel-btn close-pos" data-symbol="${r.symbol}">Clore</button>` },
  ], positions);

  document.querySelectorAll(".close-pos").forEach((btn) => {
    btn.addEventListener("click", () => sendCommand({ cmd: "CLOSE", symbol: btn.dataset.symbol }));
  });

  const events = (b.events || b.status_log || []).slice(-15).reverse();
  qs("#events").innerHTML = events.length
    ? events.map((ev) => {
        const text = typeof ev === "string" ? ev : (ev.text || ev.msg || JSON.stringify(ev));
        const ts   = typeof ev === "object" ? (ev.ts || "") : "";
        const type = typeof ev === "object" ? (ev.type || "INFO") : "INFO";
        const cls  = type === "WARN" ? "ev-warn" : type === "ERROR" ? "ev-error" : type === "PRIME" ? "ev-prime" : "";
        return `<div class="event ${cls}">${ts ? `<strong>${ts}</strong> ` : ""}${text}</div>`;
      }).join("")
    : `<div class="empty">En attente d'evenements.</div>`;

  renderBars(qs("#equity-chart"), state.performance.daily || [], "pnl");
}

// ── Binance open orders ───────────────────────────────────────────────────────

function renderOrders() {
  table(qs("#orders-table"), [
    { label: "Symbole", key: "symbol" },
    { label: "Cote",    render: (r) => r.side === "BUY"
        ? `<span class="positive">ACHAT</span>`
        : `<span class="negative">VENTE</span>` },
    { label: "Type",    key: "type" },
    { label: "Qte",     render: (r) => Number(r.origQty || 0).toFixed(6) },
    { label: "Prix",    render: (r) => Number(r.price   || 0).toFixed(6) },
    { label: "Statut",  key: "status" },
    { label: "Action",  render: (r) =>
        `<button class="cancel-btn cancel-order" data-symbol="${r.symbol}" data-id="${r.orderId}">Annuler</button>` },
  ], state.orders, "Aucun ordre ouvert sur Binance.");

  document.querySelectorAll(".cancel-order").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch("/api/binance/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: btn.dataset.symbol, order_id: Number(btn.dataset.id) }),
      });
      await refresh();
    });
  });
}

// ── Binance asset holdings ────────────────────────────────────────────────────

function renderAssets() {
  const assets = state.binance.assets || {};
  const keys   = Object.keys(assets).filter((k) => Number(assets[k]) > 0);
  if (!keys.length) {
    qs("#assets-grid").innerHTML = `<div class="muted" style="padding:8px">Aucun actif ou Binance non connecte.</div>`;
    return;
  }
  qs("#assets-grid").innerHTML = keys.map((k) => `
    <div class="mini-card">
      <span>${k}</span>
      <strong>${Number(assets[k]).toFixed(6)}</strong>
    </div>
  `).join("");
}

// ── Opportunities ─────────────────────────────────────────────────────────────

function renderOpportunities() {
  const scoreMin = Number(qs("#score-filter").value);
  const strat    = qs("#strategy-filter").value;
  qs("#score-filter-label").textContent = `Score >= ${scoreMin}`;

  let rows = state.bridge.opportunities || [];
  rows = rows.filter((r) => Number(r.score || r.composite_score || 0) >= scoreMin);
  if (strat !== "ALL") rows = rows.filter((r) => String(r.strategy || r.suggested_strategy || "") === strat);

  table(qs("#opps-table"), [
    { label: "Symbole",   key: "symbol" },
    { label: "Score",     render: (r) => `<span class="gold">${Number(r.score || r.composite_score || 0).toFixed(1)}</span>` },
    { label: "Verdict",   render: (r) => r.is_prime ? `<span class="positive">PRIME</span>` : "BUY" },
    { label: "Strategie", render: (r) => r.strategy || r.suggested_strategy || "-" },
    { label: "Prix",      render: (r) => Number(r.entry_price || r.best_price || 0).toFixed(6) },
    { label: "RSI",       render: (r) => Number(r.rsi || 0).toFixed(1) },
    { label: "VWAP",      render: (r) => signal(r.vwap_signal) },
    { label: "CVD",       render: (r) => signal(r.cvd_signal) },
    { label: "Micro",     render: (r) => signal(r.microstructure_signal) },
    { label: "Deep %",    render: (r) => r.deep_proba != null ? `${(Number(r.deep_proba) * 100).toFixed(0)}%` : "-" },
    { label: "Conseil",   render: (r) => {
        const v = r.council_verdict;
        if (!v) return "-";
        const cls = (v === "BUY" || v === "STRONG_BUY") ? "positive" : v === "SKIP" ? "negative" : "muted";
        return `<span class="${cls}">${v}</span>`;
      }
    },
    { label: "Whales",    render: (r) => r.whale_buys ? `<span class="positive">${r.whale_buys}</span>` : "-" },
    { label: "Social",    render: (r) => r.social_score != null ? Number(r.social_score).toFixed(0) : "-" },
    { label: "Duree",     render: (r) => `${Number(r.estimated_hours || 0).toFixed(1)}h` },
    { label: "Action",    render: (r) => `<button class="exec" data-symbol="${r.symbol}">GO</button>` },
  ], rows);

  document.querySelectorAll(".exec").forEach((btn) => {
    btn.addEventListener("click", () => sendCommand({ cmd: "EXECUTE", symbol: btn.dataset.symbol }));
  });
}

// ── Trades ────────────────────────────────────────────────────────────────────

function renderTrades() {
  table(qs("#trades-table"), [
    { label: "Date",      render: (r) => String(r.opened_at || "").slice(0, 16) },
    { label: "Symbole",   key: "symbol" },
    { label: "Strategie", key: "strategy" },
    { label: "Entree",    render: (r) => Number(r.entry_price || 0).toFixed(6) },
    { label: "Sortie",    render: (r) => r.exit_price ? Number(r.exit_price).toFixed(6) : "-" },
    { label: "TP1",       render: (r) => Number(r.tp1_hit || 0) ? "OK" : "-" },
    { label: "TP2",       render: (r) => Number(r.tp2_hit || 0) ? "OK" : "-" },
    { label: "Net",       render: (r) => `<span class="${clsNum(r.pnl_usdt)}">${fmtUsdt(r.pnl_usdt)}</span>` },
    { label: "Net %",     render: (r) => `${Number(r.pnl_pct || 0).toFixed(2)}%` },
  ], state.trades);
}

// ── Performance ───────────────────────────────────────────────────────────────

function renderPerformance() {
  const latest = state.performance.latest || {};
  const cards  = [
    ["Win rate",      latest.win_rate     != null ? fmtPct(latest.win_rate)      : "-"],
    ["Profit factor", latest.profit_factor                                        ?? "-"],
    ["Max drawdown",  latest.max_drawdown != null ? fmtPct(latest.max_drawdown)  : "-"],
    ["PnL total",     fmtUsdt(latest.total_pnl)],
    ["Trades",        latest.total_trades                                         ?? "0"],
    ["Avg win",       fmtUsdt(latest.avg_win)],
  ];
  qs("#performance-cards").innerHTML = cards.map(([label, value]) => (
    `<div class="mini-card"><span>${label}</span><strong>${value}</strong></div>`
  )).join("");
  renderBars(qs("#daily-chart"), state.performance.daily || [], "pnl");
}

// ── Learning ──────────────────────────────────────────────────────────────────

function renderLearning() {
  const trades = state.trades;
  if (!trades.length) {
    qs("#learning-summary").innerHTML = `<div class="empty">En attente de donnees.</div>`;
    return;
  }
  const losses = trades.filter((t) => Number(t.pnl_usdt || 0) < 0);
  const wins   = trades.filter((t) => Number(t.pnl_usdt || 0) >= 0 && t.closed_at);
  qs("#learning-summary").innerHTML = `
    <div class="mini-grid">
      <div class="mini-card"><span>Trades analyses</span><strong>${trades.length}</strong></div>
      <div class="mini-card"><span>Trades gagnants</span><strong>${wins.length}</strong></div>
      <div class="mini-card"><span>Erreurs/pertes</span><strong>${losses.length}</strong></div>
      <div class="mini-card"><span>Source</span><strong>CortexDB</strong></div>
    </div>
  `;
}

// ── Simulations ───────────────────────────────────────────────────────────────

function renderSimulations() {
  table(qs("#sim-table"), [
    { label: "Date",      render: (r) => String(r.created_at || "").slice(0, 16) },
    { label: "Symbole",   key: "symbol" },
    { label: "Strategie", key: "strategy" },
    { label: "Win rate",  render: (r) => fmtPct(r.win_rate) },
    { label: "Sharpe",    render: (r) => Number(r.sharpe || 0).toFixed(2) },
    { label: "Max DD",    render: (r) => fmtPct(r.max_drawdown) },
    { label: "Qualite",   render: (r) => `${r.quality_score || 0}/100` },
  ], state.simulations);
}

// ── Council LLM ───────────────────────────────────────────────────────────────

function renderCouncil() {
  const opps = state.bridge.opportunities || [];
  const rows = opps.filter((o) => o.council_verdict);

  qs("#council-meta").textContent = rows.length
    ? `${rows.length} opportunite(s) avec verdict LLM`
    : "";

  table(qs("#council-table"), [
    { label: "Symbole",   key: "symbol" },
    { label: "Score",     render: (r) => `<span class="gold">${Number(r.score || 0).toFixed(1)}</span>` },
    { label: "Verdict",   render: (r) => {
        const v   = r.council_verdict || "-";
        const cls = (v === "BUY" || v === "STRONG_BUY") ? "positive" : v === "SKIP" ? "negative" : "muted";
        return `<span class="${cls}">${v}</span>`;
      }
    },
    { label: "Accord %",  render: (r) => r.council_agreement != null ? `${Number(r.council_agreement).toFixed(0)}%` : "-" },
    { label: "Modeles",   render: (r) => r.council_models   != null ? String(r.council_models) : "-" },
    { label: "Strategie", render: (r) => r.strategy || "-" },
    { label: "Avis",      render: (r) => `<button class="council-detail-btn" data-sym="${r.symbol}">Detail</button>` },
  ], rows, "Aucun verdict LLM — lancez l'agent pour obtenir des analyses.");

  document.querySelectorAll(".council-detail-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const opp = opps.find((o) => o.symbol === btn.dataset.sym);
      const el  = qs("#council-detail");
      if (opp) {
        el.textContent = opp.council_text || "Pas de texte disponible.";
        el.classList.remove("empty");
      }
    });
  });
}

// ── Intelligence Macro ────────────────────────────────────────────────────────

function renderMacro() {
  const opps = state.bridge.opportunities || [];
  const src  = opps[0] || {};

  const cards = [
    ["Fear & Greed",   src.fear_greed        != null ? `${Number(src.fear_greed).toFixed(0)}/100`               : "-"],
    ["Social Score",   src.social_score      != null ? `${Number(src.social_score).toFixed(0)}/100`             : "-"],
    ["BTC Dominance",  src.btc_dominance     != null ? `${Number(src.btc_dominance).toFixed(1)}%`               : "-"],
    ["ETH/BTC Ratio",  src.eth_btc_ratio     != null ? Number(src.eth_btc_ratio).toFixed(4)                     : "-"],
    ["ETH/BTC Trend",  src.eth_btc_trend                                                                        || "-"],
    ["Funding Rate",   src.funding_rate      != null ? `${(Number(src.funding_rate) * 100).toFixed(4)}%`        : "-"],
    ["Signal Funding", src.funding_signal                                                                       || "-"],
    ["Volatilite",     src.volatility_label                                                                     || "-"],
  ];

  qs("#macro-cards").innerHTML = cards.map(([label, value]) => (
    `<div class="mini-card"><span>${label}</span><strong>${value}</strong></div>`
  )).join("");

  const macroRows = opps.filter((o) => o.btc_dominance != null || o.eth_btc_ratio != null);
  table(qs("#macro-table"), [
    { label: "Symbole",      key: "symbol" },
    { label: "BTC Dom.",     render: (r) => r.btc_dominance  != null ? `${Number(r.btc_dominance).toFixed(1)}%`          : "-" },
    { label: "ETH/BTC",      render: (r) => r.eth_btc_ratio  != null ? Number(r.eth_btc_ratio).toFixed(4)               : "-" },
    { label: "Trend ETH",    render: (r) => signal(r.eth_btc_trend) },
    { label: "Funding",      render: (r) => r.funding_rate   != null ? `${(Number(r.funding_rate) * 100).toFixed(4)}%`  : "-" },
    { label: "Signal",       render: (r) => signal(r.funding_signal) },
    { label: "Volatilite",   render: (r) => r.volatility_label                                                          || "-" },
    { label: "Liq. haut %",  render: (r) => r.liq_above_pct  != null ? `${Number(r.liq_above_pct).toFixed(1)}%`        : "-" },
    { label: "Liq. bas %",   render: (r) => r.liq_below_pct  != null ? `${Number(r.liq_below_pct).toFixed(1)}%`        : "-" },
  ], macroRows, "Aucune donnee macro disponible — relancez l'agent.");

  const fng      = Number(src.fear_greed || 50);
  const fngColor = fng >= 75 ? "#00E676" : fng >= 55 ? "#02C076" : fng >= 45 ? "#F0B90B" : fng >= 25 ? "#FF8C00" : "#F6465D";
  const fngLabel = fng >= 75 ? "Avidite extreme" : fng >= 55 ? "Avidite" : fng >= 45 ? "Neutre" : fng >= 25 ? "Peur" : "Peur extreme";
  qs("#fng-display").innerHTML = `
    <div class="fng-value" style="color:${fngColor}">${fng}</div>
    <div class="fng-label" style="color:${fngColor}">${fngLabel}</div>
    <div class="muted" style="font-size:12px;margin-top:4px">Fear &amp; Greed Index</div>
    ${src.social_score      != null ? `<div style="margin-top:8px"><span class="muted">Social:</span> <strong>${Number(src.social_score).toFixed(0)}/100</strong></div>` : ""}
    ${src.reddit_trend               ? `<div><span class="muted">Reddit:</span> <strong>${src.reddit_trend}</strong></div>` : ""}
    ${src.cryptopanic_score != null  ? `<div><span class="muted">CryptoPanic:</span> <strong>${Number(src.cryptopanic_score).toFixed(0)}/100</strong></div>` : ""}
  `;
}

// ── Master render ─────────────────────────────────────────────────────────────

function render() {
  const entry = titles[state.page] || ["—", ""];
  qs("#page-title").textContent = entry[0];
  qs("#subtitle").textContent   = entry[1];
  qs("#clock").textContent      = new Date().toLocaleTimeString();

  renderMetrics();
  renderLive();
  renderOrders();
  renderAssets();
  renderOpportunities();
  renderTrades();
  renderPerformance();
  renderLearning();
  renderSimulations();
  renderCouncil();
  renderMacro();
}

// ── Data refresh ──────────────────────────────────────────────────────────────

async function refresh() {
  const safe = (p) => p.catch(() => null);
  try {
    const [health, bridge, trades, performance, simulations, binance, orders] = await Promise.all([
      getJson("/api/health"),
      getJson("/api/state"),
      getJson("/api/trades"),
      getJson("/api/performance"),
      getJson("/api/simulations"),
      safe(getJson("/api/binance/balance")),
      safe(getJson("/api/binance/orders")),
    ]);

    qs("#health").textContent = health.ok ? "OK" : "WARN";

    const binanceOk = health.binance || false;
    const testnet   = health.testnet !== false;
    const badge     = qs("#network-badge");
    badge.textContent = testnet ? "TESTNET" : "MAINNET";
    badge.className   = `network-badge ${testnet ? "testnet" : "mainnet"}`;

    const statusEl     = qs("#binance-status");
    statusEl.textContent = binanceOk ? `Binance: ${testnet ? "TESTNET" : "LIVE"}` : "Binance: OFF";
    statusEl.className   = binanceOk ? "binance-on" : "binance-off";

    state.bridge      = bridge || {};
    state.trades      = (trades      || {}).items || [];
    state.performance = performance  || {};
    state.simulations = (simulations || {}).items || [];
    if (binance) state.binance = { ...binance, ok: binance.ok ?? false };
    if (orders)  state.orders  = orders.items || [];

    render();
  } catch (err) {
    qs("#health").textContent = "OFF";
    console.error(err);
  }
}

// ── Navigation ────────────────────────────────────────────────────────────────

function bindNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.page = button.dataset.page;
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b === button));
      document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
      qs(`#page-${state.page}`).classList.add("active");
      render();
    });
  });
}

// ── Command buttons ───────────────────────────────────────────────────────────

function bindCommands() {
  document.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => {
      const payload = { cmd: button.dataset.command };
      if (button.dataset.value) payload.value = button.dataset.value;
      if (button.dataset.days)  payload.days  = Number(button.dataset.days);
      sendCommand(payload);
    });
  });
  qs("#score-filter").addEventListener("input", renderOpportunities);
  qs("#strategy-filter").addEventListener("change", renderOpportunities);
  qs("#refresh-orders").addEventListener("click", refresh);
}

// ── Boot ──────────────────────────────────────────────────────────────────────

bindNavigation();
bindCommands();
refresh();
setInterval(refresh, 5000);
