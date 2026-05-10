"""FastAPI dashboard for CENTINA with live Binance integration.

Serves static HTML/CSS/JS and exposes a JSON API over bridge files,
SQLite and the live Binance account (balance, open orders, prices).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.connectors.binance_rest import BinanceREST

logger = logging.getLogger(__name__)

WEB_DIR   = Path(__file__).resolve().parent / "web"
DATA_DIR  = ROOT / "data"
DB_PATH   = DATA_DIR / "cortex.db"
BRIDGE_FILE = DATA_DIR / "bridge_state.json"
CMD_QUEUE   = DATA_DIR / "cmd_queue.json"


# ── Lifespan: connect/disconnect Binance ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    testnet = os.getenv("BINANCE_TESTNET", "True").lower() in ("1", "true", "yes")
    if testnet:
        key    = os.getenv("BINANCE_TESTNET_API_KEY", "") or os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_TESTNET_API_SECRET", "") or os.getenv("BINANCE_API_SECRET", "")
    else:
        key    = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")

    app.state.testnet = testnet
    app.state.binance = None

    if key and secret:
        try:
            rest = BinanceREST(key, secret, testnet=testnet)
            await rest.connect()
            app.state.binance = rest
            logger.info("Dashboard Binance connected (testnet=%s)", testnet)
        except Exception as exc:
            logger.warning("Dashboard Binance connection failed: %s", exc)
    else:
        logger.info("Dashboard: no Binance credentials — live balance disabled")

    yield

    if app.state.binance is not None:
        await app.state.binance.disconnect()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="CENTINA Web Dashboard", version="5.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


class CommandIn(BaseModel):
    cmd: str
    value: Any | None = None
    symbol: str | None = None
    text: str | None = None
    days: int | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_command(command: dict[str, Any]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    command = {k: v for k, v in command.items() if v is not None}
    command["ts"] = datetime.now(timezone.utc).isoformat()
    queue = _read_json(CMD_QUEUE, [])
    if not isinstance(queue, list):
        queue = []
    queue.append(command)
    CMD_QUEUE.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    return command


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def _state() -> dict[str, Any]:
    bridge = _read_json(BRIDGE_FILE, {})
    if not isinstance(bridge, dict):
        bridge = {}
    bridge.setdefault("mode",         "PAPER")
    bridge.setdefault("capital",      bridge.get("capital_usdt", 200))
    bridge.setdefault("capital_usdt", bridge.get("capital", 200))
    bridge.setdefault("daily_pnl",    bridge.get("pnl_day_eur", 0))
    bridge.setdefault("pnl_day_eur",  bridge.get("daily_pnl", 0))
    bridge.setdefault("total_pnl",    0)
    bridge.setdefault("cb_name",      bridge.get("cb_level", "NORMAL"))
    bridge.setdefault("positions",    [])
    bridge.setdefault("opportunities",[])
    bridge.setdefault("status_log",   bridge.get("events", []))
    bridge.setdefault("events",       bridge.get("status_log", []))
    bridge["server_ts"] = datetime.now(timezone.utc).isoformat()
    return bridge


def _get_binance(request: Request) -> BinanceREST | None:
    return getattr(request.app.state, "binance", None)


# ── Static / index ────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {
        "ok":      True,
        "db":      DB_PATH.exists(),
        "bridge":  BRIDGE_FILE.exists(),
        "queue":   CMD_QUEUE.exists(),
        "binance": _get_binance(request) is not None,
        "testnet": getattr(request.app.state, "testnet", True),
    }


# ── Bridge state ──────────────────────────────────────────────────────────────

@app.get("/api/state")
def state() -> dict[str, Any]:
    return _state()


# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/trades")
def trades(limit: int = 200) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    rows = _query("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    return {"items": rows}


# ── Performance ───────────────────────────────────────────────────────────────

@app.get("/api/performance")
def performance() -> dict[str, Any]:
    latest = _query("SELECT * FROM performance ORDER BY id DESC LIMIT 1")
    daily  = _query(
        "SELECT date(closed_at) AS day, SUM(pnl_usdt) AS pnl, COUNT(*) AS trades "
        "FROM trades WHERE closed_at IS NOT NULL GROUP BY day ORDER BY day DESC LIMIT 60"
    )
    return {"latest": latest[0] if latest else {}, "daily": daily}


# ── Simulations ───────────────────────────────────────────────────────────────

@app.get("/api/simulations")
def simulations(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(limit, 300))
    rows = _query("SELECT * FROM historical_simulations ORDER BY id DESC LIMIT ?", (limit,))
    return {"items": rows}


# ── Commands ──────────────────────────────────────────────────────────────────

@app.post("/api/command")
def command(payload: CommandIn) -> dict[str, Any]:
    cmd = payload.cmd.strip().upper()
    if not cmd:
        raise HTTPException(status_code=400, detail="cmd is required")
    written = _write_command({
        "cmd":    cmd,
        "value":  payload.value,
        "symbol": payload.symbol,
        "text":   payload.text,
        "days":   payload.days,
    })
    return {"ok": True, "command": written}


# ── Live Binance endpoints ────────────────────────────────────────────────────

@app.get("/api/binance/balance")
async def binance_balance(request: Request) -> dict[str, Any]:
    rest = _get_binance(request)
    testnet = getattr(request.app.state, "testnet", True)
    if rest is None:
        bridge = _state()
        return {
            "ok":      False,
            "reason":  "Binance not connected — check API keys in .env",
            "usdt":    float(bridge.get("capital_usdt", bridge.get("capital", 0))),
            "assets":  {},
            "testnet": testnet,
        }
    try:
        balances = await rest.get_account_balance()
        usdt   = balances.pop("USDT", 0.0)
        locked = {k: v for k, v in balances.items() if v > 0}
        return {"ok": True, "usdt": usdt, "assets": locked, "testnet": testnet}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "usdt": 0.0, "assets": {}, "testnet": testnet}


@app.get("/api/binance/orders")
async def binance_orders(request: Request, symbol: str | None = None) -> dict[str, Any]:
    rest = _get_binance(request)
    if rest is None:
        return {"ok": False, "reason": "Binance not connected", "items": []}
    try:
        items = await rest.get_open_orders(symbol.upper() if symbol else None)
        return {"ok": True, "items": items}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "items": []}


@app.get("/api/binance/price/{symbol}")
async def binance_price(symbol: str, request: Request) -> dict[str, Any]:
    rest = _get_binance(request)
    if rest is None:
        return {"ok": False, "symbol": symbol.upper(), "price": 0.0}
    try:
        price = await rest.get_symbol_price(symbol.upper())
        return {"ok": True, "symbol": symbol.upper(), "price": price}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "symbol": symbol.upper(), "price": 0.0}


@app.post("/api/binance/cancel")
async def binance_cancel(request: Request) -> dict[str, Any]:
    rest = _get_binance(request)
    if rest is None:
        raise HTTPException(status_code=503, detail="Binance not connected")
    body = await request.json()
    symbol   = str(body.get("symbol", "")).upper()
    order_id = int(body.get("order_id", 0))
    if not symbol or not order_id:
        raise HTTPException(status_code=400, detail="symbol and order_id required")
    try:
        result = await rest.cancel_order(symbol, order_id)
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8600, reload=False)
