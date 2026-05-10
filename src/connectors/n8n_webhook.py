from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import hashlib
import hmac

from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="CENTINA n8n Gateway", version="5.0.0")

_signal_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

WEBHOOK_SECRET = os.getenv("N8N_WEBHOOK_SECRET", "")


# ── Payload models ────────────────────────────────────────────────────

class NewsSignal(BaseModel):
    source:   str                    # e.g. "cointelegraph", "twitter"
    symbol:   str | None = None      # affected symbol (None = market-wide)
    headline: str
    sentiment: float = Field(0.0, ge=-1.0, le=1.0)   # -1 bearish → +1 bullish
    urgency:   str  = "normal"       # normal | high | critical
    raw_url:   str | None = None


class ExternalSignal(BaseModel):
    signal_type: str    # "news" | "onchain" | "social" | "custom"
    payload:     dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/webhook/news")
async def receive_news(
    request: Request,
    signal: NewsSignal,
    x_centina_signature: str | None = Header(None),
):
    await _verify_hmac(request, x_centina_signature)
    await _enqueue({"type": "news", **signal.model_dump()})
    logger.info("n8n webhook: news signal — %s %s", signal.symbol, signal.headline[:60])
    return {"status": "queued"}


@app.post("/webhook/signal")
async def receive_signal(
    request: Request,
    signal: ExternalSignal,
    x_centina_signature: str | None = Header(None),
):
    await _verify_hmac(request, x_centina_signature)
    await _enqueue({"type": signal.signal_type, **signal.payload})
    logger.info("n8n webhook: external signal type=%s", signal.signal_type)
    return {"status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": _signal_queue.qsize()}


# ── Consumer helper ───────────────────────────────────────────────────

async def get_next_signal(timeout: float = 1.0) -> dict | None:
    """Called by the main orchestrator to drain the signal queue."""
    try:
        return await asyncio.wait_for(_signal_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


# ── Internal ─────────────────────────────────────────────────────────

async def _verify_hmac(request: Request, signature: str | None) -> None:
    """HMAC-SHA256 verification: X-Centina-Signature: sha256=<hex_digest>"""
    if not WEBHOOK_SECRET:
        return
    if not signature:
        raise HTTPException(status_code=401, detail="Missing X-Centina-Signature header")
    body = await request.body()
    mac = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")


async def _enqueue(payload: dict) -> None:
    if _signal_queue.full():
        logger.warning("n8n signal queue full – dropping oldest signal")
        try:
            _signal_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await _signal_queue.put(payload)


# ── Server launcher ───────────────────────────────────────────────────

def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


async def start_server_async(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
