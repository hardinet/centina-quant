from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AUDIT_FILE = Path("data/logs/audit.jsonl")


class AuditLogger:
    """
    Append-only, tamper-evident audit log.

    Each record is a JSON line containing:
    - seq       : monotonic sequence number
    - ts        : UTC ISO timestamp
    - event     : event type string
    - payload   : arbitrary dict
    - prev_hash : SHA-256 of the previous raw line (chain)
    - hash      : SHA-256 of this line (without the 'hash' field itself)

    The hash chain lets any auditor verify that no entry was silently deleted
    or modified.  A break in the chain signals tampering.
    """

    def __init__(self, path: Path = AUDIT_FILE):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq, self._prev_hash = self._bootstrap()

    # ── Public API ───────────────────────────────────────────────────

    def log(self, event: str, payload: dict[str, Any] | None = None) -> str:
        """Append a record. Returns the record's hash."""
        with self._lock:
            self._seq += 1
            record = {
                "seq":       self._seq,
                "ts":        datetime.now(timezone.utc).isoformat(),
                "event":     event,
                "payload":   payload or {},
                "prev_hash": self._prev_hash,
            }
            # Hash everything except the 'hash' field
            canonical  = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record_hash = hashlib.sha256(canonical.encode()).hexdigest()
            record["hash"] = record_hash

            line = json.dumps(record, separators=(",", ":"))
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

            self._prev_hash = record_hash
            return record_hash

    def verify_chain(self) -> tuple[bool, int]:
        """
        Walk the full log and verify hash integrity.
        Returns (is_valid, first_broken_seq).  is_valid=True means all good.
        """
        prev_hash = "0" * 64
        with self._path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.rstrip("\n")
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    return False, -1

                stored_hash = record.pop("hash", "")
                prev_hash_in_record = record.get("prev_hash", "")

                canonical   = json.dumps(record, sort_keys=True, separators=(",", ":"))
                computed    = hashlib.sha256(canonical.encode()).hexdigest()

                if computed != stored_hash:
                    return False, record.get("seq", -1)
                if prev_hash_in_record != prev_hash:
                    return False, record.get("seq", -1)

                prev_hash = stored_hash

        return True, -1

    # ── Convenience event shortcuts ──────────────────────────────────

    def trade_opened(self, symbol: str, qty: float, price: float, order_id: int) -> None:
        self.log("TRADE_OPENED", {"symbol": symbol, "qty": qty, "price": price, "order_id": order_id})

    def trade_closed(self, symbol: str, pnl: float, reason: str) -> None:
        self.log("TRADE_CLOSED", {"symbol": symbol, "pnl": pnl, "reason": reason})

    def circuit_breaker_triggered(self, level: str, drawdown: float) -> None:
        self.log("CIRCUIT_BREAKER", {"level": level, "drawdown": drawdown})

    def order_rejected(self, symbol: str, reason: str) -> None:
        self.log("ORDER_REJECTED", {"symbol": symbol, "reason": reason})

    def manipulation_detected(self, symbol: str, flags: dict) -> None:
        self.log("MANIPULATION_DETECTED", {"symbol": symbol, "flags": flags})

    # ── Internal ─────────────────────────────────────────────────────

    def _bootstrap(self) -> tuple[int, str]:
        """Read the last line to recover (seq, hash) for continuation."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return 0, "0" * 64
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                last_line = ""
                for last_line in fh:
                    pass
            record = json.loads(last_line)
            return record.get("seq", 0), record.get("hash", "0" * 64)
        except Exception:
            return 0, "0" * 64
