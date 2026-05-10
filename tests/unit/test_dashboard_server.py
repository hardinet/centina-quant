"""Smoke tests for the FastAPI web dashboard."""
from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.server import app


def test_dashboard_index_serves_html():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CENTINA" in response.text


def test_dashboard_health_contract():
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert {"db", "bridge", "queue", "binance", "testnet"}.issubset(payload)


def test_dashboard_state_has_safe_defaults():
    client = TestClient(app)
    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"]
    assert isinstance(payload["positions"], list)
    assert isinstance(payload["opportunities"], list)
