"""
Tests for dashboard FastAPI endpoints (start, stop, status, metrics).
Requires dashboard server not to be running (tests use TestClient).
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app after path is set
from dashboard.server import app

client = TestClient(app)


class TestDashboardEndpoints:
    """Test HTTP API endpoints."""

    def test_root_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert "RL-Traffic" in r.text or "traffic" in r.text.lower()

    def test_api_status_returns_json(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert "training_active" in data
        assert "state" in data
        assert "recent_metrics" in data

    def test_api_metrics_returns_list(self):
        r = client.get("/api/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], list)

    def test_api_metrics_accepts_last_param(self):
        r = client.get("/api/metrics?last=10")
        assert r.status_code == 200
        data = r.json()
        assert len(data["metrics"]) <= 10

    def test_api_stop_idempotent(self):
        r = client.post("/api/stop")
        assert r.status_code == 200
        assert r.json().get("status") == "stopped"

    def test_api_start_accepts_params(self):
        # Start with few steps so test finishes; may already be running
        r = client.post("/api/start?agent_type=round_robin&steps=100&curriculum=false")
        # 200 = started, or 200 with already_running
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("started", "already_running")
        # If we started, stop so other tests don't hang
        if data.get("status") == "started":
            client.post("/api/stop")

    def test_api_replay_returns_list(self):
        r = client.get("/api/replay?last=50")
        assert r.status_code == 200
        data = r.json()
        assert "replay" in data
        assert isinstance(data["replay"], list)

    def test_api_stress_returns_ok(self):
        r = client.post("/api/stress?duration_steps=50")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert data.get("status") == "stress_scheduled"

    def test_metrics_page_returns_html(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert "metrics" in r.text.lower() or "chart" in r.text.lower()

    def test_replay_page_returns_html(self):
        r = client.get("/replay")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
