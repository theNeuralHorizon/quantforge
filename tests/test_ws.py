"""Tests for the WebSocket /ws/events endpoint."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from quantforge.api import auth
    auth.reset_keys()
    auth.register_raw_key("test_ws_key_1234567890")
    from quantforge.api.app import create_app
    return TestClient(create_app())


class TestWebSocket:
    def test_rejects_unauthorized(self, client):
        # No api_key → expect a close with policy-violation code
        try:
            with client.websocket_connect("/ws/events?topic=signals"):
                pytest.fail("should have been rejected")
        except Exception:
            # starlette raises WebSocketDisconnect with policy code 1008
            pass

    def test_accepts_with_key_and_receives_connected(self, client):
        with client.websocket_connect("/ws/events?topic=signals&api_key=test_ws_key_1234567890") as ws:
            msg = ws.receive_json()
            assert msg["topic"] == "system"
            assert msg["data"]["event"] == "connected"

    def test_ping_pong(self, client):
        with client.websocket_connect("/ws/events?topic=signals&api_key=test_ws_key_1234567890") as ws:
            ws.receive_json()  # connected
            ws.send_text(json.dumps({"action": "ping"}))
            reply = ws.receive_json()
            assert reply["data"]["event"] == "pong"

    def test_subscribe_extra_topic(self, client):
        with client.websocket_connect("/ws/events?topic=signals&api_key=test_ws_key_1234567890") as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"action": "subscribe", "topic": "jobs"}))
            reply = ws.receive_json()
            assert reply["data"]["event"] == "subscribed"
            assert reply["data"]["topic"] == "jobs"

    def test_broadcast_delivered(self, client):
        from quantforge.api.ws import hub
        with client.websocket_connect("/ws/events?topic=signals&api_key=test_ws_key_1234567890") as ws:
            ws.receive_json()  # connected
            # publish from server side
            asyncio.run(hub.publish("signals", {"symbol": "SPY", "direction": 1, "strategy": "test"}))
            msg = ws.receive_json()
            assert msg["topic"] == "signals"
            assert msg["data"]["symbol"] == "SPY"
