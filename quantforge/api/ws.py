"""WebSocket endpoint — broadcast strategy signals, job progress, and alerts in real time."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from quantforge.api.auth import _sha256, _ALLOWED_HASHES, _unauth_allowed


router = APIRouter(tags=["ws"])


@dataclass
class _Hub:
    """In-process pub/sub hub for the single-process case. For multi-replica
    deployments, swap this for a Redis pub/sub adapter (hook left below)."""
    clients: Dict[str, Set[WebSocket]] = field(default_factory=dict)  # topic -> sockets
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def subscribe(self, topic: str, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.setdefault(topic, set()).add(ws)

    async def unsubscribe(self, topic: str, ws: WebSocket) -> None:
        async with self.lock:
            sockets = self.clients.get(topic)
            if sockets:
                sockets.discard(ws)
                if not sockets:
                    self.clients.pop(topic, None)

    async def publish(self, topic: str, message: dict) -> int:
        """Returns number of deliveries."""
        async with self.lock:
            sockets = list(self.clients.get(topic, set()))
        if not sockets:
            return 0
        delivered = 0
        drop: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json({"topic": topic, "data": message, "ts": time.time()})
                delivered += 1
            except Exception:
                drop.append(ws)
        if drop:
            async with self.lock:
                sockets_now = self.clients.get(topic, set())
                for ws in drop:
                    sockets_now.discard(ws)
        return delivered


hub = _Hub()


def _auth_ok(api_key: Optional[str]) -> bool:
    if _unauth_allowed():
        return True
    if not api_key or len(api_key) < 8 or len(api_key) > 128:
        return False
    return _sha256(api_key) in _ALLOWED_HASHES


@router.websocket("/ws/events")
async def ws_events(
    ws: WebSocket,
    topic: str = Query("signals", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$"),
    api_key: Optional[str] = Query(None, max_length=128),
) -> None:
    """Subscribe to a topic. Examples: signals, jobs, alerts, market.SPY.
    The client may also send JSON `{"action": "subscribe", "topic": ...}` or
    `{"action": "publish", "topic": ..., "data": ...}` (publish is rejected).
    """
    # For WebSockets, the key comes via query param because custom headers
    # aren't always available in browser WebSocket constructors.
    if not _auth_ok(api_key):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized")
        return

    await ws.accept()
    topics: Set[str] = {topic}
    await hub.subscribe(topic, ws)
    try:
        await ws.send_json({
            "topic": "system",
            "data": {"event": "connected", "topic": topic},
            "ts": time.time(),
        })
        # heartbeat loop — keeps LBs + proxies happy
        hb_task = asyncio.create_task(_heartbeat(ws))
        try:
            while True:
                msg_raw = await ws.receive_text()
                try:
                    msg = json.loads(msg_raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                action = msg.get("action")
                if action == "ping":
                    await ws.send_json({"topic": "system", "data": {"event": "pong"}, "ts": time.time()})
                elif action == "subscribe":
                    t = str(msg.get("topic", "")).strip()
                    if t and t not in topics and len(t) <= 64:
                        await hub.subscribe(t, ws)
                        topics.add(t)
                        await ws.send_json({"topic": "system", "data": {"event": "subscribed", "topic": t}, "ts": time.time()})
                elif action == "unsubscribe":
                    t = str(msg.get("topic", "")).strip()
                    if t in topics:
                        await hub.unsubscribe(t, ws)
                        topics.discard(t)
                        await ws.send_json({"topic": "system", "data": {"event": "unsubscribed", "topic": t}, "ts": time.time()})
                # publish from clients is explicitly not supported (security)
        finally:
            hb_task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception:
        # ensure clean close on unexpected errors
        try:
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        for t in topics:
            await hub.unsubscribe(t, ws)


async def _heartbeat(ws: WebSocket, interval: float = 25.0) -> None:
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json({"topic": "system", "data": {"event": "heartbeat"}, "ts": time.time()})
    except Exception:
        pass


# Server-side helpers other modules can call to fan out events
async def broadcast_signal(symbol: str, direction: int, strategy: str, extra: dict | None = None) -> int:
    msg = {"symbol": symbol, "direction": direction, "strategy": strategy}
    if extra:
        msg.update(extra)
    return await hub.publish("signals", msg)


async def broadcast_job_event(job_id: str, status_str: str, progress: float, extra: dict | None = None) -> int:
    msg = {"job_id": job_id, "status": status_str, "progress": progress}
    if extra:
        msg.update(extra)
    return await hub.publish("jobs", msg)


async def broadcast_alert(severity: str, title: str, detail: str = "", extra: dict | None = None) -> int:
    msg = {"severity": severity, "title": title, "detail": detail}
    if extra:
        msg.update(extra)
    return await hub.publish("alerts", msg)
