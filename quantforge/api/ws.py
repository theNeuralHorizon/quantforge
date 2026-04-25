"""WebSocket endpoint — broadcast strategy signals, job progress, and alerts in real time."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from quantforge.api.auth import _unauth_allowed, check_raw_key

router = APIRouter(tags=["ws"])

# Bounds prevent a single misbehaving client from exhausting memory.
_MAX_INBOUND_MESSAGE_BYTES = 4 * 1024            # 4 KiB — control frames are tiny
_MAX_TOPICS_PER_CONNECTION = 16                  # cap fan-out per socket
_PUBLISH_PER_SOCKET_TIMEOUT = 5.0                # one slow client can't block all subscribers

_log = logging.getLogger("quantforge.api.ws")


@dataclass
class _Hub:
    """In-process pub/sub hub for the single-process case. For multi-replica
    deployments, swap this for a Redis pub/sub adapter (hook left below)."""
    clients: dict[str, set[WebSocket]] = field(default_factory=dict)  # topic -> sockets
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
        """Returns number of deliveries.

        Sends to every subscriber concurrently with a per-socket timeout
        so one slow client can't stall the broadcast for the rest. Any
        socket that errors or times out is dropped from the topic set.
        """
        async with self.lock:
            sockets = list(self.clients.get(topic, set()))
        if not sockets:
            return 0

        envelope = {"topic": topic, "data": message, "ts": time.time()}

        async def _send_one(ws: WebSocket) -> bool:
            try:
                await asyncio.wait_for(ws.send_json(envelope), _PUBLISH_PER_SOCKET_TIMEOUT)
                return True
            except (asyncio.TimeoutError, Exception):
                return False

        results = await asyncio.gather(*[_send_one(ws) for ws in sockets], return_exceptions=False)
        delivered = sum(1 for ok in results if ok)
        drop = [ws for ws, ok in zip(sockets, results, strict=False) if not ok]
        if drop:
            async with self.lock:
                sockets_now = self.clients.get(topic, set())
                for ws in drop:
                    sockets_now.discard(ws)
                if not sockets_now:
                    self.clients.pop(topic, None)
        return delivered


hub = _Hub()


def _auth_ok(api_key: str | None) -> bool:
    """Constant-time auth check that mirrors the HTTP path's timing
    profile. Previous version used `_sha256(key) in set` which bypasses
    `hmac.compare_digest` and gives a different timing signature than
    `verify_api_key`."""
    if _unauth_allowed():
        return True
    return check_raw_key(api_key) is not None


@router.websocket("/ws/events")
async def ws_events(
    ws: WebSocket,
    topic: str = Query("signals", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$"),
    api_key: str | None = Query(None, max_length=128),
) -> None:
    """Subscribe to a topic. Examples: signals, jobs, alerts, market.SPY.
    The client may also send JSON `{"action": "subscribe", "topic": ...}`
    or `{"action": "ping"}`. Client-side publish is explicitly rejected.
    """
    # For WebSockets, the key comes via query param because custom headers
    # aren't always available in browser WebSocket constructors. The query
    # string IS captured by reverse proxies; use a short-lived token for
    # production hardening.
    if not _auth_ok(api_key):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized")
        return

    await ws.accept()
    topics: set[str] = {topic}
    await hub.subscribe(topic, ws)
    hb_task: asyncio.Task | None = None
    try:
        await ws.send_json({
            "topic": "system",
            "data": {"event": "connected", "topic": topic},
            "ts": time.time(),
        })
        # heartbeat loop — keeps LBs + proxies happy
        hb_task = asyncio.create_task(_heartbeat(ws))
        while True:
            msg_raw = await ws.receive_text()
            # Bounded message size — control frames are tiny; anything
            # larger is either a misbehaving client or a memory-bomb attempt.
            if len(msg_raw) > _MAX_INBOUND_MESSAGE_BYTES:
                continue
            try:
                msg = json.loads(msg_raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            action = msg.get("action")
            if action == "ping":
                await ws.send_json(
                    {"topic": "system", "data": {"event": "pong"}, "ts": time.time()},
                )
            elif action == "subscribe":
                # Cap topics-per-connection so a malicious client can't
                # exhaust the hub's clients dict.
                if len(topics) >= _MAX_TOPICS_PER_CONNECTION:
                    continue
                t = str(msg.get("topic", "")).strip()
                if t and t not in topics and len(t) <= 64:
                    await hub.subscribe(t, ws)
                    topics.add(t)
                    await ws.send_json({
                        "topic": "system",
                        "data": {"event": "subscribed", "topic": t},
                        "ts": time.time(),
                    })
            elif action == "unsubscribe":
                t = str(msg.get("topic", "")).strip()
                if t in topics:
                    await hub.unsubscribe(t, ws)
                    topics.discard(t)
                    await ws.send_json({
                        "topic": "system",
                        "data": {"event": "unsubscribed", "topic": t},
                        "ts": time.time(),
                    })
            # publish from clients is explicitly not supported (security)
    except WebSocketDisconnect:
        pass
    except Exception:
        _log.exception("ws_events handler crashed")
        with contextlib.suppress(Exception):
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        # Cancel + drain the heartbeat so asyncio doesn't log a noisy
        # "Task exception was never retrieved" warning when the heartbeat
        # raised before we got here.
        if hb_task is not None:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await hb_task
        for t in topics:
            await hub.unsubscribe(t, ws)


async def _heartbeat(ws: WebSocket, interval: float = 25.0) -> None:
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json(
                {"topic": "system", "data": {"event": "heartbeat"}, "ts": time.time()},
            )
    except (asyncio.CancelledError, WebSocketDisconnect):
        raise
    except Exception:  # pragma: no cover - heartbeat is best-effort
        return


# Server-side helpers other modules can call to fan out events
async def broadcast_signal(
    symbol: str, direction: int, strategy: str, extra: dict | None = None,
) -> int:
    msg = {"symbol": symbol, "direction": direction, "strategy": strategy}
    if extra:
        msg.update(extra)
    return await hub.publish("signals", msg)


async def broadcast_job_event(
    job_id: str, status_str: str, progress: float, extra: dict | None = None,
) -> int:
    msg = {"job_id": job_id, "status": status_str, "progress": progress}
    if extra:
        msg.update(extra)
    return await hub.publish("jobs", msg)


async def broadcast_alert(
    severity: str, title: str, detail: str = "", extra: dict | None = None,
) -> int:
    msg = {"severity": severity, "title": title, "detail": detail}
    if extra:
        msg.update(extra)
    return await hub.publish("alerts", msg)
