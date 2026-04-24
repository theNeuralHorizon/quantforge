"""Alert delivery channels."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from quantforge.alerts.rules import AlertEvent, Severity

log = logging.getLogger("quantforge.alerts")


class Channel(ABC):
    @abstractmethod
    def send(self, event: AlertEvent) -> bool:
        ...


class NullChannel(Channel):
    """No-op (for tests/dry-run)."""
    def send(self, event: AlertEvent) -> bool:
        return True


class LogChannel(Channel):
    def send(self, event: AlertEvent) -> bool:
        getattr(log, event.severity.value.lower() if event.severity.value != "critical" else "error")(
            "alert", extra={"alert": event.to_dict()}
        )
        return True


class WebhookChannel(Channel):
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def send(self, event: AlertEvent) -> bool:
        try:
            import httpx
        except ImportError:
            return False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.url, json=event.to_dict())
                return 200 <= r.status_code < 300
        except Exception:
            return False


class SlackChannel(Channel):
    """Incoming-webhook style Slack integration."""
    def __init__(self, webhook_url: str, timeout: float = 5.0):
        self.url = webhook_url
        self.timeout = timeout

    def send(self, event: AlertEvent) -> bool:
        try:
            import httpx
        except ImportError:
            return False
        color = {
            Severity.INFO: "#00D9FF",
            Severity.WARNING: "#FFB547",
            Severity.CRITICAL: "#FF4E6B",
        }.get(event.severity, "#7A869A")
        payload = {
            "attachments": [{
                "color": color,
                "title": f"[{event.severity.value.upper()}] {event.title}",
                "text": event.detail,
                "footer": f"rule: {event.rule_name}",
            }]
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.url, json=payload)
                return 200 <= r.status_code < 300
        except Exception:
            return False
