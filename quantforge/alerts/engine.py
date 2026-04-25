"""AlertEngine: evaluates rules, deduplicates repeats, fans out to channels."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from quantforge.alerts.channels import Channel, NullChannel
from quantforge.alerts.rules import AlertEvent, AlertRule

_log = logging.getLogger("quantforge.alerts")

# Cap kept simple: ~1000 events fits the typical "last alerts" sidebar
# without growing without bound. deque(maxlen=...) gives O(1) appends and
# automatic eviction, removing the periodic slice-and-rebind that was
# previously racy under concurrent evaluate() calls.
_HISTORY_CAP = 1000


@dataclass
class AlertEngine:
    rules: list[AlertRule] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=lambda: [NullChannel()])
    dedupe_seconds: float = 300.0          # suppress duplicate alerts
    _last_fired: dict[str, float] = field(default_factory=dict)
    # `deque` swap (was: list) eliminates the manual eviction in evaluate()
    # and is atomic for append + popleft, so concurrent threads can no
    # longer drop each other's events on the eviction race.
    _history: deque[AlertEvent] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_CAP),
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def evaluate(self, context: dict[str, Any]) -> list[AlertEvent]:
        fired: list[AlertEvent] = []
        now = time.time()
        with self._lock:
            for rule in self.rules:
                event = rule.evaluate(context)
                if event is None:
                    continue
                last = self._last_fired.get(rule.name, 0.0)
                if now - last < self.dedupe_seconds:
                    continue
                self._last_fired[rule.name] = now
                fired.append(event)
                self._history.append(event)
        # Fan-out is best-effort and may make network calls; release the
        # lock before invoking channels so a slow webhook can't stall
        # concurrent callers.
        for event in fired:
            self._fanout(event)
        return fired

    def _fanout(self, event: AlertEvent) -> None:
        for ch in self.channels:
            try:
                ch.send(event)
            except Exception as e:
                # A flaky channel (broken webhook, dead SMTP) shouldn't
                # take the whole engine down. Log + move on so the next
                # channel still gets a shot.
                _log.warning("alert channel %s failed: %s", type(ch).__name__, e)

    def recent(self, limit: int = 50) -> list[AlertEvent]:
        with self._lock:
            return list(self._history)[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_fired.clear()
