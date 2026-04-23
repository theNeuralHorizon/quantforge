"""AlertEngine: evaluates rules, deduplicates repeats, fans out to channels."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quantforge.alerts.channels import Channel, NullChannel
from quantforge.alerts.rules import AlertEvent, AlertRule


@dataclass
class AlertEngine:
    rules: List[AlertRule] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=lambda: [NullChannel()])
    dedupe_seconds: float = 300.0          # suppress duplicate alerts
    _last_fired: Dict[str, float] = field(default_factory=dict)
    _history: List[AlertEvent] = field(default_factory=list)

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def evaluate(self, context: Dict[str, Any]) -> List[AlertEvent]:
        fired: List[AlertEvent] = []
        now = time.time()
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
            self._fanout(event)
        # keep history bounded
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        return fired

    def _fanout(self, event: AlertEvent) -> None:
        for ch in self.channels:
            try:
                ch.send(event)
            except Exception:
                pass

    def recent(self, limit: int = 50) -> List[AlertEvent]:
        return list(self._history[-limit:])

    def clear_history(self) -> None:
        self._history.clear()
        self._last_fired.clear()
