"""Alert rules + event model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertEvent:
    rule_name: str
    severity: Severity
    title: str
    detail: str
    value: float | None = None
    threshold: float | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name, "severity": self.severity.value,
            "title": self.title, "detail": self.detail,
            "value": self.value, "threshold": self.threshold, "tags": self.tags,
        }


class AlertRule(ABC):
    name: str
    severity: Severity = Severity.WARNING

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> AlertEvent | None:
        ...


@dataclass
class ThresholdRule(AlertRule):
    """Fires when `metric(context)` crosses threshold in the given direction."""
    name: str
    metric: Callable[[dict[str, Any]], float]
    threshold: float
    direction: str = "above"   # "above" | "below"
    severity: Severity = Severity.WARNING
    title_template: str = "threshold breached"
    detail_template: str = "{name}: {value:.4f} {direction} {threshold:.4f}"

    def evaluate(self, context: dict[str, Any]) -> AlertEvent | None:
        try:
            value = float(self.metric(context))
        except Exception:
            return None
        breached = (value >= self.threshold) if self.direction == "above" else (value <= self.threshold)
        if not breached:
            return None
        detail = self.detail_template.format(
            name=self.name, value=value, direction=self.direction, threshold=self.threshold,
        )
        return AlertEvent(
            rule_name=self.name, severity=self.severity,
            title=self.title_template, detail=detail,
            value=value, threshold=self.threshold,
        )
