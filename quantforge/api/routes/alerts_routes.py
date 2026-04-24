"""/v1/alerts — expose the alert engine over REST."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from quantforge.alerts import AlertEngine
from quantforge.alerts.rules import Severity, ThresholdRule
from quantforge.api.auth import verify_api_key

router = APIRouter(prefix="/v1/alerts", tags=["alerts"])


_ALLOWED_METRICS: dict[str, Callable[[dict], float]] = {
    "drawdown":   lambda c: float(c["drawdown"]),
    "var":        lambda c: float(c["var"]),
    "vol":        lambda c: float(c["vol"]),
    "loss":       lambda c: float(c["loss"]),
    "pnl":        lambda c: float(c["pnl"]),
    "spread_z":   lambda c: float(c["spread_z"]),
    "exposure":   lambda c: float(c["exposure"]),
}


class RuleSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    metric: str = Field(..., description="one of: " + ", ".join(_ALLOWED_METRICS))
    threshold: float
    direction: Literal["above", "below"] = "above"
    severity: Literal["info", "warning", "critical"] = "warning"

    @property
    def metric_fn(self) -> Callable[[dict], float]:
        return _ALLOWED_METRICS[self.metric]


class EvalRequest(BaseModel):
    context: dict[str, float] = Field(..., description="Metric values keyed by name")


class EvalResponse(BaseModel):
    fired: list[dict]
    n_rules: int


# Singleton engine — in prod you'd scope to user/owner
_engine: AlertEngine | None = None
_rule_specs: dict[str, RuleSpec] = {}


def _get_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine(dedupe_seconds=60.0)
    return _engine


def reset_for_tests() -> None:
    """Module-level hook for test isolation."""
    global _engine, _rule_specs
    _engine = None
    _rule_specs = {}


@router.get("/rules")
def list_rules(_owner: str = Depends(verify_api_key)) -> list[dict]:
    return [s.model_dump() for s in _rule_specs.values()]


@router.post("/rules", status_code=201)
def create_rule(spec: RuleSpec, _owner: str = Depends(verify_api_key)) -> dict:
    if spec.metric not in _ALLOWED_METRICS:
        raise HTTPException(status_code=422, detail=f"metric must be one of {sorted(_ALLOWED_METRICS)}")
    if spec.name in _rule_specs:
        raise HTTPException(status_code=409, detail=f"rule '{spec.name}' already exists")
    rule = ThresholdRule(
        name=spec.name,
        metric=spec.metric_fn,
        threshold=spec.threshold,
        direction=spec.direction,
        severity=Severity(spec.severity),
    )
    _get_engine().add_rule(rule)
    _rule_specs[spec.name] = spec
    return spec.model_dump()


@router.delete("/rules/{rule_name}")
def delete_rule(rule_name: str, _owner: str = Depends(verify_api_key)) -> dict:
    if rule_name not in _rule_specs:
        raise HTTPException(status_code=404, detail="rule not found")
    eng = _get_engine()
    eng.rules = [r for r in eng.rules if r.name != rule_name]
    _rule_specs.pop(rule_name, None)
    return {"deleted": rule_name}


@router.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest, _owner: str = Depends(verify_api_key)) -> EvalResponse:
    eng = _get_engine()
    events = eng.evaluate(req.context)
    return EvalResponse(
        fired=[e.to_dict() for e in events],
        n_rules=len(eng.rules),
    )


@router.get("/events")
def recent_events(limit: int = 50, _owner: str = Depends(verify_api_key)) -> list[dict]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be in [1, 500]")
    return [e.to_dict() for e in _get_engine().recent(limit=limit)]
