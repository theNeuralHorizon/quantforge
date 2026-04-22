"""Prometheus metrics — requests, latency, errors, cache hits."""
from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
    )
    _PROM = True
except ImportError:  # pragma: no cover
    _PROM = False


if _PROM:
    REQ_COUNT = Counter(
        "qf_requests_total", "Total HTTP requests",
        labelnames=("method", "path", "status"),
    )
    REQ_LATENCY = Histogram(
        "qf_request_latency_seconds", "Request latency",
        labelnames=("method", "path"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    CACHE_HITS = Counter("qf_cache_hits_total", "Cache hits", labelnames=("namespace",))
    CACHE_MISSES = Counter("qf_cache_misses_total", "Cache misses", labelnames=("namespace",))
    IN_FLIGHT = Gauge("qf_in_flight_requests", "In-flight requests")


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not _PROM:
            return await call_next(request)
        path = request.url.path
        method = request.method
        IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            IN_FLIGHT.dec()
            # lightweight label cardinality: group status by class
            REQ_COUNT.labels(method=method, path=path, status=str(status_code)).inc()
            REQ_LATENCY.labels(method=method, path=path).observe(elapsed)
        return response


def metrics_endpoint() -> Response:
    if not _PROM:
        return Response("prometheus_client not installed", status_code=503)
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
