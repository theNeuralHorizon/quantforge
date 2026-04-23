"""OpenTelemetry tracing — no-op if OTel SDK not installed."""
from __future__ import annotations

import logging
import os


def configure_tracing(app) -> bool:
    """Wire OTel into FastAPI if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Safe to call always: returns False if OTel isn't installed or configured.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    service_name = os.environ.get("OTEL_SERVICE_NAME", "quantforge-api")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logging.getLogger("quantforge.tracing").warning(
            "OTEL endpoint set but opentelemetry packages not installed; "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http "
            "opentelemetry-instrumentation-fastapi"
        )
        return False

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    logging.getLogger("quantforge.tracing").info("tracing enabled → %s", endpoint)
    return True
