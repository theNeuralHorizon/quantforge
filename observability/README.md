# Observability

## Metrics
Prometheus endpoint: `GET /metrics` (no auth; keep behind private network).
Label-low-cardinality counters + latency histograms:

- `qf_requests_total{method,path,status}`
- `qf_request_latency_seconds_bucket{method,path,le}`
- `qf_in_flight_requests`
- `qf_cache_hits_total{namespace}` / `qf_cache_misses_total{namespace}`

## Grafana
Import `grafana-dashboard.json` — UID `quantforge-api`.
Datasource expected: `Prometheus` (rename in the import flow if yours is named differently).

## Alerting
Load `prometheus-alerts.yaml` into your Prometheus rules directory:

```yaml
rule_files:
  - /etc/prometheus/rules/quantforge.yaml
```

## Tracing (OpenTelemetry)
Set these env vars to enable OTel span export:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
OTEL_SERVICE_NAME=quantforge-api
```

Requires (already in requirements):
- `opentelemetry-sdk`
- `opentelemetry-exporter-otlp-proto-http`
- `opentelemetry-instrumentation-fastapi`

Traces are excluded for `/healthz`, `/readyz`, `/metrics` to avoid spam.
