"""FastAPI app with hardened security + rate limiting + metrics."""
from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quantforge import __version__
from quantforge.api.auth import get_or_create_dev_key, verify_api_key
from quantforge.api.cache import cache_backend
from quantforge.api.logging_config import configure_logging
from quantforge.api.metrics import PrometheusMiddleware, metrics_endpoint
from quantforge.api.routes import backtest as backtest_route
from quantforge.api.routes import market as market_route
from quantforge.api.routes import ml as ml_route
from quantforge.api.routes import jobs as jobs_route
from quantforge.api.routes import jwt_routes
from quantforge.api.routes import options as options_route
from quantforge.api.routes import portfolio as portfolio_route
from quantforge.api.routes import risk as risk_route
from quantforge.api.routes import alerts_routes
from quantforge.api.routes import audit_routes
from quantforge.api.routes import backtest_extra
from quantforge.api.audit import AuditMiddleware
from quantforge.api.ws import router as ws_router
from quantforge.api.schemas import HealthResponse, ReadinessResponse
from quantforge.api.security import (
    MaxBodySizeMiddleware, RequestIDMiddleware, SecurityHeadersMiddleware,
    build_cors_config,
)

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    _SLOWAPI = True
except ImportError:  # pragma: no cover
    _SLOWAPI = False


_START_TIME = time.time()


def _client_key(request: Request) -> str:
    """Rate-limit by API key if present, else by IP."""
    key = request.headers.get("X-API-Key")
    if key:
        return f"k:{key[:12]}"
    return f"ip:{get_remote_address(request)}" if _SLOWAPI else "anon"


def create_app() -> FastAPI:
    configure_logging(os.environ.get("QUANTFORGE_LOG_LEVEL", "INFO"))

    app = FastAPI(
        title="QuantForge API",
        version=__version__,
        description="Hardened REST API for quant research: options, backtests, "
                    "portfolio optimization, risk, and ML.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # rate limiter
    if _SLOWAPI:
        limiter = Limiter(key_func=_client_key,
                          default_limits=["120/minute", "2000/hour"])
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

        @app.exception_handler(RateLimitExceeded)
        async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
            return JSONResponse(
                status_code=429,
                content={"error": "rate limit exceeded", "detail": str(exc.detail)},
                headers={"Retry-After": "60"},
            )

    # body size limit — 256kb for reasonable quant payloads
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=256 * 1024)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(PrometheusMiddleware)

    cors_extras = [o.strip() for o in os.environ.get("QUANTFORGE_CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, **build_cors_config(cors_extras))

    # routers
    for r in (options_route.router, backtest_route.router, portfolio_route.router,
               risk_route.router, ml_route.router, market_route.router,
               jobs_route.router, ws_router, jwt_routes.router,
               alerts_routes.router, audit_routes.router, backtest_extra.router):
        app.include_router(r)

    # Audit every authenticated call. Opt-out via QUANTFORGE_AUDIT_DISABLE=1.
    if os.environ.get("QUANTFORGE_AUDIT_DISABLE", "").lower() not in ("1", "true", "yes"):
        app.add_middleware(AuditMiddleware)

    # Meta endpoints
    @app.get("/", include_in_schema=False)
    def _root():
        return HTMLResponse(_root_html(__version__))

    @app.get("/healthz", response_model=HealthResponse)
    def healthz():
        return HealthResponse(status="ok", version=__version__,
                               uptime_s=time.time() - _START_TIME)

    @app.get("/readyz", response_model=ReadinessResponse)
    def readyz():
        checks = {"cache": bool(cache_backend().ping())}
        ready = all(checks.values())
        return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)

    @app.get("/metrics", include_in_schema=False)
    def _metrics():
        return metrics_endpoint()

    @app.get("/v1/meta/version", dependencies=[Depends(verify_api_key)])
    def _version():
        return {"version": __version__}

    # Serve the static web UI if it exists
    web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
    web_dir = os.path.abspath(web_dir)
    if os.path.isdir(web_dir):
        app.mount("/ui", StaticFiles(directory=web_dir, html=True), name="ui")

    # For dev convenience, print the auto-generated key once on startup
    dev_key = get_or_create_dev_key()
    if dev_key:
        import logging
        logging.getLogger("quantforge.api").warning(
            "dev API key generated: %s (set QUANTFORGE_API_KEYS in prod)", dev_key,
        )

    # OpenTelemetry tracing (no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set)
    try:
        from quantforge.api.tracing import configure_tracing
        configure_tracing(app)
    except Exception:
        pass

    return app


def _root_html(version: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>QuantForge API</title>
<style>body{{font:15px system-ui;background:#0B0F19;color:#F0F2F6;padding:2rem;max-width:720px;margin:auto}}
a{{color:#00D9FF}}code{{background:#1C2130;padding:2px 6px;border-radius:4px;font-family:Menlo,monospace}}
.pill{{display:inline-block;padding:3px 10px;border-radius:99px;background:#1C2130;border:1px solid #2B3142;font-size:12px;margin-right:6px}}</style></head>
<body>
<h1 style="background:linear-gradient(90deg,#00D9FF,#00D98F);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
⚡ QuantForge API</h1>
<p>Version <span class=pill>{version}</span> <a href="/docs">OpenAPI docs</a> · <a href="/ui/">Web UI</a> · <a href="/metrics">metrics</a></p>
<h3>Try it</h3>
<pre style="background:#1C2130;padding:1rem;border-radius:8px;overflow:auto;">
curl -H "X-API-Key: &lt;key&gt;" -H "Content-Type: application/json" \\
  -d '{{"S":100,"K":100,"T":1,"sigma":0.2}}' \\
  http://localhost:8000/v1/options/price
</pre>
<p style="color:#7A869A;margin-top:2rem;font-size:13px;">
Send <code>X-API-Key</code> with every request. Dev key printed to server logs on startup.
Set <code>QUANTFORGE_API_KEYS</code> env var to a comma-separated list of SHA-256 hex digests in production.
</p>
</body></html>"""


# Module-level app for uvicorn / gunicorn
app = create_app()
