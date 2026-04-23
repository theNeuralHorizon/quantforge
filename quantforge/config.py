"""Centralised configuration via pydantic-settings.

All QuantForge env vars live in one place. Loads from environment and
(optionally) a `.env` file next to the project root.
"""
from __future__ import annotations

from typing import List, Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PS = True
except ImportError:  # pragma: no cover
    # fallback if pydantic-settings isn't installed: pure-stdlib shim
    from dataclasses import dataclass
    _HAS_PS = False


if _HAS_PS:

    class Settings(BaseSettings):
        """All QuantForge runtime config, loaded from env + .env."""
        model_config = SettingsConfigDict(
            env_prefix="QUANTFORGE_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # API
        api_keys: str = Field(default="", description="comma-separated SHA-256 hex digests")
        allow_unauth: bool = Field(default=False)
        cors_origins: str = Field(default="")
        log_level: str = Field(default="INFO")
        api_host: str = Field(default="0.0.0.0")  # nosec B104 — intended for container deploys
        api_port: int = Field(default=8000)

        # Limits
        max_body_kb: int = Field(default=256, ge=1, le=10_000)
        rate_limit_per_minute: int = Field(default=120, ge=1)
        rate_limit_per_hour: int = Field(default=2000, ge=1)

        # JWT
        jwt_secret: str = Field(default="")
        jwt_issuer: str = Field(default="quantforge")
        jwt_audience: str = Field(default="quantforge-api")
        jwt_alg: str = Field(default="HS256")

        # Cache / Redis
        redis_url: str = Field(default="")

        # OpenTelemetry
        otel_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
        otel_service_name: str = Field(default="quantforge-api", alias="OTEL_SERVICE_NAME")

        # Data cache
        cache_dir: str = Field(default="data/cache")

        # Job manager
        job_max_workers: int = Field(default=4, ge=1, le=64)
        job_max_queued: int = Field(default=100, ge=1)
        job_per_owner_inflight_cap: int = Field(default=10, ge=1)

        @property
        def api_keys_list(self) -> List[str]:
            return [h.strip().lower() for h in self.api_keys.split(",") if h.strip()]

        @property
        def cors_origins_list(self) -> List[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

else:
    @dataclass
    class Settings:
        api_keys: str = ""
        allow_unauth: bool = False
        cors_origins: str = ""
        log_level: str = "INFO"
        api_host: str = "0.0.0.0"  # nosec B104 — intended for container deploys
        api_port: int = 8000
        max_body_kb: int = 256
        rate_limit_per_minute: int = 120
        rate_limit_per_hour: int = 2000
        jwt_secret: str = ""
        jwt_issuer: str = "quantforge"
        jwt_audience: str = "quantforge-api"
        jwt_alg: str = "HS256"
        redis_url: str = ""
        otel_endpoint: str = ""
        otel_service_name: str = "quantforge-api"
        cache_dir: str = "data/cache"
        job_max_workers: int = 4
        job_max_queued: int = 100
        job_per_owner_inflight_cap: int = 10

        @property
        def api_keys_list(self) -> List[str]:
            return [h.strip().lower() for h in self.api_keys.split(",") if h.strip()]

        @property
        def cors_origins_list(self) -> List[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
