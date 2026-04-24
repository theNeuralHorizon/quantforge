"""Centralised configuration via pydantic-settings.

All QuantForge env vars live in one place. Loads from environment and
(optionally) a `.env` file next to the project root.
"""
from __future__ import annotations

import os

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PS = True
except ImportError:  # pragma: no cover
    # fallback if pydantic-settings isn't installed: pure-stdlib shim
    from dataclasses import dataclass, field
    _HAS_PS = False


def _env(name: str, default: str) -> str:
    """Read `QUANTFORGE_<name>` from the process env, else return default."""
    return os.environ.get(f"QUANTFORGE_{name.upper()}", default)


def _env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    raw = os.environ.get(f"QUANTFORGE_{name.upper()}")
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except ValueError as e:
        raise ValueError(f"QUANTFORGE_{name.upper()} must be an int, got {raw!r}") from e
    if lo is not None and val < lo:
        raise ValueError(f"QUANTFORGE_{name.upper()}={val} must be >= {lo}")
    if hi is not None and val > hi:
        raise ValueError(f"QUANTFORGE_{name.upper()}={val} must be <= {hi}")
    return val


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"QUANTFORGE_{name.upper()}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
        def api_keys_list(self) -> list[str]:
            return [h.strip().lower() for h in self.api_keys.split(",") if h.strip()]

        @property
        def cors_origins_list(self) -> list[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

else:
    @dataclass
    class Settings:  # type: ignore[no-redef]
        """Fallback settings that read from os.environ when pydantic-settings
        is unavailable. Env var names mirror the pydantic model
        (QUANTFORGE_<FIELD>, upper-cased). Two exceptions mirror pydantic-
        settings aliases: OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_SERVICE_NAME.
        """
        api_keys: str = field(default_factory=lambda: _env("api_keys", ""))
        allow_unauth: bool = field(default_factory=lambda: _env_bool("allow_unauth", False))
        cors_origins: str = field(default_factory=lambda: _env("cors_origins", ""))
        log_level: str = field(default_factory=lambda: _env("log_level", "INFO"))
        api_host: str = field(default_factory=lambda: _env("api_host", "0.0.0.0"))  # nosec B104
        api_port: int = field(default_factory=lambda: _env_int("api_port", 8000, lo=1, hi=65535))
        max_body_kb: int = field(
            default_factory=lambda: _env_int("max_body_kb", 256, lo=1, hi=10_000)
        )
        rate_limit_per_minute: int = field(
            default_factory=lambda: _env_int("rate_limit_per_minute", 120, lo=1)
        )
        rate_limit_per_hour: int = field(
            default_factory=lambda: _env_int("rate_limit_per_hour", 2000, lo=1)
        )
        jwt_secret: str = field(default_factory=lambda: _env("jwt_secret", ""))
        jwt_issuer: str = field(default_factory=lambda: _env("jwt_issuer", "quantforge"))
        jwt_audience: str = field(default_factory=lambda: _env("jwt_audience", "quantforge-api"))
        jwt_alg: str = field(default_factory=lambda: _env("jwt_alg", "HS256"))
        redis_url: str = field(default_factory=lambda: _env("redis_url", ""))
        # Aliased env vars (match pydantic model aliases)
        otel_endpoint: str = field(
            default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        )
        otel_service_name: str = field(
            default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "quantforge-api")
        )
        cache_dir: str = field(default_factory=lambda: _env("cache_dir", "data/cache"))
        job_max_workers: int = field(
            default_factory=lambda: _env_int("job_max_workers", 4, lo=1, hi=64)
        )
        job_max_queued: int = field(
            default_factory=lambda: _env_int("job_max_queued", 100, lo=1)
        )
        job_per_owner_inflight_cap: int = field(
            default_factory=lambda: _env_int("job_per_owner_inflight_cap", 10, lo=1)
        )

        @property
        def api_keys_list(self) -> list[str]:
            return [h.strip().lower() for h in self.api_keys.split(",") if h.strip()]

        @property
        def cors_origins_list(self) -> list[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
