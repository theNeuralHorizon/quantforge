"""Tests for the config system."""
from __future__ import annotations

import os

import pytest

from quantforge.config import Settings, reload_settings


class TestSettings:
    def test_defaults(self):
        for k in ("QUANTFORGE_API_KEYS", "QUANTFORGE_JWT_SECRET", "REDIS_URL"):
            os.environ.pop(k, None)
        s = Settings()
        assert s.log_level == "INFO"
        assert s.api_port == 8000
        assert s.rate_limit_per_minute == 120
        assert s.api_keys_list == []
        assert s.cors_origins_list == []

    def test_env_override(self):
        os.environ["QUANTFORGE_LOG_LEVEL"] = "DEBUG"
        os.environ["QUANTFORGE_RATE_LIMIT_PER_MINUTE"] = "60"
        s = reload_settings()
        assert s.log_level == "DEBUG"
        assert s.rate_limit_per_minute == 60
        # cleanup
        os.environ.pop("QUANTFORGE_LOG_LEVEL", None)
        os.environ.pop("QUANTFORGE_RATE_LIMIT_PER_MINUTE", None)
        reload_settings()

    def test_api_keys_list_parses(self):
        os.environ["QUANTFORGE_API_KEYS"] = "abc123, def456 ,ghi789"
        s = reload_settings()
        assert s.api_keys_list == ["abc123", "def456", "ghi789"]
        os.environ.pop("QUANTFORGE_API_KEYS", None)
        reload_settings()

    def test_cors_origins_list(self):
        os.environ["QUANTFORGE_CORS_ORIGINS"] = "http://a.com,https://b.com"
        s = reload_settings()
        assert "http://a.com" in s.cors_origins_list
        assert "https://b.com" in s.cors_origins_list
        os.environ.pop("QUANTFORGE_CORS_ORIGINS", None)
        reload_settings()

    def test_body_size_bounds(self):
        # pydantic-settings enforces bounds
        try:
            from pydantic_settings import BaseSettings
            os.environ["QUANTFORGE_MAX_BODY_KB"] = "999999999"
            with pytest.raises(Exception):
                reload_settings()
        except ImportError:
            pytest.skip("pydantic-settings not installed")
        finally:
            os.environ.pop("QUANTFORGE_MAX_BODY_KB", None)
            reload_settings()
