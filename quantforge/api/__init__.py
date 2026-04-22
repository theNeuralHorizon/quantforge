"""QuantForge REST API — FastAPI service with hardened security."""
from quantforge.api.app import create_app, app

__all__ = ["create_app", "app"]
