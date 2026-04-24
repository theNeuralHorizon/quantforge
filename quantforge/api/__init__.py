"""QuantForge REST API — FastAPI service with hardened security."""
from quantforge.api.app import app, create_app

__all__ = ["create_app", "app"]
