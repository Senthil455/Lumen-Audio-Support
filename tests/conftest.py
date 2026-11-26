"""Test configuration.

Environment is set before app.config is imported, because settings are read
once at import time. TRUST_PROXY_HEADERS lets each test use its own forwarded
IP, so rate-limit buckets do not leak between tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRUST_PROXY_HEADERS", "1")
os.environ.setdefault("LOG_SALT", "test-salt-not-a-secret")
os.environ.setdefault("RATE_LIMIT", "5/minute")
os.environ.setdefault("AGENT_RATE_LIMIT", "1000/minute")
os.environ.setdefault("LOG_DIR", os.path.join(os.path.dirname(__file__), "_logs"))
os.environ.setdefault("RUNTIME_DIR", os.path.join(os.path.dirname(__file__), "_runtime"))
os.environ.pop("LLM_API_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fresh_ip():
    """A unique caller per test, so the per-IP limiter starts clean."""
    counter = {"n": 0}

    def _next() -> dict:
        counter["n"] += 1
        return {"X-Forwarded-For": f"10.0.0.{counter['n']}"}

    return _next
