"""
Pytest configuration for isolated, zero-cost, deterministic testing.
Forces AI_PROVIDER to 'mock' and EMBEDDING_PROVIDER to 'local' for all automated tests
so tests execute fast, offline, and without consuming API quotas.
"""

import pytest
from config import Config


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Ensure tests run strictly with mock AI and local zero-download similarity engine."""
    monkeypatch.setattr(Config, "AI_PROVIDER", "mock")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(Config, "FREELLMAPI_KEY", "")
    monkeypatch.setattr(Config, "GEMINI_API_KEY", "")
