"""Integration tests for REST API endpoints.

Tests the REST API with real engine implementation.
"""
from __future__ import annotations

from typing import List

import pytest
from fastapi.testclient import TestClient

from api.rest_api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """Return a test client for the REST API."""
    return TestClient(app)


@pytest.fixture
def empty_board_5x5() -> List[List[int]]:
    """Return a 5x5 empty board."""
    return [[0] * 5 for _ in range(5)]


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestRESTAPIEndpoints:
    """Tests for REST API endpoints."""

    def test_health_endpoint(self, client: TestClient):
        """Health endpoint returns ok status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_endpoint(self, client: TestClient, empty_board_5x5: List[List[int]]):
        """Predict endpoint returns valid move."""
        payload = {"input": {"board": empty_board_5x5, "color": "black"}}

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json()["output"] is not None
