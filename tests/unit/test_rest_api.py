"""Unit tests for REST API (api/rest_api.py).

Tests REST API endpoints for health check and prediction.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from api import rest_api


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """Make a synchronous request to the ASGI app."""
    async def _call() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rest_api.app),
            base_url="http://test"
        ) as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(_call())


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_returns_ok_status(self):
        """Health endpoint returns ok status."""
        response = _request("GET", "/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_success(self):
        """Predict endpoint returns prediction on success."""
        with patch("core.engine.predict", return_value={"move": "pass"}, create=True) as mock_pred:
            payload = {"input": {"board": []}}

            response = _request("POST", "/predict", json=payload)

            assert response.status_code == 200
            assert response.json() == {"output": {"move": "pass"}}
            mock_pred.assert_called_once_with(payload["input"])

    def test_predict_invalid_payload(self):
        """Predict endpoint returns 422 for invalid payload."""
        response = _request("POST", "/predict", json={})

        assert response.status_code == 422

    def test_predict_exception_handling(self):
        """Predict endpoint returns 500 on internal error."""
        with patch("core.engine.predict", side_effect=RuntimeError("boom"), create=True):
            response = _request("POST", "/predict", json={"input": {"x": 1}})

            assert response.status_code == 500
            assert response.json()["detail"] == "boom"
