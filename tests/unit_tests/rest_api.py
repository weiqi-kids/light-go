import pathlib
import sys
import asyncio
from unittest.mock import patch

import httpx
import pytest

# Allow importing from project root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from api import rest_api


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    async def _call() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rest_api.app), base_url="http://test") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(_call())


def test_health_endpoint() -> None:
    response = _request("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_success() -> None:
    with patch("core.engine.predict", return_value={"move": "pass"}, create=True) as mock_pred:
        payload = {"input": {"board": []}}
        response = _request("POST", "/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == {"output": {"move": "pass"}}
        mock_pred.assert_called_once_with(payload["input"])


def test_predict_invalid_payload() -> None:
    response = _request("POST", "/predict", json={})
    assert response.status_code == 422


def test_predict_exception_handling() -> None:
    with patch("core.engine.predict", side_effect=RuntimeError("boom"), create=True):
        response = _request("POST", "/predict", json={"input": {"x": 1}})
        assert response.status_code == 500
        assert response.json()["detail"] == "boom"
