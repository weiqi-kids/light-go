"""Unit tests for WebSocket API (api/websocket_api.py).

Tests WebSocket connection, message handling, and prediction endpoint.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest
import uvicorn
from websockets.sync.client import connect

from api.websocket_api import app


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORT = 8765


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _start_server():
    """Start the WebSocket server in a background thread."""
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    return server, thread


def _stop_server(server, thread):
    """Stop the WebSocket server."""
    server.should_exit = True
    thread.join()


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestWebSocketPredict:
    """Tests for WebSocket prediction endpoint."""

    def test_predict_success(self):
        """Predict returns output on success."""
        with patch("core.engine.predict", return_value={"move": "D4"}, create=True):
            server, thread = _start_server()
            try:
                ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
                ws.send(json.dumps({"input": {"board": []}}))
                data = json.loads(ws.recv())

                assert data == {"output": {"move": "D4"}}
                ws.close()
            finally:
                _stop_server(server, thread)

    def test_predict_default_engine(self):
        """Predict with default engine returns output."""
        server, thread = _start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            # Use 9x9 board (minimum valid Go board size)
            board_9x9 = [[0] * 9 for _ in range(9)]
            ws.send(json.dumps({"input": {"board": board_9x9, "color": "black"}}))
            data = json.loads(ws.recv())

            assert data["output"] is not None
            ws.close()
        finally:
            _stop_server(server, thread)


class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling."""

    def test_invalid_json(self):
        """Invalid JSON returns error."""
        server, thread = _start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            ws.send("not json")
            data = json.loads(ws.recv())

            assert data["error"] == "invalid_json"
            ws.close()
        finally:
            _stop_server(server, thread)

    def test_missing_input_key(self):
        """Missing input key returns error."""
        server, thread = _start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            ws.send(json.dumps({"foo": 1}))
            data = json.loads(ws.recv())

            assert data["error"] == "missing_input"
            ws.close()
        finally:
            _stop_server(server, thread)


class TestWebSocketConnection:
    """Tests for WebSocket connection handling."""

    def test_ping_pong(self):
        """Server responds pong to ping."""
        server, thread = _start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            ws.send("ping")

            assert ws.recv() == "pong"
            ws.close()
        finally:
            _stop_server(server, thread)

    def test_disconnect_and_reconnect(self):
        """Client can reconnect after disconnect."""
        server, thread = _start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            ws.send("ping")
            ws.recv()
            ws.close()

            with patch("core.engine.predict", return_value=42, create=True):
                ws2 = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
                ws2.send(json.dumps({"input": {"x": 1}}))
                data = json.loads(ws2.recv())

                assert data == {"output": 42}
                ws2.close()
        finally:
            _stop_server(server, thread)
