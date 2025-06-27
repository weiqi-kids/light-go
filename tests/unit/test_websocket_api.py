import json
import pathlib
import sys
import threading
import time
from unittest.mock import patch

import pytest
from websockets.sync.client import connect
import uvicorn

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from api.websocket_api import app


PORT = 8765


def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    return server, thread


def stop_server(server, thread):
    server.should_exit = True
    thread.join()


def test_predict_success():
    with patch("core.engine.predict", return_value={"move": "D4"}, create=True):
        server, thread = start_server()
        try:
            ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
            ws.send(json.dumps({"input": {"board": []}}))
            data = json.loads(ws.recv())
            assert data == {"output": {"move": "D4"}}
            ws.close()
        finally:
            stop_server(server, thread)


def test_invalid_json():
    server, thread = start_server()
    try:
        ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
        ws.send("not json")
        data = json.loads(ws.recv())
        assert data["error"] == "invalid_json"
        ws.close()
    finally:
        stop_server(server, thread)


def test_missing_input_key():
    server, thread = start_server()
    try:
        ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
        ws.send(json.dumps({"foo": 1}))
        data = json.loads(ws.recv())
        assert data["error"] == "missing_input"
        ws.close()
    finally:
        stop_server(server, thread)


def test_ping_pong():
    server, thread = start_server()
    try:
        ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
        ws.send("ping")
        assert ws.recv() == "pong"
        ws.close()
    finally:
        stop_server(server, thread)


def test_disconnect_and_reconnect():
    server, thread = start_server()
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
        stop_server(server, thread)


def test_predict_default_engine():
    server, thread = start_server()
    try:
        ws = connect(f"ws://127.0.0.1:{PORT}/ws/predict")
        ws.send(json.dumps({"input": {"board": [[0]], "color": "black"}}))
        data = json.loads(ws.recv())
        assert data["output"] is not None
        ws.close()
    finally:
        stop_server(server, thread)
