"""WebSocket inference service using FastAPI.

This module exposes a ``FastAPI`` application that serves real-time predictions
via the ``/ws/predict`` endpoint.  Multiple clients can connect concurrently and
send JSON messages of the form ``{"input": {...}}``.  Each message is forwarded
to :func:`core.engine.predict` and the returned value is sent back as a JSON
object ``{"output": ...}``.

Messages with the literal text ``"ping"`` will receive ``"pong"`` in response.
Invalid JSON or missing ``input`` keys are reported back with an ``error``
message.  Any internal exception is logged and returned to the client in a safe
format.

Running the module directly will start the service using ``uvicorn``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core import engine


logger = logging.getLogger(__name__)
app = FastAPI(title="Light-Go WebSocket API")


class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self) -> None:
        """Create a new manager with an empty connection set."""
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept ``websocket`` and track the connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("Client connected. Active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove ``websocket`` from the active set."""
        self.active_connections.discard(websocket)
        logger.info(
            "Client disconnected. Active: %d", len(self.active_connections)
        )


manager = ConnectionManager()


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    """Handle WebSocket prediction requests."""

    await manager.connect(websocket)
    try:
        while True:
            try:
                text = await websocket.receive_text()
            except WebSocketDisconnect:
                raise

            if text == "ping":
                await websocket.send_text("pong")
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid_json"})
                continue

            if not isinstance(payload, dict) or "input" not in payload:
                await websocket.send_json({"error": "missing_input"})
                continue

            input_data = payload["input"]
            try:
                result = engine.predict(input_data)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - log unexpected errors
                logger.exception("Prediction failed: %s", exc)
                await websocket.send_json({"error": str(exc)})
                continue

            await websocket.send_json({"output": result})
    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/predict")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unexpected error: %s", exc)
    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":  # pragma: no cover - manual start
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
