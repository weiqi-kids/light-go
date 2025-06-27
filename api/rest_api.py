"""FastAPI-based RESTful service for Light-Go predictions.

This module exposes two routes:
 - ``/predict`` accepts POST requests with JSON ``{"input": ...}`` and
   returns ``{"output": ...}`` using :func:`core.engine.predict`.
 - ``/health`` is a simple GET route for health checks.

It also enables CORS and can be run directly with Uvicorn.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import engine as core_engine


class PredictRequest(BaseModel):
    """Request model for the ``/predict`` endpoint."""

    input: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response model returned by the ``/predict`` endpoint."""

    output: Any


app = FastAPI(title="Light-Go REST API")

# Configure very permissive CORS by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Run a prediction using :func:`core.engine.predict`.

    Parameters
    ----------
    req:
        Parsed request payload containing the ``input`` data.

    Returns
    -------
    PredictResponse
        The prediction wrapped in a ``PredictResponse`` model.
    """

    try:
        result = core_engine.predict(req.input)
    except Exception as exc:  # pragma: no cover - generic safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(output=result)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple liveness probe."""

    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual start
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
