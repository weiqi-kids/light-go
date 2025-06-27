from __future__ import annotations

import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
import gradio as gr

from monitoring import health_check, performance, logging as log_utils


fastapi_app = FastAPI(title="Light-Go Dashboard")


@fastapi_app.get("/monitoring/health")
async def monitoring_health():
    dash = health_check.build_default_dashboard()
    return await dash.api_status()


@fastapi_app.get("/monitoring/performance")
async def monitoring_performance():
    with performance.PerformanceMonitor() as mon:
        pass
    return mon.stats


@fastapi_app.get("/monitoring/logging")
def monitoring_logging(file: str = "performance.json"):
    q = log_utils.PerformanceLogQuery()
    try:
        q.load_logs(file)
        return {"summary": q.summary(), "entries": q.logs}
    except Exception as exc:  # pragma: no cover - optional file
        return {"error": str(exc)}


fastapi_app.mount("/static", StaticFiles(directory="ui"), name="static")

_client = TestClient(fastapi_app)


def _fetch(path: str, **params):
    resp = _client.get(path, params=params)
    if resp.status_code == 200:
        return resp.json()
    return {"error": resp.text}


def refresh_status(file: str):
    health = _fetch("/monitoring/health")
    perf = _fetch("/monitoring/performance")
    logs = _fetch("/monitoring/logging", file=file)
    return health, perf, logs


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Light-Go\n歡迎使用 Light-Go")
        with gr.Row():
            gr.Button("自我對弈")
            gr.Button("對弈")
            gr.Button("研究")
        gr.Markdown("## 系統監控")
        file_in = gr.Textbox("performance.json", label="Log file")
        refresh = gr.Button("刷新")
        health_box = gr.JSON(label="Health")
        perf_box = gr.JSON(label="Performance")
        log_box = gr.JSON(label="Logs")
        refresh.click(refresh_status, inputs=file_in, outputs=[health_box, perf_box, log_box])
    return demo


demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
