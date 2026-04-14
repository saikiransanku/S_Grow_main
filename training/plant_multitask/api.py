from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

try:
    from .inference import PlantRecognizer
except ImportError:  # pragma: no cover
    from training.plant_multitask.inference import PlantRecognizer


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    request_count: int
    average_latency_ms: float
    low_confidence_count: int


class MonitoringWindow:
    def __init__(self, max_items: int = 256) -> None:
        self.latencies_ms: deque[float] = deque(maxlen=max_items)
        self.request_count = 0
        self.low_confidence_count = 0

    def record(self, latency_ms: float, confidence: float) -> None:
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        if confidence < 0.6:
            self.low_confidence_count += 1

    def snapshot(self) -> MetricsResponse:
        avg_latency = sum(self.latencies_ms) / max(1, len(self.latencies_ms))
        return MetricsResponse(
            request_count=self.request_count,
            average_latency_ms=round(avg_latency, 2),
            low_confidence_count=self.low_confidence_count,
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Plant Recognition Service", version="1.0.0")
    app.state.recognizer = None
    app.state.monitor = MonitoringWindow()

    @app.on_event("startup")
    def startup() -> None:
        checkpoint = os.getenv("PLANT_MODEL_CHECKPOINT")
        onnx_path = os.getenv("PLANT_MODEL_ONNX")
        if not checkpoint:
            raise RuntimeError("PLANT_MODEL_CHECKPOINT must be set.")

        app.state.recognizer = PlantRecognizer(
            checkpoint_path=Path(checkpoint),
            onnx_path=Path(onnx_path) if onnx_path else None,
            device=os.getenv("PLANT_MODEL_DEVICE"),
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=request.app.state.recognizer is not None)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(request: Request) -> MetricsResponse:
        return request.app.state.monitor.snapshot()

    @app.post("/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        question: str | None = Form(None),
        top_k: int = Form(5),
    ) -> dict[str, Any]:
        recognizer = request.app.state.recognizer
        if recognizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")

        started = time.perf_counter()
        result = recognizer.predict(image_bytes, question=question, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        confidence = float(result["species"]["confidence"])
        request.app.state.monitor.record(latency_ms, confidence)
        result["latency_ms"] = round(latency_ms, 2)
        return result

    @app.post("/predict/batch")
    async def predict_batch(
        request: Request,
        files: list[UploadFile] = File(...),
        question: str | None = Form(None),
        top_k: int = Form(5),
    ) -> dict[str, Any]:
        recognizer = request.app.state.recognizer
        if recognizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not files:
            raise HTTPException(status_code=400, detail="At least one image is required.")

        outputs = []
        for upload in files:
            image_bytes = await upload.read()
            if not image_bytes:
                continue
            started = time.perf_counter()
            result = recognizer.predict(image_bytes, question=question, top_k=top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            confidence = float(result["species"]["confidence"])
            request.app.state.monitor.record(latency_ms, confidence)
            result["filename"] = upload.filename
            result["latency_ms"] = round(latency_ms, 2)
            outputs.append(result)
        return {"items": outputs}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("training.plant_multitask.api:app", host="0.0.0.0", port=8001, reload=False)
