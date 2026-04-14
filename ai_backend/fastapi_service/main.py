from __future__ import annotations

import logging
import os
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import get_settings
from .database import Base, engine, get_db
from .predictor import PlantDiseasePredictor
from .response_builder import LocalLLMClient, PesticideCatalog, ResponseManager
from .schemas import AnalyzeResponse, HealthResponse
from .service import ImagePayload, PredictionWorkflowService

settings = get_settings()

logging.basicConfig(level=os.getenv("FASTAPI_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("sgrow_fastapi")

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)

    if settings.model_path is None:
        raise RuntimeError(
            "fastapi_service requires MODEL_PATH to be set explicitly. "
            "The Django seasonal inference pipeline now loads its models directly "
            "from training/image_prediction/runs."
        )

    app.state.predictor = PlantDiseasePredictor(
        model_path=settings.model_path,
        classes_path=settings.classes_path,
        model_arch=settings.model_arch,
        image_size=settings.image_size,
        min_confidence=settings.min_confidence,
    )
    app.state.pesticide_catalog = PesticideCatalog(settings.pesticides_csv_path)
    app.state.llm_client = LocalLLMClient(
        api_url=settings.llm_api_url,
        model_name=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    logger.info("Sgrow analyze pipeline initialized")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    files: List[UploadFile] = File(...),
    user_note: str | None = Form(None),
    plant_confirmed: str | None = Form(None),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_request} images allowed",
        )

    payloads: list[ImagePayload] = []
    for index, upload in enumerate(files, start=1):
        content = upload.file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Image {index} is empty")
        payloads.append(
            ImagePayload(
                image_index=index,
                filename=upload.filename or f"image_{index}.jpg",
                content=content,
            )
        )

    responder = ResponseManager(
        db=db,
        catalog=app.state.pesticide_catalog,
        llm_client=app.state.llm_client,
    )
    workflow = PredictionWorkflowService(
        db=db,
        predictor=app.state.predictor,
        responder=responder,
        storage_dir=settings.storage_dir,
        model_version=settings.model_version,
    )

    try:
        result = workflow.analyze_images(
            payloads,
            user_note=user_note,
            plant_confirmed=plant_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalyzeResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_service.main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        reload=False,
    )
