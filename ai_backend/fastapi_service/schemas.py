from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class AnalyzeResponse(BaseModel):
    plant: str
    disease: str
    confidence: float
    analysis: str
    cached: bool
