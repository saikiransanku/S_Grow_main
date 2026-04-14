from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    upload_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    plant_confirmed: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duplicate_of: Mapped[int | None] = mapped_column(ForeignKey("images.id"), nullable=True, index=True)

    original: Mapped[Optional["ImageRecord"]] = relationship(remote_side=[id], lazy="joined")
    predictions: Mapped[list["PredictionRecord"]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False, index=True)
    plant_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    disease_name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    image: Mapped[ImageRecord] = relationship(back_populates="predictions", lazy="joined")
    responses: Mapped[list["ResponseRecord"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ResponseRecord(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False, index=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    prediction: Mapped[PredictionRecord] = relationship(back_populates="responses", lazy="joined")
