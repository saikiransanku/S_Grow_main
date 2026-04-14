from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from .models import ImageRecord, PredictionRecord
from .predictor import PlantDiseasePredictor, normalize_token
from .response_builder import ResponseManager


@dataclass(frozen=True)
class ImagePayload:
    image_index: int
    filename: str
    content: bytes


@dataclass(frozen=True)
class ImageAnalysisItem:
    image_id: int
    prediction_id: int
    class_label: str
    plant_name: str
    disease_name: str
    confidence: float
    analysis: str
    cached: bool


def sha256_image(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_filename(filename: str) -> str:
    base = Path(filename or "leaf.jpg").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "leaf.jpg"


class PredictionWorkflowService:
    def __init__(
        self,
        *,
        db: Session,
        predictor: PlantDiseasePredictor,
        responder: ResponseManager,
        storage_dir: Path,
        model_version: str,
    ) -> None:
        self.db = db
        self.predictor = predictor
        self.responder = responder
        self.storage_dir = storage_dir
        self.model_version = model_version

    def analyze_images(
        self,
        image_payloads: Sequence[ImagePayload],
        *,
        user_note: str | None,
        plant_confirmed: str | None,
    ) -> dict[str, object]:
        if not image_payloads:
            return {
                "plant": "unknown",
                "disease": "uncertain diagnosis",
                "confidence": 0.0,
                "analysis": "No images were uploaded.",
                "cached": False,
            }

        per_image_results: list[ImageAnalysisItem] = []
        normalized_confirmed = normalize_token(plant_confirmed or "") or None

        for payload in image_payloads:
            image_hash = sha256_image(payload.content)
            original_image = self._find_original_by_hash(image_hash)

            if original_image is not None:
                image_record = ImageRecord(
                    image_hash=image_hash,
                    file_path=original_image.file_path,
                    user_note=user_note,
                    plant_confirmed=normalized_confirmed,
                    duplicate_of=original_image.id,
                )
                self.db.add(image_record)
                self.db.flush()

                cached_prediction = self._latest_prediction_for_image(original_image.id)
                if cached_prediction is not None:
                    clone_prediction = PredictionRecord(
                        image_id=image_record.id,
                        plant_name=cached_prediction.plant_name,
                        disease_name=cached_prediction.disease_name,
                        confidence=cached_prediction.confidence,
                        model_version=cached_prediction.model_version,
                    )
                    self.db.add(clone_prediction)
                    self.db.flush()

                    response_text, _ = self.responder.get_or_create_response(
                        prediction=clone_prediction,
                        class_label=f"{clone_prediction.disease_name}_{clone_prediction.plant_name}",
                    )

                    per_image_results.append(
                        ImageAnalysisItem(
                            image_id=image_record.id,
                            prediction_id=clone_prediction.id,
                            class_label=f"{clone_prediction.disease_name}_{clone_prediction.plant_name}",
                            plant_name=clone_prediction.plant_name,
                            disease_name=clone_prediction.disease_name,
                            confidence=round(float(clone_prediction.confidence), 4),
                            analysis=response_text,
                            cached=True,
                        )
                    )
                    continue
            else:
                file_path = self._store_original_image(
                    image_hash=image_hash,
                    filename=payload.filename,
                    content=payload.content,
                )
                image_record = ImageRecord(
                    image_hash=image_hash,
                    file_path=str(file_path),
                    user_note=user_note,
                    plant_confirmed=normalized_confirmed,
                    duplicate_of=None,
                )
                self.db.add(image_record)
                self.db.flush()

            prediction = self.predictor.predict(payload.content, user_note=user_note)

            prediction_row = PredictionRecord(
                image_id=image_record.id,
                plant_name=prediction.plant_name,
                disease_name=prediction.disease_name,
                confidence=prediction.confidence,
                model_version=self.model_version,
            )
            self.db.add(prediction_row)
            self.db.flush()

            response_text, response_cached = self.responder.get_or_create_response(
                prediction=prediction_row,
                class_label=prediction.class_label,
            )

            per_image_results.append(
                ImageAnalysisItem(
                    image_id=image_record.id,
                    prediction_id=prediction_row.id,
                    class_label=prediction.class_label,
                    plant_name=prediction.plant_name,
                    disease_name=prediction.disease_name,
                    confidence=prediction.confidence,
                    analysis=response_text,
                    cached=response_cached,
                )
            )

        self.db.commit()
        return self._summarize(per_image_results)

    def _summarize(self, results: Sequence[ImageAnalysisItem]) -> dict[str, object]:
        if len(results) == 1:
            item = results[0]
            return {
                "plant": item.plant_name,
                "disease": self._display_disease(item.disease_name, item.confidence),
                "confidence": round(float(item.confidence), 4),
                "analysis": item.analysis,
                "cached": item.cached,
            }

        unique_diagnoses = {(item.plant_name, item.disease_name) for item in results}
        if len(unique_diagnoses) == 1:
            plant_name, disease_name = next(iter(unique_diagnoses))
            summary_confidence = round(
                sum(item.confidence for item in results) / float(len(results)),
                4,
            )
            best_item = max(results, key=lambda item: item.confidence)
            return {
                "plant": plant_name,
                "disease": self._display_disease(disease_name, summary_confidence),
                "confidence": summary_confidence,
                "analysis": best_item.analysis,
                "cached": all(item.cached for item in results),
            }

        plant_counter: dict[str, int] = {}
        for item in results:
            plant_counter[item.plant_name] = plant_counter.get(item.plant_name, 0) + 1
        majority_plant = max(plant_counter.items(), key=lambda pair: pair[1])[0]

        plant_subset = [item for item in results if item.plant_name == majority_plant]
        disease_counter: dict[str, int] = {}
        for item in plant_subset:
            disease_counter[item.disease_name] = disease_counter.get(item.disease_name, 0) + 1
        majority_disease = max(disease_counter.items(), key=lambda pair: pair[1])[0]

        final_subset = [item for item in plant_subset if item.disease_name == majority_disease]
        if not final_subset:
            final_subset = plant_subset

        summary_confidence = round(
            sum(item.confidence for item in final_subset) / float(len(final_subset)),
            4,
        )

        best_item = max(final_subset, key=lambda item: item.confidence)
        summary_analysis = (
            f"Multi-image summary: majority plant={majority_plant}, majority disease={self._display_disease(majority_disease, summary_confidence)}.\n\n"
            f"{best_item.analysis}"
        )

        return {
            "plant": majority_plant,
            "disease": self._display_disease(majority_disease, summary_confidence),
            "confidence": summary_confidence,
            "analysis": summary_analysis,
            "cached": all(item.cached for item in results),
        }

    def _find_original_by_hash(self, image_hash: str) -> ImageRecord | None:
        stmt = (
            select(ImageRecord)
            .where(ImageRecord.image_hash == image_hash)
            .where(ImageRecord.duplicate_of.is_(None))
            .order_by(asc(ImageRecord.id))
            .limit(1)
        )
        original = self.db.scalar(stmt)
        if original is not None:
            return original

        fallback_stmt = (
            select(ImageRecord)
            .where(ImageRecord.image_hash == image_hash)
            .order_by(asc(ImageRecord.id))
            .limit(1)
        )
        return self.db.scalar(fallback_stmt)

    def _latest_prediction_for_image(self, image_id: int) -> PredictionRecord | None:
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.image_id == image_id)
            .order_by(desc(PredictionRecord.created_at), desc(PredictionRecord.id))
            .limit(1)
        )
        return self.db.scalar(stmt)

    def _store_original_image(self, *, image_hash: str, filename: str, content: bytes) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        clean_name = safe_filename(filename)
        target = self.storage_dir / f"{image_hash}_{clean_name}"
        target.write_bytes(content)
        return target

    def _display_disease(self, disease_name: str, confidence: float) -> str:
        normalized = normalize_token(disease_name)
        if confidence < self.predictor.min_confidence or normalized == "uncertain_diagnosis":
            return "uncertain diagnosis"
        return normalized
