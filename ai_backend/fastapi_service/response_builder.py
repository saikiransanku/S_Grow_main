from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from .models import PredictionRecord, ResponseRecord
from .predictor import normalize_token


SYSTEM_PROMPT = (
    "You are an agricultural assistant. Explain the detected disease and suggest safe treatment steps for farmers."
)


def humanize_token(value: str) -> str:
    normalized = normalize_token(value).replace("_", " ").strip()
    return normalized.title() if normalized else "Unknown"


@dataclass(frozen=True)
class PesticideRecommendation:
    class_label: str
    crop: str
    disease: str
    treatment_type: str
    product_name: str
    active_ingredient: str
    priority: int
    effectiveness: float
    usage_note: str


class PesticideCatalog:
    MAX_ROWS_PER_INDEX_KEY = 12

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.by_class: dict[str, list[PesticideRecommendation]] = {}
        self.by_disease: dict[str, list[PesticideRecommendation]] = {}
        self.by_crop: dict[str, list[PesticideRecommendation]] = {}
        self.by_disease_crop: dict[tuple[str, str], list[PesticideRecommendation]] = {}
        self._load_rows(csv_path)

    @staticmethod
    def _normalize_csv_key(raw_key: str | None) -> str:
        token = str(raw_key or "").replace("\ufeff", "").strip()
        return token.strip('"').strip("'").lower()

    @classmethod
    def _normalize_record(cls, record: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, value in record.items():
            key = cls._normalize_csv_key(raw_key)
            if not key:
                continue
            normalized[key] = str(value or "").strip()
        return normalized

    @staticmethod
    def _sort_key(row: PesticideRecommendation) -> tuple[int, float, str]:
        return (row.priority, -row.effectiveness, row.product_name.lower())

    @staticmethod
    def _row_identity(row: PesticideRecommendation) -> tuple[str, str, str, str]:
        return (
            row.treatment_type,
            row.product_name.lower(),
            row.active_ingredient.lower(),
            normalize_token(row.crop),
        )

    def _add_to_index(
        self,
        *,
        index: dict[str | tuple[str, str], list[PesticideRecommendation]],
        key: str | tuple[str, str],
        recommendation: PesticideRecommendation,
    ) -> None:
        if not key:
            return

        bucket = index.setdefault(key, [])
        recommendation_identity = self._row_identity(recommendation)

        for idx, existing in enumerate(bucket):
            if self._row_identity(existing) != recommendation_identity:
                continue
            if self._sort_key(recommendation) < self._sort_key(existing):
                bucket[idx] = recommendation
                bucket.sort(key=self._sort_key)
            return

        bucket.append(recommendation)
        bucket.sort(key=self._sort_key)
        if len(bucket) > self.MAX_ROWS_PER_INDEX_KEY:
            del bucket[self.MAX_ROWS_PER_INDEX_KEY :]

    def _load_rows(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_record in reader:
                record = self._normalize_record(raw_record)

                class_label = normalize_token(record.get("class_label", ""))
                crop = record.get("crop", "").strip() or humanize_token(class_label)
                disease = normalize_token(record.get("disease_type", "")) or "unknown"
                product_name = record.get("recommended_pesticide", "").strip() or "No recommendation"
                active_ingredient = record.get("active_ingredient", "").strip() or "n/a"
                usage_note = record.get("usage_note", "").strip()

                treatment_type = normalize_token(record.get("treatment_type", ""))
                if treatment_type not in {"organic", "chemical"}:
                    treatment_type = self._infer_treatment_type(product_name, active_ingredient)

                priority = self._safe_int(record.get("priority"), default=3)
                effectiveness = self._safe_float(record.get("effectiveness_score"), default=0.5)

                if not class_label and disease != "unknown":
                    class_label = normalize_token(f"{disease}_{crop}")

                recommendation = PesticideRecommendation(
                    class_label=class_label,
                    crop=crop,
                    disease=disease,
                    treatment_type=treatment_type,
                    product_name=product_name,
                    active_ingredient=active_ingredient,
                    priority=max(1, priority),
                    effectiveness=min(max(effectiveness, 0.0), 1.0),
                    usage_note=usage_note,
                )

                crop_key = normalize_token(crop)
                self._add_to_index(index=self.by_class, key=class_label, recommendation=recommendation)
                self._add_to_index(index=self.by_disease, key=disease, recommendation=recommendation)
                if crop_key:
                    self._add_to_index(index=self.by_crop, key=crop_key, recommendation=recommendation)
                    self._add_to_index(
                        index=self.by_disease_crop,
                        key=(disease, crop_key),
                        recommendation=recommendation,
                    )

    @staticmethod
    def _safe_int(value: str | None, default: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: str | None, default: float) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _infer_treatment_type(product: str, ingredient: str) -> str:
        token = f"{normalize_token(product)}_{normalize_token(ingredient)}"
        organic_keys = {
            "neem",
            "azadirachtin",
            "bacillus_subtilis",
            "trichoderma",
            "beauveria",
            "spinosad",
            "potassium_bicarbonate",
            "wettable_sulfur",
            "sulfur",
            "copper_soap",
            "no_pesticide_required",
            "compost_tea",
            "garlic_extract",
            "chitosan",
        }
        for key in organic_keys:
            if key in token:
                return "organic"
        return "chemical"

    def recommend(
        self,
        *,
        class_label: str,
        plant_name: str,
        disease_name: str,
        limit: int = 2,
    ) -> dict[str, list[PesticideRecommendation]]:
        class_key = normalize_token(class_label)
        plant_key = normalize_token(plant_name)
        disease_key = normalize_token(disease_name)

        def sort_rows(items: Sequence[PesticideRecommendation]) -> list[PesticideRecommendation]:
            return sorted(items, key=self._sort_key)

        candidates = list(self.by_class.get(class_key, []))
        if not candidates:
            candidates = self._find_by_disease_and_crop(disease_key=disease_key, plant_key=plant_key)
        if not candidates:
            candidates = list(self.by_disease.get(disease_key, []))
        if not candidates:
            candidates = self._find_by_crop(plant_key=plant_key)

        organic = sort_rows([row for row in candidates if row.treatment_type == "organic"])[:limit]
        chemical = sort_rows([row for row in candidates if row.treatment_type == "chemical"])[:limit]

        if not organic:
            organic = [
                PesticideRecommendation(
                    class_label=class_key,
                    crop=humanize_token(plant_name),
                    disease=disease_key,
                    treatment_type="organic",
                    product_name="Neem Oil 1500 ppm",
                    active_ingredient="azadirachtin",
                    priority=1,
                    effectiveness=0.6,
                    usage_note="Spray in the evening and repeat every 7 days.",
                ),
                PesticideRecommendation(
                    class_label=class_key,
                    crop=humanize_token(plant_name),
                    disease=disease_key,
                    treatment_type="organic",
                    product_name="Bacillus subtilis Bio-fungicide",
                    active_ingredient="bacillus subtilis",
                    priority=2,
                    effectiveness=0.55,
                    usage_note="Apply as preventive foliar spray after pruning infected leaves.",
                ),
            ][:limit]

        if not chemical:
            chemical = [
                PesticideRecommendation(
                    class_label=class_key,
                    crop=humanize_token(plant_name),
                    disease=disease_key,
                    treatment_type="chemical",
                    product_name="Mancozeb 75% WP",
                    active_ingredient="mancozeb",
                    priority=1,
                    effectiveness=0.65,
                    usage_note="Follow local dosage guidance and rotate fungicide groups.",
                ),
                PesticideRecommendation(
                    class_label=class_key,
                    crop=humanize_token(plant_name),
                    disease=disease_key,
                    treatment_type="chemical",
                    product_name="Copper Oxychloride 50% WP",
                    active_ingredient="copper oxychloride",
                    priority=2,
                    effectiveness=0.6,
                    usage_note="Use preventive spray and avoid excessive application.",
                ),
            ][:limit]

        return {
            "organic": organic,
            "chemical": chemical,
        }

    def _find_by_disease_and_crop(self, *, disease_key: str, plant_key: str) -> list[PesticideRecommendation]:
        if disease_key == "unknown":
            return []

        matches: list[PesticideRecommendation] = []
        for (indexed_disease, crop_key), rows in self.by_disease_crop.items():
            if indexed_disease != disease_key:
                continue
            if plant_key and plant_key not in crop_key:
                continue
            matches.extend(rows)
        return matches

    def _find_by_crop(self, *, plant_key: str) -> list[PesticideRecommendation]:
        if not plant_key:
            return []

        matches: list[PesticideRecommendation] = []
        for crop_key, rows in self.by_crop.items():
            if plant_key in crop_key:
                matches.extend(rows)
        return matches


class LocalLLMClient:
    def __init__(self, *, api_url: str, model_name: str, timeout_seconds: float) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return ""

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body.strip()

        if isinstance(parsed, dict):
            if isinstance(parsed.get("response"), str):
                return parsed["response"].strip()
            if isinstance(parsed.get("text"), str):
                return parsed["text"].strip()
            message = parsed.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
        return ""


def _format_recommendations(items: list[PesticideRecommendation]) -> str:
    lines = []
    for idx, item in enumerate(items, start=1):
        lines.append(
            (
                f"{idx}. {item.product_name} ({item.active_ingredient}) "
                f"- {item.usage_note}"
            )
        )
    return "\n".join(lines)


def build_prompt(
    *,
    plant_name: str,
    disease_name: str,
    confidence: float,
    recommendations: dict[str, list[PesticideRecommendation]],
) -> str:
    organic_block = _format_recommendations(recommendations["organic"])
    chemical_block = _format_recommendations(recommendations["chemical"])

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Use the exact sections and concise farmer-friendly language:\n"
        "1) Short Explanation\n"
        "2) Organic Treatment Option\n"
        "3) Chemical Treatment Option\n"
        "4) Safety Warning\n\n"
        f"Plant: {humanize_token(plant_name)}\n"
        f"Disease: {humanize_token(disease_name)}\n"
        f"Confidence: {confidence:.2f}\n\n"
        "Organic options (best first, top 2):\n"
        f"{organic_block}\n\n"
        "Chemical options (best first, top 2):\n"
        f"{chemical_block}\n"
    )


def build_fallback_response(
    *,
    plant_name: str,
    disease_name: str,
    confidence: float,
    recommendations: dict[str, list[PesticideRecommendation]],
) -> str:
    organic_block = _format_recommendations(recommendations["organic"])
    chemical_block = _format_recommendations(recommendations["chemical"])

    return (
        f"Short Explanation\n"
        f"Detected {humanize_token(disease_name)} on {humanize_token(plant_name)} with confidence {confidence:.2f}."
        " Act early to reduce spread and remove visibly affected leaves.\n\n"
        f"Organic Treatment Option\n{organic_block}\n\n"
        f"Chemical Treatment Option\n{chemical_block}\n\n"
        "Safety Warning\n"
        "Wear gloves, mask, and eye protection during spray. Follow label dosage, avoid windy hours, and keep"
        " children, animals, and water sources away from treated area."
    )


class ResponseManager:
    def __init__(
        self,
        *,
        db: Session,
        catalog: PesticideCatalog,
        llm_client: LocalLLMClient,
    ) -> None:
        self.db = db
        self.catalog = catalog
        self.llm_client = llm_client

    def get_or_create_response(
        self,
        *,
        prediction: PredictionRecord,
        class_label: str,
    ) -> tuple[str, bool]:
        existing_response = self._latest_response_for_prediction(prediction.id)
        if existing_response:
            return existing_response.response_text, True

        canonical_response = self._canonical_response_for_diagnosis(
            plant_name=prediction.plant_name,
            disease_name=prediction.disease_name,
        )
        if canonical_response:
            row = ResponseRecord(
                prediction_id=prediction.id,
                response_text=canonical_response.response_text,
                source="canonical",
            )
            self.db.add(row)
            self.db.flush()
            return row.response_text, True

        recommendations = self.catalog.recommend(
            class_label=class_label,
            plant_name=prediction.plant_name,
            disease_name=prediction.disease_name,
            limit=2,
        )
        prompt = build_prompt(
            plant_name=prediction.plant_name,
            disease_name=prediction.disease_name,
            confidence=prediction.confidence,
            recommendations=recommendations,
        )

        generated = self.llm_client.generate(prompt)
        if not generated:
            generated = build_fallback_response(
                plant_name=prediction.plant_name,
                disease_name=prediction.disease_name,
                confidence=prediction.confidence,
                recommendations=recommendations,
            )

        row = ResponseRecord(
            prediction_id=prediction.id,
            response_text=generated,
            source="llm",
        )
        self.db.add(row)
        self.db.flush()
        return generated, False

    def _latest_response_for_prediction(self, prediction_id: int) -> ResponseRecord | None:
        stmt = (
            select(ResponseRecord)
            .where(ResponseRecord.prediction_id == prediction_id)
            .order_by(desc(ResponseRecord.created_at), desc(ResponseRecord.id))
            .limit(1)
        )
        return self.db.scalar(stmt)

    def _canonical_response_for_diagnosis(
        self,
        *,
        plant_name: str,
        disease_name: str,
    ) -> ResponseRecord | None:
        stmt = (
            select(ResponseRecord)
            .join(PredictionRecord, PredictionRecord.id == ResponseRecord.prediction_id)
            .where(PredictionRecord.plant_name == plant_name)
            .where(PredictionRecord.disease_name == disease_name)
            .order_by(asc(ResponseRecord.created_at), asc(ResponseRecord.id))
            .limit(1)
        )
        return self.db.scalar(stmt)
