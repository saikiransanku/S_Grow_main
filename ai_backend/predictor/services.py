import base64
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import Any, Dict, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError
from zipfile import is_zipfile

import h5py
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from django.conf import settings

from .service_modules import (
    DEFAULT_ACTIVE_LEARNING_MIN_CONFIDENCE,
    DEFAULT_AGRONOMY_RETRIEVAL_MAX_REFERENCES,
    build_retrieval_tokens,
    handle_unknown_crop as _handle_unknown_crop_impl,
    load_agronomy_reference_index,
    normalize_reference_text,
    resolve_active_learning_dataset_root,
    resolve_agronomy_docs_dir,
    score_reference_snippet,
)
from .service_modules.agriculture_advisor import (
    CROP_RECOMMENDATION_ENGINE,
    FARM_PLANNING_ADVISOR,
    GENERAL_AGRICULTURE_QA,
    PLANT_DISEASE_PREDICTOR,
    build_agriculture_advisor_response,
    classify_agriculture_request,
)

try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None
    tv_models = None
    transforms = None

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

IMG_SIZE = (224, 224)
SEASON_ALIASES = {
    "all": "all_season",
    "all_season": "all_season",
    "auto": "auto",
    "kharif": "kharif",
    "rabi": "rabi",
}
VALID_SEASONS = set(SEASON_ALIASES)
SEASON_MODEL_KEYS = ("kharif", "rabi", "all_season")
LATEST_MODEL_FILE_NAME = "ssgrow_disease_model_v2.keras"
LATEST_CLASS_FILE_NAME = "class_names.txt"
LOW_CONFIDENCE_THRESHOLD = 50.0
LOW_CONFIDENCE_MESSAGE = "We are working on this plant, we will complete it soon."
UNSUPPORTED_CROP_MESSAGE = "We are working on this crop. We will be back soon."
INVALID_LEAF_OR_FRUIT_MESSAGE = "This image is not a leaf or fruit. Please upload it again."
AUTO_ALL_SEASON_CONFIDENCE_THRESHOLD = 80.0
AUTO_SEASONAL_CONFIDENCE_THRESHOLD = 60.0
EXPLICIT_SEASON_BLOCK_THRESHOLD = 95.0
PREDICTION_PIPELINE_VERSION = "multi-stage-funnel-v3"
MANUAL_REVIEW_LABEL = "manual_review_required_unrecognized_disease"
MANUAL_REVIEW_REASON = (
    "Weighted lesion evidence remained strong after penalizing a healthy model prediction."
)
DEFAULT_MIN_IMAGE_QUALITY_SCORE = 50.0
DEFAULT_MIN_IMAGE_SHARPNESS = 20.0
DEFAULT_MAX_IMAGE_GLARE_RATIO = 0.12
DEFAULT_MAX_IMAGE_DARK_RATIO = 0.45
DEFAULT_MIN_SEGMENTED_LEAF_COVERAGE_RATIO = 0.08
DEFAULT_MODEL_CONFIDENCE_TEMPERATURE = 1.0
DEFAULT_STAGE1_CROP_CONFIDENCE_THRESHOLD = 55.0
DEFAULT_HEALTHY_VISUAL_BYPASS_CONFIDENCE = 95.0
DEFAULT_HEALTHY_LESION_SIGNAL_THRESHOLD = 0.55
DEFAULT_HEALTHY_LESION_SCORE_THRESHOLD = 0.15
DEFAULT_HEALTHY_LESION_CONFIDENCE_PENALTY = 0.70
DEFAULT_TARGET_LUMINANCE = 128.0
DEFAULT_GAIN_MIN = 0.6
DEFAULT_GAIN_MAX = 2.2
DEFAULT_GAIN_EPSILON = 1.0
DEFAULT_ILLUMINATION_KERNEL = 201
DEFAULT_SPECULAR_VALUE_THRESHOLD = 245.0
DEFAULT_SPECULAR_SATURATION_THRESHOLD = 110.0
DEFAULT_SPECULAR_WARNING_RATIO = 0.10
PREPROCESSING_PIPELINE_VERSION = "adaptive-leaf-illumination-v1"
SSGROW_MODEL_VERSION = "SSGrow-CNN-v2"
SSGROW_TRAINED_SEASONS = ("Kharif", "Rabi", "All Season")
SSGROW_REQUIRED_RESPONSE_HEADINGS = (
    "Crop Identified",
    "Disease Prediction",
    "Prediction Confidence",
    "Symptoms Detected",
    "Seasonal Model Note",
    "Recommended Actions",
    "Uncertainty Note",
)
SSGROW_SEASONAL_MODEL_NOTE = (
    f"{SSGROW_MODEL_VERSION} was trained using Kharif, Rabi, and All-Season leaf datasets "
    "with augmentation including brightness, zoom, rotation, and color variation to improve "
    "performance under different field conditions."
)
SSGROW_SYSTEM_PROMPT = (
    "You are SSGrow, a crop disease assistant for farmers.\n"
    "Use simple language and short sentences.\n"
    "If the user only greets you, give a short welcome reply, use their name if provided, "
    "introduce yourself as the SSGrow AI crop assistant, and invite them to upload a leaf image.\n"
    "For crop disease help, explain CNN crop disease predictions from leaf images.\n"
    "Always answer disease-analysis questions with exactly these section titles in this order:\n"
    "Crop Identified:\n"
    "Disease Prediction:\n"
    "Prediction Confidence:\n"
    "Symptoms Detected:\n"
    "Seasonal Model Note:\n"
    "Recommended Actions:\n"
    "Uncertainty Note:\n"
    "Rules:\n"
    "- Disease Prediction must be the predicted disease name or Healthy or Unknown.\n"
    "- Prediction Confidence must be a percentage when available.\n"
    "- Symptoms Detected must use bullet points.\n"
    "- Give practical farmer-friendly treatment advice.\n"
    "- Mention the provided seasonal model note exactly.\n"
    "- If confidence is below 40%, say the result may be uncertain and ask for a clearer image.\n"
    "- If no image or no prediction context is provided, ask the user to upload a clear crop leaf image and still keep the same response format.\n"
    "- Avoid machine learning jargon, complex words, and unsupported dosage instructions.\n"
    "- Recommended Actions may use short bullet points."
)


@dataclass
class PredictionResult:
    label: str
    confidence: float
    crop_detected: str
    season_used: str
    verification_passed: bool
    verification_reason: str
    uploaded_image_data_url: str
    seasonal_comparison: Dict[str, Dict[str, object]]
    visual_analysis: str
    is_low_confidence: bool
    status_message: str
    recommended_pesticide: str
    active_ingredient: str
    usage_note: str
    leaf_visual_analysis: Dict[str, Any]
    diagnosis_status: str
    override_applied: bool
    override_reason: str
    model_label_before_override: str
    model_confidence_before_override: float
    heuristic_lesion_count: int
    preprocessing_metrics: Dict[str, Any]
    farmer_report: Dict[str, Any]
    farmer_action_plan_markdown: str


class ImageQualityError(ValueError):
    pass


@dataclass
class PreparedPredictionImage:
    image_bytes: bytes
    quality_metrics: Dict[str, float]
    segmentation_metrics: Dict[str, Any]
    preprocessing_metrics: Dict[str, Any]


@dataclass
class CropRoutingDecision:
    accepted: bool
    crop: str
    crop_confidence: float
    season: Optional[str]
    reason: str
    source: str
    disease_model_used: str = "seasonal"


@dataclass
class TorchClassifierArtifact:
    model: Any
    class_names: List[str]
    image_size: int
    artifact_path: Path
    kind: str


KNOWN_CROP_SUFFIXES = tuple(
    sorted(
        {
            "apple",
            "banana",
            "blueberry",
            "cauliflower",
            "cherry_including_sour",
            "chilli",
            "corn_maize",
            "cotton",
            "grape",
            "groundnut",
            "jackfruit",
            "mango",
            "millet",
            "okra",
            "onion",
            "orange",
            "papaya",
            "peach",
            "pearl_millet",
            "pepper_bell",
            "pigeonpea",
            "potato",
            "pumpkin",
            "raspberry",
            "rice",
            "sorghum",
            "soybean",
            "strawberry",
            "sugarcane",
            "sunflower",
            "tea",
            "tomato",
        },
        key=len,
        reverse=True,
    )
)
NON_LEAF_OR_FRUIT_STAGE1_CLASSES = frozenset(
    {
        "other",
        "others",
        "non_leaf",
        "non_leaf_or_fruit",
        "not_leaf",
        "not_leaf_or_fruit",
    }
)


class ModelRegistry:
    def __init__(self) -> None:
        self.runs_dir = _resolve_runs_dir()
        self.artifact_paths = resolve_prediction_artifact_paths()
        self.models = {
            season: self._load_model(self.artifact_paths[season]["model"])
            for season in SEASON_MODEL_KEYS
        }
        self.class_names_by_model = {
            season: self._load_model_classes(
                model_key=season,
                class_file_path=self.artifact_paths[season]["classes"],
            )
            for season in SEASON_MODEL_KEYS
        }
        self.supported_crops_by_model = {
            season: self._collect_supported_crops(self.class_names_by_model[season])
            for season in SEASON_MODEL_KEYS
        }
        self.supported_seasons_by_crop: Dict[str, set[str]] = {}
        for season, crops in self.supported_crops_by_model.items():
            for crop in crops:
                self.supported_seasons_by_crop.setdefault(crop, set()).add(season)
        self.prediction_temperature = max(
            0.05,
            _env_float(
                "MODEL_CONFIDENCE_TEMPERATURE",
                DEFAULT_MODEL_CONFIDENCE_TEMPERATURE,
            ),
        )
        self.stage1_artifact_paths = resolve_stage1_crop_artifact_paths()
        self.stage1_crop_classifier = self._load_optional_torch_classifier(
            model_path=self.stage1_artifact_paths["model"],
            class_file_path=self.stage1_artifact_paths["classes"],
            kind="plant_classifier",
        )
        self.crop_specific_models_dir = _resolve_crop_specific_models_dir()
        self.crop_specific_model_paths = self._discover_crop_specific_model_paths(
            self.crop_specific_models_dir
        )
        self.crop_specific_models: Dict[str, TorchClassifierArtifact] = {}

    def _load_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return load_compat_model(model_path)

    def _read_classes_file(self, path: Path):
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _load_model_classes(self, model_key: str, class_file_path: Path):
        class_names = self._read_classes_file(class_file_path)
        if not class_names:
            raise FileNotFoundError(f"Class names file not found or empty: {class_file_path}")
        model_units = int(self.models[model_key].output_shape[-1])

        if len(class_names) != model_units:
            raise ValueError(
                f"Class names count mismatch for {model_key}: "
                f"{len(class_names)} names for {model_units} model outputs."
            )
        return class_names

    def _collect_supported_crops(self, class_names: List[str]) -> set[str]:
        crops = {extract_crop_from_label(label) for label in class_names}
        return {crop for crop in crops if crop != "unknown"}

    def preprocess_image(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_arr = np.array(image).astype(np.float32)
        expects_0_255 = os.getenv("MODEL_INPUT_0_255", "true").lower() == "true"
        if not expects_0_255:
            image_arr = image_arr / 255.0
        image_arr = np.expand_dims(image_arr, axis=0)
        return image_arr

    def _load_torch_class_names(
        self,
        checkpoint: Dict[str, Any],
        class_file_path: Optional[Path],
    ) -> List[str]:
        class_names = checkpoint.get("class_names")
        if isinstance(class_names, list) and class_names:
            return [str(item).strip() for item in class_names if str(item).strip()]

        if class_file_path and class_file_path.exists():
            try:
                payload = json.loads(class_file_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [str(item).strip() for item in payload if str(item).strip()]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        return []

    def _create_torch_classifier_model(self, num_classes: int, kind: str):
        if tv_models is None or nn is None:
            raise RuntimeError("Torch vision inference is unavailable in this runtime.")

        model = tv_models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        if kind == "crop_disease":
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25, inplace=True),
                nn.Linear(512, num_classes),
            )
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.35, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        return model

    def _load_optional_torch_classifier(
        self,
        *,
        model_path: Path,
        class_file_path: Optional[Path],
        kind: str,
    ) -> Optional[TorchClassifierArtifact]:
        if torch is None or tv_models is None or nn is None:
            return None
        if not model_path.exists():
            return None

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception:  # pragma: no cover - depends on local artifact compatibility
            return None

        if not isinstance(checkpoint, dict):
            return None

        class_names = self._load_torch_class_names(checkpoint, class_file_path)
        if not class_names:
            return None

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return None

        resolved_kind = kind
        if checkpoint.get("plant_name"):
            resolved_kind = "crop_disease"

        image_size = int(checkpoint.get("image_size") or 300)
        try:
            model = self._create_torch_classifier_model(
                num_classes=len(class_names),
                kind=resolved_kind,
            )
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:  # pragma: no cover - depends on local artifact compatibility
            return None

        return TorchClassifierArtifact(
            model=model,
            class_names=class_names,
            image_size=image_size,
            artifact_path=model_path,
            kind=resolved_kind,
        )

    def _discover_crop_specific_model_paths(self, root: Path) -> Dict[str, Dict[str, Path]]:
        if not root.exists():
            return {}

        mappings: Dict[str, Dict[str, Path]] = {}
        for model_path in sorted(root.rglob("*_disease_model.pth")):
            crop = _normalize_label_text(model_path.stem.removesuffix("_disease_model"))
            if not crop:
                continue
            class_path = model_path.with_name(f"{crop}_disease_classes.json")
            mappings[crop] = {
                "model": model_path,
                "classes": class_path,
            }
        return mappings

    def _preprocess_torch_image(self, image_bytes: bytes, image_size: int):
        if transforms is None or torch is None:
            raise RuntimeError("Torch vision inference is unavailable in this runtime.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(image).unsqueeze(0)

    def _predict_torch_artifact(
        self,
        artifact: TorchClassifierArtifact,
        image_bytes: bytes,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        tensor = self._preprocess_torch_image(image_bytes, artifact.image_size)
        with torch.no_grad():
            raw_scores = artifact.model(tensor).detach().cpu().numpy().reshape(-1)

        calibrated_scores = _apply_temperature_scaling(
            raw_scores,
            self.prediction_temperature,
        )
        idx = int(np.argmax(calibrated_scores))
        confidence = float(calibrated_scores[idx])
        label = (
            artifact.class_names[idx]
            if idx < len(artifact.class_names)
            else "unknown"
        )
        return {
            "label": label,
            "confidence": confidence,
            "top_k": _build_top_k_predictions(
                calibrated_scores,
                artifact.class_names,
                top_k=top_k,
            ),
        }

    def predict_details(self, model_key: str, image_batch, top_k: int = 3) -> Dict[str, Any]:
        model = self.models[model_key]
        preds = model.predict(image_batch, verbose=0)
        calibrated_scores = _apply_temperature_scaling(
            np.asarray(preds).reshape(-1),
            self.prediction_temperature,
        )
        idx = int(np.argmax(calibrated_scores))
        confidence = float(calibrated_scores[idx])
        model_classes = self.class_names_by_model[model_key]
        label = model_classes[idx] if idx < len(model_classes) else "unknown"
        return {
            "label": label,
            "confidence": confidence,
            "top_k": _build_top_k_predictions(
                calibrated_scores,
                model_classes,
                top_k=top_k,
            ),
        }

    def predict(self, model_key: str, image_batch):
        details = self.predict_details(model_key, image_batch, top_k=1)
        return str(details["label"]), float(details["confidence"])

    def route_supported_crop(self, image_bytes: bytes) -> Optional[CropRoutingDecision]:
        if not _env_flag("ENABLE_HIERARCHICAL_ROUTING", True):
            return None
        if self.stage1_crop_classifier is None:
            return None

        stage1_prediction = self._predict_torch_artifact(
            self.stage1_crop_classifier,
            image_bytes,
            top_k=3,
        )
        crop = _normalize_label_text(str(stage1_prediction["label"]))
        confidence = round(float(stage1_prediction["confidence"]) * 100.0, 2)
        confidence_threshold = _env_float(
            "STAGE1_CROP_CONFIDENCE_THRESHOLD",
            DEFAULT_STAGE1_CROP_CONFIDENCE_THRESHOLD,
        )

        if _is_non_leaf_or_fruit_class(crop) and confidence >= confidence_threshold:
            return CropRoutingDecision(
                accepted=False,
                crop=crop,
                crop_confidence=confidence,
                season=None,
                reason=(
                    "Stage-1 crop gate classified this upload as a non-leaf-or-fruit image "
                    f"with {confidence}% confidence."
                ),
                source="stage1_crop_classifier",
                disease_model_used="rejected_input",
            )

        inferred_season = infer_season_from_crop(
            crop,
            seasons_by_crop=self.supported_seasons_by_crop,
        )

        if not crop or crop == "unknown" or inferred_season == "unknown":
            return CropRoutingDecision(
                accepted=False,
                crop=crop or "unknown",
                crop_confidence=confidence,
                season=None,
                reason=(
                    "Stage-1 crop gate could not match this image to a supported crop. "
                    "The image may be unsupported or out of distribution for the current models."
                ),
                source="stage1_crop_classifier",
            )

        if confidence < confidence_threshold:
            return CropRoutingDecision(
                accepted=False,
                crop=crop,
                crop_confidence=confidence,
                season=None,
                reason=(
                    "Stage-1 crop gate detected "
                    f"{crop.replace('_', ' ')} at only {confidence}% confidence, "
                    f"below the supported-crop threshold of {round(confidence_threshold, 2)}%."
                ),
                source="stage1_crop_classifier",
            )

        return CropRoutingDecision(
            accepted=True,
            crop=crop,
            crop_confidence=confidence,
            season=inferred_season,
            reason=(
                "Stage-1 crop gate routed this image to "
                f"{crop.replace('_', ' ')} with {confidence}% confidence."
            ),
            source="stage1_crop_classifier",
        )

    def get_crop_specific_model(self, crop: str) -> Optional[TorchClassifierArtifact]:
        crop_key = _normalize_label_text(crop)
        if not crop_key:
            return None
        if crop_key in self.crop_specific_models:
            return self.crop_specific_models[crop_key]

        artifact_paths = self.crop_specific_model_paths.get(crop_key)
        if not artifact_paths:
            return None

        artifact = self._load_optional_torch_classifier(
            model_path=artifact_paths["model"],
            class_file_path=artifact_paths["classes"],
            kind="crop_disease",
        )
        if artifact is not None:
            self.crop_specific_models[crop_key] = artifact
        return artifact

    def predict_crop_specific_disease(
        self,
        crop: str,
        image_bytes: bytes,
    ) -> Optional[Dict[str, Any]]:
        artifact = self.get_crop_specific_model(crop)
        if artifact is None:
            return None

        prediction = self._predict_torch_artifact(artifact, image_bytes, top_k=3)
        raw_label = str(prediction["label"] or "unknown")
        normalized_label = _normalize_crop_specific_label(crop, raw_label)
        return {
            "disease_prediction": normalized_label,
            "crop_detected": crop,
            "confidence": round(float(prediction["confidence"]) * 100.0, 2),
            "source": "crop_specific_model",
            "top_k": prediction["top_k"],
        }


def _normalize_token(raw: str) -> str:
    return "".join(ch.lower() for ch in raw if ch.isalnum())


def _normalize_label_text(raw: str) -> str:
    cleaned = (raw or "").strip().lower().replace(",", "")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _is_non_leaf_or_fruit_class(label: str) -> bool:
    return _normalize_label_text(label) in NON_LEAF_OR_FRUIT_STAGE1_CLASSES


def _titleize_label(label: str) -> str:
    return label.replace("_", " ").replace(",", "").strip().title()


def extract_crop_from_label(label: str) -> str:
    normalized_label = _normalize_label_text(label)
    if not normalized_label:
        return "unknown"

    for crop_name in KNOWN_CROP_SUFFIXES:
        if (
            normalized_label == crop_name
            or normalized_label.endswith(f"_{crop_name}")
            or normalized_label.startswith(f"{crop_name}_")
        ):
            return crop_name
    return "unknown"


def infer_season_from_crop(
    crop: str,
    seasons_by_crop: Optional[Dict[str, set[str]]] = None,
) -> str:
    crop_key = _normalize_label_text(crop)
    crop_to_seasons = seasons_by_crop or get_registry().supported_seasons_by_crop
    seasons = crop_to_seasons.get(crop_key, set())
    if not seasons:
        return "unknown"
    if len(seasons) == 1:
        return next(iter(seasons))
    if "all_season" in seasons:
        return "all_season"
    return next(iter(sorted(seasons)))


def normalize_requested_season(season: Optional[str]) -> str:
    normalized = (season or "auto").strip().lower()
    return SEASON_ALIASES.get(normalized, "auto")


class _TorchArtifactDiscoveryCNN:
    def __init__(self, registry: "ModelRegistry", artifact: TorchClassifierArtifact) -> None:
        self.registry = registry
        self.artifact = artifact

    def predict(self, image_data: Any) -> Tuple[str, float]:
        pil_image = image_data_to_pil_image(image_data)
        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        prediction = self.registry._predict_torch_artifact(
            self.artifact,
            output.getvalue(),
            top_k=1,
        )
        return str(prediction["label"] or "unknown"), float(prediction["confidence"])


def handle_unknown_crop(image_data, discovery_cnn, dataset_root_path):
    return _handle_unknown_crop_impl(
        image_data,
        discovery_cnn,
        dataset_root_path,
        minimum_confidence=_env_float(
            "ACTIVE_LEARNING_MIN_CONFIDENCE",
            DEFAULT_ACTIVE_LEARNING_MIN_CONFIDENCE,
        ),
    )


def _get_active_learning_discovery_cnn(models: ModelRegistry) -> Optional[_TorchArtifactDiscoveryCNN]:
    artifact = getattr(models, "stage1_crop_classifier", None)
    if artifact is None:
        return None
    return _TorchArtifactDiscoveryCNN(models, artifact)


def _run_active_learning_fallback(
    *,
    image_bytes: bytes,
    models: ModelRegistry,
) -> Optional[Dict[str, str]]:
    if not _env_flag("ENABLE_ACTIVE_LEARNING_FALLBACK", True):
        return None

    discovery_cnn = _get_active_learning_discovery_cnn(models)
    if discovery_cnn is None:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

    try:
        return handle_unknown_crop(
            image,
            discovery_cnn,
            resolve_active_learning_dataset_root(),
        )
    except (OSError, TypeError, ValueError):
        return None


def _apply_temperature_scaling(raw_scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.array([], dtype=np.float32)

    calibrated_temperature = max(float(temperature), 1e-3)
    if scores.size == 1:
        return np.array([1.0], dtype=np.float32)

    finite_scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    score_sum = float(np.sum(finite_scores))
    looks_like_probability = bool(
        np.all(finite_scores >= 0.0) and 0.98 <= score_sum <= 1.02
    )

    if looks_like_probability:
        logits = np.log(np.clip(finite_scores, 1e-9, 1.0))
        scaled = logits / calibrated_temperature
    else:
        scaled = finite_scores / calibrated_temperature

    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    denominator = max(float(np.sum(exp_values)), 1e-9)
    return (exp_values / denominator).astype(np.float32)


def _build_top_k_predictions(
    scores: np.ndarray,
    class_names: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    flattened = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return []

    limit = max(1, min(int(top_k), flattened.size, len(class_names)))
    ranked_indices = np.argsort(flattened)[::-1][:limit]
    return [
        {
            "label": class_names[idx] if idx < len(class_names) else "unknown",
            "confidence": round(float(flattened[idx]) * 100.0, 2),
        }
        for idx in ranked_indices.tolist()
    ]


def _normalize_crop_specific_label(crop: str, label: str) -> str:
    crop_key = _normalize_label_text(crop)
    normalized_label = _normalize_label_text(label)
    if not normalized_label:
        return f"unknown_{crop_key}" if crop_key else "unknown"
    if extract_crop_from_label(normalized_label) != "unknown":
        return normalized_label
    if normalized_label in {"healthy", "healthy_leaf", "healthyleaf"}:
        return f"healthy_{crop_key}" if crop_key else "healthy"
    if normalized_label.startswith("healthy_") and crop_key:
        return f"healthy_{crop_key}"
    if crop_key and normalized_label.endswith(f"_{crop_key}"):
        return normalized_label
    return f"{normalized_label}_{crop_key}" if crop_key else normalized_label


def build_image_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _image_bytes_to_bgr_array(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ImageQualityError(
            "Please upload a valid leaf image before prediction.",
        ) from exc

    arr = np.array(image)
    if arr.size == 0:
        raise ImageQualityError(
            "Please upload a valid leaf image before prediction.",
        )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_array_to_image_bytes(image_bgr: np.ndarray, content_type: str) -> bytes:
    rgb = cv2.cvtColor(
        np.clip(image_bgr, 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2RGB,
    )
    image = Image.fromarray(rgb)
    output = io.BytesIO()
    save_format = "PNG" if "png" in (content_type or "").lower() else "JPEG"
    image.save(output, format=save_format, quality=95)
    return output.getvalue()


def _estimate_image_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    glare_mask = (value >= 240.0) & (saturation <= 45.0)
    glare_ratio = float(np.mean(glare_mask))
    extreme_dark_ratio = float(np.mean(gray <= 25.0))

    sharpness_score = min(
        100.0,
        (np.log1p(max(sharpness, 0.0)) / np.log1p(400.0)) * 100.0,
    )
    contrast_score = min(100.0, max(0.0, (contrast / 64.0) * 100.0))
    brightness_score = max(
        0.0,
        100.0 - (abs(brightness - 135.0) / 135.0) * 100.0,
    )
    glare_score = max(0.0, 100.0 - (glare_ratio * 450.0))
    exposure_score = max(0.0, 100.0 - (extreme_dark_ratio * 250.0))

    quality_score = round(
        (0.42 * sharpness_score)
        + (0.18 * contrast_score)
        + (0.20 * brightness_score)
        + (0.12 * glare_score)
        + (0.08 * exposure_score),
        2,
    )
    return {
        "quality_score": quality_score,
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
        "glare_ratio": round(glare_ratio, 4),
        "extreme_dark_ratio": round(extreme_dark_ratio, 4),
    }


def _resolve_odd_kernel_size(
    min_dimension: int,
    requested: int,
    *,
    minimum: int = 3,
    maximum_ratio: float = 0.45,
) -> int:
    if min_dimension <= 1:
        return 1

    image_limit = min_dimension if min_dimension % 2 == 1 else max(1, min_dimension - 1)
    ratio_limit = max(3, int(min_dimension * maximum_ratio))
    if ratio_limit % 2 == 0:
        ratio_limit -= 1

    normalized_minimum = max(1, int(minimum))
    if normalized_minimum % 2 == 0:
        normalized_minimum += 1

    kernel = max(normalized_minimum, int(requested))
    if kernel % 2 == 0:
        kernel += 1

    kernel = min(kernel, image_limit, ratio_limit)
    if kernel % 2 == 0:
        kernel -= 1
    return max(1, kernel)


def _compute_lighting_metrics(image_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    return {
        "mean_value": round(float(np.mean(value)), 2),
        "bright_pixel_ratio": round(float(np.mean(value >= 220.0)), 4),
        "dark_pixel_ratio": round(float(np.mean(value <= 55.0)), 4),
    }


def _detect_specular_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    value_threshold = _env_float(
        "SPECULAR_VALUE_THRESHOLD",
        DEFAULT_SPECULAR_VALUE_THRESHOLD,
    )
    saturation_threshold = _env_float(
        "SPECULAR_SATURATION_THRESHOLD",
        DEFAULT_SPECULAR_SATURATION_THRESHOLD,
    )
    value_only_mask = value >= value_threshold
    mask = value_only_mask & (
        (saturation <= saturation_threshold) | (gray.astype(np.float32) >= value_threshold)
    )
    if not np.any(mask) and float(np.mean(value_only_mask)) > 0.02:
        mask = value_only_mask

    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel_size = _resolve_odd_kernel_size(
        min(image_bgr.shape[:2]),
        requested=5,
        minimum=3,
        maximum_ratio=0.04,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_uint8


def _smooth_gain_map(gain_map: np.ndarray, guide_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
        radius = max(4, int(_env_float("GUIDED_FILTER_RADIUS", 12.0)))
        eps = max(1e-6, _env_float("GUIDED_FILTER_EPS", 1e-3))
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide_gray,
            src=gain_map.astype(np.float32),
            radius=radius,
            eps=eps,
        )
        return smoothed, "guided_filter"

    diameter = max(3, int(_env_float("GAIN_BILATERAL_DIAMETER", 9.0)))
    if diameter % 2 == 0:
        diameter += 1
    sigma_color = max(0.01, _env_float("GAIN_BILATERAL_SIGMA_COLOR", 0.12))
    sigma_space = max(3.0, _env_float("GAIN_BILATERAL_SIGMA_SPACE", 21.0))
    smoothed = cv2.bilateralFilter(
        gain_map.astype(np.float32),
        diameter,
        sigma_color,
        sigma_space,
    )
    return smoothed, "bilateral_filter"


def _apply_local_contrast_restoration(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def _should_prefer_corrected_image(
    original_quality: Dict[str, float],
    corrected_quality: Dict[str, float],
    preprocessing_metrics: Dict[str, Any],
) -> bool:
    original_score = float(original_quality.get("quality_score") or 0.0)
    corrected_score = float(corrected_quality.get("quality_score") or 0.0)
    if corrected_score >= original_score:
        return True

    target = float(preprocessing_metrics.get("target_mean_value") or DEFAULT_TARGET_LUMINANCE)
    mean_before = float(preprocessing_metrics.get("mean_value_before") or 0.0)
    mean_after = float(preprocessing_metrics.get("mean_value_after") or 0.0)
    bright_before = float(preprocessing_metrics.get("bright_pixel_ratio_before") or 0.0)
    bright_after = float(preprocessing_metrics.get("bright_pixel_ratio_after") or 0.0)
    dark_before = float(preprocessing_metrics.get("dark_pixel_ratio_before") or 0.0)
    dark_after = float(preprocessing_metrics.get("dark_pixel_ratio_after") or 0.0)
    spec_before = float(preprocessing_metrics.get("specular_ratio_before") or 0.0)
    spec_after = float(preprocessing_metrics.get("specular_ratio_after") or 0.0)

    lighting_improved = (
        abs(mean_after - target) + 4.0 < abs(mean_before - target)
        or bright_after + 0.01 < bright_before
        or dark_after + 0.01 < dark_before
        or spec_after + 0.003 < spec_before
    )
    return corrected_score >= (original_score - 8.0) and lighting_improved


def _apply_adaptive_leaf_illumination_correction(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    initial_metrics = _compute_lighting_metrics(image_bgr)
    target_mean_value = float(
        np.clip(
            _env_float("TARGET_LUMINANCE", DEFAULT_TARGET_LUMINANCE),
            110.0,
            140.0,
        )
    )
    min_gain = _env_float("GAIN_MIN", DEFAULT_GAIN_MIN)
    max_gain = _env_float("GAIN_MAX", DEFAULT_GAIN_MAX)
    gain_epsilon = max(1e-3, _env_float("GAIN_EPSILON", DEFAULT_GAIN_EPSILON))

    if not _env_flag("ENABLE_ADAPTIVE_LEAF_CORRECTION", True):
        disabled_metrics = {
            "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "selected_variant": "original",
            "target_mean_value": round(target_mean_value, 2),
            "mean_value_before": initial_metrics["mean_value"],
            "mean_value_after": initial_metrics["mean_value"],
            "bright_pixel_ratio_before": initial_metrics["bright_pixel_ratio"],
            "bright_pixel_ratio_after": initial_metrics["bright_pixel_ratio"],
            "dark_pixel_ratio_before": initial_metrics["dark_pixel_ratio"],
            "dark_pixel_ratio_after": initial_metrics["dark_pixel_ratio"],
            "specular_ratio_before": 0.0,
            "specular_ratio_after": 0.0,
            "specular_pixels_inpainted": 0,
            "specular_inpaint_applied": False,
            "gain_min": 1.0,
            "gain_max": 1.0,
            "gain_mean": 1.0,
            "illumination_kernel": 0,
            "edge_aware_smoother": "disabled",
            "denoise_applied": False,
            "warnings": [],
        }
        return image_bgr, disabled_metrics

    working = image_bgr.copy()
    specular_mask = _detect_specular_mask(working)
    specular_pixels = int(np.count_nonzero(specular_mask))
    specular_ratio_before = float(specular_pixels / max(1, specular_mask.size))
    specular_inpaint_applied = specular_pixels > 0
    if specular_inpaint_applied:
        inpaint_radius = max(1, int(_env_float("SPECULAR_INPAINT_RADIUS", 3.0)))
        working = cv2.inpaint(working, specular_mask, inpaint_radius, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    illumination_kernel = _resolve_odd_kernel_size(
        min(working.shape[:2]),
        requested=int(_env_float("ILLUMINATION_KERNEL", DEFAULT_ILLUMINATION_KERNEL)),
        minimum=31,
        maximum_ratio=0.5,
    )
    illumination = cv2.GaussianBlur(
        l_channel,
        (illumination_kernel, illumination_kernel),
        0,
    )
    illumination = np.maximum(illumination, gain_epsilon)

    gain_map = np.clip(target_mean_value / illumination, min_gain, max_gain).astype(np.float32)
    smoothed_gain, smoother_name = _smooth_gain_map(gain_map, working)
    smoothed_gain = np.clip(smoothed_gain, min_gain, max_gain)

    corrected = np.clip(
        working.astype(np.float32) * smoothed_gain[:, :, None],
        0.0,
        255.0,
    ).astype(np.uint8)

    should_denoise = _env_flag("ENABLE_ADAPTIVE_POST_DENOISE", True) and (
        float(np.max(smoothed_gain)) > 1.35 or initial_metrics["dark_pixel_ratio"] > 0.08
    )
    if should_denoise:
        corrected = cv2.fastNlMeansDenoisingColored(corrected, None, 4, 4, 5, 15)

    corrected = _apply_local_contrast_restoration(corrected)
    corrected_metrics = _compute_lighting_metrics(corrected)
    specular_ratio_after = float(
        np.count_nonzero(_detect_specular_mask(corrected)) / max(1, corrected.shape[0] * corrected.shape[1])
    )

    warnings: List[str] = []
    if specular_ratio_before > _env_float("SPECULAR_WARNING_RATIO", DEFAULT_SPECULAR_WARNING_RATIO):
        warnings.append(
            "Strong glare covered more than 10% of the image. Retake is recommended if the result looks uncertain."
        )
    if corrected_metrics["bright_pixel_ratio"] > 0.25 and initial_metrics["bright_pixel_ratio"] > 0.30:
        warnings.append(
            "Lighting remains uneven after correction. A retake in softer light may improve accuracy."
        )

    preprocessing_metrics: Dict[str, Any] = {
        "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
        "selected_variant": "adaptive_correction",
        "target_mean_value": round(target_mean_value, 2),
        "mean_value_before": initial_metrics["mean_value"],
        "mean_value_after": corrected_metrics["mean_value"],
        "bright_pixel_ratio_before": initial_metrics["bright_pixel_ratio"],
        "bright_pixel_ratio_after": corrected_metrics["bright_pixel_ratio"],
        "dark_pixel_ratio_before": initial_metrics["dark_pixel_ratio"],
        "dark_pixel_ratio_after": corrected_metrics["dark_pixel_ratio"],
        "specular_ratio_before": round(specular_ratio_before, 4),
        "specular_ratio_after": round(specular_ratio_after, 4),
        "specular_pixels_inpainted": specular_pixels,
        "specular_inpaint_applied": specular_inpaint_applied,
        "gain_min": round(float(np.min(smoothed_gain)), 4),
        "gain_max": round(float(np.max(smoothed_gain)), 4),
        "gain_mean": round(float(np.mean(smoothed_gain)), 4),
        "illumination_kernel": illumination_kernel,
        "edge_aware_smoother": smoother_name,
        "denoise_applied": should_denoise,
        "warnings": warnings,
    }
    return corrected, preprocessing_metrics


def _validate_image_quality_gate(quality_metrics: Dict[str, float]) -> None:
    min_quality_score = _env_float(
        "MIN_IMAGE_QUALITY_SCORE",
        DEFAULT_MIN_IMAGE_QUALITY_SCORE,
    )
    min_sharpness = _env_float(
        "MIN_IMAGE_SHARPNESS",
        DEFAULT_MIN_IMAGE_SHARPNESS,
    )
    max_glare_ratio = _env_float(
        "MAX_IMAGE_GLARE_RATIO",
        DEFAULT_MAX_IMAGE_GLARE_RATIO,
    )
    max_dark_ratio = _env_float(
        "MAX_IMAGE_DARK_RATIO",
        DEFAULT_MAX_IMAGE_DARK_RATIO,
    )

    if float(quality_metrics.get("glare_ratio") or 0.0) > max_glare_ratio:
        raise ImageQualityError(
            "The image has too much glare or reflection. Retake it with softer light and avoid shiny hotspots on the leaf.",
        )
    if float(quality_metrics.get("extreme_dark_ratio") or 0.0) > max_dark_ratio:
        raise ImageQualityError(
            "The image is too dark for reliable analysis. Retake it in brighter natural light without heavy shadows.",
        )
    if (
        float(quality_metrics.get("quality_score") or 0.0) < min_quality_score
        or float(quality_metrics.get("sharpness") or 0.0) < min_sharpness
    ):
        raise ImageQualityError(
            "Please reupload a clearer image. Keep the leaf in focus and make sure it covers at least 50% of the frame.",
        )


def _segment_leaf_foreground(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not _env_flag("ENABLE_BACKGROUND_SEGMENTATION", True):
        return image_bgr, {
            "applied": False,
            "reason": "segmentation_disabled",
        }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    exg = (2.0 * g) - r - b
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    base_mask = (
        (exg >= np.percentile(exg, 55))
        & (saturation >= max(20.0, np.percentile(saturation, 35)))
        & (value > 25.0)
    )
    if float(base_mask.mean()) < 0.05:
        base_mask = exg >= np.percentile(exg, 72)

    mask = (base_mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return image_bgr, {
            "applied": False,
            "reason": "no_leaf_component",
            "leaf_coverage_ratio": 0.0,
        }

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_index = int(np.argmax(component_areas)) + 1
    x = int(stats[component_index, cv2.CC_STAT_LEFT])
    y = int(stats[component_index, cv2.CC_STAT_TOP])
    width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    image_area = max(1, image_bgr.shape[0] * image_bgr.shape[1])
    coverage_ratio = float(area / image_area)

    minimum_coverage = _env_float(
        "MIN_SEGMENTED_LEAF_COVERAGE_RATIO",
        DEFAULT_MIN_SEGMENTED_LEAF_COVERAGE_RATIO,
    )
    if coverage_ratio < minimum_coverage:
        return image_bgr, {
            "applied": False,
            "reason": "low_leaf_coverage",
            "leaf_coverage_ratio": round(coverage_ratio, 4),
            "bbox": [x, y, x + width, y + height],
        }

    padding_x = max(4, int(width * 0.08))
    padding_y = max(4, int(height * 0.08))
    x0 = max(0, x - padding_x)
    y0 = max(0, y - padding_y)
    x1 = min(image_bgr.shape[1], x + width + padding_x)
    y1 = min(image_bgr.shape[0], y + height + padding_y)

    cropped = image_bgr[y0:y1, x0:x1].copy()
    local_mask = labels[y0:y1, x0:x1] == component_index
    if not np.any(local_mask):
        return image_bgr, {
            "applied": False,
            "reason": "empty_leaf_mask",
            "leaf_coverage_ratio": round(coverage_ratio, 4),
        }

    fill_color = np.median(cropped[local_mask], axis=0).astype(np.uint8)
    cropped[~local_mask] = fill_color
    return cropped, {
        "applied": True,
        "reason": "segmented_leaf_foreground",
        "leaf_coverage_ratio": round(coverage_ratio, 4),
        "bbox": [x0, y0, x1, y1],
        "crop_shape": [int(cropped.shape[1]), int(cropped.shape[0])],
    }


def run_input_gatekeepers(image_bytes: bytes, content_type: str) -> PreparedPredictionImage:
    original = _image_bytes_to_bgr_array(image_bytes)
    original_quality = _estimate_image_quality(original)
    corrected, preprocessing_metrics = _apply_adaptive_leaf_illumination_correction(original)
    corrected_quality = _estimate_image_quality(corrected)

    best_image = original
    best_quality = original_quality
    selected_variant = "original"
    if _should_prefer_corrected_image(
        original_quality,
        corrected_quality,
        preprocessing_metrics,
    ):
        best_image = corrected
        best_quality = corrected_quality
        selected_variant = "adaptive_correction"

    segmented_image, segmentation_metrics = _segment_leaf_foreground(best_image)
    if bool(segmentation_metrics.get("applied")):
        segmented_quality = _estimate_image_quality(segmented_image)
        if segmented_quality["quality_score"] >= (best_quality["quality_score"] - 8.0):
            best_image = segmented_image
            best_quality = segmented_quality
            if selected_variant == "adaptive_correction":
                selected_variant = "adaptive_correction_with_segmentation"
            else:
                selected_variant = "segmented_leaf"

    preprocessing_metrics["quality_score_original"] = float(original_quality["quality_score"])
    preprocessing_metrics["quality_score_corrected"] = float(corrected_quality["quality_score"])
    preprocessing_metrics["quality_score_selected"] = float(best_quality["quality_score"])
    preprocessing_metrics["segmentation_applied"] = bool(segmentation_metrics.get("applied"))
    preprocessing_metrics["segmentation_selected"] = selected_variant in {
        "segmented_leaf",
        "adaptive_correction_with_segmentation",
    }
    preprocessing_metrics["selected_variant"] = selected_variant

    _validate_image_quality_gate(best_quality)

    return PreparedPredictionImage(
        image_bytes=_bgr_array_to_image_bytes(best_image, content_type),
        quality_metrics=best_quality,
        segmentation_metrics=segmentation_metrics,
        preprocessing_metrics=preprocessing_metrics,
    )


def prepare_image_for_prediction(image_bytes: bytes, content_type: str) -> bytes:
    prepared = run_input_gatekeepers(image_bytes, content_type)
    return prepared.image_bytes


def _default_pesticide_record() -> Dict[str, str]:
    return {
        "recommended_pesticide": "Expert review required",
        "active_ingredient": "n/a",
        "usage_note": "No exact pesticide match found for this disease label.",
    }


def _normalize_label_key(label: str) -> str:
    return (label or "").strip().lower()


def _load_pesticide_index() -> Dict[str, Dict[str, str]]:
    project_root = settings.BASE_DIR.parent
    csv_path = Path(
        os.getenv(
            "PESTICIDES_CSV",
            project_root / "data" / "outputs" / "project_data" / "pesticides_data.csv",
        )
    )
    if not csv_path.exists():
        return {}

    index: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = _normalize_label_key(row.get("class_label", ""))
            if not key:
                continue
            index[key] = {
                "recommended_pesticide": row.get("recommended_pesticide", "").strip()
                or "Expert review required",
                "active_ingredient": row.get("active_ingredient", "").strip() or "n/a",
                "usage_note": row.get("usage_note", "").strip()
                or "Follow local agronomy guidelines before spraying.",
            }
    return index


def get_pesticide_recommendation(disease_label: str) -> Dict[str, str]:
    global pesticide_index
    if pesticide_index is None:
        pesticide_index = _load_pesticide_index()

    key = _normalize_label_key(disease_label)
    if key in pesticide_index:
        return pesticide_index[key]
    return _default_pesticide_record()


def retrieve_agronomy_references(
    *,
    crop_detected: str,
    disease_label: str,
    pesticide: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    if not _env_flag("ENABLE_AGRONOMY_RETRIEVAL", True):
        return []

    max_references = max(
        1,
        int(
            _env_float(
                "AGRONOMY_RETRIEVAL_MAX_REFERENCES",
                DEFAULT_AGRONOMY_RETRIEVAL_MAX_REFERENCES,
            )
        ),
    )
    tokens = build_retrieval_tokens(crop_detected, disease_label)
    if not tokens:
        return []

    references: List[Dict[str, Any]] = []
    pesticide_record = pesticide or get_pesticide_recommendation(disease_label)
    chemical_name = str(pesticide_record.get("recommended_pesticide") or "").strip()
    usage_note = str(pesticide_record.get("usage_note") or "").strip()
    if chemical_name and chemical_name.lower() not in {"n/a", "expert review required"}:
        references.append(
            {
                "title": "Pesticide dataset",
                "source": "pesticides_data.csv",
                "snippet": normalize_reference_text(
                    f"Mapped pesticide: {chemical_name}. {usage_note or 'Follow local label guidance.'}"
                ),
                "score": 99.0,
            }
        )

    global agronomy_reference_index
    if agronomy_reference_index is None:
        agronomy_reference_index = load_agronomy_reference_index()

    for item in agronomy_reference_index or []:
        snippet = str(item.get("snippet") or "")
        score = score_reference_snippet(snippet, tokens)
        if score <= 0:
            continue
        references.append(
            {
                "title": str(item.get("title") or "Agronomy note"),
                "source": str(item.get("source") or ""),
                "snippet": snippet,
                "score": round(score, 3),
            }
        )

    ranked: List[Dict[str, str]] = []
    seen_signatures = set()
    for item in sorted(references, key=lambda row: float(row.get("score") or 0.0), reverse=True):
        signature = (
            str(item.get("title") or "").strip().lower(),
            str(item.get("snippet") or "").strip().lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ranked.append(
            {
                "title": str(item.get("title") or "Agronomy note"),
                "source": str(item.get("source") or ""),
                "snippet": str(item.get("snippet") or ""),
            }
        )
        if len(ranked) >= max_references:
            break

    return ranked


def _format_leaf_name(crop_detected: str) -> str:
    crop = (crop_detected or "").strip()
    if not crop or crop.lower() == "unknown":
        return "Unknown Plant"
    return crop.replace(",", "").replace("_", " ").title()


def _extract_disease_name(label: str, diagnosis_status: str) -> str:
    if diagnosis_status == "manual_review_required":
        return "Unrecognized Disease (Manual Review Required)"

    normalized = (label or "").strip()
    if not normalized:
        return "Unknown Disease"
    if _is_healthy_prediction(normalized):
        return "Healthy Leaf"

    parts = [p for p in normalized.split("_") if p]
    if len(parts) <= 1:
        return _titleize_label(normalized)
    disease_tokens = parts[:-1]
    if not disease_tokens:
        return _titleize_label(normalized)
    return " ".join(disease_tokens).replace(",", " ").title()


def _build_reason_summary(
    *,
    diagnosis_status: str,
    override_reason: str,
    model_label_before_override: str,
    model_confidence_before_override: float,
    leaf_visual_analysis: Dict[str, Any],
) -> str:
    anomalies = leaf_visual_analysis.get("anomalies_textures", {}) if isinstance(leaf_visual_analysis, dict) else {}
    lesion_summary = str(anomalies.get("lesion_summary") or "lesion pattern not clearly available")
    lesions_detected = int(anomalies.get("lesions_detected") or 0)
    chlorosis_halo = str(anomalies.get("chlorosis_halo") or "not clearly visible")

    model_text = (
        f"Model signal: '{_titleize_label(model_label_before_override)}' "
        f"with confidence {round(float(model_confidence_before_override), 2)}%."
    )
    visual_text = (
        f"Visual signal: {lesion_summary}; lesions detected={lesions_detected}; "
        f"chlorosis halo={chlorosis_halo}."
    )
    if diagnosis_status == "manual_review_required":
        return f"{model_text} {visual_text} Override applied: {override_reason}"
    return f"{model_text} {visual_text}"


def _build_organic_recommendations(disease_name: str, diagnosis_status: str) -> List[Dict[str, str]]:
    disease_key = (disease_name or "").lower()
    if disease_name == "Healthy Leaf":
        return [
            {
                "name": "No spray needed",
                "why_preferred": "Leaf is predicted healthy, so avoiding unnecessary spraying protects soil life and reduces costs.",
                "use_case": "Continue monitoring and field sanitation only.",
            }
        ]

    if diagnosis_status == "manual_review_required":
        return [
            {
                "name": "Neem oil (organic)",
                "why_preferred": "Low-residue preventive option while diagnosis is uncertain.",
                "use_case": "Use as an interim mild treatment until manual review confirms disease.",
            },
            {
                "name": "Trichoderma bio-fungicide",
                "why_preferred": "Biological control that can suppress fungal pressure with less ecological impact.",
                "use_case": "Use in integrated disease management and preventive schedules.",
            },
        ]

    if any(k in disease_key for k in {"blight", "spot", "anthracnose", "rust", "mildew", "rot"}):
        return [
            {
                "name": "Neem oil (organic)",
                "why_preferred": "Broad organic suppression and lower residue on produce compared with heavy chemical-only plans.",
                "use_case": "Early and mild symptom stages.",
            },
            {
                "name": "Bacillus/Trichoderma bio-fungicide",
                "why_preferred": "Targets disease pressure while helping microbial balance in soil and leaf surface.",
                "use_case": "Preventive + rotation treatment in repeated outbreaks.",
            },
        ]

    return [
        {
            "name": "Neem oil (organic)",
            "why_preferred": "Safer first-line option for uncertain early symptoms.",
            "use_case": "Initial response while monitoring progression.",
        }
    ]


def _build_chemical_recommendation(pesticide: Dict[str, str], diagnosis_status: str) -> Dict[str, str]:
    recommended = pesticide.get("recommended_pesticide", "Expert review required")
    active = pesticide.get("active_ingredient", "n/a")
    note = pesticide.get("usage_note", "Follow local agronomy guidance.")

    if diagnosis_status == "manual_review_required":
        why = (
            "Chemical pesticide is not preferred immediately because diagnosis is uncertain. "
            "Confirm disease first to avoid unnecessary chemical stress."
        )
    elif recommended.lower() in {"n/a", "expert review required"}:
        why = "No exact mapped chemical found in dataset; confirm with local agronomy expert."
    else:
        why = "Mapped from your trained class-to-pesticide dataset for this detected disease label."

    return {
        "name": recommended,
        "active_ingredient": active,
        "usage_note": note,
        "why_preferred": why,
    }


def _build_overuse_side_effects() -> List[str]:
    return [
        "Pathogen resistance can increase, making future sprays less effective.",
        "Residue risk on produce can rise above safe levels.",
        "Beneficial insects and soil microbes can be harmed.",
        "Leaf burn (phytotoxicity) and reduced plant vigor may occur.",
        "Soil and nearby water contamination risk increases.",
    ]


def build_farmer_report(
    *,
    crop_detected: str,
    final_label: str,
    diagnosis_status: str,
    override_reason: str,
    model_label_before_override: str,
    model_confidence_before_override: float,
    pesticide: Dict[str, str],
    leaf_visual_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    leaf_name = _format_leaf_name(crop_detected)
    disease_name = _extract_disease_name(final_label, diagnosis_status)
    reason = _build_reason_summary(
        diagnosis_status=diagnosis_status,
        override_reason=override_reason,
        model_label_before_override=model_label_before_override,
        model_confidence_before_override=model_confidence_before_override,
        leaf_visual_analysis=leaf_visual_analysis,
    )
    organic = _build_organic_recommendations(disease_name, diagnosis_status)
    chemical = _build_chemical_recommendation(pesticide, diagnosis_status)
    side_effects = _build_overuse_side_effects()
    agronomy_references = retrieve_agronomy_references(
        crop_detected=crop_detected,
        disease_label=final_label,
        pesticide=pesticide,
    )

    if diagnosis_status == "manual_review_required":
        preference = (
            "Prefer organic/interim management first. Final chemical choice should wait until manual review confirms disease type."
        )
    elif disease_name == "Healthy Leaf":
        preference = "Prefer no-spray monitoring. Use treatment only if new symptoms develop."
    else:
        preference = (
            "Prefer integrated management: start with organic/biological control for mild spread, "
            "and use mapped chemical only when disease pressure increases."
        )

    return {
        "leaf_name": leaf_name,
        "type_of_disease": disease_name,
        "reason_for_prediction": reason,
        "organic_recommendations": organic,
        "chemical_recommendation": chemical,
        "preferred_plan_why": preference,
        "overuse_side_effects": side_effects,
        "agronomy_references": agronomy_references,
    }


def build_farmer_action_plan_markdown(
    *,
    farmer_report: Dict[str, Any],
    leaf_visual_analysis: Dict[str, Any],
    diagnosis_status: str,
) -> str:
    leaf_name = str(farmer_report.get("leaf_name") or "this plant")
    disease_name = str(farmer_report.get("type_of_disease") or "Unknown Disease")
    reason = str(farmer_report.get("reason_for_prediction") or "No detailed reason available.")

    anomalies = (
        leaf_visual_analysis.get("anomalies_textures", {})
        if isinstance(leaf_visual_analysis, dict)
        else {}
    )
    lesion_summary = str(anomalies.get("lesion_summary") or "spots/lesions are visible")
    lesion_count = int(anomalies.get("lesions_detected") or 0)
    chlorosis_halo = str(anomalies.get("chlorosis_halo") or "not clearly visible")

    organic_recs = (
        farmer_report.get("organic_recommendations")
        if isinstance(farmer_report.get("organic_recommendations"), list)
        else []
    )
    chemical = (
        farmer_report.get("chemical_recommendation")
        if isinstance(farmer_report.get("chemical_recommendation"), dict)
        else {}
    )
    side_effects = (
        farmer_report.get("overuse_side_effects")
        if isinstance(farmer_report.get("overuse_side_effects"), list)
        else []
    )
    agronomy_references = (
        farmer_report.get("agronomy_references")
        if isinstance(farmer_report.get("agronomy_references"), list)
        else []
    )

    step_organic = "Start with low-toxicity options (organic first):"
    if organic_recs:
        organic_lines = [
            f"- {item.get('name', 'Organic option')}: {item.get('use_case', 'Use as per label guidance.')}"
            for item in organic_recs[:2]
            if isinstance(item, dict)
        ]
    else:
        organic_lines = ["- Neem oil or a bio-fungicide can be used as first response."]

    chemical_name = str(chemical.get("name") or "Expert review required")
    chemical_note = str(chemical.get("usage_note") or "Follow local agronomy guidance.")

    if diagnosis_status == "manual_review_required":
        chemical_step = (
            "Use chemical pesticide only after manual confirmation of disease to avoid wrong spray."
        )
    else:
        chemical_step = (
            f"If spread increases, use **{chemical_name}** carefully. {chemical_note}"
        )

    warning_items = side_effects[:3] if side_effects else [
        "Overuse can create pesticide resistance.",
        "Overuse can harm beneficial insects and soil microbes.",
        "Overuse can leave higher residue on produce.",
    ]
    warning_text = " ".join(warning_items)
    reference_lines = [
        f"- {str(item.get('title') or 'Agronomy note')}: {str(item.get('snippet') or '').strip()}"
        for item in agronomy_references[:2]
        if isinstance(item, dict) and str(item.get("snippet") or "").strip()
    ]

    return "\n".join(
        [
            f"**I understand your concern. You did the right thing by sharing the {leaf_name} image.**",
            "",
            f"**Detected issue:** {disease_name}",
            "",
            f"**Why this result:** {reason}",
            f"**What I observed in your image:** {lesion_summary}; lesion count detected: {lesion_count}; chlorosis halo: {chlorosis_halo}.",
            "",
            "**Action Plan (Start from safest):**",
            "1. Remove heavily affected leaves and keep field sanitation strong (dispose infected leaves away from field).",
            f"2. {step_organic}",
            *organic_lines,
            "3. Improve airflow and avoid overhead irrigation on infected leaves to reduce disease spread.",
            f"4. {chemical_step}",
            *(
                ["", "**Retrieved agronomy references:**", *reference_lines]
                if reference_lines
                else []
            ),
            "",
            f"**Safety warning:** {warning_text}",
        ]
    )


registry: Optional[ModelRegistry] = None
registry_namespace: Optional[str] = None
pesticide_index: Optional[Dict[str, Dict[str, str]]] = None
agronomy_reference_index: Optional[List[Dict[str, str]]] = None


def _resolve_keras_models():
    tf_keras = getattr(tf, "keras", None)
    if tf_keras is not None and getattr(tf_keras, "models", None) is not None:
        return tf_keras.models

    try:
        import keras  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        raise RuntimeError(
            "TensorFlow loaded without tf.keras, and standalone Keras could not be imported."
        ) from exc

    keras_models = getattr(keras, "models", None)
    if keras_models is None:
        raise RuntimeError(
            "TensorFlow loaded without tf.keras, and standalone Keras models are unavailable."
        )
    return keras_models


def _resolve_runs_dir() -> Path:
    project_root = settings.BASE_DIR.parent
    return Path(
        os.getenv(
            "SEASONAL_RUNS_DIR",
            project_root / "training" / "image_prediction" / "runs",
        )
    ).expanduser()


def resolve_stage1_crop_artifact_paths() -> Dict[str, Path]:
    project_root = settings.BASE_DIR.parent
    default_root = project_root / "training" / "hierarchical" / "artifacts"
    return {
        "model": Path(
            os.getenv(
                "STAGE1_CROP_MODEL_PATH",
                default_root / "plant_classifier.pth",
            )
        ).expanduser(),
        "classes": Path(
            os.getenv(
                "STAGE1_CROP_CLASS_FILE",
                default_root / "plant_classifier_classes.json",
            )
        ).expanduser(),
    }


def _resolve_crop_specific_models_dir() -> Path:
    project_root = settings.BASE_DIR.parent
    return Path(
        os.getenv(
            "CROP_SPECIFIC_MODELS_DIR",
            project_root / "training" / "hierarchical" / "disease_models",
        )
    ).expanduser()


def _resolve_model_artifact_path(season: str, runs_dir: Path) -> Path:
    env_key = {
        "all_season": "ALL_SEASON_MODEL_PATH",
        "kharif": "KHARIF_MODEL_PATH",
        "rabi": "RABI_MODEL_PATH",
    }[season]
    default_path = runs_dir / season / LATEST_MODEL_FILE_NAME
    return Path(os.getenv(env_key, default_path)).expanduser()


def _resolve_class_artifact_path(season: str, runs_dir: Path) -> Path:
    env_key = {
        "all_season": "ALL_SEASON_CLASS_FILE",
        "kharif": "KHARIF_CLASS_FILE",
        "rabi": "RABI_CLASS_FILE",
    }[season]
    default_path = runs_dir / season / LATEST_CLASS_FILE_NAME
    return Path(os.getenv(env_key, default_path)).expanduser()


def resolve_prediction_artifact_paths() -> Dict[str, Dict[str, Path]]:
    runs_dir = _resolve_runs_dir()
    return {
        season: {
            "model": _resolve_model_artifact_path(season, runs_dir),
            "classes": _resolve_class_artifact_path(season, runs_dir),
        }
        for season in SEASON_MODEL_KEYS
    }


def get_prediction_cache_namespace() -> str:
    manifest_parts: List[str] = [f"pipeline:{PREDICTION_PIPELINE_VERSION}"]
    for season, artifact_map in resolve_prediction_artifact_paths().items():
        for artifact_type, artifact_path in artifact_map.items():
            resolved_path = artifact_path.expanduser()
            try:
                stat = resolved_path.stat()
                manifest_parts.append(
                    f"{season}:{artifact_type}:{resolved_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
                )
            except OSError:
                manifest_parts.append(
                    f"{season}:{artifact_type}:{resolved_path}:missing"
                )

    for artifact_type, artifact_path in resolve_stage1_crop_artifact_paths().items():
        resolved_path = artifact_path.expanduser()
        try:
            stat = resolved_path.stat()
            manifest_parts.append(
                f"stage1:{artifact_type}:{resolved_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            )
        except OSError:
            manifest_parts.append(f"stage1:{artifact_type}:{resolved_path}:missing")

    crop_specific_dir = _resolve_crop_specific_models_dir()
    if crop_specific_dir.exists():
        for artifact_path in sorted(crop_specific_dir.rglob("*")):
            if not artifact_path.is_file():
                continue
            if artifact_path.suffix.lower() not in {".pth", ".json"}:
                continue
            try:
                stat = artifact_path.stat()
                manifest_parts.append(
                    f"crop-specific:{artifact_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
                )
            except OSError:
                manifest_parts.append(f"crop-specific:{artifact_path}:missing")
    else:
        manifest_parts.append(f"crop-specific-dir:{crop_specific_dir}:missing")

    agronomy_docs_dir = resolve_agronomy_docs_dir()
    if agronomy_docs_dir.exists():
        for doc_path in sorted(agronomy_docs_dir.rglob("*")):
            if not doc_path.is_file():
                continue
            if doc_path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                continue
            try:
                stat = doc_path.stat()
                manifest_parts.append(
                    f"agronomy-doc:{doc_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
                )
            except OSError:
                manifest_parts.append(f"agronomy-doc:{doc_path}:missing")
    else:
        manifest_parts.append(f"agronomy-doc-dir:{agronomy_docs_dir}:missing")

    for env_name in (
        "ENABLE_HIERARCHICAL_ROUTING",
        "STAGE1_CROP_CONFIDENCE_THRESHOLD",
        "MODEL_CONFIDENCE_TEMPERATURE",
        "MIN_IMAGE_QUALITY_SCORE",
        "MIN_IMAGE_SHARPNESS",
        "MAX_IMAGE_GLARE_RATIO",
        "MAX_IMAGE_DARK_RATIO",
        "ENABLE_BACKGROUND_SEGMENTATION",
        "MIN_SEGMENTED_LEAF_COVERAGE_RATIO",
        "ENABLE_AGRONOMY_RETRIEVAL",
        "AGRONOMY_DOCS_DIR",
        "AGRONOMY_RETRIEVAL_MAX_REFERENCES",
        "ENABLE_ACTIVE_LEARNING_FALLBACK",
        "ACTIVE_LEARNING_DATASET_ROOT",
        "ACTIVE_LEARNING_MIN_CONFIDENCE",
    ):
        manifest_parts.append(f"env:{env_name}:{os.getenv(env_name, '')}")

    return hashlib.sha256("|".join(manifest_parts).encode("utf-8")).hexdigest()


def _find_exported_h5_fallback(model_path: Path) -> Optional[Path]:
    summary_path = model_path.parent / "training_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            configured_path = str(
                ((payload.get("config") or {}).get("backend_model_path") or "")
            ).strip()
            if configured_path:
                candidate = Path(configured_path).expanduser()
                if candidate.exists():
                    return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    season = model_path.parent.name.strip().lower()
    if season:
        candidate = settings.BASE_DIR.parent / "assets" / "models" / f"{season}_cnn.h5"
        if candidate.exists():
            return candidate
    return None


def load_compat_model(model_path: Path):
    keras_models = _resolve_keras_models()

    def patch_keras_config(obj):
        if isinstance(obj, dict):
            patched = {}
            for key, value in obj.items():
                if key in {"optional", "quantization_config"}:
                    continue
                if key == "batch_shape":
                    patched["batch_input_shape"] = patch_keras_config(value)
                    continue
                if (
                    key == "dtype"
                    and isinstance(value, dict)
                    and value.get("class_name") == "DTypePolicy"
                ):
                    patched[key] = value.get("config", {}).get("name", "float32")
                    continue
                patched[key] = patch_keras_config(value)
            if (
                obj.get("class_name") == "DepthwiseConv2D"
                and isinstance(patched.get("config"), dict)
            ):
                patched["config"].pop("groups", None)
            return patched
        if isinstance(obj, list):
            return [patch_keras_config(item) for item in obj]
        return obj

    def direct_load(target_path: Path):
        return keras_models.load_model(target_path, compile=False)

    try:
        return direct_load(model_path)
    except Exception as direct_exc:
        if model_path.suffix.lower() == ".keras" and is_zipfile(model_path):
            fallback_path = _find_exported_h5_fallback(model_path)
            if fallback_path is not None:
                return direct_load(fallback_path)
            raise RuntimeError(
                f"Model '{model_path}' uses the newer zip-based .keras format, "
                "but this runtime cannot load it and no exported .h5 fallback was found."
            ) from direct_exc

        with h5py.File(model_path, "r") as src:
            raw_config = src.attrs.get("model_config")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            if not raw_config:
                raise
            model_config = json.loads(raw_config)
            patched_config = patch_keras_config(model_config)

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                temp_path = Path(tmp.name)

            try:
                with h5py.File(model_path, "r") as src_f, h5py.File(temp_path, "w") as dst_f:
                    for attr_key, attr_value in src_f.attrs.items():
                        dst_f.attrs[attr_key] = attr_value
                    for group_name in src_f:
                        src_f.copy(group_name, dst_f)
                    dst_f.attrs["model_config"] = json.dumps(patched_config).encode("utf-8")
                return keras_models.load_model(temp_path, compile=False)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def get_registry() -> ModelRegistry:
    global registry, registry_namespace
    current_namespace = get_prediction_cache_namespace()
    if registry is None or registry_namespace != current_namespace:
        registry = ModelRegistry()
        registry_namespace = current_namespace
    return registry


def _format_season_name(season: str) -> str:
    if season == "all_season":
        return "All Season"
    if season == "unsupported":
        return "Unsupported"
    return season.replace("_", " ").title() if season else "Unknown"


def _build_seasonal_comparison(raw_predictions: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        season: {
            "disease_prediction": str(prediction.get("disease_prediction") or ""),
            "crop_detected": str(prediction.get("crop_detected") or "unknown"),
            "confidence": float(prediction.get("confidence") or 0.0),
            "ran_model": bool(prediction.get("ran_model", True)),
        }
        for season, prediction in raw_predictions.items()
    }


def _default_seasonal_prediction_record(*, ran_model: bool = False) -> Dict[str, object]:
    return {
        "disease_prediction": "",
        "crop_detected": "unknown",
        "confidence": 0.0,
        "ran_model": ran_model,
    }


def _run_seasonal_predictions(
    models: ModelRegistry,
    image_batch,
    seasons: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Dict[str, object]]:
    selected_seasons = tuple(dict.fromkeys(seasons or SEASON_MODEL_KEYS))
    predictions = {
        season: _default_seasonal_prediction_record(ran_model=False)
        for season in SEASON_MODEL_KEYS
    }
    for season_key in selected_seasons:
        details = models.predict_details(season_key, image_batch, top_k=3)
        label = str(details["label"])
        predictions[season_key] = {
            "disease_prediction": label,
            "crop_detected": extract_crop_from_label(label),
            "confidence": round(float(details["confidence"]) * 100.0, 2),
            "ran_model": True,
            "top_k": details["top_k"],
        }
    return predictions


def _resolve_stage1_crop_route(
    *,
    models: ModelRegistry,
    image_bytes: bytes,
    requested_season: Optional[str],
) -> Optional[CropRoutingDecision]:
    route = models.route_supported_crop(image_bytes)
    if route is None:
        return None

    if not route.accepted:
        return route

    normalized_requested = normalize_requested_season(requested_season)
    if (
        normalized_requested in SEASON_MODEL_KEYS
        and normalized_requested != route.season
    ):
        return CropRoutingDecision(
            accepted=False,
            crop=route.crop,
            crop_confidence=route.crop_confidence,
            season=None,
            reason=(
                "Stage-1 crop routing mapped this image to "
                f"{_format_season_name(str(route.season))}, not {_format_season_name(normalized_requested)}."
            ),
            source=route.source,
            disease_model_used=route.disease_model_used,
        )

    return route


def _choose_auto_prediction_season(
    raw_predictions: Dict[str, Dict[str, object]],
    supported_seasons_by_crop: Dict[str, set[str]],
) -> tuple[Optional[str], str, str]:
    all_prediction = raw_predictions["all_season"]
    all_crop = str(all_prediction.get("crop_detected") or "unknown")
    all_confidence = float(all_prediction.get("confidence") or 0.0)
    all_supported_seasons = supported_seasons_by_crop.get(all_crop, set())

    if (
        all_supported_seasons == {"all_season"}
        and all_confidence >= AUTO_ALL_SEASON_CONFIDENCE_THRESHOLD
    ):
        return (
            "all_season",
            all_crop,
            f"Detected {all_crop.replace('_', ' ')} from the All Season model.",
        )

    seasonal_candidates: List[tuple[float, str, str]] = []
    for season in ("kharif", "rabi"):
        prediction = raw_predictions.get(season) or {}
        crop = str(prediction.get("crop_detected") or "unknown")
        confidence = float(prediction.get("confidence") or 0.0)
        if (
            supported_seasons_by_crop.get(crop, set()) == {season}
            and confidence >= AUTO_SEASONAL_CONFIDENCE_THRESHOLD
        ):
            seasonal_candidates.append((confidence, season, crop))

    if seasonal_candidates:
        confidence, season, crop = max(seasonal_candidates, key=lambda item: item[0])
        del confidence
        return (
            season,
            crop,
            f"Detected {crop.replace('_', ' ')} from the {_format_season_name(season)} model.",
        )

    if all_supported_seasons == {"all_season"}:
        return (
            "all_season",
            all_crop,
            f"Fallback to All Season model for {all_crop.replace('_', ' ')}.",
        )

    return (
        None,
        all_crop,
        "Crop is not covered by the current seasonal routing rules.",
    )


def _resolve_requested_prediction_season(
    *,
    requested_season: Optional[str],
    raw_predictions: Dict[str, Dict[str, object]],
    supported_seasons_by_crop: Dict[str, set[str]],
) -> tuple[Optional[str], str, str]:
    normalized_requested = normalize_requested_season(requested_season)
    auto_season, auto_crop, auto_reason = _choose_auto_prediction_season(
        raw_predictions,
        supported_seasons_by_crop,
    )

    if normalized_requested == "auto":
        return auto_season, auto_crop, auto_reason

    if normalized_requested not in SEASON_MODEL_KEYS:
        return "all_season", auto_crop, "Invalid season received. Falling back to All Season."

    if auto_season and auto_season != normalized_requested:
        all_season_confidence = float(
            (raw_predictions.get("all_season") or {}).get("confidence") or 0.0
        )
        if auto_season in {"kharif", "rabi"} or (
            auto_season == "all_season"
            and all_season_confidence >= EXPLICIT_SEASON_BLOCK_THRESHOLD
        ):
            return (
                None,
                auto_crop,
                f"This crop looks closer to the {_format_season_name(auto_season)} dataset than the {_format_season_name(normalized_requested)} dataset.",
            )

    requested_prediction = raw_predictions.get(normalized_requested) or {}
    requested_crop = str(requested_prediction.get("crop_detected") or auto_crop or "unknown")
    return (
        normalized_requested,
        requested_crop,
        f"Using the {_format_season_name(normalized_requested)} model as requested.",
    )


def _build_unsupported_farmer_report(crop_detected: str, reason: str) -> Dict[str, Any]:
    if _is_non_leaf_or_fruit_class(crop_detected):
        return {
            "leaf_name": "Leaf or fruit image required",
            "type_of_disease": "Not a leaf or fruit",
            "reason_for_prediction": reason,
            "organic_recommendations": [
                {
                    "name": "Upload another image",
                    "why_preferred": "Disease analysis only works on clear leaf or fruit photos.",
                    "use_case": "Retake one close image of a single leaf or fruit in good light.",
                }
            ],
            "chemical_recommendation": {
                "name": "No pesticide recommendation",
                "active_ingredient": "n/a",
                "usage_note": "This upload is not a leaf or fruit image, so pesticide advice is not reliable.",
                "why_preferred": "The input should be corrected before any disease treatment decision is made.",
            },
            "preferred_plan_why": "The upload was rejected before disease analysis because it does not look like a supported leaf or fruit image.",
            "overuse_side_effects": _build_overuse_side_effects(),
        }

    return {
        "leaf_name": _format_leaf_name(crop_detected),
        "type_of_disease": "Working On This Crop",
        "reason_for_prediction": reason,
        "organic_recommendations": [
            {
                "name": "Wait for model support",
                "why_preferred": "This crop is not yet mapped cleanly in the selected seasonal model.",
                "use_case": "Use local expert advice until support is added.",
            }
        ],
        "chemical_recommendation": {
            "name": "Expert review required",
            "active_ingredient": "n/a",
            "usage_note": "Avoid spraying only on this AI result because the crop is outside the current supported seasonal classes.",
            "why_preferred": "No reliable mapped seasonal prediction is available yet.",
        },
        "preferred_plan_why": "The crop is outside the supported seasonal model classes, so a safe hold-and-review path is better than forcing a wrong disease label.",
        "overuse_side_effects": _build_overuse_side_effects(),
    }


def _build_unsupported_action_plan(crop_detected: str, reason: str) -> str:
    if _is_non_leaf_or_fruit_class(crop_detected):
        return "\n".join(
            [
                f"**{INVALID_LEAF_OR_FRUIT_MESSAGE}**",
                "",
                f"**Why this happened:** {reason}",
                "",
                "**What to do now:**",
                "1. Upload one clear close photo of a single leaf or fruit.",
                "2. Keep the subject centered, in focus, and in good light.",
                "3. If crop symptoms are spreading, confirm with a local agricultural expert before spraying.",
            ]
        )

    leaf_name = _format_leaf_name(crop_detected)
    return "\n".join(
        [
            f"**We are still working on support for {leaf_name}. We will be back soon.**",
            "",
            f"**Why this happened:** {reason}",
            "",
            "**What to do now:**",
            "1. Try the correct season model if you already know the crop season.",
            "2. Upload one more clear image of the front and back of the leaf.",
            "3. If symptoms are spreading, contact a local agricultural expert before spraying.",
        ]
    )


def _build_unsupported_prediction_result(
    *,
    crop_detected: str,
    reason: str,
    seasonal_comparison: Dict[str, Dict[str, object]],
    image_bytes: bytes,
    content_type: str,
    status_message: Optional[str] = None,
    preprocessing_metrics: Optional[Dict[str, Any]] = None,
) -> PredictionResult:
    crop_hint = crop_detected if crop_detected and crop_detected != "unknown" else "unknown"
    is_non_leaf_or_fruit = _is_non_leaf_or_fruit_class(crop_hint)
    leaf_visual_analysis = build_structured_leaf_visual_analysis(
        image_bytes=image_bytes,
        content_type=content_type,
        prediction_summary={
            "crop": crop_hint,
            "disease": "working_on_it",
            "confidence": 0.0,
        },
    )
    farmer_report = _build_unsupported_farmer_report(crop_hint, reason)
    result_label = "invalid_leaf_or_fruit" if is_non_leaf_or_fruit else "working_on_it_crop_not_supported"
    return PredictionResult(
        label=result_label,
        confidence=0.0,
        crop_detected=crop_hint,
        season_used="unsupported",
        verification_passed=False,
        verification_reason=reason,
        uploaded_image_data_url="",
        seasonal_comparison=seasonal_comparison,
        visual_analysis="",
        is_low_confidence=True,
        status_message=str(
            status_message
            or (INVALID_LEAF_OR_FRUIT_MESSAGE if is_non_leaf_or_fruit else UNSUPPORTED_CROP_MESSAGE)
        ),
        recommended_pesticide="No pesticide recommendation" if is_non_leaf_or_fruit else "Expert review required",
        active_ingredient="n/a",
        usage_note=(
            "Please upload one clear leaf or fruit image for analysis."
            if is_non_leaf_or_fruit
            else "No reliable seasonal prediction is available for this crop yet."
        ),
        leaf_visual_analysis=leaf_visual_analysis,
        diagnosis_status="invalid_leaf_or_fruit" if is_non_leaf_or_fruit else "unsupported_crop",
        override_applied=False,
        override_reason=reason,
        model_label_before_override=result_label,
        model_confidence_before_override=0.0,
        heuristic_lesion_count=int(
            (
                leaf_visual_analysis.get("anomalies_textures", {})
                if isinstance(leaf_visual_analysis, dict)
                else {}
            ).get("lesions_detected")
            or 0
        ),
        preprocessing_metrics=dict(preprocessing_metrics or {}),
        farmer_report=farmer_report,
        farmer_action_plan_markdown=_build_unsupported_action_plan(crop_hint, reason),
    )


def verify_by_type(
    initial_label: str,
    chosen_season: str,
    seasonal_label: str,
    supported_seasons_by_crop: Optional[Dict[str, set[str]]] = None,
) -> Dict[str, str]:
    initial_crop = extract_crop_from_label(initial_label)
    seasonal_crop = extract_crop_from_label(seasonal_label)
    if _normalize_token(initial_crop) == _normalize_token(seasonal_crop):
        return {"passed": True, "reason": "Crop type matched across verification model."}
    if chosen_season == "all_season":
        return {"passed": True, "reason": "All-season model used for generalized crop types."}
    inferred_initial_season = infer_season_from_crop(
        initial_crop,
        seasons_by_crop=supported_seasons_by_crop,
    )
    inferred_seasonal_season = infer_season_from_crop(
        seasonal_crop,
        seasons_by_crop=supported_seasons_by_crop,
    )
    if inferred_seasonal_season == chosen_season and inferred_initial_season != chosen_season:
        return {
            "passed": True,
            "reason": "Seasonal crop was routed directly because the all-season verifier does not cover that crop.",
        }
    return {
        "passed": False,
        "reason": "Crop type mismatch between all-season and selected season model.",
    }


def _to_percent(value: float) -> float:
    return round(float(value) * 100.0, 2)


def _image_to_rgb_array(image_bytes: bytes, max_side: int = 512) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > max_side:
        ratio = max_side / float(longest)
        image = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return np.array(image).astype(np.float32)


def _extract_connected_components(binary_mask: np.ndarray, min_pixels: int) -> List[Dict[str, Any]]:
    height, width = binary_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    ys, xs = np.where(binary_mask)
    components: List[Dict[str, Any]] = []

    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        pixels = 0
        y_min = y_max = y
        x_min = x_max = x

        while stack:
            cy, cx = stack.pop()
            pixels += 1
            y_min = min(y_min, cy)
            y_max = max(y_max, cy)
            x_min = min(x_min, cx)
            x_max = max(x_max, cx)

            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and not visited[ny, nx]
                    and binary_mask[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if pixels < min_pixels:
            continue
        box_w = max(1, x_max - x_min + 1)
        box_h = max(1, y_max - y_min + 1)
        components.append(
            {
                "pixels": int(pixels),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "bbox_aspect_ratio": round(float(box_w / box_h), 3),
            }
        )

    components.sort(key=lambda item: item["pixels"], reverse=True)
    return components


def _heuristic_leaf_visual_analysis(image_bytes: bytes) -> Dict[str, Any]:
    arr = _image_to_rgb_array(image_bytes=image_bytes, max_side=512)
    height, width = arr.shape[:2]

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    exg = (2.0 * g) - r - b
    leaf_mask = exg > np.percentile(exg, 60)
    if float(leaf_mask.mean()) < 0.08:
        leaf_mask = g > np.percentile(g, 65)
    if float(leaf_mask.mean()) > 0.95:
        leaf_mask = exg > np.percentile(exg, 75)

    leaf_pixels = int(np.sum(leaf_mask))
    if leaf_pixels < 50:
        leaf_mask = g > np.percentile(g, 70)
        leaf_pixels = int(np.sum(leaf_mask))

    if leaf_pixels > 0:
        ys, xs = np.where(leaf_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
    else:
        x_min, y_min = 0, 0
        x_max, y_max = width - 1, height - 1

    box_w = max(1, x_max - x_min + 1)
    box_h = max(1, y_max - y_min + 1)
    aspect_ratio = max(box_w, box_h) / max(1, min(box_w, box_h))

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt((gx * gx) + (gy * gy))
    leaf_focus = float(grad_mag[leaf_mask].mean()) if leaf_pixels > 0 else float(grad_mag.mean())
    bg_mask = ~leaf_mask
    bg_pixels = int(np.sum(bg_mask))
    bg_focus = float(grad_mag[bg_mask].mean()) if bg_pixels > 0 else max(leaf_focus * 0.8, 1.0)
    focus_ratio = leaf_focus / max(bg_focus, 1e-6)

    leaf_rgb = arr[leaf_mask] if leaf_pixels > 0 else arr.reshape(-1, 3)
    bg_rgb = arr[bg_mask] if bg_pixels > 0 else np.zeros((1, 3), dtype=np.float32)
    mean_r, mean_g, mean_b = [float(v) for v in leaf_rgb.mean(axis=0)]
    bg_mean_r, bg_mean_g, bg_mean_b = [float(v) for v in bg_rgb.mean(axis=0)]

    if mean_g > (mean_r * 1.15) and mean_g > (mean_b * 1.15):
        dominant_leaf_color = "vibrant chlorophyll green"
    elif mean_g > mean_r and mean_g > mean_b:
        dominant_leaf_color = "green"
    else:
        dominant_leaf_color = "mixed green-brown"

    brightness_std = float(gray.std())
    if brightness_std < 25:
        lighting_desc = "diffused and even daylight"
    elif brightness_std < 45:
        lighting_desc = "moderately even lighting"
    else:
        lighting_desc = "uneven lighting with stronger contrast"

    bg_channel_spread = max(
        abs(bg_mean_r - bg_mean_g),
        abs(bg_mean_g - bg_mean_b),
        abs(bg_mean_b - bg_mean_r),
    )
    if bg_channel_spread < 8:
        background_desc = "flat light gray/off-white background"
    elif bg_mean_g > bg_mean_r and bg_mean_g > bg_mean_b:
        background_desc = "green-heavy background (possibly other foliage)"
    else:
        background_desc = "mixed background"

    venation_contrast = float(grad_mag[leaf_mask].std()) if leaf_pixels > 0 else float(grad_mag.std())
    if venation_contrast > 22:
        venation_desc = "prominent midrib with visible lateral veins"
    elif venation_contrast > 14:
        venation_desc = "moderately visible vein structure"
    else:
        venation_desc = "vein structure is faint in this image"

    if aspect_ratio >= 2.0:
        shape_desc = "elongated lanceolate leaf"
    elif aspect_ratio >= 1.4:
        shape_desc = "oval-lanceolate leaf"
    else:
        shape_desc = "broad leaf shape"

    if leaf_pixels > 0:
        leaf_gray = gray[leaf_mask]
        dark_threshold = float(np.percentile(leaf_gray, 18))
        dark_mask = leaf_mask & (gray <= dark_threshold)
    else:
        dark_mask = np.zeros_like(leaf_mask, dtype=bool)

    min_component_pixels = max(12, int((height * width) * 0.00035))
    lesions = _extract_connected_components(dark_mask, min_pixels=min_component_pixels)[:8]
    lesion_pixels = int(sum(item["pixels"] for item in lesions))
    lesion_count = len(lesions)
    lesion_area_pct = (lesion_pixels / max(leaf_pixels, 1)) * 100.0

    if leaf_pixels > 0:
        yellow_mask = leaf_mask & (g > r) & (r > (b + 5.0)) & ((g - b) > 20.0)
    else:
        yellow_mask = np.zeros_like(leaf_mask, dtype=bool)
    yellow_pct = (float(np.sum(yellow_mask)) / max(leaf_pixels, 1)) * 100.0

    if lesion_count >= 2 and lesion_area_pct > 0.5:
        lesion_summary = "distinct necrotic lesions detected"
    elif lesion_count == 1:
        lesion_summary = "single notable dark lesion detected"
    else:
        lesion_summary = "no strong circular necrotic pattern detected"

    chlorosis_summary = (
        "faint chlorotic halo likely around lesions"
        if yellow_pct > 2.0
        else "chlorotic halo is not strongly visible"
    )

    framing = {
        "subject": "single detached green leaf" if (leaf_pixels / (height * width)) > 0.15 else "leaf in mixed scene",
        "camera_angle": "close-up top-down or near top-down estimate",
        "focus": (
            "leaf texture appears sharper than background"
            if focus_ratio >= 1.15
            else "focus separation is limited"
        ),
        "background": background_desc,
        "metrics": {
            "image_size": {"width": int(width), "height": int(height)},
            "leaf_area_ratio_percent": _to_percent(leaf_pixels / max(height * width, 1)),
            "focus_ratio_leaf_to_background": round(float(focus_ratio), 3),
        },
    }

    geometry = {
        "shape": shape_desc,
        "venation": venation_desc,
        "midrib_visibility": "central vein appears detectable" if venation_contrast >= 14 else "midrib not clearly separable",
        "metrics": {
            "bounding_box_aspect_ratio": round(float(aspect_ratio), 3),
            "venation_contrast_score": round(float(venation_contrast), 3),
        },
    }

    colors = {
        "dominant_leaf_color": dominant_leaf_color,
        "lighting": lighting_desc,
        "background_tone": background_desc,
        "metrics": {
            "mean_leaf_rgb": [round(mean_r, 2), round(mean_g, 2), round(mean_b, 2)],
            "mean_background_rgb": [round(bg_mean_r, 2), round(bg_mean_g, 2), round(bg_mean_b, 2)],
            "brightness_stddev": round(brightness_std, 3),
        },
    }

    anomalies = {
        "lesion_summary": lesion_summary,
        "dark_center_presence": lesion_count > 0,
        "chlorosis_halo": chlorosis_summary,
        "lesions_detected": lesion_count,
        "metrics": {
            "lesion_area_percent_of_leaf": round(float(lesion_area_pct), 3),
            "yellow_chlorosis_percent_of_leaf": round(float(yellow_pct), 3),
            "largest_lesions": lesions[:3],
        },
    }

    return {
        "source": "heuristic_cv",
        "primary_subject_framing": framing,
        "geometry_structure": geometry,
        "color_palette_lighting": colors,
        "anomalies_textures": anomalies,
    }


def _extract_first_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _llm_structured_leaf_visual_analysis(
    image_data_url: str,
    prediction_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    llm_reply = _openai_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agronomy image analysis assistant. Return strict JSON only. "
                    "Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this uploaded leaf image with the exact structure below.\n"
                            "1) primary_subject_framing: subject, camera_angle, focus, background\n"
                            "2) geometry_structure: shape, venation, midrib_visibility\n"
                            "3) color_palette_lighting: dominant_leaf_color, lighting, background_tone\n"
                            "4) anomalies_textures: lesion_summary, dark_center_presence, chlorosis_halo, lesions_detected\n"
                            "Keep it factual and concise.\n"
                            f"Model hint: crop={prediction_summary.get('crop')}, "
                            f"disease={prediction_summary.get('disease')}, "
                            f"confidence={prediction_summary.get('confidence')}%"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        model=model,
        temperature=0.2,
        timeout_seconds=25,
    )
    parsed = _extract_first_json_block(llm_reply or "")
    if not parsed:
        return None
    parsed["source"] = "llm_vision"
    return parsed


def build_structured_leaf_visual_analysis(
    image_bytes: bytes,
    content_type: str,
    prediction_summary: Dict[str, Any],
) -> Dict[str, Any]:
    enabled = os.getenv("ENABLE_STRUCTURED_VISUAL_ANALYSIS", "true").lower() == "true"
    if not enabled:
        return {}

    use_llm = os.getenv("STRUCTURED_VISUAL_USE_LLM", "false").lower() == "true"
    if use_llm and _get_openai_api_key():
        try:
            image_data_url = build_image_data_url(image_bytes, content_type)
            llm_result = _llm_structured_leaf_visual_analysis(image_data_url, prediction_summary)
            if llm_result:
                return llm_result
        except (HTTPError, URLError, KeyError, ValueError):
            pass

    return _heuristic_leaf_visual_analysis(image_bytes)


def _get_lesion_count_from_analysis(leaf_visual_analysis: Dict[str, Any]) -> int:
    if not isinstance(leaf_visual_analysis, dict):
        return 0
    anomalies = leaf_visual_analysis.get("anomalies_textures")
    if not isinstance(anomalies, dict):
        return 0
    raw_count = anomalies.get("lesions_detected", 0)
    try:
        return max(0, int(raw_count))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value.strip())
    except (TypeError, ValueError):
        return float(default)


def _get_anomalies_from_analysis(leaf_visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(leaf_visual_analysis, dict):
        return {}
    anomalies = leaf_visual_analysis.get("anomalies_textures")
    return anomalies if isinstance(anomalies, dict) else {}


def _get_anomaly_metrics_from_analysis(leaf_visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
    anomalies = _get_anomalies_from_analysis(leaf_visual_analysis)
    metrics = anomalies.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _get_lesion_area_percent_from_analysis(leaf_visual_analysis: Dict[str, Any]) -> float:
    metrics = _get_anomaly_metrics_from_analysis(leaf_visual_analysis)
    return max(0.0, _coerce_float(metrics.get("lesion_area_percent_of_leaf"), 0.0))


def _get_yellow_chlorosis_percent_from_analysis(leaf_visual_analysis: Dict[str, Any]) -> float:
    metrics = _get_anomaly_metrics_from_analysis(leaf_visual_analysis)
    return max(0.0, _coerce_float(metrics.get("yellow_chlorosis_percent_of_leaf"), 0.0))


def _has_dark_center_signal(leaf_visual_analysis: Dict[str, Any]) -> bool:
    anomalies = _get_anomalies_from_analysis(leaf_visual_analysis)
    return bool(anomalies.get("dark_center_presence"))


def _has_visible_chlorosis_signal(leaf_visual_analysis: Dict[str, Any]) -> bool:
    yellow_pct = _get_yellow_chlorosis_percent_from_analysis(leaf_visual_analysis)
    if yellow_pct > 2.0:
        return True

    anomalies = _get_anomalies_from_analysis(leaf_visual_analysis)
    chlorosis = str(anomalies.get("chlorosis_halo") or "").strip().lower()
    if not chlorosis:
        return False
    return chlorosis not in {
        "chlorotic halo is not strongly visible",
        "not clearly visible",
    } and "not" not in chlorosis


def _score_healthy_lesion_signal(leaf_visual_analysis: Dict[str, Any]) -> Dict[str, float]:
    lesion_count = _get_lesion_count_from_analysis(leaf_visual_analysis)
    lesion_area_pct = _get_lesion_area_percent_from_analysis(leaf_visual_analysis)
    yellow_pct = _get_yellow_chlorosis_percent_from_analysis(leaf_visual_analysis)

    lesion_count_score = min(1.0, lesion_count / 2.0)
    lesion_area_score = min(1.0, lesion_area_pct / 0.75)
    dark_center_score = 1.0 if _has_dark_center_signal(leaf_visual_analysis) else 0.0
    chlorosis_score = 1.0 if _has_visible_chlorosis_signal(leaf_visual_analysis) else 0.0

    lesion_signal_score = (
        (0.45 * lesion_count_score)
        + (0.35 * lesion_area_score)
        + (0.10 * dark_center_score)
        + (0.10 * chlorosis_score)
    )

    return {
        "lesion_count": float(lesion_count),
        "lesion_area_percent": round(float(lesion_area_pct), 4),
        "yellow_chlorosis_percent": round(float(yellow_pct), 4),
        "lesion_count_score": round(float(lesion_count_score), 4),
        "lesion_area_score": round(float(lesion_area_score), 4),
        "dark_center_score": round(float(dark_center_score), 4),
        "chlorosis_score": round(float(chlorosis_score), 4),
        "lesion_signal_score": round(float(lesion_signal_score), 4),
    }


def _healthy_visual_bypass_threshold() -> float:
    return _env_float(
        "HEALTHY_VISUAL_BYPASS_CONFIDENCE",
        DEFAULT_HEALTHY_VISUAL_BYPASS_CONFIDENCE,
    )


def _should_bypass_healthy_visual_analysis(
    predicted_label: str,
    predicted_confidence: float,
) -> bool:
    gate_enabled = _env_flag("ENABLE_HEALTHY_LESION_OVERRIDE", True)
    if not gate_enabled or not _is_healthy_prediction(predicted_label):
        return False
    return float(predicted_confidence) >= _healthy_visual_bypass_threshold()


def _build_suppressed_healthy_leaf_visual_analysis(
    *,
    reason: str,
    source: str,
) -> Dict[str, Any]:
    return {
        "source": source,
        "gate_decision": {
            "reason": reason,
        },
        "primary_subject_framing": {},
        "geometry_structure": {},
        "color_palette_lighting": {},
        "anomalies_textures": {
            "lesion_summary": "no strong circular necrotic pattern detected",
            "dark_center_presence": False,
            "chlorosis_halo": "chlorotic halo is not strongly visible",
            "lesions_detected": 0,
            "metrics": {
                "lesion_area_percent_of_leaf": 0.0,
                "yellow_chlorosis_percent_of_leaf": 0.0,
                "largest_lesions": [],
                "suppressed_by_logic_gate": True,
            },
        },
    }


def _is_healthy_prediction(label: str) -> bool:
    normalized = (label or "").strip().lower()
    if not normalized:
        return False
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", normalized) if tok]
    return "healthy" in tokens or "healthy" in normalized


def apply_logic_gate_override(
    predicted_label: str,
    predicted_confidence: float,
    leaf_visual_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    gate_enabled = _env_flag("ENABLE_HEALTHY_LESION_OVERRIDE", True)
    lesion_count = _get_lesion_count_from_analysis(leaf_visual_analysis)
    if gate_enabled and lesion_count > 0 and _is_healthy_prediction(predicted_label):
        lesion_scores = _score_healthy_lesion_signal(leaf_visual_analysis)
        confidence_penalty = min(max(float(predicted_confidence), 0.0), 100.0) / 100.0
        confidence_penalty *= _env_float(
            "HEALTHY_LESION_CONFIDENCE_PENALTY",
            DEFAULT_HEALTHY_LESION_CONFIDENCE_PENALTY,
        )
        weighted_lesion_score = lesion_scores["lesion_signal_score"] - confidence_penalty
        raw_signal_threshold = _env_float(
            "HEALTHY_LESION_SIGNAL_THRESHOLD",
            DEFAULT_HEALTHY_LESION_SIGNAL_THRESHOLD,
        )
        weighted_score_threshold = _env_float(
            "HEALTHY_LESION_SCORE_THRESHOLD",
            DEFAULT_HEALTHY_LESION_SCORE_THRESHOLD,
        )

        if (
            lesion_scores["lesion_signal_score"] >= raw_signal_threshold
            and weighted_lesion_score >= weighted_score_threshold
        ):
            return {
                "override_applied": True,
                "diagnosis_status": "manual_review_required",
                "override_reason": (
                    f"{MANUAL_REVIEW_REASON} "
                    f"Lesion count={lesion_count}; "
                    f"lesion_signal_score={round(lesion_scores['lesion_signal_score'], 3)}; "
                    f"weighted_lesion_score={round(weighted_lesion_score, 3)}."
                ),
                "label": MANUAL_REVIEW_LABEL,
                "confidence": 0.0,
                "heuristic_lesion_count": lesion_count,
                "suppress_lesion_output": False,
                "suppression_reason": "",
            }
        should_suppress_output = float(predicted_confidence) >= LOW_CONFIDENCE_THRESHOLD
        return {
            "override_applied": False,
            "diagnosis_status": "ok",
            "override_reason": "",
            "label": predicted_label,
            "confidence": float(predicted_confidence),
            "heuristic_lesion_count": 0 if should_suppress_output else lesion_count,
            "suppress_lesion_output": should_suppress_output,
            "suppression_reason": (
                "Healthy-confidence logic gate suppressed weak lesion output. "
                f"Healthy confidence={round(float(predicted_confidence), 2)}%; "
                f"lesion_signal_score={round(lesion_scores['lesion_signal_score'], 3)}; "
                f"weighted_lesion_score={round(weighted_lesion_score, 3)}."
                if should_suppress_output
                else ""
            ),
        }
    return {
        "override_applied": False,
        "diagnosis_status": "ok",
        "override_reason": "",
        "label": predicted_label,
        "confidence": float(predicted_confidence),
        "heuristic_lesion_count": lesion_count,
        "suppress_lesion_output": False,
        "suppression_reason": "",
    }


def _openai_chat_completion(
    messages,
    model: str,
    temperature: float = 0.3,
    timeout_seconds: int = 45,
) -> Optional[str]:
    api_key = _get_openai_api_key()
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        try:
            raw = json.loads(resp.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON response from LLM API.") from exc

    if not isinstance(raw, dict):
        return None

    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)
    return None


def _generate_visual_analysis(image_data_url: str, prediction_summary: dict) -> str:
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    use_vision = os.getenv("ENABLE_VISION_SUMMARY", "true").lower() == "true"
    if not use_vision:
        return (
            f"Visual analysis: likely {_titleize_label(prediction_summary['disease'])} on "
            f"{prediction_summary['crop'].title()} crop. Review leaf color changes, spotting, "
            "and lesion spread before treatment."
        )

    try:
        llm_reply = _openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agronomy vision assistant. Give a short visual analysis of the "
                        "plant image and align with provided model prediction without exaggeration."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this leaf image briefly in 3-4 lines. "
                                f"Model hint: crop={prediction_summary['crop']}, "
                                f"disease={prediction_summary['disease']}, "
                                f"confidence={prediction_summary['confidence']}%."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            model=model,
            temperature=0.2,
        )
        if llm_reply:
            return llm_reply
    except (HTTPError, URLError, KeyError, ValueError):
        pass

    return (
        f"Visual analysis: likely {_titleize_label(prediction_summary['disease'])} on "
        f"{prediction_summary['crop'].title()} crop. Review leaf texture, patches, and lesion "
        "boundaries for confirmation."
    )


def run_prediction(image_bytes: bytes, content_type: str, requested_season: Optional[str]):
    models = get_registry()
    prepared_image = run_input_gatekeepers(image_bytes, content_type)
    processed_image_bytes = prepared_image.image_bytes
    segmentation_note = (
        "Leaf foreground segmentation was applied before inference."
        if bool(prepared_image.segmentation_metrics.get("applied"))
        else ""
    )

    def build_unsupported_result(*, crop_detected: str, reason: str, seasonal_comparison: Dict[str, Dict[str, object]]):
        active_learning_result = None
        if not _is_non_leaf_or_fruit_class(crop_detected):
            active_learning_result = _run_active_learning_fallback(
                image_bytes=image_bytes,
                models=models,
            )
        return _build_unsupported_prediction_result(
            crop_detected=crop_detected,
            reason=reason,
            seasonal_comparison=seasonal_comparison,
            image_bytes=processed_image_bytes,
            content_type=content_type,
            status_message=(
                str(active_learning_result.get("message") or "")
                if isinstance(active_learning_result, dict)
                else None
            ),
            preprocessing_metrics=prepared_image.preprocessing_metrics,
        )

    def build_unsupported_reason(crop_detected: str, base_reason: str) -> str:
        fallback_message = (
            INVALID_LEAF_OR_FRUIT_MESSAGE
            if _is_non_leaf_or_fruit_class(crop_detected)
            else UNSUPPORTED_CROP_MESSAGE
        )
        return f"{base_reason} {fallback_message}".strip()

    stage1_route = _resolve_stage1_crop_route(
        models=models,
        image_bytes=processed_image_bytes,
        requested_season=requested_season,
    )
    raw_predictions: Dict[str, Dict[str, object]]
    route_reason = ""

    if stage1_route is not None:
        route_reason = stage1_route.reason
        if segmentation_note:
            route_reason = f"{route_reason} {segmentation_note}".strip()

        if not stage1_route.accepted:
            unsupported_reason = build_unsupported_reason(stage1_route.crop, route_reason)
            return build_unsupported_result(
                crop_detected=stage1_route.crop,
                reason=unsupported_reason,
                seasonal_comparison={},
            )

        season = str(stage1_route.season or "all_season")
        seasons_to_run = ("all_season",) if season == "all_season" else ("all_season", season)
        batch = models.preprocess_image(processed_image_bytes)
        raw_predictions = _run_seasonal_predictions(models, batch, seasons=seasons_to_run)
        seasonal_comparison = _build_seasonal_comparison(raw_predictions)

        crop_specific_prediction = models.predict_crop_specific_disease(
            stage1_route.crop,
            processed_image_bytes,
        )
        if crop_specific_prediction is not None:
            final_prediction = crop_specific_prediction
            route_reason = (
                f"{route_reason} Crop-specific disease model was used for "
                f"{stage1_route.crop.replace('_', ' ')}."
            ).strip()
        else:
            final_prediction = raw_predictions.get(season) or _default_seasonal_prediction_record()
            seasonal_crop = str(final_prediction.get("crop_detected") or "unknown")
            if seasonal_crop not in {"unknown", stage1_route.crop}:
                conflict_reason = (
                    f"{route_reason} Stage-2 disease routing was stopped because the seasonal model "
                    f"predicted {seasonal_crop.replace('_', ' ')} instead of "
                    f"{stage1_route.crop.replace('_', ' ')}."
                ).strip()
                return build_unsupported_result(
                    crop_detected=stage1_route.crop,
                    reason=conflict_reason,
                    seasonal_comparison=seasonal_comparison,
                )
            final_prediction = {
                **final_prediction,
                "crop_detected": stage1_route.crop,
            }
    else:
        batch = models.preprocess_image(processed_image_bytes)
        raw_predictions = _run_seasonal_predictions(models, batch)
        seasonal_comparison = _build_seasonal_comparison(raw_predictions)
        season, routed_crop, route_reason = _resolve_requested_prediction_season(
            requested_season=requested_season,
            raw_predictions=raw_predictions,
            supported_seasons_by_crop=models.supported_seasons_by_crop,
        )
        if segmentation_note:
            route_reason = f"{route_reason} {segmentation_note}".strip()

        if season is None:
            unsupported_reason = build_unsupported_reason(routed_crop, route_reason)
            return build_unsupported_result(
                crop_detected=routed_crop,
                reason=unsupported_reason,
                seasonal_comparison=seasonal_comparison,
            )

        final_prediction = raw_predictions[season]
        final_crop = str(final_prediction.get("crop_detected") or "unknown")
        if final_crop == "unknown":
            unsupported_reason = build_unsupported_reason(routed_crop, route_reason)
            return build_unsupported_result(
                crop_detected=routed_crop,
                reason=unsupported_reason,
                seasonal_comparison=seasonal_comparison,
            )

    final_label = str(final_prediction.get("disease_prediction") or "")
    final_confidence = float(final_prediction.get("confidence") or 0.0)
    final_crop = str(final_prediction.get("crop_detected") or "unknown")
    model_label_before_override = final_label
    model_confidence_before_override = final_confidence

    bypassed_visual_analysis = _should_bypass_healthy_visual_analysis(
        model_label_before_override,
        model_confidence_before_override,
    )
    if bypassed_visual_analysis:
        leaf_visual_analysis = _build_suppressed_healthy_leaf_visual_analysis(
            reason=(
                "Healthy-confidence bypass skipped lesion analysis. "
                f"Healthy confidence={round(model_confidence_before_override, 2)}%; "
                f"bypass_threshold={round(_healthy_visual_bypass_threshold(), 2)}%."
            ),
            source="logic_gate_bypass",
        )
    else:
        leaf_visual_analysis = build_structured_leaf_visual_analysis(
            image_bytes=processed_image_bytes,
            content_type=content_type,
            prediction_summary={
                "crop": final_crop,
                "disease": model_label_before_override,
                "confidence": model_confidence_before_override,
            },
        )

    logic_gate = apply_logic_gate_override(
        predicted_label=model_label_before_override,
        predicted_confidence=model_confidence_before_override,
        leaf_visual_analysis=leaf_visual_analysis,
    )
    if bool(logic_gate.get("suppress_lesion_output")) and not bypassed_visual_analysis:
        leaf_visual_analysis = _build_suppressed_healthy_leaf_visual_analysis(
            reason=str(logic_gate.get("suppression_reason") or "Healthy-confidence logic gate suppressed lesion output."),
            source="logic_gate_suppressed",
        )
    final_label = str(logic_gate["label"])
    final_confidence = float(logic_gate["confidence"])
    override_applied = bool(logic_gate["override_applied"])
    override_reason = str(logic_gate["override_reason"])
    diagnosis_status = str(logic_gate["diagnosis_status"])
    heuristic_lesion_count = int(logic_gate["heuristic_lesion_count"])
    pesticide = get_pesticide_recommendation(final_label)
    farmer_report = build_farmer_report(
        crop_detected=final_crop,
        final_label=final_label,
        diagnosis_status=diagnosis_status,
        override_reason=override_reason,
        model_label_before_override=model_label_before_override,
        model_confidence_before_override=model_confidence_before_override,
        pesticide=pesticide,
        leaf_visual_analysis=leaf_visual_analysis,
    )
    farmer_action_plan_markdown = build_farmer_action_plan_markdown(
        farmer_report=farmer_report,
        leaf_visual_analysis=leaf_visual_analysis,
        diagnosis_status=diagnosis_status,
    )

    verification = verify_by_type(
        str(raw_predictions["all_season"]["disease_prediction"]),
        season,
        model_label_before_override,
        supported_seasons_by_crop=models.supported_seasons_by_crop,
    )
    if route_reason:
        verification = {
            "passed": bool(verification["passed"]),
            "reason": f"{route_reason} {str(verification['reason'])}".strip(),
        }
    if override_applied:
        verification = {
            "passed": False,
            "reason": (
                f"{route_reason} Manual review required due to healthy-vs-lesion conflict."
                if route_reason
                else "Manual review required due to healthy-vs-lesion conflict."
            ).strip(),
        }

    is_low_confidence = override_applied or final_confidence < LOW_CONFIDENCE_THRESHOLD
    if override_applied:
        status_message = "Manual review required."
    elif is_low_confidence:
        status_message = LOW_CONFIDENCE_MESSAGE
    else:
        status_message = "Prediction completed successfully."

    if is_low_confidence and not override_applied:
        active_learning_result = _run_active_learning_fallback(
            image_bytes=image_bytes,
            models=models,
        )
        if (
            isinstance(active_learning_result, dict)
            and str(active_learning_result.get("status") or "").strip().lower() == "saved"
        ):
            status_message = str(active_learning_result.get("message") or status_message)

    preprocessing_warnings = prepared_image.preprocessing_metrics.get("warnings")
    if isinstance(preprocessing_warnings, list):
        primary_warning = next(
            (str(item).strip() for item in preprocessing_warnings if str(item).strip()),
            "",
        )
        if primary_warning and primary_warning not in status_message:
            status_message = f"{status_message} {primary_warning}".strip()

    enable_visual = os.getenv("ENABLE_VISION_SUMMARY", "false").lower() == "true"
    image_data_url = ""
    visual_analysis = ""
    if enable_visual:
        image_data_url = build_image_data_url(processed_image_bytes, content_type)
        visual_analysis = _generate_visual_analysis(
            image_data_url=image_data_url,
            prediction_summary={
                "crop": final_crop,
                "disease": model_label_before_override,
                "confidence": model_confidence_before_override,
            },
        )
        if override_applied:
            visual_analysis = (
                "Manual review required: lesion signals conflict with a healthy model output.\n\n"
                f"{visual_analysis}"
            )

    return PredictionResult(
        label=final_label,
        confidence=final_confidence,
        crop_detected=final_crop,
        season_used=season,
        verification_passed=bool(verification["passed"]),
        verification_reason=str(verification["reason"]),
        uploaded_image_data_url=image_data_url,
        seasonal_comparison=seasonal_comparison,
        visual_analysis=visual_analysis,
        is_low_confidence=is_low_confidence,
        status_message=status_message,
        recommended_pesticide=pesticide["recommended_pesticide"],
        active_ingredient=pesticide["active_ingredient"],
        usage_note=pesticide["usage_note"],
        leaf_visual_analysis=leaf_visual_analysis,
        diagnosis_status=diagnosis_status,
        override_applied=override_applied,
        override_reason=override_reason,
        model_label_before_override=model_label_before_override,
        model_confidence_before_override=model_confidence_before_override,
        heuristic_lesion_count=heuristic_lesion_count,
        preprocessing_metrics=prepared_image.preprocessing_metrics,
        farmer_report=farmer_report,
        farmer_action_plan_markdown=farmer_action_plan_markdown,
    )


def _is_greeting(message: str) -> bool:
    normalized = re.sub(r"[^a-zA-Z ]+", " ", message).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    base_greetings = {
        "hi",
        "hii",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if normalized in base_greetings:
        return True
    tokens = normalized.split()
    if not tokens:
        return False
    if "good" in tokens and any(
        token in {"morning", "mrng", "afternoon", "aftn", "evening", "evng"}
        for token in tokens
    ):
        return True
    return tokens[0] in {"hi", "hii", "hello", "hey"} and len(tokens) <= 4


def _build_greeting(profile_name: str) -> str:
    clean_name = profile_name.strip()
    greeting = f"Hello {clean_name}!" if clean_name else "Hello!"
    return (
        f"{greeting}\n\n"
        "Welcome back to SSGrow.\n\n"
        "I'm your AI crop assistant. I can help with crop health, disease detection, and farming advice.\n"
        "You can upload a leaf image anytime, and I'll analyze it for possible diseases.\n\n"
        "How can I help you today?"
    )



def _get_openai_api_key() -> str:
    raw = os.getenv("OPENAI_API_KEY", "").strip()
    if raw.lower() in {"", "replace_me", "your_key_here"}:
        return ""
    return raw


def _format_plain_text(value: Any, default: str = "Not available") -> str:
    text = str(value or "").replace("_", " ").replace(",", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return default
    if text.lower() == text:
        return text.title()
    return text


def _confidence_to_percentage(raw_value: Any) -> Optional[float]:
    if raw_value is None or isinstance(raw_value, bool):
        return None

    value: Optional[float]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        had_percent = stripped.endswith("%")
        stripped = stripped.rstrip("%").strip()
        try:
            value = float(stripped)
        except ValueError:
            return None
        if had_percent:
            return round(value, 2)
    elif isinstance(raw_value, (int, float)):
        value = float(raw_value)
    else:
        return None

    if 0.0 <= value <= 1.0:
        value *= 100.0
    return round(value, 2)


def _format_percentage(raw_value: Any, default: str = "Not available") -> str:
    percentage = _confidence_to_percentage(raw_value)
    if percentage is None:
        return default
    return f"{percentage:.2f}".rstrip("0").rstrip(".") + "%"


def _extract_prediction_rows(context: Optional[dict]) -> List[Dict[str, Any]]:
    if not isinstance(context, dict):
        return []

    rows: List[Dict[str, Any]] = []
    prediction_keys = {
        "crop_detected",
        "disease_type",
        "disease_prediction",
        "disease_name",
        "confidence_score",
        "confidence",
        "farmer_report",
        "leaf_visual_analysis",
        "diagnosis_status",
    }
    if any(key in context for key in prediction_keys):
        rows.append(context)

    raw_results = context.get("results")
    if isinstance(raw_results, list):
        rows.extend(item for item in raw_results if isinstance(item, dict))
    return rows


def _select_primary_prediction_row(context: Optional[dict]) -> Dict[str, Any]:
    rows = _extract_prediction_rows(context)
    if not rows:
        return {}

    def row_score(row: Dict[str, Any]) -> tuple[int, int, int, float]:
        confidence = (
            _confidence_to_percentage(row.get("confidence_score"))
            or _confidence_to_percentage(row.get("confidence"))
            or _confidence_to_percentage(row.get("disease_confidence"))
            or _confidence_to_percentage(row.get("prediction_score"))
            or -1.0
        )
        has_farmer_report = 1 if isinstance(row.get("farmer_report"), dict) else 0
        has_disease = 1 if any(row.get(key) for key in ("disease_type", "disease_prediction", "disease_name")) else 0
        has_crop = 1 if row.get("crop_detected") else 0
        return (has_farmer_report, has_disease, has_crop, confidence)

    return max(rows, key=row_score)


def _has_prediction_context(primary_ctx: Dict[str, Any]) -> bool:
    if not primary_ctx:
        return False
    keys = (
        "crop_detected",
        "disease_type",
        "disease_prediction",
        "disease_name",
        "confidence_score",
        "confidence",
        "farmer_report",
    )
    return any(primary_ctx.get(key) for key in keys)


def _build_llm_prediction_context(context: Optional[dict]) -> Dict[str, Any]:
    primary_ctx = _select_primary_prediction_row(context)
    farmer_report = primary_ctx.get("farmer_report") if isinstance(primary_ctx.get("farmer_report"), dict) else {}
    diagnosis_status = str(primary_ctx.get("diagnosis_status") or "").strip().lower()
    raw_label = str(
        primary_ctx.get("disease_type")
        or primary_ctx.get("disease_prediction")
        or primary_ctx.get("disease_name")
        or ""
    ).strip()

    crop_name = _format_plain_text(
        primary_ctx.get("crop_detected")
        or farmer_report.get("leaf_name")
        or (extract_crop_from_label(raw_label) if raw_label else ""),
        default="Not available",
    )

    disease_name = str(farmer_report.get("type_of_disease") or "").strip()
    if not disease_name and raw_label:
        disease_name = _extract_disease_name(raw_label, diagnosis_status)
    disease_name = _format_plain_text(disease_name, default="Not available")

    confidence_text = _format_percentage(
        primary_ctx.get("confidence_score")
        if primary_ctx.get("confidence_score") is not None
        else primary_ctx.get("confidence")
        if primary_ctx.get("confidence") is not None
        else primary_ctx.get("disease_confidence")
        if primary_ctx.get("disease_confidence") is not None
        else primary_ctx.get("prediction_score"),
        default="Not available",
    )
    agronomy_references = (
        farmer_report.get("agronomy_references")
        if isinstance(farmer_report.get("agronomy_references"), list)
        else []
    )

    return {
        "crop": crop_name,
        "disease": disease_name,
        "confidence": confidence_text,
        "model_version": SSGROW_MODEL_VERSION,
        "trained_seasons": list(SSGROW_TRAINED_SEASONS),
        "agronomy_references": [
            {
                "title": str(item.get("title") or "Agronomy note"),
                "snippet": str(item.get("snippet") or ""),
            }
            for item in agronomy_references[:2]
            if isinstance(item, dict)
        ],
    }


def _default_symptom_hint(disease_name: str) -> str:
    normalized = _normalize_token(disease_name)
    if not normalized or disease_name == "Not available":
        return "No symptoms were checked yet because no image was uploaded."
    if normalized in {"healthy", "healthyleaf"}:
        return "No strong disease spots were clearly visible on the leaf."
    if "sootymould" in normalized or "sootymold" in normalized:
        return "Possible dark black fungal growth can appear on the leaf surface."
    if "blight" in normalized:
        return "Possible brown or burnt patches can spread on the leaf."
    if "mildew" in normalized:
        return "Possible white or gray powdery growth can appear on the leaf."
    if "rust" in normalized:
        return "Possible orange or rust-colored spots can appear on the leaf."
    if "spot" in normalized:
        return "Possible round brown or black spots can appear on the leaf."
    if "rot" in normalized:
        return "Possible dark soft tissue damage can appear on the leaf."
    if "uncertaindiagnosis" in normalized or "manualreviewrequired" in normalized:
        return "The image shows stress signs, but the disease pattern is not fully clear."
    return f"Possible leaf changes linked to {disease_name.lower()} can be present."


def _build_symptoms_detected(
    primary_ctx: Dict[str, Any],
    disease_name: str,
    has_prediction: bool,
) -> List[str]:
    if not has_prediction:
        return ["Please upload a clear crop leaf image so symptoms can be checked."]

    leaf_visual = (
        primary_ctx.get("leaf_visual_analysis")
        if isinstance(primary_ctx.get("leaf_visual_analysis"), dict)
        else {}
    )
    anomalies = leaf_visual.get("anomalies_textures") if isinstance(leaf_visual, dict) else {}
    symptoms: List[str] = []

    lesion_summary = str(anomalies.get("lesion_summary") or "").strip()
    if lesion_summary:
        if lesion_summary == "no strong circular necrotic pattern detected":
            if _normalize_token(disease_name) in {"healthy", "healthyleaf"}:
                symptoms.append("No strong disease spots were clearly visible on the leaf.")
        else:
            symptoms.append(lesion_summary.capitalize().rstrip(".") + ".")

    chlorosis_halo = str(anomalies.get("chlorosis_halo") or "").strip()
    if chlorosis_halo and chlorosis_halo != "chlorotic halo is not strongly visible":
        symptoms.append(chlorosis_halo.capitalize().rstrip(".") + ".")

    if not symptoms:
        symptoms.append(_default_symptom_hint(disease_name).rstrip(".") + ".")
    return symptoms[:3]


def _append_unique_action(actions: List[str], text: str) -> None:
    clean = re.sub(r"\s+", " ", text.strip())
    if clean and clean not in actions:
        actions.append(clean)


def _build_recommended_actions(
    primary_ctx: Dict[str, Any],
    disease_name: str,
    diagnosis_status: str,
    has_prediction: bool,
) -> List[str]:
    if not has_prediction:
        return [
            "Upload one clear leaf image in good light.",
            "Keep the leaf flat, close, and in focus.",
            "If you know the crop name, send it with the image.",
        ]

    normalized = _normalize_token(disease_name)
    if normalized in {"healthy", "healthyleaf"}:
        return [
            "No spray is needed right now.",
            "Keep watching the crop for new spots or yellowing.",
            "Maintain field hygiene and avoid unnecessary spraying.",
        ]

    actions: List[str] = []
    farmer_report = primary_ctx.get("farmer_report") if isinstance(primary_ctx.get("farmer_report"), dict) else {}
    organic_recommendations = (
        farmer_report.get("organic_recommendations")
        if isinstance(farmer_report.get("organic_recommendations"), list)
        else []
    )
    for item in organic_recommendations[:2]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        use_case = str(item.get("use_case") or "").strip()
        if name and use_case:
            _append_unique_action(actions, f"Try {name}. Best for {use_case.rstrip('.').lower()}.")
        elif name:
            _append_unique_action(actions, f"Try {name} as an early treatment option.")

    chemical_recommendation = (
        farmer_report.get("chemical_recommendation")
        if isinstance(farmer_report.get("chemical_recommendation"), dict)
        else {}
    )
    chemical_name = str(
        chemical_recommendation.get("name")
        or primary_ctx.get("recommended_pesticide")
        or ""
    ).strip()
    usage_note = str(
        chemical_recommendation.get("usage_note")
        or primary_ctx.get("usage_note")
        or ""
    ).strip()
    if chemical_name and chemical_name.lower() not in {"n/a", "expert review required"}:
        if usage_note:
            _append_unique_action(
                actions,
                f"If the problem keeps spreading, {chemical_name} may help; {usage_note.rstrip('.')}.",
            )
        else:
            _append_unique_action(
                actions,
                f"If the problem keeps spreading, use {chemical_name} only as per label and local guidance.",
            )

    if diagnosis_status == "manual_review_required" or normalized in {"uncertaindiagnosis", "manualreviewrequiredunrecognizeddisease"}:
        _append_unique_action(actions, "Upload one more close-up image of the front and back of the leaf.")
        _append_unique_action(actions, "Avoid strong chemical spray until the disease is confirmed.")
        _append_unique_action(actions, "Talk to a local agricultural expert if the damage is spreading.")
        return actions[:3]

    if "sootymould" in normalized or "sootymold" in normalized:
        _append_unique_action(actions, "Spray neem oil on the affected area if it matches local guidance.")
        _append_unique_action(actions, "Control aphids or whiteflies because they often support this problem.")
        _append_unique_action(actions, "Wash dirty leaf surfaces with clean water where practical.")
    elif any(keyword in normalized for keyword in {"blight", "spot", "anthracnose", "rust", "mildew", "rot"}):
        _append_unique_action(actions, "Remove badly affected leaves to slow down spread.")
        _append_unique_action(actions, "Keep the leaves dry and improve airflow around the crop.")
        _append_unique_action(actions, "Use a locally recommended fungicide only if the spread increases.")
    else:
        _append_unique_action(actions, "Remove badly affected leaves and keep the field clean.")
        _append_unique_action(actions, "Watch nearby plants for the same symptoms.")
        _append_unique_action(actions, "Consult a local agricultural expert if symptoms keep spreading.")
    return actions[:3]


def _build_disease_prediction_text(
    prediction_context: Dict[str, Any],
    diagnosis_status: str,
    has_prediction: bool,
) -> str:
    disease_name = prediction_context["disease"]
    normalized = _normalize_token(disease_name)

    if not has_prediction:
        return "Awaiting clear crop leaf image"
    if diagnosis_status == "manual_review_required":
        return "Uncertain diagnosis"
    if normalized in {"healthy", "healthyleaf"}:
        return "Healthy"
    if disease_name == "Not available":
        return "Unknown"
    return disease_name


def _extract_prediction_confidence_value(primary_ctx: Dict[str, Any]) -> Optional[float]:
    return _confidence_to_percentage(
        primary_ctx.get("confidence_score")
        if primary_ctx.get("confidence_score") is not None
        else primary_ctx.get("confidence")
        if primary_ctx.get("confidence") is not None
        else primary_ctx.get("disease_confidence")
        if primary_ctx.get("disease_confidence") is not None
        else primary_ctx.get("prediction_score"),
    )


def _build_prediction_confidence_text(
    prediction_context: Dict[str, Any],
    has_prediction: bool,
    confidence_value: Optional[float],
) -> str:
    confidence_text = str(prediction_context.get("confidence") or "Not available")
    if not has_prediction or confidence_text == "Not available" or confidence_value is None:
        return "Not available"
    return confidence_text


def _build_uncertainty_note(
    *,
    has_prediction: bool,
    diagnosis_status: str,
    confidence_value: Optional[float],
    cloud_fallback_reason: str = "",
) -> str:
    if not has_prediction:
        note = "Please upload a clear crop leaf image so I can start the analysis."
    elif confidence_value is not None and confidence_value < 40.0:
        note = (
            "Prediction confidence is below 40%, so this result may be uncertain. "
            "Please upload another clearer leaf image."
        )
    elif diagnosis_status == "manual_review_required":
        note = (
            "The leaf image was analyzed, but the disease pattern is still uncertain. "
            "Please upload another clearer leaf image or confirm it with a local agricultural expert."
        )
    else:
        note = (
            "This AI prediction may still need field confirmation if symptoms continue to spread."
        )
    if cloud_fallback_reason:
        note = f"{note} {cloud_fallback_reason}".strip()
    return note


def _format_ssgrow_response(
    *,
    prediction_context: Dict[str, Any],
    disease_prediction_text: str,
    prediction_confidence_text: str,
    symptoms_text: List[str],
    recommended_actions: List[str],
    uncertainty_note: str,
) -> str:
    symptoms_block = "\n".join(f"* {item}" for item in symptoms_text) or "* Visual symptoms could not be confirmed clearly."
    actions_block = "\n".join(f"* {action}" for action in recommended_actions) or "* Upload one clear leaf image."
    return (
        "Crop Identified:\n"
        f"{prediction_context['crop']}\n\n"
        "Disease Prediction:\n"
        f"{disease_prediction_text}\n\n"
        "Prediction Confidence:\n"
        f"{prediction_confidence_text}\n\n"
        "Symptoms Detected:\n"
        f"{symptoms_block}\n\n"
        "Seasonal Model Note:\n"
        f"{SSGROW_SEASONAL_MODEL_NOTE}\n\n"
        "Recommended Actions:\n"
        f"{actions_block}\n\n"
        "Uncertainty Note:\n"
        f"{uncertainty_note}"
    )


def _is_structured_ssgrow_response(text: str) -> bool:
    if not text.strip():
        return False

    position = -1
    for heading in SSGROW_REQUIRED_RESPONSE_HEADINGS:
        token = f"{heading}:"
        next_position = text.find(token)
        if next_position < 0 or next_position < position:
            return False
        position = next_position
    return True


def _extract_http_error_detail(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            raw_error = parsed.get("error")
            if isinstance(raw_error, dict):
                detail = raw_error.get("message")
                if isinstance(detail, str):
                    return detail.strip()
    except Exception:
        return ""
    return ""


def _fallback_assistant_reply(
    prompt: str,
    context: Optional[dict],
    profile_name: str,
    *,
    cloud_fallback_reason: str = "",
) -> str:
    if _is_greeting(prompt):
        return _build_greeting(profile_name)

    primary_ctx = _select_primary_prediction_row(context)
    prediction_context = _build_llm_prediction_context(context)
    has_prediction = _has_prediction_context(primary_ctx)
    diagnosis_status = str(primary_ctx.get("diagnosis_status") or "").strip().lower()
    confidence_value = _extract_prediction_confidence_value(primary_ctx)

    return _format_ssgrow_response(
        prediction_context=prediction_context,
        disease_prediction_text=_build_disease_prediction_text(
            prediction_context,
            diagnosis_status,
            has_prediction,
        ),
        prediction_confidence_text=_build_prediction_confidence_text(
            prediction_context,
            has_prediction,
            confidence_value,
        ),
        symptoms_text=_build_symptoms_detected(
            primary_ctx,
            str(prediction_context.get("disease") or "Not available"),
            has_prediction,
        ),
        recommended_actions=_build_recommended_actions(
            primary_ctx,
            str(prediction_context.get("disease") or "Not available"),
            diagnosis_status,
            has_prediction,
        ),
        uncertainty_note=_build_uncertainty_note(
            has_prediction=has_prediction,
            diagnosis_status=diagnosis_status,
            confidence_value=confidence_value,
            cloud_fallback_reason=cloud_fallback_reason,
        ),
    )


def call_llm(
    prompt: str,
    context: Optional[dict] = None,
    profile_name: str = "",
    profile_context: Optional[dict] = None,
    advisor_context: Optional[dict] = None,
    conversation_history: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    if _is_greeting(prompt):
        return {"answer": _build_greeting(profile_name)}

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    primary_ctx = _select_primary_prediction_row(context)
    prediction_context = _build_llm_prediction_context(context)
    has_prediction = _has_prediction_context(primary_ctx)
    route = classify_agriculture_request(prompt, has_prediction_context=has_prediction)
    advisor_reply = build_agriculture_advisor_response(
        prompt=prompt,
        profile_name=profile_name,
        profile_context=profile_context,
        advisor_context=advisor_context,
        conversation_history=conversation_history,
        has_prediction_context=has_prediction,
    )

    if route in {CROP_RECOMMENDATION_ENGINE, FARM_PLANNING_ADVISOR, GENERAL_AGRICULTURE_QA}:
        return {
            "answer": str(advisor_reply.get("answer") or "Unable to generate a response right now."),
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }

    if route != PLANT_DISEASE_PREDICTOR and str(advisor_reply.get("answer") or "").strip():
        return {
            "answer": str(advisor_reply.get("answer") or "Unable to generate a response right now."),
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": str(advisor_reply.get("route") or route),
        }

    if route == PLANT_DISEASE_PREDICTOR and not has_prediction:
        return {
            "answer": (
                "This looks like a plant disease or pest question. "
                "Please use the Plant Disease Predictor and upload a clear leaf image."
            ),
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }

    api_key = _get_openai_api_key()
    if not api_key:
        return {
            "answer": _fallback_assistant_reply(prompt, context, profile_name),
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }

    diagnosis_status = str(primary_ctx.get("diagnosis_status") or "").strip().lower()
    confidence_value = _extract_prediction_confidence_value(primary_ctx)
    symptom_hint = _build_symptoms_detected(
        primary_ctx,
        str(prediction_context.get("disease") or "Not available"),
        has_prediction,
    )
    action_hints = _build_recommended_actions(
        primary_ctx,
        str(prediction_context.get("disease") or "Not available"),
        diagnosis_status,
        has_prediction,
    )
    name_note = f"User profile name: {profile_name.strip() or 'Not provided'}"
    llm_user_payload = (
        f"{name_note}\n"
        f"Prediction Context: {json.dumps(prediction_context, ensure_ascii=True)}\n"
        f"Diagnosis Status: {diagnosis_status or 'not_provided'}\n"
        f"Prediction Available: {'yes' if has_prediction else 'no'}\n"
        f"Prediction Confidence Value: {json.dumps(confidence_value)}\n"
        f"Symptoms Hint: {json.dumps(symptom_hint, ensure_ascii=True)}\n"
        f"Action Hints: {json.dumps(action_hints, ensure_ascii=True)}\n"
        f"Question: {prompt.strip() or 'Please explain the result.'}"
    )

    try:
        text = _openai_chat_completion(
            messages=[
                {"role": "system", "content": SSGROW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": llm_user_payload,
                },
            ],
            model=model,
            temperature=0.3,
        )
        if text and _is_structured_ssgrow_response(text):
            return {
                "answer": text,
                "advisor_context": advisor_reply.get("advisor_context"),
                "route": route,
            }
        return {
            "answer": _fallback_assistant_reply(prompt, context, profile_name),
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }
    except HTTPError as exc:
        error_detail = _extract_http_error_detail(exc)
        if exc.code == 429:
            return {
                "answer": _fallback_assistant_reply(
                    prompt,
                    context,
                    profile_name,
                    cloud_fallback_reason=(
                        "This answer uses the CNN prediction only because the cloud AI quota is currently reached."
                        if not error_detail
                        else (
                            "This answer uses the CNN prediction only because the cloud AI quota is currently "
                            f"reached: {error_detail}"
                        )
                    ),
                )
                ,
                "advisor_context": advisor_reply.get("advisor_context"),
                "route": route,
            }
        if exc.code in {401, 403}:
            if error_detail:
                return {
                    "answer": f"LLM API authentication failed: {error_detail}",
                    "advisor_context": advisor_reply.get("advisor_context"),
                    "route": route,
                }
            return {
                "answer": "LLM API authentication failed. Check OPENAI_API_KEY and model access.",
                "advisor_context": advisor_reply.get("advisor_context"),
                "route": route,
            }
        if error_detail:
            return {
                "answer": _fallback_assistant_reply(
                    prompt,
                    context,
                    profile_name,
                    cloud_fallback_reason=(
                        f"This answer uses the CNN prediction because the cloud AI returned error {exc.code}: {error_detail}"
                    ),
                )
                ,
                "advisor_context": advisor_reply.get("advisor_context"),
                "route": route,
            }
        return {
            "answer": _fallback_assistant_reply(
                prompt,
                context,
                profile_name,
                cloud_fallback_reason=(
                    f"This answer uses the CNN prediction because the cloud AI returned error {exc.code}."
                ),
            )
            ,
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }
    except (URLError, SocketTimeout, TimeoutError):
        return {
            "answer": _fallback_assistant_reply(
                prompt,
                context,
                profile_name,
                cloud_fallback_reason=(
                    "This answer uses the CNN prediction because the cloud AI service is not reachable right now."
                ),
            )
            ,
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "answer": _fallback_assistant_reply(
                prompt,
                context,
                profile_name,
                cloud_fallback_reason=(
                    "This answer uses the CNN prediction because the cloud AI response could not be processed."
                ),
            )
            ,
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }
    except Exception:
        return {
            "answer": _fallback_assistant_reply(
                prompt,
                context,
                profile_name,
                cloud_fallback_reason=(
                    "This answer uses the CNN prediction because the cloud AI assistant is temporarily unavailable."
                ),
            )
            ,
            "advisor_context": advisor_reply.get("advisor_context"),
            "route": route,
        }
    
