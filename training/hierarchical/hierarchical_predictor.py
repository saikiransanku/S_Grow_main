from __future__ import annotations

import argparse
import io
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PIPELINE_ROOT / "artifacts"
DEFAULT_DISEASE_MODELS_DIR = PIPELINE_ROOT / "disease_models"


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


@dataclass(frozen=True)
class PredictionCandidate:
    label: str
    probability: float


@dataclass
class LoadedModel:
    model: nn.Module
    class_names: List[str]
    image_size: int
    lock: threading.Lock


class HierarchicalPredictor:
    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        plant_model_path: Path | None = None,
        plant_classes_path: Path | None = None,
        disease_models_dir: Path | None = None,
        plant_confidence_threshold: float | None = None,
        disease_confidence_threshold: float | None = None,
        top_k: int = 3,
    ) -> None:
        self.manifest_path = manifest_path
        self.top_k = max(1, int(top_k))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.manifest: Dict[str, object] = {}

        if manifest_path is not None:
            self.manifest = self._load_manifest(manifest_path)
            thresholds = self.manifest.get("recommended_thresholds", {})
            self.plant_confidence_threshold = float(
                plant_confidence_threshold
                if plant_confidence_threshold is not None
                else thresholds.get("plant_confidence", 0.55)
            )
            self.disease_confidence_threshold = float(
                disease_confidence_threshold
                if disease_confidence_threshold is not None
                else thresholds.get("disease_confidence", 0.60)
            )

            plant_entry = self.manifest.get("plant_classifier")
            if not isinstance(plant_entry, dict):
                raise ValueError("Pipeline manifest does not contain a plant classifier entry.")

            self.plant_model = self._load_model(
                model_path=Path(plant_entry["model_path"]),
                classes_path=Path(plant_entry["classes_path"]) if plant_entry.get("classes_path") else None,
            )
            self.disease_entries = {
                normalize_token(plant_name): entry
                for plant_name, entry in (self.manifest.get("disease_models") or {}).items()
                if isinstance(entry, dict)
            }
        else:
            resolved_plant_model = plant_model_path or DEFAULT_ARTIFACTS_DIR / "plant_classifier.pth"
            resolved_plant_classes = plant_classes_path or DEFAULT_ARTIFACTS_DIR / "plant_classifier_classes.json"
            resolved_disease_models_dir = disease_models_dir or DEFAULT_DISEASE_MODELS_DIR

            self.plant_confidence_threshold = float(plant_confidence_threshold if plant_confidence_threshold is not None else 0.55)
            self.disease_confidence_threshold = float(
                disease_confidence_threshold if disease_confidence_threshold is not None else 0.60
            )
            self.plant_model = self._load_model(
                model_path=resolved_plant_model,
                classes_path=resolved_plant_classes,
            )
            self.disease_entries = self._discover_disease_entries(resolved_disease_models_dir)

        self._disease_models: Dict[str, LoadedModel] = {}

    def _load_manifest(self, manifest_path: Path) -> Dict[str, object]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pipeline manifest must be a JSON object.")
        return payload

    def _discover_disease_entries(self, disease_models_dir: Path) -> Dict[str, Dict[str, object]]:
        entries: Dict[str, Dict[str, object]] = {}
        if not disease_models_dir.exists():
            return entries

        for model_path in sorted(disease_models_dir.rglob("*_disease_model.pth")):
            plant_name = normalize_token(model_path.stem.removesuffix("_disease_model"))
            if not plant_name:
                continue

            classes_path = model_path.parent / f"{plant_name}_disease_classes.json"
            entries[plant_name] = {
                "model_path": str(model_path),
                "classes_path": str(classes_path) if classes_path.exists() else None,
            }

        return entries

    def _create_model(self, num_classes: int, kind: str) -> nn.Module:
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features

        if kind == "disease":
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25, inplace=True),
                nn.Linear(512, num_classes),
            )
            return model

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model

    def _read_class_names(self, classes_path: Path | None) -> List[str]:
        if classes_path is None or not classes_path.exists():
            return []

        if classes_path.suffix.lower() == ".json":
            payload = json.loads(classes_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [normalize_token(item) for item in payload if normalize_token(item)]

        return [normalize_token(line) for line in classes_path.read_text(encoding="utf-8").splitlines() if normalize_token(line)]

    def _load_model(self, *, model_path: Path, classes_path: Path | None) -> LoadedModel:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {model_path}")

        class_names = checkpoint.get("class_names")
        if not isinstance(class_names, list) or not class_names:
            class_names = self._read_class_names(classes_path)
        class_names = [normalize_token(item) for item in class_names if normalize_token(item)]
        if not class_names:
            raise ValueError(f"Class names missing for checkpoint: {model_path}")

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            state_dict = checkpoint

        if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
            state_dict = {str(key)[len("module.") :]: value for key, value in state_dict.items()}

        kind = "disease" if checkpoint.get("plant_name") else "plant"
        image_size = int(checkpoint.get("image_size") or 300)
        model = self._create_model(num_classes=len(class_names), kind=kind)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        return LoadedModel(
            model=model,
            class_names=class_names,
            image_size=image_size,
            lock=threading.Lock(),
        )

    def _preprocess(self, image_bytes: bytes, image_size: int) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid image file") from exc

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0).to(self.device)

    def _predict_top_k(self, loaded_model: LoadedModel, image_bytes: bytes) -> Tuple[PredictionCandidate, ...]:
        tensor = self._preprocess(image_bytes, loaded_model.image_size)
        with loaded_model.lock:
            with torch.inference_mode():
                logits = loaded_model.model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = torch.softmax(logits, dim=1).squeeze(0)

        limit = min(self.top_k, len(loaded_model.class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=limit)
        return tuple(
            PredictionCandidate(
                label=loaded_model.class_names[int(idx)],
                probability=float(prob),
            )
            for prob, idx in zip(top_probabilities.tolist(), top_indices.tolist())
        )

    def _get_disease_model(self, plant_name: str) -> LoadedModel | None:
        plant_key = normalize_token(plant_name)
        if plant_key in self._disease_models:
            return self._disease_models[plant_key]

        entry = self.disease_entries.get(plant_key)
        if not entry:
            return None

        model_path = entry.get("model_path")
        if not model_path:
            return None

        loaded_model = self._load_model(
            model_path=Path(str(model_path)),
            classes_path=Path(str(entry["classes_path"])) if entry.get("classes_path") else None,
        )
        self._disease_models[plant_key] = loaded_model
        return loaded_model

    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        plant_top_k = self._predict_top_k(self.plant_model, image_bytes)
        top_plant = plant_top_k[0]
        plant_name = normalize_token(top_plant.label) or "unknown"
        accepted = top_plant.probability >= self.plant_confidence_threshold
        reasons: List[str] = []

        if not accepted:
            reasons.append("Plant confidence is low.")

        disease_top_k: Tuple[PredictionCandidate, ...] = ()
        disease_name = "uncertain_diagnosis"
        disease_confidence = 0.0

        disease_model = self._get_disease_model(plant_name)
        if disease_model is None:
            accepted = False
            reasons.append(f"No crop-specific disease model is available for {plant_name}.")
        else:
            disease_top_k = self._predict_top_k(disease_model, image_bytes)
            top_disease = disease_top_k[0]
            disease_name = normalize_token(top_disease.label) or "unknown"
            disease_confidence = float(top_disease.probability)
            if disease_confidence < self.disease_confidence_threshold:
                accepted = False
                reasons.append("Disease confidence is low.")

        health_status = "healthy" if disease_name == "healthy" else "unhealthy"
        final_result = (
            f"healthy_{plant_name}" if accepted and disease_name == "healthy" else f"{disease_name}_{plant_name}"
        )
        if not accepted:
            final_result = "uncertain_diagnosis"

        message = (
            "Prediction accepted."
            if accepted
            else "Uncertain result. Show top predictions and ask for a clearer leaf image."
        )

        return {
            "accepted": accepted,
            "message": message,
            "reasons": reasons,
            "plant_prediction": {
                "label": plant_name,
                "confidence": round(float(top_plant.probability), 4),
            },
            "disease_prediction": {
                "label": disease_name,
                "confidence": round(disease_confidence, 4),
            },
            "health_status": health_status,
            "final_result": final_result,
            "top_predictions": {
                "plant": [
                    {"label": candidate.label, "probability": round(candidate.probability, 4)}
                    for candidate in plant_top_k
                ],
                "disease": [
                    {"label": candidate.label, "probability": round(candidate.probability, 4)}
                    for candidate in disease_top_k
                ],
            },
        }

    def predict_minimal(self, image_bytes: bytes) -> Dict[str, object]:
        detailed = self.predict(image_bytes)
        plant_confidence = float(detailed["plant_prediction"]["confidence"])
        disease_confidence = float(detailed["disease_prediction"]["confidence"])
        final_confidence = round(plant_confidence * disease_confidence, 4)

        return {
            "plant_name": detailed["plant_prediction"]["label"],
            "disease_name": detailed["disease_prediction"]["label"],
            "confidence": final_confidence,
        }

    def predict_file(self, image_path: Path) -> Dict[str, object]:
        return self.predict(image_path.read_bytes())

    def predict_minimal_file(self, image_path: Path) -> Dict[str, object]:
        return self.predict_minimal(image_path.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical plant -> disease inference.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plant-model", type=Path, default=None)
    parser.add_argument("--plant-classes", type=Path, default=None)
    parser.add_argument("--disease-models-dir", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--plant-confidence-threshold", type=float, default=None)
    parser.add_argument("--disease-confidence-threshold", type=float, default=None)
    parser.add_argument("--minimal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = HierarchicalPredictor(
        manifest_path=args.manifest,
        plant_model_path=args.plant_model,
        plant_classes_path=args.plant_classes,
        disease_models_dir=args.disease_models_dir,
        plant_confidence_threshold=args.plant_confidence_threshold,
        disease_confidence_threshold=args.disease_confidence_threshold,
        top_k=args.top_k,
    )
    result = predictor.predict_minimal_file(args.image) if args.minimal else predictor.predict_file(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
