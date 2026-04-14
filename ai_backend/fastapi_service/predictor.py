from __future__ import annotations

import io
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


def normalize_token(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return re.sub(r"_+", "_", cleaned)


def create_model(arch: str, num_classes: int) -> nn.Module:
    model_key = normalize_token(arch)

    if model_key == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_features, num_classes))
        return model

    if model_key == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_features, num_classes))
        return model

    if model_key == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


def load_class_names(class_file: Path | None) -> List[str]:
    if class_file is None or not class_file.exists():
        return []

    if class_file.suffix.lower() == ".json":
        payload = json.loads(class_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [normalize_token(str(item)) for item in payload if normalize_token(str(item))]

    return [normalize_token(line) for line in class_file.read_text(encoding="utf-8").splitlines() if normalize_token(line)]


def _extract_known_plants(class_names: Iterable[str]) -> set[str]:
    normalized_classes = [normalize_token(item) for item in class_names if normalize_token(item)]

    plants = {
        label[len("healthy_") :]
        for label in normalized_classes
        if label.startswith("healthy_") and len(label) > len("healthy_")
    }
    if plants:
        return plants

    suffix_counts: Counter[str] = Counter()
    for label in normalized_classes:
        tokens = label.split("_")
        limit = min(4, len(tokens))
        for size in range(1, limit + 1):
            suffix = "_".join(tokens[-size:])
            suffix_counts[suffix] += 1

    for suffix, count in suffix_counts.items():
        if count >= 2 and suffix != "healthy":
            plants.add(suffix)

    if plants:
        return plants

    fallback = set()
    for label in normalized_classes:
        tokens = label.split("_")
        if len(tokens) > 1:
            fallback.add(tokens[-1])
    return fallback


def parse_label(label: str, known_plants: set[str]) -> tuple[str, str]:
    normalized = normalize_token(label)
    if not normalized:
        return "unknown", "unknown"

    if normalized.startswith("healthy_"):
        plant = normalized[len("healthy_") :]
        return (plant or "unknown", "healthy")

    candidates = [
        plant
        for plant in known_plants
        if normalized == plant or normalized.endswith(f"_{plant}")
    ]
    if candidates:
        plant = max(candidates, key=len)
        if normalized == plant:
            disease = "unknown"
        else:
            disease = normalized[: -(len(plant) + 1)]
            disease = disease or "unknown"
        return plant, disease

    parts = normalized.split("_")
    if len(parts) == 1:
        return "unknown", parts[0]

    plant = parts[-1]
    disease = "_".join(parts[:-1]) or "unknown"
    return plant, disease


def _build_alias_map(plants: set[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}

    for plant in plants:
        tokenized = plant.split("_")
        variants = {
            plant,
            plant.replace("_", " "),
            "".join(tokenized),
        }
        variants.update(tokenized)
        if len(tokenized) > 1:
            variants.add(" ".join(tokenized))

        for variant in variants:
            normalized = normalize_token(variant)
            if normalized and normalized not in aliases:
                aliases[normalized] = plant

    manual_aliases = {
        "corn": "corn_maize",
        "maize": "corn_maize",
        "pepper": "pepper_bell",
        "bellpepper": "pepper_bell",
        "capsicum": "pepper_bell",
        "ground nut": "groundnut",
        "chilli": "chilli",
        "chili": "chilli",
        "paddy": "rice",
    }
    for alias, plant in manual_aliases.items():
        normalized_alias = normalize_token(alias)
        normalized_plant = normalize_token(plant)
        if normalized_alias and normalized_plant in plants:
            aliases[normalized_alias] = normalized_plant

    return aliases


def resolve_user_note_plant(user_note: str | None, alias_map: Dict[str, str]) -> str | None:
    normalized_note = normalize_token(user_note or "")
    if not normalized_note:
        return None

    if normalized_note in alias_map:
        return alias_map[normalized_note]

    note_parts = normalized_note.split("_")
    for part in note_parts:
        if part in alias_map:
            return alias_map[part]

    for alias, plant in alias_map.items():
        if alias and alias in normalized_note:
            return plant

    return None


@dataclass(frozen=True)
class ClassScore:
    class_label: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    class_label: str
    plant_name: str
    disease_name: str
    confidence: float
    all_scores: tuple[ClassScore, ...]
    resolved_note_plant: str | None
    uncertain: bool


@dataclass
class ModelBundle:
    model: nn.Module
    class_names: List[str]
    transform: transforms.Compose
    device: torch.device
    lock: threading.Lock


class PlantDiseasePredictor:
    def __init__(
        self,
        *,
        model_path: Path,
        classes_path: Path | None,
        model_arch: str,
        image_size: int,
        min_confidence: float,
    ) -> None:
        self.model_path = model_path
        self.classes_path = classes_path
        self.model_arch = model_arch
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bundle = self._load_model_bundle()
        self.known_plants = _extract_known_plants(self.bundle.class_names)
        self.alias_map = _build_alias_map(self.known_plants)

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model_bundle(self) -> ModelBundle:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            class_names = load_class_names(self.classes_path)
            if not class_names:
                raise ValueError("Class names are required when loading a serialized model object.")
        elif isinstance(checkpoint, dict):
            class_names = checkpoint.get("class_names")
            if not isinstance(class_names, list) or not class_names:
                class_names = load_class_names(self.classes_path)
            class_names = [normalize_token(item) for item in class_names if normalize_token(item)]
            if not class_names:
                raise ValueError("Class names missing from checkpoint and classes file.")

            arch = str(checkpoint.get("arch") or self.model_arch)
            image_size = int(checkpoint.get("image_size") or self.image_size)
            self.image_size = image_size

            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, dict):
                state_dict = checkpoint

            if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
                state_dict = {str(key)[len("module.") :]: value for key, value in state_dict.items()}

            model = create_model(arch=arch, num_classes=len(class_names))
            model.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError("Unsupported checkpoint format.")

        model.to(self.device)
        model.eval()

        return ModelBundle(
            model=model,
            class_names=class_names,
            transform=self._build_transform(),
            device=self.device,
            lock=threading.Lock(),
        )

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid image file") from exc

        tensor = self.bundle.transform(image).unsqueeze(0)
        return tensor.to(self.bundle.device)

    def predict_probabilities(self, image_bytes: bytes) -> tuple[ClassScore, ...]:
        tensor = self._preprocess(image_bytes)

        with self.bundle.lock:
            with torch.inference_mode():
                logits = self.bundle.model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

        scores = [
            ClassScore(class_label=label, probability=float(prob))
            for label, prob in zip(self.bundle.class_names, probabilities)
        ]
        scores.sort(key=lambda item: item.probability, reverse=True)
        return tuple(scores)

    def predict(self, image_bytes: bytes, user_note: str | None = None) -> PredictionResult:
        all_scores = self.predict_probabilities(image_bytes)

        resolved_note_plant = resolve_user_note_plant(user_note, self.alias_map)
        candidate_scores = all_scores

        if resolved_note_plant:
            filtered = [
                score
                for score in all_scores
                if parse_label(score.class_label, self.known_plants)[0] == resolved_note_plant
            ]
            if filtered:
                candidate_scores = tuple(filtered)

        top_score = candidate_scores[0]
        plant_name, disease_name = parse_label(top_score.class_label, self.known_plants)

        confidence = round(float(top_score.probability), 4)
        uncertain = confidence < float(self.min_confidence)

        if uncertain:
            disease_name = "uncertain_diagnosis"

        return PredictionResult(
            class_label=top_score.class_label,
            plant_name=normalize_token(resolved_note_plant or plant_name) or "unknown",
            disease_name=normalize_token(disease_name) or "unknown",
            confidence=confidence,
            all_scores=all_scores,
            resolved_note_plant=resolved_note_plant,
            uncertain=uncertain,
        )
