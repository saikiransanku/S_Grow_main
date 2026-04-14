import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
from django.conf import settings

DEFAULT_ACTIVE_LEARNING_MIN_CONFIDENCE = 0.50


def resolve_active_learning_dataset_root() -> str:
    project_root = settings.BASE_DIR.parent
    return os.path.expanduser(
        os.getenv(
            "ACTIVE_LEARNING_DATASET_ROOT",
            str(project_root / "data" / "datasets" / "active_learning_crops"),
        )
    )


def image_data_to_pil_image(image_data: Any) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")

    if isinstance(image_data, np.ndarray):
        array = np.asarray(image_data)
        if array.size == 0:
            raise ValueError("Image array is empty.")
        if array.ndim == 2:
            return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] == 1:
            return Image.fromarray(
                np.clip(array[:, :, 0], 0, 255).astype(np.uint8),
                mode="L",
            ).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 3:
            rgb = cv2.cvtColor(np.clip(array, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        if array.ndim == 3 and array.shape[2] == 4:
            rgba = cv2.cvtColor(np.clip(array, 0, 255).astype(np.uint8), cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba).convert("RGB")
        raise ValueError("Unsupported image array shape for active-learning storage.")

    raise TypeError("image_data must be a PIL Image or OpenCV-compatible numpy array.")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_discovery_prediction(raw_prediction: Any) -> Tuple[str, float]:
    plant_class_name = ""
    confidence = 0.0

    if isinstance(raw_prediction, dict):
        plant_class_name = str(
            raw_prediction.get("plant_class_name")
            or raw_prediction.get("class_name")
            or raw_prediction.get("label")
            or raw_prediction.get("plant")
            or ""
        ).strip()
        confidence = _coerce_float(
            raw_prediction.get("confidence")
            if raw_prediction.get("confidence") is not None
            else raw_prediction.get("score"),
            0.0,
        )
    elif isinstance(raw_prediction, (tuple, list)) and len(raw_prediction) >= 2:
        plant_class_name = str(raw_prediction[0] or "").strip()
        confidence = _coerce_float(raw_prediction[1], 0.0)
    else:
        raise ValueError("discovery_cnn.predict() must return a (plant_class_name, confidence) pair.")

    if not plant_class_name:
        raise ValueError("Discovery CNN did not return a plant class name.")
    return plant_class_name, float(confidence)


def handle_unknown_crop(
    image_data,
    discovery_cnn,
    dataset_root_path,
    minimum_confidence: float = DEFAULT_ACTIVE_LEARNING_MIN_CONFIDENCE,
):
    # Use the broader plant identifier as a second-pass gate before saving anything.
    prediction = discovery_cnn.predict(image_data)
    plant_class_name, confidence = _parse_discovery_prediction(prediction)

    # Reject images that do not look like recognizable plant material.
    if confidence < float(minimum_confidence):
        return {
            "status": "rejected",
            "message": "No recognizable plant features detected.",
        }

    root_path = os.path.abspath(os.fspath(dataset_root_path))
    os.makedirs(root_path, exist_ok=True)

    # Save recognized plants into plant-named folders so they can be reused for retraining.
    folder_name = plant_class_name.strip()
    if not folder_name:
        raise ValueError("Plant class name cannot be empty when saving active-learning data.")

    plant_folder = os.path.abspath(os.path.join(root_path, folder_name))
    if os.path.commonpath([root_path, plant_folder]) != root_path:
        raise ValueError("Resolved plant folder escapes the configured dataset root path.")
    os.makedirs(plant_folder, exist_ok=True)

    # Build a collision-resistant filename from plant name, UTC timestamp, and a short UUID.
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    short_uuid = uuid.uuid4().hex[:8]
    plant_slug = re.sub(r"[^A-Za-z0-9]+", "_", folder_name).strip("_") or "plant"
    filename = f"{plant_slug}_{timestamp}_{short_uuid}.png"
    save_path = os.path.join(plant_folder, filename)

    image = image_data_to_pil_image(image_data)
    image.save(save_path, format="PNG")

    return {
        "status": "saved",
        "message": (
            f"SGrow does not currently support disease diagnosis for {plant_class_name}, "
            "but we have saved this image to improve our systems in the future!"
        ),
    }
