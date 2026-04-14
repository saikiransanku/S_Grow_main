from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def compute_blur_score(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def iter_image_paths(root: Path) -> Iterable[Path]:
    for image_path in root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def infer_split_from_path(image_path: Path, data_root: Path) -> str | None:
    try:
        parts = [normalize_token(part) for part in image_path.relative_to(data_root).parts]
    except ValueError:
        parts = [normalize_token(part) for part in image_path.parts]
    for part in parts:
        if part in {"train", "val", "test"}:
            return part
        if part in {"valid", "validation"}:
            return "val"
    return None


def assign_splits_by_species(
    records: Sequence[object],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[int, str]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[getattr(record, "species")].append(index)

    rng = random.Random(seed)
    assigned: dict[int, str] = {}

    for indices in grouped.values():
        bucket = list(indices)
        rng.shuffle(bucket)
        total = len(bucket)

        if total == 1:
            assigned[bucket[0]] = "train"
            continue

        train_count = max(1, int(round(total * train_ratio)))
        val_count = int(round(total * val_ratio))
        test_count = total - train_count - val_count

        if total >= 3 and val_count == 0:
            val_count = 1
            train_count = max(1, train_count - 1)
        if total >= 5 and test_count == 0:
            test_count = 1
            train_count = max(1, train_count - 1)

        while train_count + val_count + test_count > total:
            if train_count > 1:
                train_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                test_count -= 1
        while train_count + val_count + test_count < total:
            train_count += 1

        for index in bucket[:train_count]:
            assigned[index] = "train"
        for index in bucket[train_count : train_count + val_count]:
            assigned[index] = "val"
        for index in bucket[train_count + val_count :]:
            assigned[index] = "test"

    return assigned


def compute_dataset_mean_std(
    image_paths: Sequence[Path],
    *,
    image_size: int = 224,
    max_items: int = 1024,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not image_paths:
        raise ValueError("At least one image is required to compute mean/std.")

    sampled = list(image_paths[:max_items]) if len(image_paths) > max_items else list(image_paths)
    pixels: list[np.ndarray] = []

    for path in sampled:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        pixels.append(image.astype(np.float32) / 255.0)

    if not pixels:
        raise RuntimeError("Could not read any images for dataset statistics.")

    stacked = np.concatenate([item.reshape(-1, 3) for item in pixels], axis=0)
    mean = tuple(float(x) for x in stacked.mean(axis=0).tolist())
    std = tuple(float(x) for x in stacked.std(axis=0).tolist())
    return mean, std
