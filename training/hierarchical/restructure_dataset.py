from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_CLASSES = {"unknown_disease", "unknown"}
DEFAULT_SPLIT_NAMES = ("train", "val", "test")
STAGE1_ONLY_CLASSES = {
    "other",
    "others",
    "non_leaf",
    "non_leaf_or_fruit",
    "not_leaf",
    "not_leaf_or_fruit",
}


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def clean_plant_token(token: str) -> str:
    value = token.strip("_")
    if value.startswith("leaf_"):
        value = value[len("leaf_") :]
    if value.endswith("_leaf"):
        value = value[: -len("_leaf")]
    return value.strip("_")


def is_stage1_only_class(class_name: str) -> bool:
    return normalize_token(class_name) in STAGE1_ONLY_CLASSES


def iter_image_files(path: Path) -> Iterable[Path]:
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield file_path


def collect_class_dirs(dataset_root: Path) -> List[Path]:
    return sorted([entry for entry in dataset_root.iterdir() if entry.is_dir()], key=lambda p: p.name.lower())


def detect_dataset_splits(dataset_root: Path) -> List[str]:
    return [split for split in DEFAULT_SPLIT_NAMES if (dataset_root / split).is_dir()]


def build_known_plants(class_names: List[str]) -> List[str]:
    plants = set()

    for raw in class_names:
        label = normalize_token(raw)
        if label in SKIP_CLASSES:
            continue

        if label.startswith("healthy_"):
            plant = clean_plant_token(label[len("healthy_") :])
            if plant:
                plants.add(plant)
            continue

        if label.endswith("_healthy"):
            plant = clean_plant_token(label[: -len("_healthy")])
            if plant:
                plants.add(plant)
            continue

        match = re.match(r"^(?P<plant>.+?)_healthy(?:_.+)?$", label)
        if match:
            plant = clean_plant_token(match.group("plant"))
            if plant:
                plants.add(plant)

    if plants:
        return sorted(plants, key=len, reverse=True)

    for raw in class_names:
        label = normalize_token(raw)
        if label in SKIP_CLASSES:
            continue
        parts = [part for part in label.split("_") if part]
        if len(parts) >= 2:
            plants.add(parts[-1])

    return sorted(plants, key=len, reverse=True)


def parse_label(class_name: str, known_plants: List[str]) -> Tuple[Optional[str], Optional[str]]:
    label = normalize_token(class_name)
    if label in SKIP_CLASSES:
        return None, None

    if label.startswith("healthy_"):
        plant = clean_plant_token(label[len("healthy_") :])
        return (plant or None), "healthy"

    if label.endswith("_healthy"):
        plant = clean_plant_token(label[: -len("_healthy")])
        return (plant or None), "healthy"

    healthy_match = re.match(r"^(?P<plant>.+?)_healthy(?:_.+)?$", label)
    if healthy_match:
        plant = clean_plant_token(healthy_match.group("plant"))
        return (plant or None), "healthy"

    for plant in known_plants:
        suffix = f"_{plant}"
        prefix = f"{plant}_"
        if label.endswith(suffix):
            disease = label[: -len(suffix)]
            return plant, disease or None
        if label.startswith(prefix):
            disease = label[len(prefix) :]
            return plant, disease or None

    parts = [part for part in label.split("_") if part]
    if len(parts) >= 2:
        return parts[-1], "_".join(parts[:-1])

    return None, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    base = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{base}__dup{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def safe_copy_with_verify(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = ensure_unique_path(destination)
    temp_path = destination.with_name(f".{destination.name}.tmp_{uuid.uuid4().hex}")

    shutil.copy2(source, temp_path)

    if source.stat().st_size != temp_path.stat().st_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch after copy: {source} -> {destination}")

    if sha256_file(source) != sha256_file(temp_path):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Hash mismatch after copy: {source} -> {destination}")

    os.replace(temp_path, destination)
    return destination


def prune_empty_dirs(root: Path) -> None:
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                next(directory.iterdir())
            except StopIteration:
                directory.rmdir()


@dataclass
class Stats:
    total_classes_seen: int = 0
    total_classes_skipped: int = 0
    total_images_seen: int = 0
    total_images_processed: int = 0
    total_images_failed: int = 0


def build_target_name(class_name: str, source_file: Path, class_root: Path) -> str:
    rel = source_file.relative_to(class_root)
    rel_token = normalize_token(str(rel.parent))
    stem_token = normalize_token(source_file.stem)
    class_token = normalize_token(class_name)
    ext = source_file.suffix.lower()

    if rel_token and rel_token != "":
        return f"{class_token}__{rel_token}__{stem_token}{ext}"
    return f"{class_token}__{stem_token}{ext}"


def restructure_dataset(
    *,
    source_dataset: Path,
    plant_dataset: Path,
    disease_dataset: Path,
    operation: str,
    clean: bool,
    dry_run: bool,
    manifest_path: Path,
) -> Dict[str, object]:
    if clean and not dry_run:
        if plant_dataset.exists():
            shutil.rmtree(plant_dataset)
        if disease_dataset.exists():
            shutil.rmtree(disease_dataset)

    if not dry_run:
        plant_dataset.mkdir(parents=True, exist_ok=True)
        disease_dataset.mkdir(parents=True, exist_ok=True)

    source_splits = detect_dataset_splits(source_dataset)
    split_roots = (
        [(split, source_dataset / split) for split in source_splits]
        if source_splits
        else [(None, source_dataset)]
    )

    class_names: List[str] = []
    for _, split_root in split_roots:
        class_names.extend(item.name for item in collect_class_dirs(split_root))
    known_plants = build_known_plants(class_names)

    stats = Stats()
    skipped_classes: Dict[str, str] = {}
    failures: List[Dict[str, str]] = []
    by_plant: Dict[str, int] = defaultdict(int)
    stage1_only_by_class: Dict[str, int] = defaultdict(int)
    by_disease: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_split_plant: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_split_disease: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for split_name, split_root in split_roots:
        target_split = split_name or "all"
        class_dirs = collect_class_dirs(split_root)

        for class_dir in class_dirs:
            stats.total_classes_seen += 1
            class_key = normalize_token(class_dir.name)
            plant, disease = parse_label(class_dir.name, known_plants)
            skip_key = f"{target_split}:{class_dir.name}" if split_name else class_dir.name

            if is_stage1_only_class(class_key):
                plant_root = plant_dataset / split_name if split_name else plant_dataset

                for image_path in iter_image_files(class_dir):
                    stats.total_images_seen += 1
                    target_name = build_target_name(class_dir.name, image_path, class_dir)
                    plant_target = plant_root / class_key / target_name

                    if dry_run:
                        stats.total_images_processed += 1
                        by_plant[class_key] += 1
                        stage1_only_by_class[class_key] += 1
                        by_split_plant[target_split][class_key] += 1
                        continue

                    copied_target: Optional[Path] = None
                    try:
                        copied_target = safe_copy_with_verify(image_path, plant_target)

                        if operation == "move":
                            image_path.unlink(missing_ok=True)

                        stats.total_images_processed += 1
                        by_plant[class_key] += 1
                        stage1_only_by_class[class_key] += 1
                        by_split_plant[target_split][class_key] += 1
                    except Exception as exc:
                        stats.total_images_failed += 1
                        if copied_target is not None:
                            copied_target.unlink(missing_ok=True)
                        failures.append({"file": str(image_path), "error": str(exc)})

                continue

            if not plant or not disease:
                stats.total_classes_skipped += 1
                skipped_classes[skip_key] = "unable_to_parse"
                continue

            plant_key = normalize_token(plant)
            disease_key = normalize_token(disease)

            if disease_key in SKIP_CLASSES:
                stats.total_classes_skipped += 1
                skipped_classes[skip_key] = "unknown_disease_removed"
                continue

            plant_root = plant_dataset / split_name if split_name else plant_dataset
            disease_root = disease_dataset / split_name if split_name else disease_dataset

            for image_path in iter_image_files(class_dir):
                stats.total_images_seen += 1
                target_name = build_target_name(class_dir.name, image_path, class_dir)

                plant_target = plant_root / plant_key / target_name
                disease_target = disease_root / plant_key / disease_key / target_name

                if dry_run:
                    stats.total_images_processed += 1
                    by_plant[plant_key] += 1
                    by_disease[plant_key][disease_key] += 1
                    by_split_plant[target_split][plant_key] += 1
                    by_split_disease[target_split][plant_key][disease_key] += 1
                    continue

                copied_targets: List[Path] = []
                try:
                    copied_targets.append(safe_copy_with_verify(image_path, plant_target))
                    copied_targets.append(safe_copy_with_verify(image_path, disease_target))

                    if operation == "move":
                        image_path.unlink(missing_ok=True)

                    stats.total_images_processed += 1
                    by_plant[plant_key] += 1
                    by_disease[plant_key][disease_key] += 1
                    by_split_plant[target_split][plant_key] += 1
                    by_split_disease[target_split][plant_key][disease_key] += 1
                except Exception as exc:
                    stats.total_images_failed += 1
                    for copied in copied_targets:
                        copied.unlink(missing_ok=True)
                    failures.append({"file": str(image_path), "error": str(exc)})

    if operation == "move" and not dry_run:
        prune_empty_dirs(source_dataset)

    manifest = {
        "source_dataset": str(source_dataset),
        "plant_dataset": str(plant_dataset),
        "disease_dataset": str(disease_dataset),
        "dataset_layout": "split" if source_splits else "flat",
        "source_splits": source_splits or ["all"],
        "operation": operation,
        "dry_run": dry_run,
        "stats": {
            "total_classes_seen": stats.total_classes_seen,
            "total_classes_skipped": stats.total_classes_skipped,
            "total_images_seen": stats.total_images_seen,
            "total_images_processed": stats.total_images_processed,
            "total_images_failed": stats.total_images_failed,
        },
        "skipped_classes": skipped_classes,
        "failures": failures,
        "stage1_only_classes": dict(sorted(stage1_only_by_class.items())),
        "plants": {
            plant: {
                "total_images": by_plant[plant],
                "diseases": dict(sorted(by_disease[plant].items())),
            }
            for plant in sorted(by_plant.keys())
        },
        "splits": {
            split: {
                "plants": {
                    plant: {
                        "total_images": by_split_plant[split][plant],
                        "diseases": dict(sorted(by_split_disease[split][plant].items())),
                    }
                    for plant in sorted(by_split_plant[split].keys())
                }
            }
            for split in sorted(by_split_plant.keys())
        },
    }

    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restructure a flat or split disease dataset into hierarchical plant and disease datasets."
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--plant-dataset", type=Path, required=True)
    parser.add_argument("--disease-dataset", type=Path, required=True)
    parser.add_argument("--operation", choices=["move", "copy"], default="move")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/outputs/project_data/hierarchical_dataset_manifest.json"),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source_dataset}")

    report = restructure_dataset(
        source_dataset=args.source_dataset,
        plant_dataset=args.plant_dataset,
        disease_dataset=args.disease_dataset,
        operation=args.operation,
        clean=args.clean,
        dry_run=args.dry_run,
        manifest_path=args.manifest_path,
    )

    logging.info("Completed restructuring. Processed images: %s", report["stats"]["total_images_processed"])
    logging.info("Failed images: %s", report["stats"]["total_images_failed"])


if __name__ == "__main__":
    main()
