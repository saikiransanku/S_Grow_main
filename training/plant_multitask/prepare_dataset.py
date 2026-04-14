from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .schemas import LabelVocab, PlantSample, save_manifest
    from .utils import (
        assign_splits_by_species,
        compute_blur_score,
        compute_dataset_mean_std,
        infer_split_from_path,
        iter_image_paths,
        normalize_token,
        sha256_file,
    )
except ImportError:  # pragma: no cover
    from training.plant_multitask.schemas import LabelVocab, PlantSample, save_manifest
    from training.plant_multitask.utils import (
        assign_splits_by_species,
        compute_blur_score,
        compute_dataset_mean_std,
        infer_split_from_path,
        iter_image_paths,
        normalize_token,
        sha256_file,
    )


DEFAULT_BINOMIAL_MAP = {
    "apple": "Malus domestica",
    "banana": "Musa acuminata",
    "barley": "Hordeum vulgare",
    "cherry_including_sour": "Prunus cerasus",
    "coconut": "Cocos nucifera",
    "corn_maize": "Zea mays",
    "cotton": "Gossypium hirsutum",
    "egg": "Solanum melongena",
    "eggplant": "Solanum melongena",
    "grape": "Vitis vinifera",
    "groundnut": "Arachis hypogaea",
    "maize": "Zea mays",
    "peach": "Prunus persica",
    "pepper": "Capsicum annuum",
    "pepper_bell": "Capsicum annuum",
    "pepper_pepper_bell": "Capsicum annuum",
    "potato": "Solanum tuberosum",
    "rice": "Oryza sativa",
    "strawberry": "Fragaria x ananassa",
    "sugarcane": "Saccharum officinarum",
    "sugarcrane": "Saccharum officinarum",
    "tomato": "Solanum lycopersicum",
    "turmeric": "Curcuma longa",
    "wheat": "Triticum aestivum",
    "coconut_tree_disease_dataset": "Cocos nucifera",
}

PART_KEYWORDS = {
    "leaf": "leaf",
    "flower": "flower",
    "fruit": "fruit",
    "fruitscarring": "fruit",
    "split_peel": "fruit",
    "peel": "fruit",
    "stem": "stem",
    "bud": "stem",
    "bark": "bark",
    "root": "root",
}
PEST_KEYWORDS = ("pest", "mite", "aphid", "thrip", "borer", "caterpillar", "beetle", "skipper", "insect")
HEALTHY_ALIASES = {"healthy", "healthy_leaf", "sugarcane_leaf", "leaf"}
QUESTION_TEMPLATES = (
    "What condition is affecting this {plant_part}?",
    "What disease or pest does this {plant_part} show?",
    "Is this {species} sample healthy or diseased?",
)


def load_species_map(path: Path | None) -> dict[str, str]:
    mapping = dict(DEFAULT_BINOMIAL_MAP)
    if path is None:
        return mapping

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping.update({normalize_token(key): str(value) for key, value in dict(payload).items()})
        return mapping

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[normalize_token(str(row["common_name"]))] = str(row["binomial_name"])
    return mapping


def parse_optional_bbox(raw_value: Any) -> list[float] | None:
    if raw_value in (None, "", "null"):
        return None
    if isinstance(raw_value, list):
        return [float(x) for x in raw_value]
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
            if isinstance(payload, list):
                return [float(x) for x in payload]
        except json.JSONDecodeError:
            parts = [item.strip() for item in raw_value.split(",") if item.strip()]
            if len(parts) == 4:
                return [float(x) for x in parts]
    raise ValueError(f"Could not parse bbox value: {raw_value}")


def resolve_species(common_name: str, species_map: dict[str, str]) -> str:
    key = normalize_token(common_name)
    return species_map.get(key, key.replace("_", " "))


def normalize_species_key(value: str) -> str:
    key = normalize_token(value)
    replacements = {
        "egg": "eggplant",
        "pepper_pepper_bell": "pepper_bell",
        "pepper_bell": "pepper_bell",
        "corn_maize": "maize",
        "sugarcrane": "sugarcane",
        "coconut_tree_disease_dataset": "coconut",
    }
    return replacements.get(key, key)


def detect_layout(data_root: Path) -> str:
    child_dirs = [item for item in data_root.iterdir() if item.is_dir()]
    season_dirs = [item for item in child_dirs if (item / "train").is_dir() or (item / "val").is_dir() or (item / "test").is_dir()]
    if season_dirs:
        return "seasonal_classification"
    return "flat_scan"


def infer_health_from_condition(condition: str) -> str:
    if condition in HEALTHY_ALIASES:
        return "healthy"
    if any(keyword in condition for keyword in PEST_KEYWORDS):
        return "pest_affected"
    return "diseased"


def infer_part_from_condition(condition: str) -> str:
    if condition in HEALTHY_ALIASES:
        return "leaf"
    for key, part in PART_KEYWORDS.items():
        if key in condition:
            return part
    return "leaf"


def clean_condition(species_key: str, remainder: str) -> str:
    condition = normalize_token(remainder)
    if not condition or condition in {"healthy", "healthy_leaf"}:
        return "healthy"

    replacements = {
        "sugarcane_leaf": "healthy",
        "healthy_leaf": "healthy",
        "fruitscarring_beetle": "fruit_scarring_beetle",
        "chewing_insect_damage_on_leaf": "chewing_insect_damage",
    }
    condition = replacements.get(condition, condition)
    if species_key == "coconut" and condition.startswith("tree_disease_dataset_"):
        condition = condition.replace("tree_disease_dataset_", "", 1)
    return condition


def parse_seasonal_class_label(class_name: str) -> tuple[str, str, str, str]:
    label = normalize_token(class_name)
    candidates = sorted(DEFAULT_BINOMIAL_MAP.keys(), key=len, reverse=True)
    species_key = None
    remainder = ""

    for candidate in candidates:
        if label == candidate or label.startswith(candidate + "_"):
            species_key = candidate
            remainder = label[len(candidate) :].lstrip("_")
            break

    if species_key is None:
        parts = label.split("_")
        species_key = parts[0]
        remainder = "_".join(parts[1:])

    species_key = normalize_species_key(species_key)
    condition = clean_condition(species_key, remainder)
    plant_part = infer_part_from_condition(condition)
    health_status = infer_health_from_condition(condition)
    return species_key, condition, plant_part, health_status


def build_question(species: str, plant_part: str, condition: str) -> str:
    template_index = hash((species, plant_part, condition)) % len(QUESTION_TEMPLATES)
    return QUESTION_TEMPLATES[template_index].format(
        species=species,
        plant_part=plant_part.replace("_", " "),
    )


def build_caption(species: str, plant_part: str, condition: str, season: str) -> str:
    condition_text = condition.replace("_", " ")
    season_text = season.replace("_", " ")
    return f"{plant_part} image of {species} collected in {season_text} season showing {condition_text}"


def infer_record_from_flat_path(image_path: Path, data_root: Path, species_map: dict[str, str]) -> PlantSample:
    relative_parts = [normalize_token(part) for part in image_path.relative_to(data_root).parts[:-1]]
    split = infer_split_from_path(image_path, data_root) or "unspecified"
    species_key = normalize_species_key(relative_parts[-1])
    species = resolve_species(species_key, species_map)
    plant_part = "leaf"
    health_status = "healthy" if "healthy" in "_".join(relative_parts) else "diseased"
    answer = "healthy" if health_status == "healthy" else "unknown_condition"
    season = relative_parts[0] if relative_parts else "unknown"
    return PlantSample(
        image_path=str(image_path.resolve()),
        species=species,
        plant_part=plant_part,
        health_status=health_status,
        split=split,
        source="directory_scan",
        caption=build_caption(species, plant_part, answer, season),
        question=build_question(species, plant_part, answer),
        answer=answer,
        metadata={"season": season, "condition": answer},
    )


def load_seasonal_classification_records(data_root: Path, species_map: dict[str, str]) -> list[PlantSample]:
    records: list[PlantSample] = []

    for season_dir in sorted([item for item in data_root.iterdir() if item.is_dir()], key=lambda item: item.name.lower()):
        for split_dir in sorted([item for item in season_dir.iterdir() if item.is_dir()], key=lambda item: item.name.lower()):
            split = normalize_token(split_dir.name)
            if split not in {"train", "val", "test"}:
                continue

            for class_dir in sorted([item for item in split_dir.iterdir() if item.is_dir()], key=lambda item: item.name.lower()):
                species_key, condition, plant_part, health_status = parse_seasonal_class_label(class_dir.name)
                species = resolve_species(species_key, species_map)
                season = normalize_token(season_dir.name)
                answer = condition

                for image_path in class_dir.rglob("*"):
                    if not image_path.is_file():
                        continue
                    records.append(
                        PlantSample(
                            image_path=str(image_path.resolve()),
                            species=species,
                            plant_part=plant_part,
                            health_status=health_status,
                            split=split,
                            source="seasonal_classification",
                            caption=build_caption(species, plant_part, condition, season),
                            question=build_question(species, plant_part, condition),
                            answer=answer,
                            metadata={
                                "season": season,
                                "condition": condition,
                                "raw_class_name": normalize_token(class_dir.name),
                            },
                        )
                    )
    return records


def load_annotation_manifest(
    annotations_path: Path,
    data_root: Path,
    species_map: dict[str, str],
    *,
    min_consensus_score: float,
) -> list[PlantSample]:
    if annotations_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif annotations_path.suffix.lower() == ".json":
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        rows = list(payload if isinstance(payload, list) else payload.get("records", []))
    else:
        with annotations_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    records: list[PlantSample] = []
    for row in rows:
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = data_root / image_path
        consensus_score = float(row.get("consensus_score", 1.0) or 1.0)
        if consensus_score < min_consensus_score:
            continue
        answer = normalize_token(str(row["answer"])) if row.get("answer") else None
        season = normalize_token(str(row.get("season") or "unknown"))
        species = resolve_species(str(row["species"]), species_map)
        plant_part = normalize_token(str(row["plant_part"] or "leaf"))
        record = PlantSample(
            image_path=str(image_path.resolve()),
            species=species,
            plant_part=plant_part,
            health_status=normalize_token(str(row["health_status"] or "healthy")),
            split=normalize_token(str(row.get("split") or infer_split_from_path(image_path, data_root) or "unspecified")),
            source=str(row.get("source") or annotations_path.stem),
            caption=str(row["caption"]) if row.get("caption") else build_caption(species, plant_part, answer or "healthy", season),
            question=str(row["question"]) if row.get("question") else build_question(species, plant_part, answer or "healthy"),
            answer=answer,
            bbox=parse_optional_bbox(row.get("bbox")) if row.get("bbox") else None,
            mask_path=str(row["mask_path"]) if row.get("mask_path") else None,
            metadata={
                "expert_validated": bool(row.get("expert_validated", False)),
                "consensus_score": consensus_score,
                "season": season,
                "condition": answer,
            },
        )
        records.append(record)
    return records


def validate_and_filter_records(
    records: list[PlantSample],
    *,
    min_blur_score: float,
    min_image_size: int,
) -> tuple[list[PlantSample], dict[str, Any]]:
    accepted: list[PlantSample] = []
    seen_hashes: set[str] = set()
    skipped = Counter()

    for record in records:
        image_path = Path(record.image_path)
        if not image_path.exists():
            skipped["missing_file"] += 1
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            skipped["invalid_image"] += 1
            continue

        if min(width, height) < min_image_size:
            skipped["low_resolution"] += 1
            continue

        blur_score = compute_blur_score(image_path)
        if blur_score < min_blur_score:
            skipped["blurry"] += 1
            continue

        digest = sha256_file(image_path)
        if digest in seen_hashes:
            skipped["duplicate"] += 1
            continue
        seen_hashes.add(digest)

        record.metadata["sha256"] = digest
        record.metadata["blur_score"] = round(blur_score, 4)
        record.metadata["width"] = width
        record.metadata["height"] = height
        accepted.append(record)

    return accepted, {"accepted_records": len(accepted), "skipped_records": dict(skipped)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a multi-task plant recognition manifest.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--species-map", type=Path, default=None)
    parser.add_argument("--input-layout", choices=["auto", "seasonal_classification", "flat_scan", "annotations"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-blur-score", type=float, default=25.0)
    parser.add_argument("--min-image-size", type=int, default=128)
    parser.add_argument("--min-consensus-score", type=float, default=0.5)
    parser.add_argument("--compute-stats", action="store_true")
    parser.add_argument("--stats-image-size", type=int, default=224)
    parser.add_argument("--max-stats-images", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    species_map = load_species_map(args.species_map)

    layout = args.input_layout
    if layout == "auto":
        layout = "annotations" if args.annotations else detect_layout(args.data_root)

    if layout == "annotations":
        if args.annotations is None:
            raise ValueError("--annotations is required when --input-layout=annotations")
        records = load_annotation_manifest(
            args.annotations,
            args.data_root,
            species_map,
            min_consensus_score=args.min_consensus_score,
        )
    elif layout == "seasonal_classification":
        records = load_seasonal_classification_records(args.data_root, species_map)
    else:
        records = [infer_record_from_flat_path(path, args.data_root, species_map) for path in iter_image_paths(args.data_root)]

    filtered_records, filter_report = validate_and_filter_records(
        records,
        min_blur_score=args.min_blur_score,
        min_image_size=args.min_image_size,
    )

    split_assignments = assign_splits_by_species(
        filtered_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for index, record in enumerate(filtered_records):
        if record.split not in {"train", "val", "test"}:
            record.split = split_assignments[index]

    vocab = LabelVocab.build(filtered_records)
    manifest_path = args.output_dir / "manifest.jsonl"
    vocab_path = args.output_dir / "label_vocab.json"
    report_path = args.output_dir / "prepare_report.json"
    save_manifest(manifest_path, filtered_records)
    vocab.save(vocab_path)

    split_counts = Counter(record.split for record in filtered_records)
    species_counts = Counter(record.species for record in filtered_records)
    part_counts = Counter(record.plant_part for record in filtered_records)
    health_counts = Counter(record.health_status for record in filtered_records)
    season_counts = Counter(str(record.metadata.get("season", "unknown")) for record in filtered_records)
    by_season_species: dict[str, Counter[str]] = defaultdict(Counter)
    for record in filtered_records:
        by_season_species[str(record.metadata.get("season", "unknown"))][record.species] += 1

    report: dict[str, Any] = {
        "layout": layout,
        "manifest_path": str(manifest_path),
        "label_vocab_path": str(vocab_path),
        "records": len(filtered_records),
        "split_counts": dict(split_counts),
        "species_count": len(species_counts),
        "top_species": dict(species_counts.most_common(20)),
        "part_distribution": dict(part_counts),
        "health_distribution": dict(health_counts),
        "season_distribution": dict(season_counts),
        "season_species_preview": {
            season: dict(counter.most_common(10))
            for season, counter in by_season_species.items()
        },
        "filter_report": filter_report,
    }

    if args.compute_stats:
        train_paths = [Path(record.image_path) for record in filtered_records if record.split == "train"]
        mean, std = compute_dataset_mean_std(
            train_paths,
            image_size=args.stats_image_size,
            max_items=args.max_stats_images,
        )
        stats = {"mean": mean, "std": std}
        (args.output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        report["dataset_stats"] = stats

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
