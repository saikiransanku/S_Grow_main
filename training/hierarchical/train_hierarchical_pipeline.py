from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

try:
    from restructure_dataset import restructure_dataset
    from train_plant_classifier import PlantClassifierArtifacts, train_plant_classifier
    from train_plant_disease_model import DiseaseModelArtifacts, train_all_disease_models
except ImportError:  # pragma: no cover - enables module execution from repo root
    from training.hierarchical.restructure_dataset import restructure_dataset
    from training.hierarchical.train_plant_classifier import PlantClassifierArtifacts, train_plant_classifier
    from training.hierarchical.train_plant_disease_model import DiseaseModelArtifacts, train_all_disease_models


PIPELINE_VERSION = "hierarchical_pipeline_v1"
PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_PLANT_DATASET_DIR = PIPELINE_ROOT / "datasets" / "plant_dataset"
DEFAULT_DISEASE_DATASET_DIR = PIPELINE_ROOT / "datasets" / "disease_dataset"
DEFAULT_ARTIFACTS_DIR = PIPELINE_ROOT / "artifacts"
DEFAULT_DISEASE_MODELS_DIR = PIPELINE_ROOT / "disease_models"


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def artifact_to_manifest_entry(artifacts: PlantClassifierArtifacts | DiseaseModelArtifacts) -> Dict[str, object]:
    return {
        "model_path": str(artifacts.model_path),
        "classes_path": str(artifacts.classes_path),
        "summary_path": str(artifacts.summary_path),
        "class_names": list(artifacts.class_names),
        "best_val_acc": artifacts.best_val_acc,
        "test_acc": artifacts.test_acc,
        "image_size": artifacts.image_size,
    }


def collect_existing_plant_artifacts(artifacts_dir: Path) -> Dict[str, object] | None:
    model_path = artifacts_dir / "plant_classifier.pth"
    classes_path = artifacts_dir / "plant_classifier_classes.json"
    summary_path = artifacts_dir / "plant_classifier_summary.json"
    if not model_path.exists() or not classes_path.exists():
        return None

    class_names = json.loads(classes_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    return {
        "model_path": str(model_path),
        "classes_path": str(classes_path),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "class_names": class_names if isinstance(class_names, list) else [],
        "best_val_acc": metrics.get("best_val_acc"),
        "test_acc": metrics.get("test_acc"),
        "image_size": summary.get("image_size"),
    }


def collect_existing_disease_artifacts(disease_models_dir: Path) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    if not disease_models_dir.exists():
        return entries

    for model_path in sorted(disease_models_dir.rglob("*_disease_model.pth")):
        plant_name = normalize_token(model_path.stem.removesuffix("_disease_model"))
        model_dir = model_path.parent
        classes_path = model_dir / f"{plant_name}_disease_classes.json"
        summary_path = model_dir / f"{plant_name}_disease_summary.json"

        class_names = []
        if classes_path.exists():
            payload = json.loads(classes_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                class_names = payload

        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        entries[plant_name] = {
            "model_path": str(model_path),
            "classes_path": str(classes_path) if classes_path.exists() else None,
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "class_names": class_names,
            "best_val_acc": metrics.get("best_val_acc"),
            "test_acc": metrics.get("test_acc"),
            "image_size": summary.get("image_size"),
        }

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full hierarchical plant disease training pipeline.")
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--plant-dataset-dir", type=Path, default=DEFAULT_PLANT_DATASET_DIR)
    parser.add_argument("--disease-dataset-dir", type=Path, default=DEFAULT_DISEASE_DATASET_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--disease-models-dir", type=Path, default=DEFAULT_DISEASE_MODELS_DIR)
    parser.add_argument("--operation", choices=["copy", "move"], default="copy")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-restructure", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--dataset-manifest-path", type=Path, default=None)
    parser.add_argument("--pipeline-manifest-path", type=Path, default=None)

    parser.add_argument("--plant-epochs", type=int, default=30)
    parser.add_argument("--plant-batch-size", type=int, default=24)
    parser.add_argument("--plant-image-size", type=int, default=300)
    parser.add_argument("--plant-freeze-epochs", type=int, default=4)
    parser.add_argument("--plant-head-lr", type=float, default=1e-3)
    parser.add_argument("--plant-finetune-lr", type=float, default=2e-4)

    parser.add_argument("--disease-epochs", type=int, default=25)
    parser.add_argument("--disease-batch-size", type=int, default=20)
    parser.add_argument("--disease-image-size", type=int, default=300)
    parser.add_argument("--disease-freeze-epochs", type=int, default=3)
    parser.add_argument("--disease-head-lr", type=float, default=1e-3)
    parser.add_argument("--disease-finetune-lr", type=float, default=2e-4)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source_dataset}")

    if args.dry_run and (args.skip_restructure or not args.skip_stage1 or not args.skip_stage2):
        raise ValueError("--dry-run can only be used for restructure preview. Use --skip-stage1 --skip-stage2 with it.")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path = args.dataset_manifest_path or artifacts_dir / "hierarchical_dataset_manifest.json"
    pipeline_manifest_path = args.pipeline_manifest_path or artifacts_dir / "hierarchical_pipeline_manifest.json"

    restructure_report: Dict[str, object] | None = None
    if not args.skip_restructure:
        restructure_report = restructure_dataset(
            source_dataset=args.source_dataset,
            plant_dataset=args.plant_dataset_dir,
            disease_dataset=args.disease_dataset_dir,
            operation=args.operation,
            clean=args.clean,
            dry_run=args.dry_run,
            manifest_path=dataset_manifest_path,
        )
        logging.info(
            "Restructuring completed | processed=%s failed=%s",
            restructure_report["stats"]["total_images_processed"],
            restructure_report["stats"]["total_images_failed"],
        )

    if args.dry_run:
        logging.info("Dry run finished. Dataset preview available in memory only.")
        return

    plant_artifacts_entry: Dict[str, object] | None = None
    if args.skip_stage1:
        plant_artifacts_entry = collect_existing_plant_artifacts(artifacts_dir)
    else:
        plant_artifacts = train_plant_classifier(
            data_dir=args.plant_dataset_dir,
            output_dir=artifacts_dir,
            epochs=args.plant_epochs,
            batch_size=args.plant_batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            image_size=args.plant_image_size,
            head_lr=args.plant_head_lr,
            finetune_lr=args.plant_finetune_lr,
            weight_decay=args.weight_decay,
            freeze_epochs=args.plant_freeze_epochs,
            seed=args.seed,
            pretrained=not args.no_pretrained,
        )
        plant_artifacts_entry = artifact_to_manifest_entry(plant_artifacts)

    disease_artifacts_entries: Dict[str, Dict[str, object]]
    if args.skip_stage2:
        disease_artifacts_entries = collect_existing_disease_artifacts(args.disease_models_dir)
    else:
        disease_artifacts = train_all_disease_models(
            disease_root=args.disease_dataset_dir,
            output_root=args.disease_models_dir,
            epochs=args.disease_epochs,
            batch_size=args.disease_batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            image_size=args.disease_image_size,
            freeze_epochs=args.disease_freeze_epochs,
            head_lr=args.disease_head_lr,
            finetune_lr=args.disease_finetune_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            pretrained=not args.no_pretrained,
        )
        disease_artifacts_entries = {
            plant_name: artifact_to_manifest_entry(artifact) for plant_name, artifact in disease_artifacts.items()
        }

    pipeline_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.source_dataset),
        "plant_dataset_dir": str(args.plant_dataset_dir),
        "disease_dataset_dir": str(args.disease_dataset_dir),
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path.exists() else None,
        "artifacts_dir": str(artifacts_dir),
        "disease_models_dir": str(args.disease_models_dir),
        "plant_classifier": plant_artifacts_entry,
        "disease_models": disease_artifacts_entries,
        "supported_plants": sorted(disease_artifacts_entries.keys()),
        "recommended_thresholds": {
            "plant_confidence": 0.55,
            "disease_confidence": 0.60,
        },
    }

    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, indent=2), encoding="utf-8")
    logging.info("Hierarchical pipeline manifest written to %s", pipeline_manifest_path)


if __name__ == "__main__":
    main()
