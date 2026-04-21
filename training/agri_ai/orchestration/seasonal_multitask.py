from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.plant_multitask import evaluate, export, inference, prepare_dataset, schemas, train, utils

__all__ = [
    "TF_TO_MULTITASK_BACKBONE",
    "normalize_season_name",
    "resolve_season_dataset_root",
    "load_single_season_classification_records",
    "prepare_single_season_manifest",
    "prepare_dataset",
    "train",
    "evaluate",
    "export",
    "inference",
    "build_parser",
    "run_multitask_training_for_season",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEASONAL_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "image_prediction_seasonal_dataset"
DEFAULT_MULTITASK_RUNS_DIR = REPO_ROOT / "training" / "image_prediction" / "runs"
SEASON_ALIASES = {
    "all": "all_season",
    "all_season": "all_season",
    "kharif": "kharif",
    "rabi": "rabi",
}
TF_TO_MULTITASK_BACKBONE = {
    "efficientnetb0": "efficientnet_v2_s",
    "resnet50": "convnext_tiny",
}


def normalize_season_name(season: str) -> str:
    normalized = str(season or "").strip().lower()
    if normalized not in SEASON_ALIASES:
        valid = ", ".join(sorted(SEASON_ALIASES))
        raise ValueError(f"Unsupported season '{season}'. Expected one of: {valid}")
    return SEASON_ALIASES[normalized]


def resolve_season_dataset_root(season: str) -> Path:
    dataset_root = (DEFAULT_SEASONAL_DATASET_ROOT / season).resolve()
    required_dirs = [dataset_root / split for split in ("train", "val")]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset is missing required folders: {missing_text}")
    return dataset_root


def _run_module_main(main_func: Any, argv: list[str]) -> Any:
    previous_argv = sys.argv[:]
    sys.argv = [str(Path(__file__).resolve())] + [str(item) for item in argv]
    try:
        return main_func()
    finally:
        sys.argv = previous_argv


def load_single_season_classification_records(
    data_root: Path,
    species_map: dict[str, str],
) -> list[schemas.PlantSample]:
    records: list[schemas.PlantSample] = []
    season = utils.normalize_token(data_root.name)

    split_dirs = [item for item in data_root.iterdir() if item.is_dir()]
    for split_dir in sorted(split_dirs, key=lambda item: item.name.lower()):
        split = utils.normalize_token(split_dir.name)
        if split not in {"train", "val", "test"}:
            continue

        class_dirs = [item for item in split_dir.iterdir() if item.is_dir()]
        for class_dir in sorted(class_dirs, key=lambda item: item.name.lower()):
            species_key, condition, plant_part, health_status = prepare_dataset.parse_seasonal_class_label(class_dir.name)
            species = prepare_dataset.resolve_species(species_key, species_map)
            answer = condition

            for image_path in class_dir.rglob("*"):
                if not image_path.is_file():
                    continue
                records.append(
                    schemas.PlantSample(
                        image_path=str(image_path.resolve()),
                        species=species,
                        plant_part=plant_part,
                        health_status=health_status,
                        split=split,
                        source="seasonal_classification",
                        caption=prepare_dataset.build_caption(species, plant_part, condition, season),
                        question=prepare_dataset.build_question(species, plant_part, condition),
                        answer=answer,
                        metadata={
                            "season": season,
                            "condition": condition,
                            "raw_class_name": utils.normalize_token(class_dir.name),
                        },
                    )
                )
    return records


def prepare_single_season_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    min_blur_score: float = 25.0,
    min_image_size: int = 128,
    compute_stats: bool = True,
    stats_image_size: int = 224,
    max_stats_images: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    del seed
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    species_map = prepare_dataset.load_species_map(None)
    records = load_single_season_classification_records(dataset_root, species_map)
    filtered_records, filter_report = prepare_dataset.validate_and_filter_records(
        records,
        min_blur_score=min_blur_score,
        min_image_size=min_image_size,
    )

    vocab = schemas.LabelVocab.build(filtered_records)
    manifest_path = output_dir / "manifest.jsonl"
    vocab_path = output_dir / "label_vocab.json"
    report_path = output_dir / "prepare_report.json"
    schemas.save_manifest(manifest_path, filtered_records)
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
        "layout": "seasonal_classification",
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
            season_name: dict(counter.most_common(10))
            for season_name, counter in by_season_species.items()
        },
        "filter_report": filter_report,
    }

    stats_path = output_dir / "dataset_stats.json"
    if compute_stats:
        train_paths = [Path(record.image_path) for record in filtered_records if record.split == "train"]
        mean, std = utils.compute_dataset_mean_std(
            train_paths,
            image_size=stats_image_size,
            max_items=max_stats_images,
        )
        stats = {"mean": mean, "std": std}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        report["dataset_stats"] = stats

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "label_vocab_path": vocab_path,
        "report_path": report_path,
        "stats_path": stats_path if compute_stats else None,
        "report": report,
    }


def run_multitask_training_for_season(
    *,
    season: str,
    workspace: str | Path | None = None,
    backbone: str = "efficientnet_v2_s",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    epochs: int = 45,
    warmup_epochs: int = 4,
    freeze_backbone_epochs: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    mix_probability: float = 0.35,
    mix_alpha: float = 0.4,
    embedding_dim: int = 512,
    dropout: float = 0.25,
    attention_heads: int = 8,
    task_hidden_dim: int = 512,
    cross_attention_layers: int = 2,
    text_width: int = 256,
    text_heads: int = 4,
    text_layers: int = 2,
    grad_accumulation_steps: int = 1,
    ema_decay: float = 0.9995,
    label_smoothing: float = 0.05,
    species_loss_type: str = "focal",
    part_loss_type: str = "cross_entropy",
    health_loss_type: str = "cross_entropy",
    focal_gamma: float = 1.5,
    seed: int = 42,
    device: str | None = None,
    no_pretrained: bool = False,
    disable_weighted_sampling: bool = False,
    disable_uncertainty_weighting: bool = False,
    sampler_species_power: float = 1.0,
    sampler_part_power: float = 0.2,
    sampler_health_power: float = 0.25,
    sampler_answer_power: float = 0.2,
    sampler_season_power: float = 0.1,
    min_blur_score: float = 25.0,
    min_image_size: int = 128,
    stats_image_size: int = 224,
    max_stats_images: int = 1024,
) -> Path:
    normalized_season = normalize_season_name(season)
    dataset_root = resolve_season_dataset_root(normalized_season)
    workspace_path = Path(workspace) if workspace else (DEFAULT_MULTITASK_RUNS_DIR / normalized_season / "multitask")
    workspace_path = workspace_path.resolve()
    processed_dir = workspace_path / "processed"
    model_dir = workspace_path / "model"

    prepare_result = prepare_single_season_manifest(
        dataset_root,
        processed_dir,
        min_blur_score=min_blur_score,
        min_image_size=min_image_size,
        compute_stats=True,
        stats_image_size=stats_image_size,
        max_stats_images=max_stats_images,
        seed=seed,
    )

    argv = [
        "--manifest-path",
        str(prepare_result["manifest_path"]),
        "--stats-path",
        str(prepare_result["stats_path"]),
        "--output-dir",
        str(model_dir),
        "--backbone",
        str(backbone),
        "--image-size",
        str(image_size),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--epochs",
        str(epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--freeze-backbone-epochs",
        str(freeze_backbone_epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--patience",
        str(patience),
        "--mix-probability",
        str(mix_probability),
        "--mix-alpha",
        str(mix_alpha),
        "--embedding-dim",
        str(embedding_dim),
        "--dropout",
        str(dropout),
        "--attention-heads",
        str(attention_heads),
        "--task-hidden-dim",
        str(task_hidden_dim),
        "--cross-attention-layers",
        str(cross_attention_layers),
        "--text-width",
        str(text_width),
        "--text-heads",
        str(text_heads),
        "--text-layers",
        str(text_layers),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
        "--ema-decay",
        str(ema_decay),
        "--label-smoothing",
        str(label_smoothing),
        "--species-loss-type",
        str(species_loss_type),
        "--part-loss-type",
        str(part_loss_type),
        "--health-loss-type",
        str(health_loss_type),
        "--focal-gamma",
        str(focal_gamma),
        "--seed",
        str(seed),
        "--sampler-species-power",
        str(sampler_species_power),
        "--sampler-part-power",
        str(sampler_part_power),
        "--sampler-health-power",
        str(sampler_health_power),
        "--sampler-answer-power",
        str(sampler_answer_power),
        "--sampler-season-power",
        str(sampler_season_power),
    ]

    if device:
        argv.extend(["--device", str(device)])
    if no_pretrained:
        argv.append("--no-pretrained")
    if disable_weighted_sampling:
        argv.append("--disable-weighted-sampling")
    if disable_uncertainty_weighting:
        argv.append("--disable-uncertainty-weighting")

    _run_module_main(train.main, argv)
    return model_dir / "best_checkpoint.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agri-AI seasonal multitask orchestration helpers.")
    parser.add_argument(
        "command",
        nargs="?",
        default="season-train",
        choices=["season-train", "prepare", "train", "evaluate", "export", "inference"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    known_args, remaining = parser.parse_known_args()

    if known_args.command == "prepare":
        _run_module_main(prepare_dataset.main, remaining)
        return
    if known_args.command == "train":
        _run_module_main(train.main, remaining)
        return
    if known_args.command == "evaluate":
        _run_module_main(evaluate.main, remaining)
        return
    if known_args.command == "export":
        _run_module_main(export.main, remaining)
        return
    if known_args.command == "inference":
        _run_module_main(inference.main, remaining)
        return

    season_parser = argparse.ArgumentParser(description="Train the multitask model on a seasonal dataset.")
    season_parser.add_argument("--season", default="all_season")
    season_parser.add_argument("--workspace", type=Path, default=None)
    season_parser.add_argument("--backbone", choices=train.BACKBONE_CHOICES, default="efficientnet_v2_s")
    season_parser.add_argument("--image-size", type=int, default=224)
    season_parser.add_argument("--batch-size", type=int, default=16)
    season_parser.add_argument("--num-workers", type=int, default=4)
    season_parser.add_argument("--epochs", type=int, default=45)
    season_parser.add_argument("--warmup-epochs", type=int, default=4)
    season_parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    season_parser.add_argument("--lr", type=float, default=3e-4)
    season_parser.add_argument("--weight-decay", type=float, default=1e-4)
    season_parser.add_argument("--patience", type=int, default=8)
    season_parser.add_argument("--mix-probability", type=float, default=0.35)
    season_parser.add_argument("--mix-alpha", type=float, default=0.4)
    season_parser.add_argument("--embedding-dim", type=int, default=512)
    season_parser.add_argument("--dropout", type=float, default=0.25)
    season_parser.add_argument("--attention-heads", type=int, default=8)
    season_parser.add_argument("--task-hidden-dim", type=int, default=512)
    season_parser.add_argument("--cross-attention-layers", type=int, default=2)
    season_parser.add_argument("--text-width", type=int, default=256)
    season_parser.add_argument("--text-heads", type=int, default=4)
    season_parser.add_argument("--text-layers", type=int, default=2)
    season_parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    season_parser.add_argument("--ema-decay", type=float, default=0.9995)
    season_parser.add_argument("--label-smoothing", type=float, default=0.05)
    season_parser.add_argument("--species-loss-type", choices=["cross_entropy", "focal"], default="focal")
    season_parser.add_argument("--part-loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    season_parser.add_argument("--health-loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    season_parser.add_argument("--focal-gamma", type=float, default=1.5)
    season_parser.add_argument("--seed", type=int, default=42)
    season_parser.add_argument("--device", default=None)
    season_parser.add_argument("--no-pretrained", action="store_true")
    season_parser.add_argument("--disable-weighted-sampling", action="store_true")
    season_parser.add_argument("--disable-uncertainty-weighting", action="store_true")
    season_parser.add_argument("--sampler-species-power", type=float, default=1.0)
    season_parser.add_argument("--sampler-part-power", type=float, default=0.2)
    season_parser.add_argument("--sampler-health-power", type=float, default=0.25)
    season_parser.add_argument("--sampler-answer-power", type=float, default=0.2)
    season_parser.add_argument("--sampler-season-power", type=float, default=0.1)
    season_parser.add_argument("--min-blur-score", type=float, default=25.0)
    season_parser.add_argument("--min-image-size", type=int, default=128)
    season_parser.add_argument("--stats-image-size", type=int, default=224)
    season_parser.add_argument("--max-stats-images", type=int, default=1024)
    season_args = season_parser.parse_args(remaining)

    final_model_path = run_multitask_training_for_season(
        season=season_args.season,
        workspace=season_args.workspace,
        backbone=season_args.backbone,
        image_size=season_args.image_size,
        batch_size=season_args.batch_size,
        num_workers=season_args.num_workers,
        epochs=season_args.epochs,
        warmup_epochs=season_args.warmup_epochs,
        freeze_backbone_epochs=season_args.freeze_backbone_epochs,
        lr=season_args.lr,
        weight_decay=season_args.weight_decay,
        patience=season_args.patience,
        mix_probability=season_args.mix_probability,
        mix_alpha=season_args.mix_alpha,
        embedding_dim=season_args.embedding_dim,
        dropout=season_args.dropout,
        attention_heads=season_args.attention_heads,
        task_hidden_dim=season_args.task_hidden_dim,
        cross_attention_layers=season_args.cross_attention_layers,
        text_width=season_args.text_width,
        text_heads=season_args.text_heads,
        text_layers=season_args.text_layers,
        grad_accumulation_steps=season_args.grad_accumulation_steps,
        ema_decay=season_args.ema_decay,
        label_smoothing=season_args.label_smoothing,
        species_loss_type=season_args.species_loss_type,
        part_loss_type=season_args.part_loss_type,
        health_loss_type=season_args.health_loss_type,
        focal_gamma=season_args.focal_gamma,
        seed=season_args.seed,
        device=season_args.device,
        no_pretrained=season_args.no_pretrained,
        disable_weighted_sampling=season_args.disable_weighted_sampling,
        disable_uncertainty_weighting=season_args.disable_uncertainty_weighting,
        sampler_species_power=season_args.sampler_species_power,
        sampler_part_power=season_args.sampler_part_power,
        sampler_health_power=season_args.sampler_health_power,
        sampler_answer_power=season_args.sampler_answer_power,
        sampler_season_power=season_args.sampler_season_power,
        min_blur_score=season_args.min_blur_score,
        min_image_size=season_args.min_image_size,
        stats_image_size=season_args.stats_image_size,
        max_stats_images=season_args.max_stats_images,
    )
    print(f"Training finished. Final checkpoint saved to: {final_model_path}")


if __name__ == "__main__":
    main()
