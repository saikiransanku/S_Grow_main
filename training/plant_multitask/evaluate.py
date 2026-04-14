from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .augmentations import build_transforms
    from .dataset import PlantVisionDataset, plant_collate_fn
    from .inference import load_checkpoint_bundle
    from .metrics import (
        confusion_matrix_from_predictions,
        expected_calibration_error,
        macro_f1,
        multiclass_brier_score,
        per_class_f1,
        save_confusion_matrix_csv,
        topk_accuracy,
        topk_membership,
    )
    from .schemas import load_manifest
    from .utils import choose_device
except ImportError:  # pragma: no cover
    from training.plant_multitask.augmentations import build_transforms
    from training.plant_multitask.dataset import PlantVisionDataset, plant_collate_fn
    from training.plant_multitask.inference import load_checkpoint_bundle
    from training.plant_multitask.metrics import (
        confusion_matrix_from_predictions,
        expected_calibration_error,
        macro_f1,
        multiclass_brier_score,
        per_class_f1,
        save_confusion_matrix_csv,
        topk_accuracy,
        topk_membership,
    )
    from training.plant_multitask.schemas import load_manifest
    from training.plant_multitask.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the multi-task plant recognition model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-failure-cases", type=int, default=100)
    parser.add_argument("--rare-class-threshold", type=int, default=20)
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def topk_rows(logits: torch.Tensor, labels: dict[int, str], k: int = 5) -> list[list[dict[str, float | str]]]:
    probabilities = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probabilities, k=min(k, probabilities.size(1)), dim=1)
    rows: list[list[dict[str, float | str]]] = []
    for row_values, row_indices in zip(values.tolist(), indices.tolist()):
        rows.append(
            [
                {"label": labels[int(index)], "confidence": round(float(value), 4)}
                for value, index in zip(row_values, row_indices)
            ]
        )
    return rows


def summarize_topk_breakdown(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, float | str | int]]:
    top1_correct = logits.argmax(dim=1).eq(labels)
    top5_correct = topk_membership(logits, labels, k=5)
    support = Counter(labels.tolist())
    top1_hits = Counter(labels[top1_correct].tolist())
    top5_hits = Counter(labels[top5_correct].tolist())

    rows = []
    for index, class_name in enumerate(class_names):
        class_support = support.get(index, 0)
        top1_error = 1.0 - (top1_hits.get(index, 0) / max(1, class_support))
        top5_error = 1.0 - (top5_hits.get(index, 0) / max(1, class_support))
        rows.append(
            {
                "class_name": class_name,
                "support": class_support,
                "top1_error": round(top1_error, 4),
                "top5_error": round(top5_error, 4),
            }
        )
    rows.sort(key=lambda item: (-int(item["top1_error"] * 10000), int(item["support"])))
    return rows


def find_similar_species_confusions(
    species_cm: Any,
    species_names: list[str],
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    *,
    top_n: int = 30,
) -> list[dict[str, float | str | int]]:
    prototype_sums: dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(embeddings.size(1)))
    prototype_counts: Counter[int] = Counter()
    for label, embedding in zip(species_labels.tolist(), embeddings):
        prototype_sums[int(label)] += embedding.cpu()
        prototype_counts[int(label)] += 1

    prototype_matrix = []
    for index in range(len(species_names)):
        if prototype_counts[index] == 0:
            prototype_matrix.append(torch.zeros(embeddings.size(1)))
        else:
            prototype_matrix.append(F.normalize(prototype_sums[index] / prototype_counts[index], dim=0))
    prototype_matrix = torch.stack(prototype_matrix)
    similarity = prototype_matrix @ prototype_matrix.t()

    rows = []
    for i in range(len(species_names)):
        for j in range(i + 1, len(species_names)):
            confusion_count = int(species_cm[i, j] + species_cm[j, i])
            if confusion_count == 0:
                continue
            rows.append(
                {
                    "species_a": species_names[i],
                    "species_b": species_names[j],
                    "mutual_confusions": confusion_count,
                    "prototype_cosine_similarity": round(float(similarity[i, j].item()), 4),
                }
            )
    rows.sort(key=lambda item: (int(item["mutual_confusions"]), float(item["prototype_cosine_similarity"])), reverse=True)
    return rows[:top_n]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    bundle = load_checkpoint_bundle(args.checkpoint, device=str(device))

    mean = tuple(bundle.runtime_metadata.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(bundle.runtime_metadata.get("std", (0.229, 0.224, 0.225)))
    image_size = int(bundle.runtime_metadata.get("image_size", 224))
    max_text_length = int(bundle.runtime_metadata.get("max_text_length", 48))

    _, eval_transform = build_transforms(image_size, mean, std)
    samples = [sample for sample in load_manifest(args.manifest_path) if sample.split == args.split]
    if not samples:
        raise ValueError(f"No samples found for split: {args.split}")

    dataset = PlantVisionDataset(
        samples,
        bundle.vocab,
        transform=eval_transform,
        tokenizer=bundle.tokenizer,
        max_text_length=max_text_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=plant_collate_fn,
    )

    model = bundle.model.to(device).eval()
    species_logits_all: list[torch.Tensor] = []
    part_logits_all: list[torch.Tensor] = []
    health_logits_all: list[torch.Tensor] = []
    answer_logits_all: list[torch.Tensor] = []
    species_labels_all: list[torch.Tensor] = []
    part_labels_all: list[torch.Tensor] = []
    health_labels_all: list[torch.Tensor] = []
    answer_labels_all: list[torch.Tensor] = []
    image_paths: list[str] = []
    embeddings_all: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            image_paths.extend(batch["image_path"])
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["images"],
                caption_ids=batch["caption_ids"],
                caption_mask=batch["caption_mask"],
                question_ids=batch["question_ids"],
                question_mask=batch["question_mask"],
            )
            species_logits_all.append(outputs["species_logits"].cpu())
            part_logits_all.append(outputs["part_logits"].cpu())
            health_logits_all.append(outputs["health_logits"].cpu())
            species_labels_all.append(batch["species_id"].cpu())
            part_labels_all.append(batch["part_id"].cpu())
            health_labels_all.append(batch["health_id"].cpu())
            embeddings_all.append(outputs["normalized_embedding"].cpu())

            if "answer_logits" in outputs:
                valid_mask = batch["answer_id"] >= 0
                if torch.any(valid_mask):
                    answer_logits_all.append(outputs["answer_logits"][valid_mask].cpu())
                    answer_labels_all.append(batch["answer_id"][valid_mask].cpu())

    species_logits = torch.cat(species_logits_all)
    part_logits = torch.cat(part_logits_all)
    health_logits = torch.cat(health_logits_all)
    species_labels = torch.cat(species_labels_all)
    part_labels = torch.cat(part_labels_all)
    health_labels = torch.cat(health_labels_all)
    embeddings = torch.cat(embeddings_all)

    species_prob = torch.softmax(species_logits, dim=1)
    part_prob = torch.softmax(part_logits, dim=1)
    health_prob = torch.softmax(health_logits, dim=1)
    species_pred = species_prob.argmax(dim=1)
    part_pred = part_prob.argmax(dim=1)
    health_pred = health_prob.argmax(dim=1)

    species_cm = confusion_matrix_from_predictions(species_labels.tolist(), species_pred.tolist(), len(bundle.vocab.species_to_idx))
    part_cm = confusion_matrix_from_predictions(part_labels.tolist(), part_pred.tolist(), len(bundle.vocab.part_to_idx))
    health_cm = confusion_matrix_from_predictions(health_labels.tolist(), health_pred.tolist(), len(bundle.vocab.health_to_idx))

    species_names = [bundle.vocab.idx_to_species[idx] for idx in range(len(bundle.vocab.species_to_idx))]
    part_names = [bundle.vocab.idx_to_part[idx] for idx in range(len(bundle.vocab.part_to_idx))]
    health_names = [bundle.vocab.idx_to_health[idx] for idx in range(len(bundle.vocab.health_to_idx))]

    species_rows = per_class_f1(species_cm, species_names)
    part_rows = per_class_f1(part_cm, part_names)
    health_rows = per_class_f1(health_cm, health_names)

    metrics: dict[str, Any] = {
        "species_top1": round(topk_accuracy(species_logits, species_labels, ks=(1,))[1], 4),
        "species_top5": round(topk_accuracy(species_logits, species_labels, ks=(5,))[5], 4),
        "species_macro_f1": round(macro_f1(species_rows), 4),
        "part_top1": round(topk_accuracy(part_logits, part_labels, ks=(1,))[1], 4),
        "part_macro_f1": round(macro_f1(part_rows), 4),
        "health_top1": round(topk_accuracy(health_logits, health_labels, ks=(1,))[1], 4),
        "health_macro_f1": round(macro_f1(health_rows), 4),
        "species_ece": round(expected_calibration_error(species_prob, species_labels), 4),
        "species_brier": round(multiclass_brier_score(species_prob, species_labels), 4),
        "part_ece": round(expected_calibration_error(part_prob, part_labels), 4),
        "health_ece": round(expected_calibration_error(health_prob, health_labels), 4),
    }

    if answer_logits_all:
        answer_logits = torch.cat(answer_logits_all)
        answer_labels = torch.cat(answer_labels_all)
        answer_prob = torch.softmax(answer_logits, dim=1)
        metrics["answer_top1"] = round(topk_accuracy(answer_logits, answer_labels, ks=(1,))[1], 4)
        metrics["answer_ece"] = round(expected_calibration_error(answer_prob, answer_labels), 4)

    rare_species = [row for row in species_rows if int(row["support"]) <= args.rare_class_threshold]
    metrics["rare_species_macro_f1"] = round(macro_f1(rare_species), 4) if rare_species else 0.0
    metrics["rare_species_count"] = len(rare_species)

    save_confusion_matrix_csv(args.output_dir / "species_confusion_matrix.csv", species_cm, species_names)
    save_confusion_matrix_csv(args.output_dir / "part_confusion_matrix.csv", part_cm, part_names)
    save_confusion_matrix_csv(args.output_dir / "health_confusion_matrix.csv", health_cm, health_names)

    species_top5 = topk_rows(species_logits, bundle.vocab.idx_to_species, k=5)
    part_top3 = topk_rows(part_logits, bundle.vocab.idx_to_part, k=3)
    health_top3 = topk_rows(health_logits, bundle.vocab.idx_to_health, k=3)
    species_conf = species_prob.max(dim=1).values

    failures = []
    for index, path in enumerate(image_paths):
        if int(species_pred[index]) == int(species_labels[index]) and int(health_pred[index]) == int(health_labels[index]):
            continue
        failures.append(
            {
                "image_path": path,
                "true_species": species_names[int(species_labels[index])],
                "pred_species": species_names[int(species_pred[index])],
                "species_confidence": round(float(species_conf[index]), 4),
                "true_part": part_names[int(part_labels[index])],
                "pred_part": part_names[int(part_pred[index])],
                "true_health": health_names[int(health_labels[index])],
                "pred_health": health_names[int(health_pred[index])],
                "species_top5": species_top5[index],
                "part_top3": part_top3[index],
                "health_top3": health_top3[index],
            }
        )
    failures.sort(key=lambda item: item["species_confidence"], reverse=True)
    failures = failures[: args.max_failure_cases]

    topk_breakdown = summarize_topk_breakdown(species_logits, species_labels, species_names)
    similar_confusions = find_similar_species_confusions(species_cm, species_names, embeddings, species_labels)

    (args.output_dir / "per_class_f1.json").write_text(
        json.dumps({"species": species_rows, "part": part_rows, "health": health_rows}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "failure_cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (args.output_dir / "topk_error_breakdown.json").write_text(json.dumps(topk_breakdown, indent=2), encoding="utf-8")
    (args.output_dir / "similar_species_confusions.json").write_text(json.dumps(similar_confusions, indent=2), encoding="utf-8")
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
