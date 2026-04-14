from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...] = (1,)) -> dict[int, float]:
    max_k = min(max(ks), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))

    results: dict[int, float] = {}
    for k in ks:
        limit = min(k, logits.size(1))
        correct_k = correct[:limit].reshape(-1).float().sum(0)
        results[k] = float(correct_k.item() / max(1, labels.numel()))
    return results


def topk_membership(logits: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    top_indices = logits.topk(min(k, logits.size(1)), dim=1).indices
    return (top_indices == labels.unsqueeze(1)).any(dim=1)


def confusion_matrix_from_predictions(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(true_labels, pred_labels):
        matrix[int(truth), int(pred)] += 1
    return matrix


def per_class_f1(confusion_matrix: np.ndarray, class_names: Sequence[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, class_name in enumerate(class_names):
        tp = float(confusion_matrix[index, index])
        fp = float(confusion_matrix[:, index].sum() - tp)
        fn = float(confusion_matrix[index, :].sum() - tp)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        rows.append(
            {
                "class_name": class_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(confusion_matrix[index, :].sum()),
            }
        )
    return rows


def macro_f1(per_class_rows: Sequence[dict[str, float | str]]) -> float:
    if not per_class_rows:
        return 0.0
    return float(sum(float(item["f1"]) for item in per_class_rows) / len(per_class_rows))


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_bins: int = 15,
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    ece = torch.tensor(0.0)

    for bin_index in range(num_bins):
        lower = bin_boundaries[bin_index]
        upper = bin_boundaries[bin_index + 1]
        mask = (confidences > lower) & (confidences <= upper if bin_index < num_bins - 1 else confidences <= upper)
        if not torch.any(mask):
            continue
        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].float().mean()
        ece += (mask.float().mean()) * torch.abs(avg_confidence - avg_accuracy)
    return float(ece.item())


def multiclass_brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probabilities.size(1)).float()
    return float(torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=1)).item())


def save_confusion_matrix_csv(path: str | Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + list(class_names))
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name] + row)
