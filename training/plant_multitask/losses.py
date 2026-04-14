from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import MixBatchMetadata


def contrastive_alignment_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
    return 0.5 * (
        F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
    )


def multiclass_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 1.5,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        labels,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


class PlantRecognitionLoss(nn.Module):
    def __init__(
        self,
        *,
        species_weights: torch.Tensor | None = None,
        part_weights: torch.Tensor | None = None,
        health_weights: torch.Tensor | None = None,
        answer_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        species_loss_type: str = "focal",
        part_loss_type: str = "cross_entropy",
        health_loss_type: str = "cross_entropy",
        focal_gamma: float = 1.5,
        species_loss_weight: float = 1.0,
        part_loss_weight: float = 0.35,
        health_loss_weight: float = 0.45,
        contrastive_loss_weight: float = 0.2,
        vqa_loss_weight: float = 0.35,
        use_uncertainty_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.species_weights = species_weights
        self.part_weights = part_weights
        self.health_weights = health_weights
        self.answer_weights = answer_weights
        self.label_smoothing = label_smoothing
        self.species_loss_type = species_loss_type
        self.part_loss_type = part_loss_type
        self.health_loss_type = health_loss_type
        self.focal_gamma = focal_gamma
        self.species_loss_weight = species_loss_weight
        self.part_loss_weight = part_loss_weight
        self.health_loss_weight = health_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.vqa_loss_weight = vqa_loss_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            self.task_log_vars = nn.ParameterDict(
                {
                    "species": nn.Parameter(torch.zeros(1)),
                    "part": nn.Parameter(torch.zeros(1)),
                    "health": nn.Parameter(torch.zeros(1)),
                    "contrastive": nn.Parameter(torch.zeros(1)),
                    "answer": nn.Parameter(torch.zeros(1)),
                }
            )
        else:
            self.task_log_vars = None

    def _base_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        loss_type: str,
    ) -> torch.Tensor:
        if loss_type == "focal":
            return multiclass_focal_loss(
                logits,
                labels,
                weight=weight,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        loss_type: str,
        mix: MixBatchMetadata | None,
    ) -> torch.Tensor:
        if mix is None:
            return self._base_loss(logits, labels, weight=weight, loss_type=loss_type)

        return mix.lambda_value * self._base_loss(
            logits,
            labels,
            weight=weight,
            loss_type=loss_type,
        ) + (1.0 - mix.lambda_value) * self._base_loss(
            logits,
            labels[mix.permutation],
            weight=weight,
            loss_type=loss_type,
        )

    def _apply_task_weight(self, loss: torch.Tensor, task_name: str, base_weight: float) -> torch.Tensor:
        if not self.use_uncertainty_weighting or self.task_log_vars is None:
            return base_weight * loss
        log_var = self.task_log_vars[task_name]
        precision = torch.exp(-log_var)
        return base_weight * (precision * loss + log_var)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        mix: MixBatchMetadata | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        species_loss = self._classification_loss(
            outputs["species_logits"],
            batch["species_id"],
            weight=self.species_weights,
            loss_type=self.species_loss_type,
            mix=mix,
        )
        part_loss = self._classification_loss(
            outputs["part_logits"],
            batch["part_id"],
            weight=self.part_weights,
            loss_type=self.part_loss_type,
            mix=mix,
        )
        health_loss = self._classification_loss(
            outputs["health_logits"],
            batch["health_id"],
            weight=self.health_weights,
            loss_type=self.health_loss_type,
            mix=mix,
        )

        total = (
            self._apply_task_weight(species_loss, "species", self.species_loss_weight)
            + self._apply_task_weight(part_loss, "part", self.part_loss_weight)
            + self._apply_task_weight(health_loss, "health", self.health_loss_weight)
        )
        metrics = {
            "species_loss": float(species_loss.detach().item()),
            "part_loss": float(part_loss.detach().item()),
            "health_loss": float(health_loss.detach().item()),
        }

        if mix is None and "caption_embedding" in outputs:
            contrastive = contrastive_alignment_loss(
                outputs["normalized_embedding"],
                outputs["caption_embedding"],
                outputs["logit_scale"],
            )
            total = total + self._apply_task_weight(contrastive, "contrastive", self.contrastive_loss_weight)
            metrics["contrastive_loss"] = float(contrastive.detach().item())

        if "answer_logits" in outputs:
            valid_mask = batch["answer_id"] >= 0
            if torch.any(valid_mask):
                answer_loss = F.cross_entropy(
                    outputs["answer_logits"][valid_mask],
                    batch["answer_id"][valid_mask],
                    weight=self.answer_weights if self.answer_weights is not None and self.answer_weights.numel() else None,
                )
                total = total + self._apply_task_weight(answer_loss, "answer", self.vqa_loss_weight)
                metrics["answer_loss"] = float(answer_loss.detach().item())

        if self.task_log_vars is not None:
            for key, value in self.task_log_vars.items():
                metrics[f"{key}_task_weight"] = round(float(torch.exp(-value.detach()).item()), 4)

        metrics["total_loss"] = float(total.detach().item())
        return total, metrics
