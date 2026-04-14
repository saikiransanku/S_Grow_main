from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from .augmentations import apply_batch_mixing, build_transforms
    from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
    from .dataset import PlantVisionDataset, build_caption, build_class_weights, build_weighted_sampler, plant_collate_fn
    from .inference import load_checkpoint_bundle
    from .losses import PlantRecognitionLoss
    from .metrics import confusion_matrix_from_predictions, macro_f1, per_class_f1, topk_accuracy
    from .model import PlantMultiTaskModel
    from .schemas import LabelVocab, load_manifest
    from .text import SimpleTokenizer
    from .utils import choose_device, seed_everything
except ImportError:  # pragma: no cover
    from training.plant_multitask.augmentations import apply_batch_mixing, build_transforms
    from training.plant_multitask.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
    from training.plant_multitask.dataset import PlantVisionDataset, build_caption, build_class_weights, build_weighted_sampler, plant_collate_fn
    from training.plant_multitask.inference import load_checkpoint_bundle
    from training.plant_multitask.losses import PlantRecognitionLoss
    from training.plant_multitask.metrics import confusion_matrix_from_predictions, macro_f1, per_class_f1, topk_accuracy
    from training.plant_multitask.model import PlantMultiTaskModel
    from training.plant_multitask.schemas import LabelVocab, load_manifest
    from training.plant_multitask.text import SimpleTokenizer
    from training.plant_multitask.utils import choose_device, seed_everything


BACKBONE_CHOICES = [
    "efficientnet_b4",
    "efficientnet_v2_s",
    "mobilenet_v3_large",
    "convnext_tiny",
    "vit_b_16",
]


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.model.state_dict()
            for key, value in ema_state.items():
                if not torch.is_floating_point(value):
                    value.copy_(model_state[key])
                    continue
                value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task plant recognition model.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default="efficientnet_v2_s")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--mix-probability", type=float, default=0.35)
    parser.add_argument("--mix-alpha", type=float, default=0.4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--task-hidden-dim", type=int, default=512)
    parser.add_argument("--cross-attention-layers", type=int, default=2)
    parser.add_argument("--text-width", type=int, default=256)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--species-loss-type", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--part-loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--health-loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    parser.add_argument("--disable-uncertainty-weighting", action="store_true")
    parser.add_argument("--sampler-species-power", type=float, default=1.0)
    parser.add_argument("--sampler-part-power", type=float, default=0.2)
    parser.add_argument("--sampler-health-power", type=float, default=0.25)
    parser.add_argument("--sampler-answer-power", type=float, default=0.2)
    parser.add_argument("--sampler-season-power", type=float, default=0.1)
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def summarize_predictions(
    species_logits: torch.Tensor,
    species_labels: torch.Tensor,
    part_logits: torch.Tensor,
    part_labels: torch.Tensor,
    health_logits: torch.Tensor,
    health_labels: torch.Tensor,
    vocab: LabelVocab,
    answer_logits: torch.Tensor | None = None,
    answer_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    species_topk = topk_accuracy(species_logits, species_labels, ks=(1, 5))
    part_top1 = topk_accuracy(part_logits, part_labels, ks=(1,))[1]
    health_top1 = topk_accuracy(health_logits, health_labels, ks=(1,))[1]

    species_rows = per_class_f1(
        confusion_matrix_from_predictions(
            species_labels.tolist(),
            species_logits.argmax(dim=1).tolist(),
            len(vocab.species_to_idx),
        ),
        [vocab.idx_to_species[idx] for idx in range(len(vocab.species_to_idx))],
    )
    part_rows = per_class_f1(
        confusion_matrix_from_predictions(part_labels.tolist(), part_logits.argmax(dim=1).tolist(), len(vocab.part_to_idx)),
        [vocab.idx_to_part[idx] for idx in range(len(vocab.part_to_idx))],
    )
    health_rows = per_class_f1(
        confusion_matrix_from_predictions(health_labels.tolist(), health_logits.argmax(dim=1).tolist(), len(vocab.health_to_idx)),
        [vocab.idx_to_health[idx] for idx in range(len(vocab.health_to_idx))],
    )

    metrics = {
        "species_top1": round(species_topk[1], 4),
        "species_top5": round(species_topk[5], 4),
        "species_macro_f1": round(macro_f1(species_rows), 4),
        "part_top1": round(part_top1, 4),
        "part_macro_f1": round(macro_f1(part_rows), 4),
        "health_top1": round(health_top1, 4),
        "health_macro_f1": round(macro_f1(health_rows), 4),
    }

    if answer_logits is not None and answer_labels is not None and answer_labels.numel() > 0:
        metrics["answer_top1"] = round(topk_accuracy(answer_logits, answer_labels, ks=(1,))[1], 4)

    metrics["selection_score"] = round(
        0.45 * metrics["species_top1"]
        + 0.25 * metrics["species_macro_f1"]
        + 0.15 * metrics["part_top1"]
        + 0.15 * metrics["health_top1"],
        4,
    )
    return metrics


def run_epoch(
    *,
    model: PlantMultiTaskModel,
    loader: DataLoader,
    criterion: PlantRecognitionLoss,
    optimizer: AdamW | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    mix_probability: float,
    mix_alpha: float,
    accumulation_steps: int,
    gradient_clip_norm: float,
    vocab: LabelVocab,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_steps = 0
    loss_terms: dict[str, list[float]] = {}
    species_logits_all: list[torch.Tensor] = []
    species_labels_all: list[torch.Tensor] = []
    part_logits_all: list[torch.Tensor] = []
    part_labels_all: list[torch.Tensor] = []
    health_logits_all: list[torch.Tensor] = []
    health_labels_all: list[torch.Tensor] = []
    answer_logits_all: list[torch.Tensor] = []
    answer_labels_all: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        images = batch["images"]
        mix = None
        if is_train:
            images, mix = apply_batch_mixing(images, probability=mix_probability, alpha=mix_alpha)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(
                images,
                caption_ids=batch["caption_ids"],
                caption_mask=batch["caption_mask"],
                question_ids=batch["question_ids"] if mix is None else None,
                question_mask=batch["question_mask"] if mix is None else None,
            )
            loss, breakdown = criterion(outputs, batch, mix=mix)

        total_loss += float(loss.detach().item())
        total_steps += 1
        for key, value in breakdown.items():
            loss_terms.setdefault(key, []).append(value)

        if is_train and optimizer is not None:
            scaled_loss = loss / max(1, accumulation_steps)
            scaler.scale(scaled_loss).backward()

            should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

        species_logits_all.append(outputs["species_logits"].detach().cpu())
        species_labels_all.append(batch["species_id"].detach().cpu())
        part_logits_all.append(outputs["part_logits"].detach().cpu())
        part_labels_all.append(batch["part_id"].detach().cpu())
        health_logits_all.append(outputs["health_logits"].detach().cpu())
        health_labels_all.append(batch["health_id"].detach().cpu())

        if "answer_logits" in outputs:
            valid_mask = batch["answer_id"] >= 0
            if torch.any(valid_mask):
                answer_logits_all.append(outputs["answer_logits"][valid_mask].detach().cpu())
                answer_labels_all.append(batch["answer_id"][valid_mask].detach().cpu())

    metrics = summarize_predictions(
        torch.cat(species_logits_all),
        torch.cat(species_labels_all),
        torch.cat(part_logits_all),
        torch.cat(part_labels_all),
        torch.cat(health_logits_all),
        torch.cat(health_labels_all),
        vocab,
        torch.cat(answer_logits_all) if answer_logits_all else None,
        torch.cat(answer_labels_all) if answer_labels_all else None,
    )
    metrics["loss"] = round(total_loss / max(1, total_steps), 4)
    for key, values in loss_terms.items():
        metrics[key] = round(sum(values) / max(1, len(values)), 4)
    return metrics


def build_experiment_config(args: argparse.Namespace, mean: tuple[float, float, float], std: tuple[float, float, float]) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            manifest_path=str(args.manifest_path),
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mean=mean,
            std=std,
            weighted_sampling=not args.disable_weighted_sampling,
            sampler_species_power=args.sampler_species_power,
            sampler_part_power=args.sampler_part_power,
            sampler_health_power=args.sampler_health_power,
            sampler_answer_power=args.sampler_answer_power,
            sampler_season_power=args.sampler_season_power,
        ),
        model=ModelConfig(
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            attention_heads=args.attention_heads,
            task_hidden_dim=args.task_hidden_dim,
            text_width=args.text_width,
            text_heads=args.text_heads,
            text_layers=args.text_layers,
            cross_attention_layers=args.cross_attention_layers,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            mix_probability=args.mix_probability,
            mix_alpha=args.mix_alpha,
            early_stopping_patience=args.patience,
            ema_decay=args.ema_decay,
            use_uncertainty_weighting=not args.disable_uncertainty_weighting,
            species_loss_type=args.species_loss_type,
            part_loss_type=args.part_loss_type,
            health_loss_type=args.health_loss_type,
            focal_gamma=args.focal_gamma,
        ),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = choose_device(args.device)

    all_records = load_manifest(args.manifest_path)
    train_records = [record for record in all_records if record.split == "train"]
    val_records = [record for record in all_records if record.split == "val"]
    test_records = [record for record in all_records if record.split == "test"]
    if not train_records or not val_records:
        raise ValueError("Manifest must contain both train and val samples.")

    stats_path = args.stats_path or args.manifest_path.parent / "dataset_stats.json"
    if stats_path.exists():
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        mean = tuple(float(x) for x in stats_payload["mean"])
        std = tuple(float(x) for x in stats_payload["std"])
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    experiment_config = build_experiment_config(args, mean, std)
    experiment_config.save(args.output_dir / "experiment_config.json")

    vocab = LabelVocab.build(all_records)
    vocab.save(args.output_dir / "label_vocab.json")

    text_corpus = [build_caption(record) for record in train_records]
    text_corpus.extend(record.question for record in train_records if record.question)
    text_corpus.extend(record.species for record in train_records)
    tokenizer = SimpleTokenizer.build(text_corpus) if text_corpus else None
    if tokenizer:
        tokenizer.save(args.output_dir / "tokenizer.json")

    train_transform, eval_transform = build_transforms(args.image_size, mean, std)
    train_dataset = PlantVisionDataset(
        train_records,
        vocab,
        transform=train_transform,
        tokenizer=tokenizer,
        max_text_length=experiment_config.data.max_text_length,
    )
    val_dataset = PlantVisionDataset(
        val_records,
        vocab,
        transform=eval_transform,
        tokenizer=tokenizer,
        max_text_length=experiment_config.data.max_text_length,
    )
    test_dataset = (
        PlantVisionDataset(
            test_records,
            vocab,
            transform=eval_transform,
            tokenizer=tokenizer,
            max_text_length=experiment_config.data.max_text_length,
        )
        if test_records
        else None
    )

    sampler = None
    if experiment_config.data.weighted_sampling:
        sampler = build_weighted_sampler(
            train_records,
            species_power=experiment_config.data.sampler_species_power,
            part_power=experiment_config.data.sampler_part_power,
            health_power=experiment_config.data.sampler_health_power,
            answer_power=experiment_config.data.sampler_answer_power,
            season_power=experiment_config.data.sampler_season_power,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=plant_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=plant_collate_fn,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=plant_collate_fn,
        )
        if test_dataset is not None
        else None
    )

    class_weights = build_class_weights(train_records, vocab)
    model = PlantMultiTaskModel(
        backbone_name=args.backbone,
        num_species=len(vocab.species_to_idx),
        num_parts=len(vocab.part_to_idx),
        num_health_states=len(vocab.health_to_idx),
        num_answers=len(vocab.answer_to_idx),
        pretrained=not args.no_pretrained,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        task_hidden_dim=args.task_hidden_dim,
        text_vocab_size=tokenizer.vocab_size if tokenizer else None,
        text_width=args.text_width,
        text_heads=args.text_heads,
        text_layers=args.text_layers,
        cross_attention_layers=args.cross_attention_layers,
        max_text_length=experiment_config.data.max_text_length,
    ).to(device)

    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone()

    criterion = PlantRecognitionLoss(
        species_weights=class_weights["species"].to(device),
        part_weights=class_weights["part"].to(device),
        health_weights=class_weights["health"].to(device),
        answer_weights=class_weights["answer"].to(device) if class_weights["answer"].numel() else None,
        label_smoothing=experiment_config.train.label_smoothing,
        species_loss_type=experiment_config.train.species_loss_type,
        part_loss_type=experiment_config.train.part_loss_type,
        health_loss_type=experiment_config.train.health_loss_type,
        focal_gamma=experiment_config.train.focal_gamma,
        species_loss_weight=experiment_config.train.species_loss_weight,
        part_loss_weight=experiment_config.train.part_loss_weight,
        health_loss_weight=experiment_config.train.health_loss_weight,
        contrastive_loss_weight=experiment_config.train.contrastive_loss_weight,
        vqa_loss_weight=experiment_config.train.vqa_loss_weight,
        use_uncertainty_weighting=experiment_config.train.use_uncertainty_weighting,
    ).to(device)

    optim_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accumulation_steps))
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    warmup_steps = optimizer_steps_per_epoch * args.warmup_epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=experiment_config.train.min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    history: list[dict[str, Any]] = []
    best_score = -1.0
    patience_counter = 0
    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            model.unfreeze_backbone()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            mix_probability=args.mix_probability,
            mix_alpha=args.mix_alpha,
            accumulation_steps=args.grad_accumulation_steps,
            gradient_clip_norm=experiment_config.train.gradient_clip_norm,
            vocab=vocab,
            ema=ema,
        )

        validation_model = ema.model if ema is not None else model
        val_metrics = run_epoch(
            model=validation_model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            device=device,
            mix_probability=0.0,
            mix_alpha=args.mix_alpha,
            accumulation_steps=1,
            gradient_clip_norm=experiment_config.train.gradient_clip_norm,
            vocab=vocab,
        )

        epoch_payload = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_payload)
        print(json.dumps(epoch_payload))

        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            patience_counter = 0
            checkpoint = {
                "model_state": copy.deepcopy(model.state_dict()),
                "ema_model_state": copy.deepcopy(ema.model.state_dict()) if ema is not None else None,
                "model_config": {
                    "backbone": args.backbone,
                    "embedding_dim": args.embedding_dim,
                    "dropout": args.dropout,
                    "attention_heads": args.attention_heads,
                    "task_hidden_dim": args.task_hidden_dim,
                    "text_width": args.text_width,
                    "text_heads": args.text_heads,
                    "text_layers": args.text_layers,
                    "cross_attention_layers": args.cross_attention_layers,
                    "max_text_length": experiment_config.data.max_text_length,
                },
                "label_vocab": vocab.to_dict(),
                "tokenizer": tokenizer.to_dict() if tokenizer else None,
                "runtime_metadata": {
                    "image_size": args.image_size,
                    "mean": mean,
                    "std": std,
                    "max_text_length": experiment_config.data.max_text_length,
                    "best_val_selection_score": best_score,
                    "backbone": args.backbone,
                },
                "history": history,
            }
            torch.save(checkpoint, best_checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    summary: dict[str, Any] = {
        "best_val_selection_score": best_score,
        "epochs_completed": len(history),
        "history": history,
    }

    if best_score < 0:
        raise RuntimeError("Training did not produce a checkpoint.")

    if test_loader is not None:
        bundle = load_checkpoint_bundle(best_checkpoint_path, device=str(device))
        best_model = bundle.model.to(device)
        test_metrics = run_epoch(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            device=device,
            mix_probability=0.0,
            mix_alpha=args.mix_alpha,
            accumulation_steps=1,
            gradient_clip_norm=experiment_config.train.gradient_clip_norm,
            vocab=vocab,
        )
        summary["test"] = test_metrics

    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
