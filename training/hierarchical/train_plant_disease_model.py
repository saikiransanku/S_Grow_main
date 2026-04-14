from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_CLASSES = {"unknown", "unknown_disease"}
DEFAULT_SPLIT_NAMES = ("train", "val", "test")
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "datasets" / "disease_dataset"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "disease_models"


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logging.info(
            "Using CUDA device: %s | torch=%s | cuda_build=%s",
            torch.cuda.get_device_name(0),
            torch.__version__,
            torch.version.cuda,
        )
        return device

    logging.warning(
        "CUDA is not available. Training will run on CPU | torch=%s | cuda_build=%s",
        torch.__version__,
        torch.version.cuda,
    )
    return torch.device("cpu")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


@dataclass(frozen=True)
class DatasetSplits:
    train_samples: List[Sample]
    val_samples: List[Sample]
    test_samples: List[Sample]
    idx_to_class: Dict[int, str]
    source_layout: str


@dataclass(frozen=True)
class DiseaseModelArtifacts:
    plant_name: str
    model_path: Path
    classes_path: Path
    summary_path: Path
    class_names: Tuple[str, ...]
    best_val_acc: float
    test_acc: float | None
    image_size: int


class LeafDiseaseDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: transforms.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label


def detect_dataset_splits(data_dir: Path) -> List[str]:
    return [split for split in DEFAULT_SPLIT_NAMES if (data_dir / split).is_dir()]


def iter_class_dirs(class_root: Path) -> List[Path]:
    if not class_root.exists():
        return []
    return sorted([entry for entry in class_root.iterdir() if entry.is_dir()], key=lambda p: p.name.lower())


def discover_plants(disease_root: Path) -> List[str]:
    split_names = detect_dataset_splits(disease_root)
    plant_names = set()
    roots = [disease_root / split for split in split_names] if split_names else [disease_root]

    for root in roots:
        for plant_dir in iter_class_dirs(root):
            plant_name = normalize_token(plant_dir.name)
            if plant_name and plant_name not in SKIP_CLASSES:
                plant_names.add(plant_name)

    if not plant_names:
        raise RuntimeError(f"No plant folders found under: {disease_root}")

    return sorted(plant_names)


def discover_disease_classes(plant_roots: Sequence[Path]) -> List[str]:
    class_names = {
        normalize_token(class_dir.name)
        for plant_root in plant_roots
        for class_dir in iter_class_dirs(plant_root)
        if normalize_token(class_dir.name) not in SKIP_CLASSES
    }
    if not class_names:
        raise RuntimeError("No disease classes found for the requested plant.")
    return sorted(class_names)


def collect_samples_from_plant_root(plant_root: Path, class_to_idx: Dict[str, int]) -> List[Sample]:
    samples: List[Sample] = []
    for class_dir in iter_class_dirs(plant_root):
        disease_name = normalize_token(class_dir.name)
        if disease_name in SKIP_CLASSES or disease_name not in class_to_idx:
            continue

        label_idx = class_to_idx[disease_name]
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(path=image_path, label=label_idx))
    return samples


def collect_flat_samples(plant_dir: Path) -> Tuple[List[Sample], Dict[int, str]]:
    class_names = discover_disease_classes([plant_dir])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    samples = collect_samples_from_plant_root(plant_dir, class_to_idx)
    if not samples:
        raise RuntimeError(f"No image files found under: {plant_dir}")
    return samples, idx_to_class


def stratified_split(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    by_label: Dict[int, List[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_samples.extend(shuffled[:val_size])
        train_samples.extend(shuffled[val_size:])

    if not train_samples:
        raise RuntimeError("Training split is empty. Lower --val-ratio.")
    if not val_samples:
        raise RuntimeError("Validation split is empty. Add more data or lower --val-ratio.")

    return train_samples, val_samples


def resolve_plant_dataset_splits(disease_root: Path, plant_name: str, val_ratio: float, seed: int) -> DatasetSplits:
    plant_key = normalize_token(plant_name)
    split_names = detect_dataset_splits(disease_root)

    if not split_names:
        plant_dir = disease_root / plant_key
        if not plant_dir.exists():
            raise FileNotFoundError(f"Plant disease dataset not found: {plant_dir}")
        samples, idx_to_class = collect_flat_samples(plant_dir)
        train_samples, val_samples = stratified_split(samples, val_ratio=val_ratio, seed=seed)
        return DatasetSplits(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=[],
            idx_to_class=idx_to_class,
            source_layout="flat",
        )

    plant_roots = [disease_root / split / plant_key for split in split_names if (disease_root / split / plant_key).exists()]
    if not plant_roots:
        raise FileNotFoundError(f"Plant disease dataset not found for '{plant_key}' under: {disease_root}")

    class_names = discover_disease_classes(plant_roots)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    train_samples = (
        collect_samples_from_plant_root(disease_root / "train" / plant_key, class_to_idx)
        if "train" in split_names and (disease_root / "train" / plant_key).exists()
        else []
    )
    if not train_samples:
        raise RuntimeError(f"Training split is missing or empty for plant '{plant_key}'.")

    val_samples = (
        collect_samples_from_plant_root(disease_root / "val" / plant_key, class_to_idx)
        if "val" in split_names and (disease_root / "val" / plant_key).exists()
        else []
    )
    test_samples = (
        collect_samples_from_plant_root(disease_root / "test" / plant_key, class_to_idx)
        if "test" in split_names and (disease_root / "test" / plant_key).exists()
        else []
    )

    if not val_samples:
        logging.warning("[%s] Validation split not found. Falling back to stratified split from training data.", plant_key)
        train_samples, val_samples = stratified_split(train_samples, val_ratio=val_ratio, seed=seed)

    return DatasetSplits(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        idx_to_class=idx_to_class,
        source_layout="split",
    )


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(0.65, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.06),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(512, num_classes),
    )

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((preds == labels).sum().item())

    return total_loss / max(1, total_items), total_correct / max(1, total_items)


def build_loader(
    samples: Sequence[Sample],
    transform: transforms.Compose,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = LeafDiseaseDataset(samples, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def save_model_artifacts(
    *,
    model: nn.Module,
    plant_name: str,
    idx_to_class: Dict[int, str],
    image_size: int,
    output_dir: Path,
    best_val_acc: float,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    model_path = output_dir / f"{plant_name}_disease_model.pth"
    classes_path = output_dir / f"{plant_name}_disease_classes.json"

    checkpoint = {
        "arch": "efficientnet_b3",
        "image_size": image_size,
        "class_names": class_names,
        "plant_name": plant_name,
        "best_val_acc": best_val_acc,
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, model_path)
    classes_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    return model_path, classes_path


def load_checkpoint_model(model_path: Path, device: torch.device) -> Tuple[nn.Module, List[str], int]:
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = [normalize_token(item) for item in checkpoint["class_names"] if normalize_token(item)]
    image_size = int(checkpoint.get("image_size") or 300)
    model = create_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, class_names, image_size


def save_training_summary(
    *,
    output_dir: Path,
    plant_name: str,
    data_root: Path,
    source_layout: str,
    class_names: Sequence[str],
    image_size: int,
    best_val_acc: float,
    test_metrics: Dict[str, float] | None,
    train_count: int,
    val_count: int,
    test_count: int,
) -> Path:
    summary = {
        "plant_name": plant_name,
        "data_root": str(data_root),
        "source_layout": source_layout,
        "image_size": image_size,
        "class_names": list(class_names),
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "metrics": {
            "best_val_acc": best_val_acc,
            **(test_metrics or {}),
        },
    }

    summary_path = output_dir / f"{plant_name}_disease_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def train_disease_model(
    plant_name: str,
    *,
    disease_root: Path = DEFAULT_DATASET_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = 25,
    batch_size: int = 20,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    image_size: int = 300,
    freeze_epochs: int = 3,
    head_lr: float = 1e-3,
    finetune_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
) -> DiseaseModelArtifacts:
    plant_key = normalize_token(plant_name)
    if not disease_root.exists():
        raise FileNotFoundError(f"Disease dataset root not found: {disease_root}")

    seed_everything(seed)
    dataset_splits = resolve_plant_dataset_splits(disease_root, plant_key, val_ratio=val_ratio, seed=seed)

    train_transform, eval_transform = build_transforms(image_size)
    train_loader = build_loader(
        dataset_splits.train_samples,
        train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_loader(
        dataset_splits.val_samples,
        eval_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = (
        build_loader(
            dataset_splits.test_samples,
            eval_transform,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if dataset_splits.test_samples
        else None
    )

    device = resolve_device()
    model = create_model(num_classes=len(dataset_splits.idx_to_class), pretrained=pretrained).to(device)
    set_backbone_trainable(model, trainable=False)

    criterion = nn.CrossEntropyLoss()
    optimizer: Adam = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr,
        weight_decay=weight_decay,
    )

    best_val_acc = -1.0
    model_output_dir = output_root / plant_key
    model_path = model_output_dir / f"{plant_key}_disease_model.pth"
    classes_path = model_output_dir / f"{plant_key}_disease_classes.json"

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = Adam(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        logging.info(
            "[%s] epoch %s/%s | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            plant_key,
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path, classes_path = save_model_artifacts(
                model=model,
                plant_name=plant_key,
                idx_to_class=dataset_splits.idx_to_class,
                image_size=image_size,
                output_dir=model_output_dir,
                best_val_acc=best_val_acc,
            )

    test_metrics: Dict[str, float] | None = None
    if test_loader is not None and model_path.exists():
        best_model, _, _ = load_checkpoint_model(model_path, device)
        test_loss, test_acc = run_epoch(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        test_metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        logging.info("[%s] test metrics | loss=%.4f acc=%.4f", plant_key, test_loss, test_acc)

    class_names = tuple(dataset_splits.idx_to_class[idx] for idx in sorted(dataset_splits.idx_to_class.keys()))
    summary_path = save_training_summary(
        output_dir=model_output_dir,
        plant_name=plant_key,
        data_root=disease_root,
        source_layout=dataset_splits.source_layout,
        class_names=class_names,
        image_size=image_size,
        best_val_acc=best_val_acc,
        test_metrics=test_metrics,
        train_count=len(dataset_splits.train_samples),
        val_count=len(dataset_splits.val_samples),
        test_count=len(dataset_splits.test_samples),
    )

    return DiseaseModelArtifacts(
        plant_name=plant_key,
        model_path=model_path,
        classes_path=classes_path,
        summary_path=summary_path,
        class_names=class_names,
        best_val_acc=best_val_acc,
        test_acc=test_metrics["test_acc"] if test_metrics else None,
        image_size=image_size,
    )


def train_all_disease_models(
    *,
    disease_root: Path = DEFAULT_DATASET_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    plants: Sequence[str] | None = None,
    epochs: int = 25,
    batch_size: int = 20,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    image_size: int = 300,
    freeze_epochs: int = 3,
    head_lr: float = 1e-3,
    finetune_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
) -> Dict[str, DiseaseModelArtifacts]:
    selected_plants = [normalize_token(plant) for plant in plants] if plants else discover_plants(disease_root)
    artifacts: Dict[str, DiseaseModelArtifacts] = {}

    for plant_name in selected_plants:
        artifacts[plant_name] = train_disease_model(
            plant_name,
            disease_root=disease_root,
            output_root=output_root,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            val_ratio=val_ratio,
            image_size=image_size,
            freeze_epochs=freeze_epochs,
            head_lr=head_lr,
            finetune_lr=finetune_lr,
            weight_decay=weight_decay,
            seed=seed,
            pretrained=pretrained,
        )

    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plant-specific disease model(s) with EfficientNet-B3.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--plant-name")
    selection.add_argument("--all-plants", action="store_true")
    parser.add_argument("--disease-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    if args.all_plants:
        artifacts = train_all_disease_models(
            disease_root=args.disease_root,
            output_root=args.output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            image_size=args.image_size,
            freeze_epochs=args.freeze_epochs,
            head_lr=args.head_lr,
            finetune_lr=args.finetune_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            pretrained=not args.no_pretrained,
        )
        logging.info("Trained %s crop-specific disease model(s).", len(artifacts))
        return

    artifacts = train_disease_model(
        args.plant_name,
        disease_root=args.disease_root,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        freeze_epochs=args.freeze_epochs,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )
    logging.info("Best model saved to %s", artifacts.model_path)
    logging.info("Training summary written to %s", artifacts.summary_path)


if __name__ == "__main__":
    main()
