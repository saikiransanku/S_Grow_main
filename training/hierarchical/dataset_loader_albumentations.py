from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


# === REPLACE START: your old transform definitions ===
# Replace your old torchvision/PIL augmentation pipeline with this.
def build_transforms(image_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            # Forces random zoom/crop so the model focuses on leaf regions, not static backgrounds.
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.55, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            # Adds geometric variability so orientation/position does not dominate learning.
            A.ShiftScaleRotate(
                shift_limit=0.12,
                scale_limit=0.20,
                rotate_limit=35,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            # Doubles perceived samples via mirroring.
            A.HorizontalFlip(p=0.5),
            # Breaks lighting shortcut learning from acquisition conditions.
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.30,
                p=0.7,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Validation/test: deterministic preprocessing only.
    eval_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, eval_transform
# === REPLACE END: your old transform definitions ===


# === REPLACE START: your old Dataset class ===
# Replace your existing dataset class with this Albumentations-powered version.
class AlbumentationsImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        transform: A.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]

        # Standard image loading (OpenCV -> RGB).
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations pipeline.
        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        return image_tensor, int(label)
# === REPLACE END: your old Dataset class ===


# === REPLACE START: your old DataLoader wiring ===
# Replace your old DataLoader creation block with this helper.
def build_dataloaders(
    train_samples: Sequence[Tuple[Path, int]],
    val_samples: Sequence[Tuple[Path, int]],
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_tfms, eval_tfms = build_transforms(image_size=image_size)

    train_ds = AlbumentationsImageDataset(train_samples, transform=train_tfms)
    val_ds = AlbumentationsImageDataset(val_samples, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
# === REPLACE END: your old DataLoader wiring ===
