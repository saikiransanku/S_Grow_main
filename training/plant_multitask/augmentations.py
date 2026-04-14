from __future__ import annotations

import random
from dataclasses import dataclass

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


@dataclass
class MixBatchMetadata:
    permutation: torch.Tensor
    lambda_value: float
    mode: str


def build_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.55, 1.0),
                ratio=(0.7, 1.35),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.Affine(
                scale=(0.9, 1.12),
                translate_percent=(-0.08, 0.08),
                rotate=(-30, 30),
                shear=(-10, 10),
                p=0.55,
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.08, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
                    A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ],
                p=0.75,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 60.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.28,
            ),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_lower=0.08, fog_coef_upper=0.2, alpha_coef=0.08, p=1.0),
                    A.RandomShadow(shadow_roi=(0.0, 0.4, 1.0, 1.0), p=1.0),
                    A.RandomRain(blur_value=3, brightness_coefficient=0.95, p=1.0),
                ],
                p=0.12,
            ),
            A.OneOf(
                [
                    A.Downscale(scale_min=0.55, scale_max=0.85, p=1.0),
                    A.ImageCompression(quality_lower=35, quality_upper=90, p=1.0),
                ],
                p=0.22,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_transform, eval_transform


def mixup_batch(images: torch.Tensor, *, alpha: float) -> tuple[torch.Tensor, MixBatchMetadata]:
    permutation = torch.randperm(images.size(0), device=images.device)
    lam = float(np.random.beta(alpha, alpha))
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    return mixed_images, MixBatchMetadata(permutation=permutation, lambda_value=lam, mode="mixup")


def _rand_bbox(image_size: tuple[int, int], lambda_value: float) -> tuple[int, int, int, int]:
    height, width = image_size
    cut_ratio = np.sqrt(1.0 - lambda_value)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = np.random.randint(width)
    center_y = np.random.randint(height)
    x1 = int(np.clip(center_x - cut_width // 2, 0, width))
    y1 = int(np.clip(center_y - cut_height // 2, 0, height))
    x2 = int(np.clip(center_x + cut_width // 2, 0, width))
    y2 = int(np.clip(center_y + cut_height // 2, 0, height))
    return x1, y1, x2, y2


def cutmix_batch(images: torch.Tensor, *, alpha: float) -> tuple[torch.Tensor, MixBatchMetadata]:
    permutation = torch.randperm(images.size(0), device=images.device)
    lam = float(np.random.beta(alpha, alpha))
    _, _, height, width = images.shape
    x1, y1, x2, y2 = _rand_bbox((height, width), lam)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    adjusted_lam = 1.0 - (bbox_area / float(height * width))
    return mixed_images, MixBatchMetadata(permutation=permutation, lambda_value=adjusted_lam, mode="cutmix")


def apply_batch_mixing(
    images: torch.Tensor,
    *,
    probability: float,
    alpha: float,
) -> tuple[torch.Tensor, MixBatchMetadata | None]:
    if images.size(0) < 2 or random.random() > probability:
        return images, None
    if random.random() < 0.5:
        return mixup_batch(images, alpha=alpha)
    return cutmix_batch(images, alpha=alpha)
