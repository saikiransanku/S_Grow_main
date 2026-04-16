from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import json
import cv2
import numpy as np


# ============================================================
# SGrow Robust Preprocessing Pipeline
# - Reads images
# - Converts to RGB
# - Resizes to target size
# - Enhances contrast with CLAHE
# - Optionally suppresses background
# - Scores image quality
# - Routes good images to seasonal folder
# - Routes bad images to manual_review folder
# - Writes a JSONL manifest for traceability
# ============================================================


VALID_SEASONS = {"kharif", "rabi", "all_season"}


@dataclass
class PreprocessConfig:
    image_size: int = 512

    # Enhancement
    use_clahe: bool = True
    use_background_suppression: bool = True
    use_white_balance: bool = False
    use_denoise: bool = False

    # Background suppression strength
    background_dim_strength: float = 0.20

    # CLAHE parameters
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8

    # Quality gates
    min_blur_variance: float = 60.0
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    min_contrast: float = 18.0
    min_saturation: float = 12.0

    # Saving
    output_extension: str = ".jpg"
    jpeg_quality: int = 95

    # Keep class folder structure if present
    preserve_class_structure: bool = True

    # Useful for tracing
    save_manifest: bool = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_season(season: str) -> str:
    season = season.strip().lower()
    if season not in VALID_SEASONS:
        raise ValueError(f"Invalid season '{season}'. Use one of: {sorted(VALID_SEASONS)}")
    return season


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_image(image_rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    Optional white balance using the gray-world assumption.
    Useful when phone white-balance is off.
    """
    img = image_rgb.astype(np.float32)
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])

    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = mean_gray / (mean_r + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_b = mean_gray / (mean_b + 1e-6)

    img[:, :, 0] *= scale_r
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_b

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB color space.
    Improves local contrast without turning the image grayscale.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced


def suppress_background(
    image_rgb: np.ndarray,
    season: str,
    dim_strength: float = 0.20,
) -> np.ndarray:
    """
    Dims non-plant pixels using HSV masks.
    Keeps plant tissue visible while reducing soil/sky/hand/tray noise.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Green plant tissue
    mask_green = cv2.inRange(
        hsv,
        np.array([20, 25, 25]),
        np.array([95, 255, 255]),
    )

    # Brown/yellow/dry tissue
    mask_brown = cv2.inRange(
        hsv,
        np.array([5, 20, 25]),
        np.array([35, 255, 255]),
    )

    # Dark necrotic areas
    mask_dark = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 80, 90]),
    )

    plant_mask = cv2.bitwise_or(mask_green, mask_brown)
    plant_mask = cv2.bitwise_or(plant_mask, mask_dark)

    # Slightly more aggressive for kharif because backgrounds are often lush and noisy
    if season == "kharif":
        kernel_size = 17
        iterations = 2
    elif season == "rabi":
        kernel_size = 13
        iterations = 1
    else:
        kernel_size = 15
        iterations = 2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    plant_mask = cv2.dilate(plant_mask, kernel, iterations=iterations)

    result = image_rgb.astype(np.float32).copy()
    background = plant_mask == 0
    result[background] *= dim_strength

    return np.clip(result, 0, 255).astype(np.uint8)


def denoise_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Optional denoise step for phone-camera grain.
    Bilateral keeps edges better than aggressive blur.
    """
    return cv2.bilateralFilter(image_rgb, d=7, sigmaColor=40, sigmaSpace=40)


def compute_quality_metrics(image_rgb: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))

    # Optional extra signal: edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    return {
        "blur_variance": blur_variance,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "edge_density": edge_density,
    }


def is_good_image(metrics: Dict[str, float], cfg: PreprocessConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if metrics["blur_variance"] < cfg.min_blur_variance:
        reasons.append(f"blur<{cfg.min_blur_variance}")
    if metrics["brightness"] < cfg.min_brightness:
        reasons.append(f"dark<{cfg.min_brightness}")
    if metrics["brightness"] > cfg.max_brightness:
        reasons.append(f"bright>{cfg.max_brightness}")
    if metrics["contrast"] < cfg.min_contrast:
        reasons.append(f"low_contrast<{cfg.min_contrast}")
    if metrics["saturation"] < cfg.min_saturation:
        reasons.append(f"low_saturation<{cfg.min_saturation}")

    return len(reasons) == 0, reasons


def preprocess_image(
    image_bgr: np.ndarray,
    season: str,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Dict[str, float], bool, List[str]]:
    """
    Full deterministic preprocessing pipeline.
    Returns:
        processed_rgb, metrics, is_good, reasons
    """
    season = validate_season(season)

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image received")

    # Convert to RGB
    image_rgb = to_rgb(image_bgr)

    # Optional white balance
    if cfg.use_white_balance:
        image_rgb = gray_world_white_balance(image_rgb)

    # Resize first so all later steps work at a stable resolution
    image_rgb = resize_image(image_rgb, cfg.image_size)

    # Optional CLAHE
    if cfg.use_clahe:
        image_rgb = apply_clahe(
            image_rgb,
            clip_limit=cfg.clahe_clip_limit,
            tile_grid=cfg.clahe_tile_grid,
        )

    # Optional denoise
    if cfg.use_denoise:
        image_rgb = denoise_image(image_rgb)

    # Optional background suppression
    if cfg.use_background_suppression:
        image_rgb = suppress_background(
            image_rgb,
            season=season,
            dim_strength=cfg.background_dim_strength,
        )

    # Quality metrics after enhancement
    metrics = compute_quality_metrics(image_rgb)
    good, reasons = is_good_image(metrics, cfg)

    return image_rgb, metrics, good, reasons


def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    season: str,
    good: bool,
    cfg: PreprocessConfig,
) -> Path:
    """
    Good images:
        output_root/season/<class_or_relative_path>/image.jpg

    Bad images:
        output_root/manual_review/season/<class_or_relative_path>/image.jpg
    """
    input_path = input_path.resolve()
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    season = validate_season(season)

    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        # If input_root is not a parent, fall back to filename only
        rel = Path(input_path.name)

    if cfg.preserve_class_structure and len(rel.parts) > 1:
        relative_subpath = Path(*rel.parts[:-1])
        filename = rel.stem + "_preprocessed" + cfg.output_extension
    else:
        relative_subpath = Path()
        filename = input_path.stem + "_preprocessed" + cfg.output_extension

    if good:
        base_dir = output_root / season
    else:
        base_dir = output_root / "manual_review" / season

    return base_dir / relative_subpath / filename


def save_rgb_image(image_rgb: np.ndarray, output_path: Path, jpeg_quality: int = 95) -> None:
    ensure_dir(output_path.parent)

    ext = output_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), params)
    else:
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def append_manifest_row(manifest_path: Path, row: Dict) -> None:
    ensure_dir(manifest_path.parent)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def route_and_save_image(
    input_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    season: str,
    cfg: Optional[PreprocessConfig] = None,
) -> Dict:
    """
    Process one image and route it to:
      - seasonal folder if good
      - manual_review folder if not good
    """
    cfg = cfg or PreprocessConfig()
    season = validate_season(season)

    input_path = Path(input_path)
    input_root = Path(input_root)
    output_root = Path(output_root)

    image_bgr = read_image_bgr(input_path)
    processed_rgb, metrics, good, reasons = preprocess_image(
        image_bgr=image_bgr,
        season=season,
        cfg=cfg,
    )

    output_path = build_output_path(
        input_path=input_path,
        input_root=input_root,
        output_root=output_root,
        season=season,
        good=good,
        cfg=cfg,
    )

    save_rgb_image(processed_rgb, output_path, jpeg_quality=cfg.jpeg_quality)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "season": season,
        "routed_to": "seasonal" if good else "manual_review",
        "quality_good": good,
        "reasons": reasons,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def process_dataset(
    input_root: str | Path,
    output_root: str | Path,
    season: str,
    cfg: Optional[PreprocessConfig] = None,
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> Dict[str, int]:
    """
    Walk through a dataset folder, preprocess every image, and save outputs.
    Keeps folder structure if preserve_class_structure=True.

    Example:
        input_root/
            corn/
                a.jpg
            sugarcane/
                b.jpg

        output_root/
            kharif/
                corn/
                    a_preprocessed.jpg
                sugarcane/
                    b_preprocessed.jpg
            manual_review/
                kharif/
                    ...
    """
    cfg = cfg or PreprocessConfig()
    season = validate_season(season)

    input_root = Path(input_root)
    output_root = Path(output_root)

    manifest_path = output_root / "_manifests" / f"{season}_manifest.jsonl"

    good_count = 0
    review_count = 0
    skipped_count = 0

    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in valid_extensions:
            continue

        try:
            row = route_and_save_image(
                input_path=file_path,
                input_root=input_root,
                output_root=output_root,
                season=season,
                cfg=cfg,
            )

            if row["quality_good"]:
                good_count += 1
            else:
                review_count += 1

            if cfg.save_manifest:
                append_manifest_row(manifest_path, row)

        except Exception as e:
            skipped_count += 1
            print(f"[SKIP] {file_path} -> {e}")

    summary = {
        "good_count": good_count,
        "review_count": review_count,
        "skipped_count": skipped_count,
    }

    if cfg.save_manifest:
        append_manifest_row(
            manifest_path,
            {
                "summary": summary,
                "season": season,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )

    return summary


# ============================================================
# Season-specific tuning helper
# ============================================================

def make_season_config(season: str) -> PreprocessConfig:
    season = validate_season(season)

    if season == "kharif":
        return PreprocessConfig(
            image_size=512,
            use_clahe=True,
            use_background_suppression=True,
            use_white_balance=False,
            use_denoise=False,
            background_dim_strength=0.15,
            clahe_clip_limit=3.0,
            min_blur_variance=60.0,
            min_brightness=35.0,
            max_brightness=230.0,
            min_contrast=18.0,
            min_saturation=12.0,
            preserve_class_structure=True,
            save_manifest=True,
        )

    if season == "rabi":
        return PreprocessConfig(
            image_size=512,
            use_clahe=True,
            use_background_suppression=True,
            use_white_balance=False,
            use_denoise=False,
            background_dim_strength=0.22,
            clahe_clip_limit=2.5,
            min_blur_variance=60.0,
            min_brightness=40.0,
            max_brightness=220.0,
            min_contrast=18.0,
            min_saturation=10.0,
            preserve_class_structure=True,
            save_manifest=True,
        )

    # all_season
    return PreprocessConfig(
        image_size=512,
        use_clahe=True,
        use_background_suppression=True,
        use_white_balance=False,
        use_denoise=False,
        background_dim_strength=0.20,
        clahe_clip_limit=2.7,
        min_blur_variance=60.0,
        min_brightness=38.0,
        max_brightness=225.0,
        min_contrast=18.0,
        min_saturation=11.0,
        preserve_class_structure=True,
        save_manifest=True,
    )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    SEASON = "kharif"
    INPUT_ROOT = REPO_ROOT / "data" / "datasets" / "image_prediction_seasonal_dataset" / SEASON
    OUTPUT_ROOT = REPO_ROOT / "data" / "datasets" / "Pre_train_data"

    cfg = make_season_config(SEASON)

    summary = process_dataset(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        season=SEASON,
        cfg=cfg,
    )

    print("\nDone.")
    print(summary)
