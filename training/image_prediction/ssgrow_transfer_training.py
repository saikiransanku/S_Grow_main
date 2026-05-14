from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# === START OF TENSORFLOW 2.10 SAVE BUG FIX ===
_original_get_config = layers.Normalization.get_config
_original_rescaling_get_config = layers.Rescaling.get_config


def _to_serializable_config_value(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable_config_value(item) for key, item in value.items()}
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

def _patched_get_config(self):
    config = _original_get_config(self)
    for key in ["mean", "variance"]:
        if key in config:
            config[key] = _to_serializable_config_value(config.get(key))
    return config

layers.Normalization.get_config = _patched_get_config


def _patched_rescaling_get_config(self):
    config = _original_rescaling_get_config(self)
    for key in ["scale", "offset"]:
        if key in config:
            config[key] = _to_serializable_config_value(config.get(key))
    return config


layers.Rescaling.get_config = _patched_rescaling_get_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEASONAL_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "image_prediction_seasonal_dataset"
DEFAULT_MODEL_DIR = REPO_ROOT / "assets" / "models"
DEFAULT_LABEL_DIR = REPO_ROOT / "assets" / "labels"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
FINAL_MODEL_NAME = "ssgrow_disease_model_v2.keras"

SEASON_ALIASES = {
    "all": "all_season",
    "all_season": "all_season",
    "kharif": "kharif",
    "rabi": "rabi",
}
BACKEND_MODEL_NAMES = {
    "all_season": "all_season_cnn.h5",
    "kharif": "kharif_cnn.h5",
    "rabi": "rabi_cnn.h5",
}
BACKEND_LABEL_NAMES = {
    "all_season": "all_season_classes.txt",
    "kharif": "kharif_classes.txt",
    "rabi": "rabi_classes.txt",
}


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    builder: Callable[..., tf.keras.Model]
    preprocess_input: Callable[[np.ndarray], np.ndarray]
    fine_tune_fraction: float


@dataclass(frozen=True)
class TrainingConfig:
    season: str
    dataset_root: Path
    output_dir: Path
    backbone: str
    image_size: int
    batch_size: int
    head_epochs: int
    fine_tune_epochs: int
    head_learning_rate: float
    fine_tune_learning_rate: float
    dense_units: int
    dropout_rate: float
    l2_weight: float
    patience: int
    seed: int
    export_backend_artifacts: bool
    backend_model_path: Optional[Path]
    backend_label_path: Optional[Path]


BACKBONES: Dict[str, BackboneSpec] = {
    "efficientnetb0": BackboneSpec(
        name="efficientnetb0",
        builder=tf.keras.applications.EfficientNetB0,
        preprocess_input=tf.keras.applications.efficientnet.preprocess_input,
        fine_tune_fraction=0.35,
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        builder=tf.keras.applications.ResNet50,
        preprocess_input=tf.keras.applications.resnet50.preprocess_input,
        fine_tune_fraction=0.30,
    ),
}


def normalize_season_name(season: str) -> str:
    normalized = str(season or "").strip().lower()
    if normalized not in SEASON_ALIASES:
        valid = ", ".join(sorted(SEASON_ALIASES))
        raise ValueError(f"Unsupported season '{season}'. Expected one of: {valid}")
    return SEASON_ALIASES[normalized]


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            continue


def resolve_dataset_root(season: str, dataset_root: Optional[Path]) -> Path:
    candidate = Path(dataset_root) if dataset_root else DEFAULT_SEASONAL_DATASET_ROOT / season
    candidate = candidate.resolve()
    required_dirs = [candidate / split for split in ("train", "val")]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset is missing required folders: {missing_text}")
    return candidate


def resolve_output_dir(season: str, output_dir: Optional[Path]) -> Path:
    resolved = Path(output_dir) if output_dir else DEFAULT_RUNS_DIR / season
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def resolve_backend_paths(
    season: str,
    export_backend_artifacts: bool,
    backend_model_path: Optional[Path],
    backend_label_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if not export_backend_artifacts:
        return None, None

    model_path = Path(backend_model_path) if backend_model_path else DEFAULT_MODEL_DIR / BACKEND_MODEL_NAMES[season]
    label_path = Path(backend_label_path) if backend_label_path else DEFAULT_LABEL_DIR / BACKEND_LABEL_NAMES[season]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path.resolve(), label_path.resolve()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    season = normalize_season_name(args.season)
    dataset_root = resolve_dataset_root(season, args.dataset_root)
    output_dir = resolve_output_dir(season, args.output_dir)
    backend_model_path, backend_label_path = resolve_backend_paths(
        season=season,
        export_backend_artifacts=args.export_backend_artifacts,
        backend_model_path=args.backend_model_path,
        backend_label_path=args.backend_label_path,
    )

    return TrainingConfig(
        season=season,
        dataset_root=dataset_root,
        output_dir=output_dir,
        backbone=args.backbone.lower(),
        image_size=args.image_size,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        head_learning_rate=args.head_learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        l2_weight=args.l2_weight,
        patience=args.patience,
        seed=args.seed,
        export_backend_artifacts=args.export_backend_artifacts,
        backend_model_path=backend_model_path,
        backend_label_path=backend_label_path,
    )


def run_multitask_training(args: argparse.Namespace) -> Path:
    try:
        from training.agri_ai.orchestration.seasonal_multitask import (
            TF_TO_MULTITASK_BACKBONE,
            run_multitask_training_for_season,
        )
    except ImportError:  # pragma: no cover
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from training.agri_ai.orchestration.seasonal_multitask import (
            TF_TO_MULTITASK_BACKBONE,
            run_multitask_training_for_season,
        )

    season = normalize_season_name(args.season)
    output_dir = resolve_output_dir(season, args.output_dir)
    backbone = TF_TO_MULTITASK_BACKBONE.get(args.backbone.lower(), "efficientnet_v2_s")
    total_epochs = max(1, int(args.head_epochs) + int(args.fine_tune_epochs))
    freeze_backbone_epochs = max(0, int(args.head_epochs))
    warmup_epochs = min(max(1, int(args.head_epochs)), total_epochs)
    return run_multitask_training_for_season(
        season=season,
        workspace=output_dir / "multitask",
        backbone=backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        freeze_backbone_epochs=freeze_backbone_epochs,
        lr=args.head_learning_rate,
        weight_decay=args.l2_weight,
        patience=args.patience,
        dropout=args.dropout_rate,
        seed=args.seed,
    )


def create_generators(
    *,
    config: TrainingConfig,
    preprocess_input: Callable[[np.ndarray], np.ndarray],
) -> tuple[DirectoryIterator, DirectoryIterator, Optional[DirectoryIterator]]:
    
    # 1. Base augmentation (Standard intensity for 'all_season')
    aug_params = {
        "preprocessing_function": preprocess_input,
        "rotation_range": 30,
        "width_shift_range": 0.10,
        "height_shift_range": 0.10,
        "horizontal_flip": True,
        "fill_mode": "nearest",
        "zoom_range": 0.35,
        "brightness_range": (0.75, 1.25),
        "channel_shift_range": 10.0,
        "shear_range": 0.10,
    }

    # 2. Dynamically adjust intensity based on the specific season
    if config.season == "kharif":
        # Monsoon: Unpredictable overcast lighting, very lush green backgrounds.
        # We need high channel shifts to stop the model from relying on "greenness"
        # and heavier zoom to crop out background weeds.
        aug_params.update({
            "brightness_range": (0.50, 1.50), # Much wider range for dark clouds vs sudden sun
            "channel_shift_range": 30.0,      # Aggressive color shifting
            "zoom_range": 0.50,               # Force the model closer to the leaf
        })
    elif config.season == "rabi":
        # Winter/Dry: Harsher direct sunlight, dry/brown soil backgrounds.
        # We need to simulate harsh shadows and different viewing angles.
        aug_params.update({
            "brightness_range": (0.80, 1.20), # Tighter, brighter range
            "shear_range": 0.25,              # Simulate slanted angles from dry, drooping leaves
            "channel_shift_range": 20.0,
            "zoom_range": 0.45,
        })

    # 3. Initialize the generator with the dynamic parameters
    augmentation = ImageDataGenerator(**aug_params)
    evaluation = ImageDataGenerator(preprocessing_function=preprocess_input)

    common_flow_kwargs = {
        "target_size": (config.image_size, config.image_size),
        "class_mode": "categorical",
        "color_mode": "rgb",
    }

    train_generator = augmentation.flow_from_directory(
        directory=str(config.dataset_root / "train"),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        **common_flow_kwargs,
    )
    class_names = list(train_generator.class_indices.keys())

    validation_generator = evaluation.flow_from_directory(
        directory=str(config.dataset_root / "val"),
        batch_size=config.batch_size,
        shuffle=False,
        classes=class_names,
        **common_flow_kwargs,
    )

    test_dir = config.dataset_root / "test"
    test_generator: Optional[DirectoryIterator] = None
    if test_dir.exists():
        test_generator = evaluation.flow_from_directory(
            directory=str(test_dir),
            batch_size=config.batch_size,
            shuffle=False,
            classes=class_names,
            **common_flow_kwargs,
        )

    return train_generator, validation_generator, test_generator

def compute_class_weights(generator: DirectoryIterator) -> Dict[int, float]:
    num_classes = len(generator.class_indices)
    class_counts = np.bincount(generator.classes, minlength=num_classes)
    total = float(np.sum(class_counts))
    weights: Dict[int, float] = {}
    for class_index, class_count in enumerate(class_counts):
        if class_count == 0:
            weights[class_index] = 1.0
            continue
        # Ensure we cast the final calculation to a pure Python float
        calc = total / (num_classes * float(class_count))
        weights[class_index] = float(calc)
    return weights


def build_transfer_model(
    *,
    config: TrainingConfig,
    backbone_spec: BackboneSpec,
    num_classes: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    input_shape = (config.image_size, config.image_size, 3)
    base_model = backbone_spec.builder(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="leaf_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(config.dropout_rate, name="head_dropout")(x)
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(config.l2_weight),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_dense_batch_norm")(x)
    x = layers.Dropout(max(config.dropout_rate - 0.10, 0.20), name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"ssgrow_{backbone_spec.name}_{config.season}",
    )
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0      ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


def create_callbacks(config: TrainingConfig, checkpoint_path: Path) -> list[callbacks.Callback]:
    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.30,
            patience=max(2, config.patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        # callbacks.CSVLogger(str(config.output_dir / "training_log.csv"), append=False),
    ]


def unfreeze_top_layers(base_model: tf.keras.Model, backbone_spec: BackboneSpec) -> int:
    base_model.trainable = True
    train_from = int(len(base_model.layers) * (1.0 - backbone_spec.fine_tune_fraction))
    for index, layer in enumerate(base_model.layers):
        should_train = index >= train_from and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = should_train
    return train_from


def merge_histories(histories):
    merged = {}
    for history in histories:
        for metric_name, values in history.history.items():
            merged.setdefault(metric_name, []).extend([float(v) for v in values])
    return merged


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    if not history:
        return

    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not list(epochs):
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(epochs, history.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("loss", []), label="Train Loss")
    axes[1].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    *,
    matrix: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    split_name: str,
) -> None:
    csv_path = output_dir / f"{split_name}_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + [int(value) for value in row])

    figure, axis = plt.subplots(figsize=(max(10, len(class_names) * 0.6), max(8, len(class_names) * 0.45)))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(f"{split_name.title()} Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=7)
    axis.set_yticklabels(class_names, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate_generator(
    *,
    model: tf.keras.Model,
    generator: DirectoryIterator,
    split_name: str,
) -> tuple[dict[str, float], np.ndarray]:
    generator.reset()
    metrics = model.evaluate(generator, verbose=1, return_dict=True)
    metrics = {k: float(v) for k, v in metrics.items()}

    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = generator.classes[: len(predicted_labels)]
    matrix = tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=len(generator.class_indices),
    ).numpy()

    results = {f"{split_name}_{metric_name}": float(metric_value) for metric_name, metric_value in metrics.items()}
    return results, matrix


def save_class_names(class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

def save_training_summary(
    *,
    config: TrainingConfig,
    history: dict[str, list[float]],
    class_weights: Dict[int, float],
    class_names: list[str],
    metrics: dict[str, float],
    checkpoint_path: Path,
    final_model_path: Path,
) -> None:

    def json_safe(obj):
        if hasattr(obj, "numpy"):
            obj = obj.numpy()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    summary = {
        "config": {
            **asdict(config),
            "dataset_root": str(config.dataset_root),
            "output_dir": str(config.output_dir),
            "backend_model_path": str(config.backend_model_path) if config.backend_model_path else None,
            "backend_label_path": str(config.backend_label_path) if config.backend_label_path else None,
        },
        "class_weights": {str(key): float(value) for key, value in class_weights.items()},
        "class_names": class_names,
        "metrics": metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "final_model_path": str(final_model_path),
    }

    (config.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_safe),
        encoding="utf-8",
    )

def train_and_evaluate(config: TrainingConfig) -> Path:
    if config.backbone not in BACKBONES:
        valid = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unsupported backbone '{config.backbone}'. Expected one of: {valid}")

    configure_runtime(config.seed)
    backbone_spec = BACKBONES[config.backbone]

    train_generator, validation_generator, test_generator = create_generators(
        config=config,
        preprocess_input=backbone_spec.preprocess_input,
    )
    class_names = list(train_generator.class_indices.keys())
    class_weights = compute_class_weights(train_generator)

    model, base_model = build_transfer_model(
        config=config,
        backbone_spec=backbone_spec,
        num_classes=len(class_names),
    )
    checkpoint_path = config.output_dir / "best_checkpoint.keras"

    compile_model(model, config.head_learning_rate)
    model.summary()

    stage_histories: list[tf.keras.callbacks.History] = []
    stage_callbacks = create_callbacks(config, checkpoint_path)
    stage_histories.append(
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.head_epochs,
            class_weight=class_weights,
            callbacks=stage_callbacks,
            verbose=1,
        )
    )

    unfreeze_top_layers(base_model, backbone_spec)
    compile_model(model, config.fine_tune_learning_rate)
    stage_histories.append(
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.head_epochs + config.fine_tune_epochs,
            initial_epoch=config.head_epochs,
            class_weight=class_weights,
            callbacks=create_callbacks(config, checkpoint_path),
            verbose=1,
        )
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    history = merge_histories(stage_histories)
    plot_training_history(history, config.output_dir)

    validation_metrics, validation_matrix = evaluate_generator(
        model=best_model,
        generator=validation_generator,
        split_name="val",
    )
    save_confusion_matrix(
        matrix=validation_matrix,
        class_names=class_names,
        output_dir=config.output_dir,
        split_name="val",
    )

    metrics = dict(validation_metrics)
    if test_generator is not None:
        test_metrics, test_matrix = evaluate_generator(
            model=best_model,
            generator=test_generator,
            split_name="test",
        )
        metrics.update(test_metrics)
        save_confusion_matrix(
            matrix=test_matrix,
            class_names=class_names,
            output_dir=config.output_dir,
            split_name="test",
        )

    final_model_path = config.output_dir / FINAL_MODEL_NAME
    best_model.save(final_model_path)

    if config.export_backend_artifacts and config.backend_model_path:
        best_model.save(config.backend_model_path)
    if config.export_backend_artifacts and config.backend_label_path:
        save_class_names(class_names, config.backend_label_path)

    save_class_names(class_names, config.output_dir / "class_names.txt")
    save_training_summary(
        config=config,
        history=history,
        class_weights=class_weights,
        class_names=class_names,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        final_model_path=final_model_path,
    )

    return final_model_path


def build_parser(default_season: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a robust SSGrow crop disease model with transfer learning and strong augmentation.",
    )
    parser.add_argument(
        "--season",
        default=default_season or "all_season",
        help="Dataset season to train. Supported: kharif, rabi, all_season.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional direct path to a dataset folder that contains train/val/test subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where training artifacts should be written.",
    )
    parser.add_argument(
        "--backbone",
        default="efficientnetb0",
        choices=sorted(BACKBONES.keys()),
        help="Pretrained backbone to use for transfer learning.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-6)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.35)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-multitask",
        action="store_true",
        help="Run the merged PyTorch multitask training pipeline instead of the TensorFlow pipeline.",
    )
    parser.add_argument(
        "--export-backend-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export backend-compatible .h5 model and class-name labels to assets/.",
    )
    parser.add_argument("--backend-model-path", type=Path, default=None)
    parser.add_argument("--backend-label-path", type=Path, default=None)
    return parser


def main(default_season: Optional[str] = None) -> None:
    parser = build_parser(default_season=default_season)
    args = parser.parse_args()
    if args.use_multitask:
        final_model_path = run_multitask_training(args)
        print(f"Training finished. Final checkpoint saved to: {final_model_path}")
        return
    config = build_training_config(args)
    final_model_path = train_and_evaluate(config)
    print(f"Training finished. Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
