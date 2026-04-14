from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and restart training from the seasonal plant dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets/image_prediction_seasonal_dataset"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("training/plant_multitask/runs/seasonal_restart"),
    )
    parser.add_argument("--backbone", default="mobilenet_v3_large")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    processed_dir = args.workspace / "processed"
    run_dir = args.workspace / "model"

    if not args.skip_prepare:
        prepare_cmd = [
            sys.executable,
            "-m",
            "training.plant_multitask.prepare_dataset",
            "--data-root",
            str(args.dataset_root),
            "--output-dir",
            str(processed_dir),
            "--input-layout",
            "seasonal_classification",
            "--compute-stats",
            "--min-image-size",
            "96",
            "--min-blur-score",
            "10",
        ]
        run_command(prepare_cmd)

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "training.plant_multitask.train",
            "--manifest-path",
            str(processed_dir / "manifest.jsonl"),
            "--stats-path",
            str(processed_dir / "dataset_stats.json"),
            "--output-dir",
            str(run_dir),
            "--backbone",
            str(args.backbone),
            "--image-size",
            str(args.image_size),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--epochs",
            str(args.epochs),
            "--warmup-epochs",
            "3",
            "--freeze-backbone-epochs",
            "2",
            "--species-loss-type",
            "focal",
            "--grad-accumulation-steps",
            "2",
            "--ema-decay",
            "0.9995",
        ]
        if args.device:
            train_cmd.extend(["--device", str(args.device)])
        run_command(train_cmd)


if __name__ == "__main__":
    main()
