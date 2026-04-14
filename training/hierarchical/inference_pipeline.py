from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from hierarchical_predictor import HierarchicalPredictor
except ImportError:  # pragma: no cover - enables module execution from repo root
    from training.hierarchical.hierarchical_predictor import HierarchicalPredictor


PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_PLANT_MODEL_PATH = PIPELINE_ROOT / "artifacts" / "plant_classifier.pth"
DEFAULT_PLANT_CLASSES_PATH = PIPELINE_ROOT / "artifacts" / "plant_classifier_classes.json"
DEFAULT_DISEASE_MODELS_DIR = PIPELINE_ROOT / "disease_models"


class TwoStageInferencePipeline:
    def __init__(
        self,
        *,
        plant_model_path: Path = DEFAULT_PLANT_MODEL_PATH,
        plant_classes_path: Path = DEFAULT_PLANT_CLASSES_PATH,
        disease_models_dir: Path = DEFAULT_DISEASE_MODELS_DIR,
        plant_confidence_threshold: float = 0.55,
        disease_confidence_threshold: float = 0.60,
    ) -> None:
        self.predictor = HierarchicalPredictor(
            plant_model_path=plant_model_path,
            plant_classes_path=plant_classes_path,
            disease_models_dir=disease_models_dir,
            plant_confidence_threshold=plant_confidence_threshold,
            disease_confidence_threshold=disease_confidence_threshold,
            top_k=3,
        )

    def predict(self, image_bytes: bytes) -> dict[str, object]:
        return self.predictor.predict_minimal(image_bytes)

    def predict_file(self, image_path: Path) -> dict[str, object]:
        return self.predictor.predict_minimal_file(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage plant -> disease inference pipeline.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--plant-model", type=Path, default=DEFAULT_PLANT_MODEL_PATH)
    parser.add_argument("--plant-classes", type=Path, default=DEFAULT_PLANT_CLASSES_PATH)
    parser.add_argument("--disease-models-dir", type=Path, default=DEFAULT_DISEASE_MODELS_DIR)
    parser.add_argument("--plant-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--disease-confidence-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TwoStageInferencePipeline(
        plant_model_path=args.plant_model,
        plant_classes_path=args.plant_classes,
        disease_models_dir=args.disease_models_dir,
        plant_confidence_threshold=args.plant_confidence_threshold,
        disease_confidence_threshold=args.disease_confidence_threshold,
    )
    result = pipeline.predict_file(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
