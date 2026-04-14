"""
Local smoke test runner for the Django AI backend inference service.

Usage:
python app.py path/to/image.jpg --season all_season
"""

import argparse
import json
import sys
from pathlib import Path

import os

REPO_ROOT = Path(__file__).resolve().parent
DJANGO_ROOT = REPO_ROOT / "ai_backend"
if str(DJANGO_ROOT) not in sys.path:
    sys.path.insert(0, str(DJANGO_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_backend.settings")

# ASGI application export for deployment entrypoints.
# For local development, prefer:
#   .\venv\Scripts\python.exe ai_backend\manage.py runserver 0.0.0.0:8000
from ai_backend.asgi import application as app


def main():
    import django

    django.setup()

    from predictor.services import run_prediction

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument(
        "--season",
        default="all_season",
        choices=["all_season", "all", "auto", "kharif", "rabi"],
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    result = run_prediction(
        image_bytes=image_bytes,
        content_type="image/jpeg",
        requested_season=args.season,
    )
    print(
        json.dumps(
            {
                "disease_type": result.label,
                "crop_detected": result.crop_detected,
                "confidence_score": result.confidence,
                "prediction_score": round(result.confidence / 100.0, 4),
                "diagnosis_status": result.diagnosis_status,
                "override_applied": result.override_applied,
                "override_reason": result.override_reason,
                "model_label_before_override": result.model_label_before_override,
                "model_confidence_before_override": result.model_confidence_before_override,
                "heuristic_lesion_count": result.heuristic_lesion_count,
                "preprocessing_metrics": result.preprocessing_metrics,
                "recommended_pesticide": result.recommended_pesticide,
                "active_ingredient": result.active_ingredient,
                "usage_note": result.usage_note,
                "visual_analysis": result.visual_analysis,
                "leaf_visual_analysis": result.leaf_visual_analysis,
                "farmer_report": result.farmer_report,
                "farmer_action_plan_markdown": result.farmer_action_plan_markdown,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
