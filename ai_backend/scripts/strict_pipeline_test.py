import os
import time
import uuid
from pathlib import Path
import sys

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client


def pick_first_image(folder: Path) -> Path:
    for path in sorted(folder.iterdir()):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found in {folder}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    django_root = repo_root / "ai_backend"
    if str(django_root) not in sys.path:
        sys.path.insert(0, str(django_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_backend.settings")
    import django  # lazy import after env setup

    django.setup()

    project_root = repo_root
    test_root = (
        project_root
        / "data"
        / "datasets"
        / "image_prediction_seasonal_dataset"
        / "all_season"
        / "test"
    )
    wait_seconds = 1.5

    classes = [
        "banana_healthy",
        "banana_cordana",
        "banana_pestalotiopsis",
        "banana_sigatoka",
        "anthracnose_mango",
        "powdery_mildew_mango",
        "healthy_mango",
        "black_rot_grape",
        "healthy_tomato",
        "redrot_sugarcane",
    ]

    images = [pick_first_image(test_root / cls) for cls in classes]
    client = Client()

    print("STRICT PIPELINE TEST (10 sequential uploads)")
    print("=" * 60)
    sugarcane_false_positive = False

    for idx, image_path in enumerate(images, start=1):
        request_id = str(uuid.uuid4())
        with open(image_path, "rb") as f:
            upload = SimpleUploadedFile(
                name=image_path.name,
                content=f.read(),
                content_type="image/jpeg",
            )
            response = client.post(
                "/api/ai/predict",
                {
                    "season": "auto",
                    "request_id": request_id,
                    "file": upload,
                },
            )

        payload = response.json()
        response_id = payload.get("request_id", "")
        disease = payload.get("disease_type", "")
        confidence = float(payload.get("confidence_score", 0))
        id_match = request_id == response_id
        expected_class = image_path.parent.name
        is_banana = expected_class.startswith("banana_")
        banana_to_sugarcane_high_conf = (
            is_banana
            and "sugarcane" in disease.lower()
            and confidence >= 99.0
        )
        if banana_to_sugarcane_high_conf:
            sugarcane_false_positive = True

        print(
            f"{idx:02d}. image={expected_class}/{image_path.name} | "
            f"id_match={id_match} | predicted={disease} | confidence={confidence:.2f}%"
        )
        time.sleep(wait_seconds)

    print("=" * 60)
    print(f"banana->sugarcane (>=99%) observed: {sugarcane_false_positive}")
    print("DONE")


if __name__ == "__main__":
    main()
