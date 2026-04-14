from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTAPI_ENV_PATH = REPO_ROOT / "ai_backend" / ".env"
DEFAULT_FASTAPI_SQLITE_PATH = REPO_ROOT / "ai_backend" / "fastapi_predictions.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_FASTAPI_SQLITE_PATH.as_posix()}"
DEFAULT_STORAGE_DIR = REPO_ROOT / "ai_backend" / "storage" / "uploads"
DEFAULT_PESTICIDES_CSV_PATH = REPO_ROOT / "data" / "outputs" / "project_data" / "pesticides_data.csv"


@dataclass(frozen=True)
class Settings:
    app_name: str
    database_url: str
    storage_dir: Path
    max_files_per_request: int

    model_path: Path | None
    classes_path: Path | None
    model_arch: str
    model_version: str
    image_size: int
    min_confidence: float

    llm_api_url: str
    llm_model: str
    llm_timeout_seconds: float

    pesticides_csv_path: Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_fastapi_env_file() -> None:
    env_file = Path(os.getenv("FASTAPI_ENV_FILE", str(DEFAULT_FASTAPI_ENV_PATH))).expanduser()
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_wrapping_quotes(raw_value.strip())
        os.environ.setdefault(key, value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_fastapi_env_file()

    raw_model_path = os.getenv("MODEL_PATH", "").strip()
    model_path = Path(raw_model_path) if raw_model_path else None

    raw_classes_path = os.getenv("CLASS_NAMES_PATH", "").strip()
    classes_path = Path(raw_classes_path) if raw_classes_path else None

    return Settings(
        app_name=os.getenv("FASTAPI_APP_NAME", "Sgrow Agricultural AI"),
        database_url=os.getenv("FASTAPI_DATABASE_URL", DEFAULT_DATABASE_URL),
        storage_dir=Path(os.getenv("IMAGE_STORAGE_DIR", str(DEFAULT_STORAGE_DIR))),
        max_files_per_request=max(1, int(os.getenv("MAX_FILES_PER_REQUEST", "10"))),
        model_path=model_path,
        classes_path=classes_path,
        model_arch=os.getenv("MODEL_ARCH", "efficientnet_b0"),
        model_version=os.getenv("MODEL_VERSION", "SSGrow-CNN-v2"),
        image_size=max(64, int(os.getenv("MODEL_IMAGE_SIZE", "224"))),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.60")),
        llm_api_url=os.getenv("LOCAL_LLM_API_URL", "http://localhost:11434/api/generate"),
        llm_model=os.getenv("LOCAL_LLM_MODEL", "llama3.1"),
        llm_timeout_seconds=float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "20")),
        pesticides_csv_path=Path(os.getenv("PESTICIDES_CSV_PATH", str(DEFAULT_PESTICIDES_CSV_PATH))),
    )
