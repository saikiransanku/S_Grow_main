import os
import re
from pathlib import Path
from typing import Dict, List

from django.conf import settings

DEFAULT_AGRONOMY_RETRIEVAL_MAX_REFERENCES = 3


def resolve_agronomy_docs_dir() -> Path:
    project_root = settings.BASE_DIR.parent
    return Path(
        os.getenv(
            "AGRONOMY_DOCS_DIR",
            project_root / "data" / "datasets" / "language_dataset" / "raw_data",
        )
    ).expanduser()


def normalize_reference_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _split_reference_chunks(text: str, chunk_size: int = 800) -> List[str]:
    normalized_text = normalize_reference_text(text)
    if not normalized_text:
        return []

    paragraphs = re.split(r"(?<=[.?!])\s+|\n{2,}", normalized_text)
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        if len(current) + len(cleaned) + 1 <= chunk_size:
            current = f"{current} {cleaned}".strip()
            continue
        if current:
            chunks.append(current)
        current = cleaned

    if current:
        chunks.append(current)

    return chunks


def _read_reference_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return ""

        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""

    return ""


def load_agronomy_reference_index() -> List[Dict[str, str]]:
    root = resolve_agronomy_docs_dir()
    if not root.exists():
        return []

    references: List[Dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        raw_text = _read_reference_document(path)
        if not raw_text.strip():
            continue
        for chunk in _split_reference_chunks(raw_text):
            references.append(
                {
                    "title": path.stem.replace("_", " ").strip(),
                    "source": str(path),
                    "snippet": chunk,
                }
            )
    return references


def build_retrieval_tokens(*values: str) -> List[str]:
    tokens: List[str] = []
    for value in values:
        normalized = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower())).strip("_")
        if not normalized:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
        for token in normalized.split("_"):
            if len(token) >= 4 and token not in tokens:
                tokens.append(token)
    return tokens


def score_reference_snippet(snippet: str, tokens: List[str]) -> float:
    lowered = snippet.lower()
    score = 0.0
    for token in tokens:
        if token in lowered:
            score += 2.5 if "_" in token else 1.0
    return score
