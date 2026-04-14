from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import normalize_token, read_jsonl, write_jsonl


@dataclass
class PlantSample:
    image_path: str
    species: str
    plant_part: str
    health_status: str
    split: str
    source: str | None = None
    caption: str | None = None
    question: str | None = None
    answer: str | None = None
    bbox: list[float] | None = None
    mask_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "species": self.species,
            "plant_part": self.plant_part,
            "health_status": self.health_status,
            "split": self.split,
            "source": self.source,
            "caption": self.caption,
            "question": self.question,
            "answer": self.answer,
            "bbox": self.bbox,
            "mask_path": self.mask_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlantSample":
        return cls(
            image_path=str(payload["image_path"]),
            species=str(payload["species"]),
            plant_part=str(payload["plant_part"]),
            health_status=str(payload["health_status"]),
            split=normalize_token(str(payload["split"])),
            source=str(payload["source"]) if payload.get("source") else None,
            caption=str(payload["caption"]) if payload.get("caption") else None,
            question=str(payload["question"]) if payload.get("question") else None,
            answer=str(payload["answer"]) if payload.get("answer") else None,
            bbox=list(payload["bbox"]) if payload.get("bbox") else None,
            mask_path=str(payload["mask_path"]) if payload.get("mask_path") else None,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class LabelVocab:
    species_to_idx: dict[str, int]
    part_to_idx: dict[str, int]
    health_to_idx: dict[str, int]
    answer_to_idx: dict[str, int] = field(default_factory=dict)

    @property
    def idx_to_species(self) -> dict[int, str]:
        return {index: label for label, index in self.species_to_idx.items()}

    @property
    def idx_to_part(self) -> dict[int, str]:
        return {index: label for label, index in self.part_to_idx.items()}

    @property
    def idx_to_health(self) -> dict[int, str]:
        return {index: label for label, index in self.health_to_idx.items()}

    @property
    def idx_to_answer(self) -> dict[int, str]:
        return {index: label for label, index in self.answer_to_idx.items()}

    @classmethod
    def build(cls, samples: list[PlantSample]) -> "LabelVocab":
        species = sorted({sample.species for sample in samples})
        parts = sorted({sample.plant_part for sample in samples})
        health = sorted({sample.health_status for sample in samples})
        answers = sorted({sample.answer for sample in samples if sample.answer})
        return cls(
            species_to_idx={label: idx for idx, label in enumerate(species)},
            part_to_idx={label: idx for idx, label in enumerate(parts)},
            health_to_idx={label: idx for idx, label in enumerate(health)},
            answer_to_idx={label: idx for idx, label in enumerate(answers)},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "species_to_idx": self.species_to_idx,
            "part_to_idx": self.part_to_idx,
            "health_to_idx": self.health_to_idx,
            "answer_to_idx": self.answer_to_idx,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LabelVocab":
        return cls(
            species_to_idx={str(key): int(value) for key, value in dict(payload["species_to_idx"]).items()},
            part_to_idx={str(key): int(value) for key, value in dict(payload["part_to_idx"]).items()},
            health_to_idx={str(key): int(value) for key, value in dict(payload["health_to_idx"]).items()},
            answer_to_idx={str(key): int(value) for key, value in dict(payload.get("answer_to_idx") or {}).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LabelVocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Label vocab file must contain a JSON object.")
        return cls.from_dict(payload)


def load_manifest(path: str | Path) -> list[PlantSample]:
    rows = read_jsonl(Path(path))
    return [PlantSample.from_dict(row) for row in rows]


def save_manifest(path: str | Path, samples: list[PlantSample]) -> None:
    write_jsonl(Path(path), [sample.to_dict() for sample in samples])
