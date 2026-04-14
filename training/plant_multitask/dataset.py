from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .schemas import LabelVocab, PlantSample
from .text import SimpleTokenizer


QUESTION_TEMPLATES = (
    "What condition is affecting this {plant_part}?",
    "Is this {species} sample healthy or diseased?",
    "Which disease or pest is visible on this {plant_part}?",
)


def build_caption(sample: PlantSample) -> str:
    if sample.caption:
        return sample.caption

    season = sample.metadata.get("season") if isinstance(sample.metadata, dict) else None
    condition = sample.answer or sample.health_status
    season_text = f" during {str(season).replace('_', ' ')} season" if season else ""
    return (
        f"{sample.plant_part} image of {sample.species}"
        f"{season_text} showing {str(condition).replace('_', ' ')}"
    )


def build_question(sample: PlantSample) -> str | None:
    if sample.question:
        return sample.question
    if not sample.answer:
        return None
    template_index = hash((sample.species, sample.plant_part, sample.answer)) % len(QUESTION_TEMPLATES)
    return QUESTION_TEMPLATES[template_index].format(
        species=sample.species,
        plant_part=sample.plant_part.replace("_", " "),
    )


class PlantVisionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PlantSample],
        vocab: LabelVocab,
        *,
        transform: Any,
        tokenizer: SimpleTokenizer | None = None,
        max_text_length: int = 48,
    ) -> None:
        self.samples = list(samples)
        self.vocab = vocab
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_text(self, text: str | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.tokenizer is None or not text:
            return None
        token_ids, attention_mask = self.tokenizer.encode(text, max_length=self.max_text_length)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(sample.image_path)
        if image is None:
            raise ValueError(f"Could not read image: {sample.image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        caption_tokens = self._encode_text(build_caption(sample))
        question_tokens = self._encode_text(build_question(sample))

        return {
            "image": transformed["image"],
            "species_id": torch.tensor(self.vocab.species_to_idx[sample.species], dtype=torch.long),
            "part_id": torch.tensor(self.vocab.part_to_idx[sample.plant_part], dtype=torch.long),
            "health_id": torch.tensor(self.vocab.health_to_idx[sample.health_status], dtype=torch.long),
            "caption_ids": caption_tokens[0] if caption_tokens else None,
            "caption_mask": caption_tokens[1] if caption_tokens else None,
            "question_ids": question_tokens[0] if question_tokens else None,
            "question_mask": question_tokens[1] if question_tokens else None,
            "answer_id": torch.tensor(self.vocab.answer_to_idx.get(sample.answer, -100), dtype=torch.long),
            "image_path": sample.image_path,
            "metadata": sample.metadata,
        }


def plant_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images": torch.stack([item["image"] for item in batch]),
        "species_id": torch.stack([item["species_id"] for item in batch]),
        "part_id": torch.stack([item["part_id"] for item in batch]),
        "health_id": torch.stack([item["health_id"] for item in batch]),
        "answer_id": torch.stack([item["answer_id"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }

    for token_key, mask_key in (("caption_ids", "caption_mask"), ("question_ids", "question_mask")):
        token_values = [item[token_key] for item in batch]
        mask_values = [item[mask_key] for item in batch]
        if all(value is None for value in token_values):
            payload[token_key] = None
            payload[mask_key] = None
            continue

        template_tokens = next(value for value in token_values if value is not None)
        template_mask = next(value for value in mask_values if value is not None)
        payload[token_key] = torch.stack(
            [value if value is not None else torch.zeros_like(template_tokens) for value in token_values]
        )
        payload[mask_key] = torch.stack(
            [value if value is not None else torch.zeros_like(template_mask) for value in mask_values]
        )
    return payload


def build_weighted_sampler(
    samples: Sequence[PlantSample],
    *,
    species_power: float = 1.0,
    part_power: float = 0.2,
    health_power: float = 0.25,
    answer_power: float = 0.2,
    season_power: float = 0.1,
) -> WeightedRandomSampler:
    species_counts = Counter(sample.species for sample in samples)
    part_counts = Counter(sample.plant_part for sample in samples)
    health_counts = Counter(sample.health_status for sample in samples)
    answer_counts = Counter(sample.answer for sample in samples if sample.answer)
    season_counts = Counter(str(sample.metadata.get("season", "unknown")) for sample in samples)

    weights: list[float] = []
    for sample in samples:
        weight = 0.0
        weight += 0.65 * (species_counts[sample.species] ** (-species_power))
        weight += 0.10 * (part_counts[sample.plant_part] ** (-part_power))
        weight += 0.10 * (health_counts[sample.health_status] ** (-health_power))
        if sample.answer:
            weight += 0.10 * (answer_counts[sample.answer] ** (-answer_power))
        season_key = str(sample.metadata.get("season", "unknown"))
        weight += 0.05 * (season_counts[season_key] ** (-season_power))
        weights.append(float(weight))

    tensor_weights = torch.tensor(weights, dtype=torch.float32)
    tensor_weights = tensor_weights / tensor_weights.mean().clamp_min(1e-8)
    return WeightedRandomSampler(weights=tensor_weights.tolist(), num_samples=len(samples), replacement=True)


def build_class_weights(samples: Sequence[PlantSample], vocab: LabelVocab, beta: float = 0.999) -> dict[str, torch.Tensor]:
    def effective_num_weights(counter: Counter[str], labels: list[str]) -> torch.Tensor:
        values = []
        for label in labels:
            count = max(1, counter.get(label, 1))
            effective_num = 1.0 - (beta ** count)
            values.append((1.0 - beta) / max(effective_num, 1e-8))
        tensor = torch.tensor(values, dtype=torch.float32)
        return tensor / tensor.mean().clamp_min(1e-8)

    return {
        "species": effective_num_weights(
            Counter(sample.species for sample in samples),
            sorted(vocab.species_to_idx, key=vocab.species_to_idx.get),
        ),
        "part": effective_num_weights(
            Counter(sample.plant_part for sample in samples),
            sorted(vocab.part_to_idx, key=vocab.part_to_idx.get),
        ),
        "health": effective_num_weights(
            Counter(sample.health_status for sample in samples),
            sorted(vocab.health_to_idx, key=vocab.health_to_idx.get),
        ),
        "answer": effective_num_weights(
            Counter(sample.answer for sample in samples if sample.answer),
            sorted(vocab.answer_to_idx, key=vocab.answer_to_idx.get),
        )
        if vocab.answer_to_idx
        else torch.empty(0, dtype=torch.float32),
    }
