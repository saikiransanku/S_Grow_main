from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    from .model import PlantMultiTaskModel
    from .schemas import LabelVocab
    from .text import SimpleTokenizer
    from .utils import choose_device
except ImportError:  # pragma: no cover
    from training.plant_multitask.model import PlantMultiTaskModel
    from training.plant_multitask.schemas import LabelVocab
    from training.plant_multitask.text import SimpleTokenizer
    from training.plant_multitask.utils import choose_device


SPECIES_PROMPT_TEMPLATES = (
    "a field photo of {species}",
    "a close-up image of a {species} leaf",
    "{species} plant showing natural variation",
)


@dataclass
class LoadedBundle:
    model: PlantMultiTaskModel
    vocab: LabelVocab
    tokenizer: SimpleTokenizer | None
    runtime_metadata: dict[str, Any]


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> LoadedBundle:
    resolved_path = Path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {resolved_path}")

    vocab = LabelVocab.from_dict(dict(checkpoint["label_vocab"]))
    tokenizer = SimpleTokenizer.from_dict(dict(checkpoint["tokenizer"])) if checkpoint.get("tokenizer") else None
    model_config = dict(checkpoint.get("model_config") or {})
    runtime_metadata = dict(checkpoint.get("runtime_metadata") or {})

    model = PlantMultiTaskModel(
        backbone_name=str(model_config.get("backbone", "efficientnet_v2_s")),
        num_species=len(vocab.species_to_idx),
        num_parts=len(vocab.part_to_idx),
        num_health_states=len(vocab.health_to_idx),
        num_answers=len(vocab.answer_to_idx),
        pretrained=False,
        embedding_dim=int(model_config.get("embedding_dim", 512)),
        dropout=float(model_config.get("dropout", 0.25)),
        attention_heads=int(model_config.get("attention_heads", 8)),
        task_hidden_dim=int(model_config.get("task_hidden_dim", 512)),
        text_vocab_size=tokenizer.vocab_size if tokenizer else None,
        text_width=int(model_config.get("text_width", 256)),
        text_heads=int(model_config.get("text_heads", 4)),
        text_layers=int(model_config.get("text_layers", 2)),
        cross_attention_layers=int(model_config.get("cross_attention_layers", 2)),
        max_text_length=int(runtime_metadata.get("max_text_length", model_config.get("max_text_length", 48))),
    )
    state_dict = checkpoint.get("ema_model_state") or checkpoint.get("model_state")
    model.load_state_dict(state_dict, strict=True)
    model.to(choose_device(device))
    model.eval()

    return LoadedBundle(model=model, vocab=vocab, tokenizer=tokenizer, runtime_metadata=runtime_metadata)


class PlantRecognizer:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        onnx_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.bundle = load_checkpoint_bundle(checkpoint_path, device=device)
        self.device = choose_device(device)
        self.model = self.bundle.model.to(self.device)
        self.vocab = self.bundle.vocab
        self.tokenizer = self.bundle.tokenizer
        self.runtime_metadata = self.bundle.runtime_metadata
        self.image_size = int(self.runtime_metadata.get("image_size", 224))
        self.mean = tuple(self.runtime_metadata.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(self.runtime_metadata.get("std", (0.229, 0.224, 0.225)))
        self.max_text_length = int(self.runtime_metadata.get("max_text_length", 48))

        self.onnx_session = None
        if onnx_path and ort is not None and Path(onnx_path).exists():
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        self.species_prompt_bank = self._build_species_prompt_bank()

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid image input.") from exc

        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image).astype(np.float32) / 255.0
        array = (array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)

    def _encode_text(self, text: str | None) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not text or self.tokenizer is None:
            return None, None
        token_ids, attention_mask = self.tokenizer.encode(text, max_length=self.max_text_length)
        return (
            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device),
        )

    def _build_species_prompt_bank(self) -> torch.Tensor | None:
        if self.tokenizer is None:
            return None

        species_names = [self.vocab.idx_to_species[idx] for idx in range(len(self.vocab.species_to_idx))]
        prompt_texts = []
        for species in species_names:
            for template in SPECIES_PROMPT_TEMPLATES:
                prompt_texts.append(template.format(species=species))

        token_batches = [self.tokenizer.encode(text, max_length=self.max_text_length) for text in prompt_texts]
        input_ids = torch.tensor([item[0] for item in token_batches], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([item[1] for item in token_batches], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            _, pooled = self.model.encode_text(input_ids, attention_mask)

        pooled = pooled.view(len(species_names), len(SPECIES_PROMPT_TEMPLATES), -1).mean(dim=1)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def _predict_torch(self, image_tensor: torch.Tensor, question: str | None) -> dict[str, torch.Tensor]:
        question_ids, question_mask = self._encode_text(question)
        with torch.inference_mode():
            return self.model(
                image_tensor.to(self.device),
                caption_ids=None,
                caption_mask=None,
                question_ids=question_ids,
                question_mask=question_mask,
            )

    def _predict_onnx(self, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.onnx_session is None:
            raise RuntimeError("ONNX session is not initialized.")
        species_logits, part_logits, health_logits = self.onnx_session.run(None, {"images": image_tensor.numpy()})
        return {
            "species_logits": torch.from_numpy(species_logits),
            "part_logits": torch.from_numpy(part_logits),
            "health_logits": torch.from_numpy(health_logits),
        }

    @staticmethod
    def _topk(probabilities: torch.Tensor, labels: dict[int, str], k: int) -> list[dict[str, float | str]]:
        values, indices = torch.topk(probabilities, k=min(k, probabilities.numel()))
        return [
            {"label": labels[int(index)], "confidence": round(float(value), 4)}
            for value, index in zip(values.tolist(), indices.tolist())
        ]

    def _prompt_scores(self, normalized_embedding: torch.Tensor, top_k: int) -> list[dict[str, float | str]] | None:
        if self.species_prompt_bank is None:
            return None
        similarities = (normalized_embedding @ self.species_prompt_bank.t()).squeeze(0).cpu()
        probabilities = torch.softmax(similarities, dim=0)
        return self._topk(probabilities, self.vocab.idx_to_species, top_k)

    def predict(self, image_bytes: bytes, *, question: str | None = None, top_k: int = 5) -> dict[str, Any]:
        image_tensor = self._preprocess_image(image_bytes)
        use_onnx = self.onnx_session is not None and question is None
        outputs = self._predict_onnx(image_tensor) if use_onnx else self._predict_torch(image_tensor, question)

        species_probs = torch.softmax(outputs["species_logits"].squeeze(0), dim=0).cpu()
        part_probs = torch.softmax(outputs["part_logits"].squeeze(0), dim=0).cpu()
        health_probs = torch.softmax(outputs["health_logits"].squeeze(0), dim=0).cpu()

        species_topk = self._topk(species_probs, self.vocab.idx_to_species, top_k)
        part_topk = self._topk(part_probs, self.vocab.idx_to_part, top_k)
        health_topk = self._topk(health_probs, self.vocab.idx_to_health, top_k)

        response: dict[str, Any] = {
            "species": {"label": species_topk[0]["label"], "confidence": species_topk[0]["confidence"], "top_k": species_topk},
            "plant_part": {"label": part_topk[0]["label"], "confidence": part_topk[0]["confidence"], "top_k": part_topk},
            "health_status": {"label": health_topk[0]["label"], "confidence": health_topk[0]["confidence"], "top_k": health_topk},
        }

        if "answer_logits" in outputs and self.vocab.answer_to_idx:
            answer_probs = torch.softmax(outputs["answer_logits"].squeeze(0), dim=0).cpu()
            answer_topk = self._topk(answer_probs, self.vocab.idx_to_answer, min(3, top_k))
            response["visual_answer"] = {
                "question": question,
                "answer": answer_topk[0]["label"],
                "confidence": answer_topk[0]["confidence"],
                "top_k": answer_topk,
            }

        if "normalized_embedding" in outputs:
            normalized_embedding = outputs["normalized_embedding"].detach().cpu()
            response["embedding"] = normalized_embedding.squeeze(0).tolist()
            prompt_topk = self._prompt_scores(normalized_embedding.to(self.device), top_k=min(3, top_k))
            if prompt_topk is not None:
                response["prompt_species"] = {
                    "label": prompt_topk[0]["label"],
                    "confidence": prompt_topk[0]["confidence"],
                    "top_k": prompt_topk,
                }

        return response

    def predict_file(self, image_path: str | Path, *, question: str | None = None, top_k: int = 5) -> dict[str, Any]:
        return self.predict(Path(image_path).read_bytes(), question=question, top_k=top_k)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run plant multi-task recognition inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    recognizer = PlantRecognizer(checkpoint_path=args.checkpoint, onnx_path=args.onnx, device=args.device)
    result = recognizer.predict_file(args.image, question=args.question, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
