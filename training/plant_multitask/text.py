from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[^\s]")


@dataclass
class SimpleTokenizer:
    token_to_id: dict[str, int]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "SimpleTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in cls.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        tokens.extend(sorted(token for token, count in counts.items() if count >= min_freq))
        token_to_id = {token: index for index, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id)

    @staticmethod
    def tokenize(text: str | None) -> list[str]:
        return TOKEN_PATTERN.findall((text or "").lower())

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def encode(self, text: str | None, max_length: int) -> tuple[list[int], list[int]]:
        tokens = [self.bos_token]
        tokens.extend(self.tokenize(text))
        tokens.append(self.eos_token)
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_length]]
        attention_mask = [1] * len(token_ids)

        while len(token_ids) < max_length:
            token_ids.append(self.pad_id)
            attention_mask.append(0)

        return token_ids, attention_mask

    def to_dict(self) -> dict[str, object]:
        return {
            "token_to_id": self.token_to_id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SimpleTokenizer":
        token_to_id = payload.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("Tokenizer payload must contain token_to_id.")
        return cls(
            token_to_id={str(key): int(value) for key, value in token_to_id.items()},
            pad_token=str(payload.get("pad_token", "<pad>")),
            unk_token=str(payload.get("unk_token", "<unk>")),
            bos_token=str(payload.get("bos_token", "<bos>")),
            eos_token=str(payload.get("eos_token", "<eos>")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tokenizer file must contain a JSON object.")
        return cls.from_dict(payload)
