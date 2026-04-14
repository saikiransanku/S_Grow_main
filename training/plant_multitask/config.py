from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    manifest_path: str
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    weighted_sampling: bool = True
    max_text_length: int = 48
    sampler_species_power: float = 1.0
    sampler_part_power: float = 0.2
    sampler_health_power: float = 0.25
    sampler_answer_power: float = 0.2
    sampler_season_power: float = 0.1


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_v2_s"
    pretrained: bool = True
    embedding_dim: int = 512
    dropout: float = 0.25
    attention_heads: int = 8
    task_hidden_dim: int = 512
    text_width: int = 256
    text_heads: int = 4
    text_layers: int = 2
    cross_attention_layers: int = 2
    use_text_alignment: bool = True
    use_vqa: bool = True


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mix_probability: float = 0.35
    mix_alpha: float = 0.4
    early_stopping_patience: int = 10
    ema_decay: float = 0.9995
    use_uncertainty_weighting: bool = True
    species_loss_type: str = "focal"
    part_loss_type: str = "cross_entropy"
    health_loss_type: str = "cross_entropy"
    focal_gamma: float = 1.5
    species_loss_weight: float = 1.0
    part_loss_weight: float = 0.35
    health_loss_weight: float = 0.45
    contrastive_loss_weight: float = 0.2
    vqa_loss_weight: float = 0.35
    rare_class_threshold: int = 20


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "training/plant_multitask/runs/default"
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ExperimentConfig":
        data = DataConfig(**payload["data"])
        model = ModelConfig(**payload.get("model", {}))
        train = TrainConfig(**payload.get("train", {}))
        return cls(
            data=data,
            model=model,
            train=train,
            output_dir=str(payload.get("output_dir", "training/plant_multitask/runs/default")),
            seed=int(payload.get("seed", 42)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment config must be a JSON object.")
        return cls.from_dict(payload)
