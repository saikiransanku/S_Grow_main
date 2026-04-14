from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _resolve_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int, str]:
    if backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        return model, 1792, "cnn"

    if backbone_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        return model, 1280, "cnn"

    if backbone_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        return model, 960, "cnn"

    if backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        return model, 768, "cnn"

    if backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        return model, 768, "vit"

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.size(0), -1, -1)
        normalized = self.norm(tokens)
        pooled, _ = self.attn(query, normalized, normalized, need_weights=False)
        return pooled.squeeze(1)


class SimpleTextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(width)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = input_ids.size(1)
        x = self.token_embedding(input_ids) + self.position_embedding[:, :sequence_length, :]
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return x, pooled


class TaskAdapter(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class CrossModalFusionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            need_weights=False,
        )
        fused = query_tokens + attended
        fused = fused + self.ffn(fused)
        if query_mask is None:
            return fused
        return fused * query_mask.unsqueeze(-1)


class PlantMultiTaskModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        num_species: int,
        num_parts: int,
        num_health_states: int,
        num_answers: int = 0,
        pretrained: bool = True,
        embedding_dim: int = 512,
        dropout: float = 0.25,
        attention_heads: int = 8,
        task_hidden_dim: int = 512,
        text_vocab_size: int | None = None,
        text_width: int = 256,
        text_heads: int = 4,
        text_layers: int = 2,
        cross_attention_layers: int = 2,
        max_text_length: int = 48,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim, self.backbone_type = _resolve_backbone(backbone_name, pretrained=pretrained)

        self.attention_pool = AttentionPool(feature_dim, num_heads=attention_heads)
        self.token_projection = nn.Linear(feature_dim, embedding_dim)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        self.species_adapter = TaskAdapter(embedding_dim, task_hidden_dim, dropout)
        self.part_adapter = TaskAdapter(embedding_dim, task_hidden_dim // 2 if task_hidden_dim > 1 else task_hidden_dim, dropout)
        self.health_adapter = TaskAdapter(embedding_dim, task_hidden_dim // 2 if task_hidden_dim > 1 else task_hidden_dim, dropout)

        self.species_head = nn.Linear(embedding_dim, num_species)
        self.part_head = nn.Linear(embedding_dim, num_parts)
        self.health_head = nn.Linear(embedding_dim, num_health_states)

        self.text_encoder = None
        self.text_projection = None
        self.cross_modal_layers = None
        self.answer_head = None
        if text_vocab_size and text_vocab_size > 0:
            self.text_encoder = SimpleTextEncoder(
                vocab_size=text_vocab_size,
                width=text_width,
                num_heads=text_heads,
                num_layers=text_layers,
                max_length=max_text_length,
            )
            self.text_projection = nn.Linear(text_width, embedding_dim)
            self.cross_modal_layers = nn.ModuleList(
                [CrossModalFusionLayer(embedding_dim, num_heads=text_heads, dropout=dropout) for _ in range(cross_attention_layers)]
            )
            if num_answers > 0:
                self.answer_head = nn.Sequential(
                    nn.LayerNorm(embedding_dim * 3),
                    nn.Linear(embedding_dim * 3, embedding_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim, num_answers),
                )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def extract_token_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == "cnn":
            features = self.backbone.features(images)
            if features.ndim != 4:
                raise RuntimeError("CNN backbone must return feature maps with shape [B, C, H, W].")
            tokens = features.flatten(2).transpose(1, 2)
            pooled = self.attention_pool(tokens)
            return tokens, pooled

        if self.backbone_type == "vit":
            x = self.backbone._process_input(images)
            batch_size = x.size(0)
            class_token = self.backbone.class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            x = self.backbone.encoder(x)
            tokens = x[:, 1:]
            pooled = 0.5 * (self.attention_pool(tokens) + x[:, 0])
            return tokens, pooled

        raise RuntimeError(f"Unsupported backbone type: {self.backbone_type}")

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_features, pooled_features = self.extract_token_features(images)
        embedding = self.embedding_norm(self.shared_projection(pooled_features))
        normalized_embedding = F.normalize(embedding, dim=-1)
        projected_tokens = F.normalize(self.token_projection(token_features), dim=-1)
        return embedding, normalized_embedding, projected_tokens

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.text_projection is None:
            raise RuntimeError("Text encoder is not enabled for this model.")
        token_features, pooled = self.text_encoder(input_ids, attention_mask)
        token_embeddings = self.text_projection(token_features)
        pooled_embedding = F.normalize(self.text_projection(pooled), dim=-1)
        return token_embeddings, pooled_embedding

    def _masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = tokens * mask.unsqueeze(-1)
        return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(
        self,
        images: torch.Tensor,
        *,
        caption_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        question_ids: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding, normalized_embedding, image_tokens = self.encode_image(images)
        species_features = self.species_adapter(embedding)
        part_features = self.part_adapter(embedding)
        health_features = self.health_adapter(embedding)

        outputs = {
            "embedding": embedding,
            "normalized_embedding": normalized_embedding,
            "species_logits": self.species_head(species_features),
            "part_logits": self.part_head(part_features),
            "health_logits": self.health_head(health_features),
            "logit_scale": self.logit_scale.exp(),
        }

        if caption_ids is not None and caption_mask is not None and self.text_encoder is not None:
            _, caption_embedding = self.encode_text(caption_ids, caption_mask)
            outputs["caption_embedding"] = caption_embedding

        if (
            question_ids is not None
            and question_mask is not None
            and self.text_encoder is not None
            and self.answer_head is not None
            and self.cross_modal_layers is not None
        ):
            question_tokens, question_embedding = self.encode_text(question_ids, question_mask)
            fused_tokens = question_tokens
            for layer in self.cross_modal_layers:
                fused_tokens = layer(fused_tokens, image_tokens, query_mask=question_mask)
            fused_embedding = F.normalize(self._masked_mean(fused_tokens, question_mask), dim=-1)
            answer_input = torch.cat([normalized_embedding, question_embedding, fused_embedding], dim=-1)
            outputs["question_embedding"] = question_embedding
            outputs["fused_question_embedding"] = fused_embedding
            outputs["answer_logits"] = self.answer_head(answer_input)

        return outputs
