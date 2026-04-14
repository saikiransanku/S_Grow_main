from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:  # pragma: no cover
    quantize_dynamic = None
    QuantType = None

try:
    from .inference import load_checkpoint_bundle
except ImportError:  # pragma: no cover
    from training.plant_multitask.inference import load_checkpoint_bundle


class ImageOnlyExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        return outputs["species_logits"], outputs["part_logits"], outputs["health_logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the plant multi-task model for deployment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize-int8", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_checkpoint_bundle(args.checkpoint, device="cpu")
    model = bundle.model.cpu().eval()
    wrapper = ImageOnlyExportWrapper(model)

    image_size = int(bundle.runtime_metadata.get("image_size", 224))
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output_path),
        opset_version=args.opset,
        dynamo=False,
        input_names=["images"],
        output_names=["species_logits", "part_logits", "health_logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "species_logits": {0: "batch"},
            "part_logits": {0: "batch"},
            "health_logits": {0: "batch"},
        },
    )

    quantized_path = None
    if args.quantize_int8 and quantize_dynamic is not None and QuantType is not None:
        quantized_path = args.output_path.with_name(args.output_path.stem + "_int8.onnx")
        quantize_dynamic(
            model_input=str(args.output_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )

    runtime_metadata = {
        **bundle.runtime_metadata,
        "onnx_path": str(args.output_path),
        "quantized_onnx_path": str(quantized_path) if quantized_path else None,
        "species_labels": bundle.vocab.idx_to_species,
        "part_labels": bundle.vocab.idx_to_part,
        "health_labels": bundle.vocab.idx_to_health,
    }
    (args.output_path.parent / "mobile_runtime.json").write_text(json.dumps(runtime_metadata, indent=2), encoding="utf-8")
    print(json.dumps(runtime_metadata, indent=2))


if __name__ == "__main__":
    main()
