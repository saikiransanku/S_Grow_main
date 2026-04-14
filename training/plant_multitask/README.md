# Plant Multitask Pipeline

## Architecture

```text
Data Sources
  iNaturalist / PlantNet / Herbarium / Field Photos / Expert CSV
        |
        v
prepare_dataset.py
  - schema validation
  - duplicate removal
  - blur / resolution filtering
  - split assignment (70 / 15 / 15)
  - dataset mean / std estimation
        |
        v
manifest.jsonl + label_vocab.json + dataset_stats.json
        |
        v
train.py
  images -> backbone (EfficientNet-V2-S / MobileNetV3 / ConvNeXt / ViT)
         -> attention pooling over image tokens
         -> shared embedding
         -> task-specific adapters
         -> species / part / health heads
         -> CLIP-style text projection
         -> cross-attention VQA fusion head
        |
        v
best_checkpoint.pt
        |
        +--> evaluate.py
        |     - top-1 / top-5
        |     - per-class F1
        |     - confusion matrices
        |     - failure cases
        |
        +--> export.py
        |     - ONNX export
        |     - mobile runtime metadata
        |
        +--> api.py
              - /predict
              - /predict/batch
              - /metrics
```

## Manifest Schema

Each JSONL row should contain:

```json
{
  "image_path": "C:/dataset/tomato_leaf_001.jpg",
  "species": "solanum lycopersicum",
  "plant_part": "leaf",
  "health_status": "healthy",
  "split": "train",
  "source": "inat",
  "caption": "Healthy tomato leaf in daylight",
  "question": "What disease is this leaf showing?",
  "answer": "healthy",
  "bbox": [10, 20, 200, 220],
  "mask_path": "C:/dataset/masks/tomato_leaf_001.png",
  "metadata": {
    "expert_validated": true,
    "consensus_score": 0.92
  }
}
```

## Typical Workflow

```bash
python -m training.plant_multitask.prepare_dataset ^
  --data-root data/datasets/plant_data ^
  --output-dir training/plant_multitask/processed ^
  --annotations data/datasets/annotations.csv ^
  --species-map data/datasets/species_map.csv ^
  --compute-stats

python -m training.plant_multitask.train ^
  --manifest-path training/plant_multitask/processed/manifest.jsonl ^
  --output-dir training/plant_multitask/runs/plant_v1 ^
  --backbone efficientnet_v2_s

python -m training.plant_multitask.evaluate ^
  --checkpoint training/plant_multitask/runs/plant_v1/best_checkpoint.pt ^
  --manifest-path training/plant_multitask/processed/manifest.jsonl ^
  --output-dir training/plant_multitask/runs/plant_v1/eval

python -m training.plant_multitask.export ^
  --checkpoint training/plant_multitask/runs/plant_v1/best_checkpoint.pt ^
  --output-path training/plant_multitask/runs/plant_v1/model.onnx

python -m training.plant_multitask.restart_seasonal_training ^
  --dataset-root data/datasets/image_prediction_seasonal_dataset ^
  --workspace training/plant_multitask/runs/seasonal_restart
```

## Mobile Deployment Notes

- `export.py` produces an ONNX model suitable for ONNX Runtime Mobile and can optionally emit an INT8 dynamic-quantized variant.
- For strict TFLite targets, convert the exported ONNX model in CI with an ONNX-to-TFLite toolchain and validate numerical drift on the evaluation split before shipping.
- Keep `image_size=224` for mobile-first runs, prefer `mobilenet_v3_large` or `efficientnet_v2_s` for edge deployment, and quantize in deployment if latency is the bottleneck.

## Scaling Suggestions

- Add source-aware sampling so field photos are not overwhelmed by lab-style images.
- Store expert validation and crowdsourced consensus in the manifest metadata and down-weight low-consensus samples.
- Build species prototype banks from the shared embedding for few-shot extension to new species.
- Add a segmentation stage for leaf localization when backgrounds are cluttered.
- Benchmark exported ONNX with representative low-light mobile captures before release.
