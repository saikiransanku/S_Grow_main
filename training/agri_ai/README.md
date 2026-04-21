# Agri-AI Training Workspace

This folder organizes the plant and disease training system by phase so the repo can grow from the current seasonal and multitask trainers into the full architecture you outlined.

```text
training/
|- agri_ai/
|  |- configs/                    # Shared pipeline templates and experiment settings
|  |- foundation/                # Self-supervised pretraining + RL policy learning
|  |- feature_extraction/        # Preprocessing, segmentation, and backbone feature loaders
|  |- feature_fusion/            # Adaptive fusion, handcrafted descriptors, cross-attention
|  |- embedding_learning/        # Metric learning, PCA, hard example mining
|  |- classification/            # XGBoost, multitask NN, prototype, ensemble heads
|  |- feedback/                  # RLHF, active learning, reward shaping
|  |- evaluation/                # Calibration, explainability, reports
|  |- orchestration/             # Cross-phase runners and bridge utilities
|- hierarchical/                 # Current crop-routing / hierarchical training pipeline
|- image_prediction/             # Runtime-compatible seasonal TensorFlow pipeline
|- plant_multitask/              # Current PyTorch multitask training package
```

## Compatibility

- `training/pre_process.py` remains available as a compatibility entrypoint and now forwards to `training/agri_ai/feature_extraction/preprocessing.py`.
- `training/image_prediction/plant_multitask_training.py` remains available as a compatibility entrypoint and now forwards to the orchestrator inside `training/agri_ai/orchestration/`.
- Seasonal runtime artifacts stay under `training/image_prediction/runs/` so the backend does not need path changes.

