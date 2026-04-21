# Training Workspace

This repo now has a phase-based training layout for the advanced Agri-AI pipeline while keeping the existing runtime-compatible trainers in place.

## Main folders

- `training/agri_ai/`: new architecture-first workspace for Phase 0 to Phase 6.
- `training/agri_ai/feature_extraction/preprocessing.py`: moved home for the reusable preprocessing pipeline.
- `training/agri_ai/orchestration/`: bridge code that prepares seasonal datasets and launches multitask training.
- `training/plant_multitask/`: current PyTorch multitask training implementation.
- `training/image_prediction/`: seasonal TensorFlow training and inference artifacts used by the backend today.
- `training/hierarchical/`: crop-routing and crop-specific disease pipeline.

## Naming conventions

- New folders use lowercase snake_case.
- Phase folders are grouped by responsibility instead of mixing scripts, outputs, and caches together.
- Generated artifacts should stay out of source folders and go into `runs/`, `outputs/`, or dataset roots.

