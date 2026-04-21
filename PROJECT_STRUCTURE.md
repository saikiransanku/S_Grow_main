# Project Structure

This workspace is organized into product apps, model-serving backends, and a dedicated training area.

```text
C:\S_Grow_main
|- ai_backend/                  # Django + FastAPI AI backend
|- backend/                     # Node/Express app backend
|- Frontend/                    # Next.js frontend
|- data/                        # Datasets and generated project data
|- outputs/                     # Exported outputs and reports
|- training/
|  |- agri_ai/                  # New phase-based training architecture
|  |- image_prediction/         # Seasonal TensorFlow training + runtime artifacts
|  |- plant_multitask/          # PyTorch multitask training package
|  |- hierarchical/             # Hierarchical crop-routing pipeline
|- assets/                      # Runtime label/model assets still used by older paths
|- app.py                       # Local ASGI entrypoint + smoke runner
```

## Notes

- Backend inference still reads seasonal runtime artifacts from `training/image_prediction/runs/{kharif,rabi,all_season}`.
- The new `training/agri_ai/` workspace mirrors the requested multi-stage pipeline: foundation, feature extraction, fusion, embeddings, classification, feedback, and evaluation.
- Legacy script entrypoints are kept as compatibility wrappers so existing commands do not break during the cleanup.
