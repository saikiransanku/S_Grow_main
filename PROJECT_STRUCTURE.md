# Project Structure

This workspace is now organized by function:

```
C:\AI_Preditions
|- ai_backend/                  # Django API backend
|- SSGrow/                      # Frontend workspace (Next.js)
|- data/
|  |- datasets/
|     |- image_prediction_seasonal_dataset/  # Active seasonal training datasets
|  |- outputs/
|     |- project_data/          # Generated CSV/TXT outputs
|- training/
|  |- image_prediction/
|     |- runs/                  # Latest seasonal inference artifacts (.keras + class_names.txt)
|     |- ...                    # Training scripts
|- venv/                        # Python virtual environment
|- app.py                       # ASGI entrypoint + local smoke CLI
|- PROJECT_STRUCTURE.md         # This structure guide
```

## Notes

- Backend inference now loads directly from `training/image_prediction/runs/{kharif,rabi,all_season}`.
- The `assets/` folder is no longer part of the inference path.
- `.env` and `.env.example` in `ai_backend/` are already updated to match this layout.
- Training scripts and dataset utility scripts were updated to the new dataset/model paths.
