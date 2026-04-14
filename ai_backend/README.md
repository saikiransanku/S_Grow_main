# AI Backend (Django)

This backend provides:
- `POST /api/ai/predict` for image disease prediction + type verification
- `POST /api/ai/chat` for LLM responses using prediction context
- `GET /api/ai/health` for health check

## Run locally

1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Set env (PowerShell example):
   ```powershell
   Copy-Item .env.example .env
   $env:OPENAI_API_KEY="your_key_here"
   ```
3. Start server:
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

For local Windows development, prefer `python manage.py runserver` over `uvicorn app:app --reload`.
The Django image upload flow in this repo is stable with `runserver`, while mixed `uvicorn` reload
processes can leave port `8000` accepting connections without returning responses.

## Multi-stage prediction pipeline

The backend now runs a staged funnel instead of a single flat classifier:

1. Input gatekeepers reject blurry, heavily glared, or badly lit images.
2. Background segmentation isolates the leaf foreground before inference.
3. Optional stage-1 crop routing can reject unsupported crops before disease prediction.
4. Stage-2 disease inference uses the routed seasonal model, or a crop-specific disease model if one exists.
5. Confidence scores are temperature-scaled through `MODEL_CONFIDENCE_TEMPERATURE`.
6. Unsupported or low-confidence images can be routed into an active-learning dataset through `handle_unknown_crop(...)`.
7. Farmer advice is built from the diagnosis plus retrieved agronomy references and pesticide mappings.

Optional hierarchical artifacts:

```env
ENABLE_HIERARCHICAL_ROUTING=true
STAGE1_CROP_MODEL_PATH=C:/AI_Preditions/training/hierarchical/artifacts/plant_classifier.pth
STAGE1_CROP_CLASS_FILE=C:/AI_Preditions/training/hierarchical/artifacts/plant_classifier_classes.json
CROP_SPECIFIC_MODELS_DIR=C:/AI_Preditions/training/hierarchical/disease_models
```

If these artifacts are missing, the API falls back to the current seasonal models without crashing.

Optional active-learning fallback:

```env
ENABLE_ACTIVE_LEARNING_FALLBACK=true
ACTIVE_LEARNING_DATASET_ROOT=C:/AI_Preditions/data/datasets/active_learning_crops
ACTIVE_LEARNING_MIN_CONFIDENCE=0.50
```

When the main disease pipeline returns an unsupported crop or a low-confidence prediction, the backend can run the broad plant classifier again as a discovery CNN and save recognized plant images into crop-specific folders for future retraining.

## FastAPI database + storage configuration

Add these values in `ai_backend/.env` (this file is auto-read by `fastapi_service/config.py`):

```env
FASTAPI_DATABASE_URL=sqlite:///C:/AI_Preditions/ai_backend/fastapi_predictions.db
IMAGE_STORAGE_DIR=C:/AI_Preditions/ai_backend/storage/uploads
PESTICIDES_CSV_PATH=C:/AI_Preditions/data/outputs/project_data/pesticides_data_1m.csv
```

- `FASTAPI_DATABASE_URL` is where you place your database link (PostgreSQL/MySQL/SQLite URL).
- Uploaded images, prediction rows, and generated/cached responses are stored in this DB.
- Start FastAPI with:
  ```bash
  uvicorn fastapi_service.main:app --host 0.0.0.0 --port 8001
  ```

## Generate pesticide CSV with 1,000,000+ rows

From `ai_backend/` run:

```bash
python scripts/generate_large_pesticides_csv.py --rows 1000001
```

This creates:
- `data/outputs/project_data/pesticides_data_1m.csv`

Note: loading the 1M CSV at FastAPI startup is slower; keep `PESTICIDES_CSV_PATH` pointed to the smaller file for normal runtime if needed.

## Frontend integration (`SSGrow/Frontend`)

Set:
```bash
NEXT_PUBLIC_AI_API_URL=http://localhost:8000/api/ai
```

## API examples

Prediction:
```bash
curl -X POST http://localhost:8000/api/ai/predict -F "file=@leaf.jpg" -F "season=all_season"
```

Chat:
```bash
curl -X POST http://localhost:8000/api/ai/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What treatment do you suggest?\",\"context\":{\"disease_prediction\":\"early_blight_tomato\"}}"
```

## Hugging Face Spaces deployment

1. Create a new **Docker Space**.
2. Upload this `ai_backend` folder contents.
3. Include the latest seasonal training run artifacts:
   - `training/image_prediction/runs/kharif/ssgrow_disease_model_v2.keras`
   - `training/image_prediction/runs/kharif/class_names.txt`
   - `training/image_prediction/runs/rabi/ssgrow_disease_model_v2.keras`
   - `training/image_prediction/runs/rabi/class_names.txt`
   - `training/image_prediction/runs/all_season/ssgrow_disease_model_v2.keras`
   - `training/image_prediction/runs/all_season/class_names.txt`
4. Set Space Variables:
   - `ALL_SEASON_MODEL_PATH=/app/training/image_prediction/runs/all_season/ssgrow_disease_model_v2.keras`
   - `ALL_SEASON_CLASS_FILE=/app/training/image_prediction/runs/all_season/class_names.txt`
   - `KHARIF_MODEL_PATH=/app/training/image_prediction/runs/kharif/ssgrow_disease_model_v2.keras`
   - `KHARIF_CLASS_FILE=/app/training/image_prediction/runs/kharif/class_names.txt`
   - `RABI_MODEL_PATH=/app/training/image_prediction/runs/rabi/ssgrow_disease_model_v2.keras`
   - `RABI_CLASS_FILE=/app/training/image_prediction/runs/rabi/class_names.txt`
   - `OPENAI_API_KEY=...`
   - `OPENAI_MODEL=gpt-4o-mini`
5. Space listens on port `7860` via Dockerfile command.

For production, replace wildcard CORS/hosts with exact domains.
