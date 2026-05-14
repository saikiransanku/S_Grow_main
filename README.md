# SSGrow Project Guide

This is the single source of truth for the repo.

## 1. What This Project Is

SSGrow is an agriculture application with four main parts:

1. `Frontend/`
   Next.js app for farmers.
   Main user-facing pages:
   - `/ai-grow`: Disease AI for leaf-image disease analysis
   - `/suggestion-ai`: Suggestion AI for crop recommendation and farm planning
   - `/profile`: farmer land and crop-profile details
   - `/dashboard`: laws and activity views

2. `backend/`
   Node.js + Express + Prisma backend for:
   - authentication
   - user profile storage
   - laws
   - history / usage logs

3. `ai_backend/`
   Django AI backend for:
   - `/api/ai/predict`: disease prediction from uploaded image
   - `/api/ai/chat`: disease follow-up chat and crop suggestion chat routing
   - `/api/ai/health`: AI health check

4. `training/`
   model training and retraining code for seasonal, multitask, preprocessing, and hierarchical experiments.

## 2. Runtime Architecture

### Disease AI flow

1. User opens `Frontend/app/ai-grow/page.tsx`
2. User uploads one or more crop images
3. Frontend sends files to `POST /api/ai/predict`
4. Django AI backend runs image validation + disease pipeline in `ai_backend/predictor/services.py`
5. Result is returned with:
   - crop name
   - disease name
   - confidence
   - treatment hints
   - visual-analysis metadata
6. Follow-up questions in the same chat go to `POST /api/ai/chat`

### Suggestion AI flow

1. User opens `Frontend/app/suggestion-ai/page.tsx`
2. Frontend loads saved user profile fields from the Node backend
3. User asks for crop recommendation, intercropping, same-land planning, or risk-aware suggestions
4. Frontend sends the text prompt plus profile context to `POST /api/ai/chat`
5. Django AI backend routes the question through `ai_backend/predictor/service_modules/agriculture_advisor.py`
6. Response comes back as crop suggestions and planning advice

### Profile flow

1. Frontend calls `Frontend/lib/api.ts`
2. Node backend stores user + profile data through Prisma in `backend/src/routes/users.ts`
3. Suggestion AI reuses that profile context

## 3. Folder Map

```text
S_Grow_main/
|- Frontend/                        Next.js frontend
|  |- app/
|  |  |- ai-grow/                   Disease AI page
|  |  |- suggestion-ai/             Crop suggestion page
|  |  |- profile/                   Farmer profile page
|  |  |- dashboard/                 Dashboard page
|  |- components/                   Shared UI
|  |- lib/                          API helpers and auth helpers
|  |- public/                       Static assets
|
|- backend/                         Node + Express + Prisma backend
|  |- src/routes/                   Auth, users, laws, history
|  |- src/middleware/               Auth middleware
|  |- src/lib/                      Prisma client
|  |- prisma/schema.prisma          Database schema
|
|- ai_backend/                      Django AI backend
|  |- ai_backend/                   Django project settings/urls
|  |- predictor/                    AI API views, models, services, tests
|  |  |- service_modules/           Advisor, agronomy, active-learning helpers
|  |- scripts/                      Utility scripts
|
|- training/
|  |- image_prediction/             Seasonal TensorFlow training entrypoints
|  |- plant_multitask/              PyTorch multitask package
|  |- agri_ai/                      Shared preprocessing + orchestration code
|  |- hierarchical/                 Optional hierarchical experiments
|
|- assets/                          Legacy exported runtime artifacts
|- app.py                           Local Django AI smoke-test runner
|- README.md                        This file
```

## 4. Important Files

### Frontend

- `Frontend/app/ai-grow/page.tsx`
  Main disease-analysis UI.

- `Frontend/app/suggestion-ai/page.tsx`
  Main crop recommendation UI.

- `Frontend/app/profile/page.tsx`
  Stores land, soil, water, season, budget, and crop preference details.

- `Frontend/components/Navbar.tsx`
  Shared navigation between Disease AI and Suggestion AI.

- `Frontend/lib/api.ts`
  Axios client for the Node backend.

- `Frontend/lib/currentUser.ts`
  Converts profile data into advisor-ready context.

### Node backend

- `backend/src/index.ts`
  Express server entrypoint.

- `backend/src/routes/users.ts`
  User/profile read and update routes.

- `backend/src/routes/auth.ts`
  Login/register routes.

- `backend/src/routes/laws.ts`
  Farmer-laws endpoints.

- `backend/src/routes/history.ts`
  Usage history routes.

- `backend/prisma/schema.prisma`
  Main DB schema.

### AI backend

- `ai_backend/predictor/views.py`
  Public AI endpoints: `health`, `predict`, `chat`.

- `ai_backend/predictor/services.py`
  Core disease-inference + advisor routing logic.

- `ai_backend/predictor/service_modules/agriculture_advisor.py`
  Crop recommendation, same-land questions, intercropping suggestions.

- `ai_backend/predictor/service_modules/agronomy.py`
  Agronomy reference loading and retrieval helpers.

- `ai_backend/predictor/service_modules/active_learning.py`
  Low-confidence / unknown-crop fallback helpers.

### Training

- `training/image_prediction/ssgrow_transfer_training.py`
  Main seasonal TensorFlow training script.

- `training/image_prediction/train_kharif.py`
- `training/image_prediction/train_rabi.py`
- `training/image_prediction/train_all_season.py`
  Thin season-specific entrypoints.

- `training/image_prediction/RETRAINING_NOTES.md`
  Current retraining direction notes.

- `training/agri_ai/orchestration/seasonal_multitask.py`
  Shared multitask orchestration code.

- `training/agri_ai/feature_extraction/preprocessing.py`
  Shared preprocessing entrypoint.

- `training/plant_multitask/`
  Dataset prep, training, evaluation, export, and inference package.

- `training/hierarchical/`
  Optional experimental crop-routing and crop-specific disease training.

## 5. Local Run Guide

### Frontend

From `Frontend/`:

```bash
npm install
npm run dev
```

Expected env:

```env
NEXT_PUBLIC_AI_API_URL=http://localhost:8000/api/ai
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

Note:
- `Frontend/lib/api.ts` currently falls back to `http://localhost:2000/api`
- `backend/src/index.ts` defaults to port `5000`
- set `NEXT_PUBLIC_API_URL` explicitly so both sides match

### Node backend

From `backend/`:

```bash
npm install
npm run build
npm run start
```

Important env:

```env
DATABASE_URL=postgresql://...
JWT_SECRET=...
PORT=5000
CORS_ORIGIN=http://localhost:3000
```

### Django AI backend

From `ai_backend/`:

```bash
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

Important env:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
DJANGO_ALLOWED_HOSTS=*
```

## 6. AI API Endpoints

### `GET /api/ai/health`

Simple health check.

### `POST /api/ai/predict`

Purpose:
- disease analysis for uploaded crop images

Important form fields:
- `file` or `files`
- `season`
- `request_id`
- `request_name`
- `message`
- `file_sources`

### `POST /api/ai/chat`

Purpose:
- disease follow-up chat
- crop recommendation chat
- same-land planning
- intercropping guidance

Important JSON fields:
- `message`
- `context`
- `profile_name`
- `profile_context`
- `advisor_context`
- `conversation_history`

## 7. Training Layout

### Active seasonal training path

Use:
- `training/image_prediction/train_kharif.py`
- `training/image_prediction/train_rabi.py`
- `training/image_prediction/train_all_season.py`

These call the main trainer:
- `training/image_prediction/ssgrow_transfer_training.py`

### Shared multitask path

Use:
- `training/agri_ai/orchestration/seasonal_multitask.py`

This bridges seasonal classification data into the multitask package in:
- `training/plant_multitask/`

### Experimental path

Use:
- `training/hierarchical/`

Only if you want crop-routing / crop-specific disease experiments. It is not the main live runtime path.

## 8. Assets And Data

- `assets/models/` and `assets/labels/`
  legacy exported artifacts still referenced by parts of training/tests.

- `data/`
  dataset storage area. Ignored by git.

- `training/**/runs/`
  generated training outputs. Ignored by git.

## 9. What Was Cleaned Up

This repo was simplified to reduce duplicate and unused structure:

- removed the unused root npm package files
- removed the unused alternate FastAPI service
- removed old compatibility wrapper scripts that only forwarded to newer files
- removed empty placeholder training folders that contained only `.gitkeep`
- removed duplicate per-folder READMEs and replaced them with this single guide

## 10. If You Need To Continue Development

Use this mental model:

1. Frontend handles UI and calls two backends
2. Node backend handles auth/profile/laws/history
3. Django AI backend handles image prediction and advisor chat
4. Training code is separate from runtime serving code
5. `training/image_prediction/` is the main seasonal retraining area today

