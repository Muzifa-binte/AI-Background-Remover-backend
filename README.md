# AI Background Remover — Backend

> **Team:** Web Team (API)  
> **Repo:** `AI-Background-Remover-backend`  
> **Parent repo:** `AI-Background-Remover` (this is a submodule)  
> **Tech:** Python 3.11 · FastAPI · Uvicorn · MongoDB (Motor) · aiofiles

---

## What This Repo Is

The FastAPI backend for the AI Background Remover application.  
It receives image uploads from the React frontend, calls the AI pipeline to remove the background, stores job metadata in MongoDB, and serves the results back.

It acts as the **bridge between the UI and the AI pipeline**.  
It does not contain any model code — it calls the `AI-Background-Remover-AI` package via the `services/bg_removal.py` wrapper.

---

## Folder Structure

```
backend/
│
├── app.py                      ← FastAPI app, CORS, router registration,
│                                 MongoDB lifespan hooks (startup/shutdown)
│
├── routes/                     ← one file per feature area
│   ├── __init__.py
│   ├── remove_bg.py            ← POST /api/remove-background
│   │                             validates upload, saves file, calls AI,
│   │                             writes history to MongoDB, returns result
│   ├── download.py             ← GET /api/download/{filename}
│   │                             serves the processed PNG file
│   ├── history.py              ← GET /api/history
│   │                             queries MongoDB, returns last 50 jobs
│   └── images.py               ← DELETE /api/image/{id}
│                                 removes file from disk + MongoDB record
│
├── services/                   ← business logic, separated from routes
│   ├── __init__.py
│   ├── bg_removal.py           ← async wrapper around AI inference
│   │                             runs inference in thread pool (non-blocking)
│   └── database.py             ← Motor (async MongoDB) connection helpers
│                                 connect_db(), close_db(), get_collection()
│
├── uploads/                    ← incoming images are saved here temporarily
│   └── .gitkeep
│
└── output/                     ← processed transparent PNGs live here
    └── .gitkeep
```

---

## API Reference

### `POST /api/remove-background`
Upload an image and get a transparent PNG back.

**Request:** `multipart/form-data`
| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | File | Yes | JPEG, PNG, or WebP. Max 10 MB. |

**Response:** `200 OK`
```json
{
  "output_filename": "abc123_result.png",
  "download_url": "/api/download/abc123_result.png"
}
```

**Errors:**
| Code | Reason |
|------|--------|
| 400 | Unsupported file type |
| 413 | File exceeds 10 MB |
| 500 | AI inference failed |

---

### `GET /api/download/{filename}`
Download a processed image file.

**Response:** PNG file stream (`image/png`)  
**Errors:** `404` if file not found.

---

### `GET /api/history`
Returns the last 50 processing jobs, newest first.

**Response:** `200 OK`
```json
[
  {
    "upload_id":       "abc123",
    "original_name":   "photo.jpg",
    "output_filename": "abc123_result.png",
    "created_at":      "2026-08-03T10:22:00+00:00"
  }
]
```

---

### `DELETE /api/image/{id}`
Deletes a processed image from disk and removes its MongoDB record.

**Path param:** `id` — the `upload_id` (UUID portion before `_result.png`)  
**Response:** `200 OK`
```json
{ "message": "Image abc123 deleted successfully." }
```
**Errors:** `404` if file not found.

---

### `GET /`
Health check.
```json
{ "status": "ok", "message": "AI Background Remover API is running." }
```

---

## How a Request Flows Through the Code

```
POST /api/remove-background
        │
        ▼
routes/remove_bg.py
  1. Validate content_type (JPEG/PNG/WebP only)
  2. Validate file size (≤ 10 MB)
  3. Save upload to uploads/<uuid>_<filename>
  4. Call services/bg_removal.py → remove_background()
        │
        ▼
  services/bg_removal.py
    Runs AI inference in thread pool executor
    (keeps FastAPI event loop unblocked)
        │
        ▼
  AI pipeline (AI-Background-Remover-AI submodule)
    preprocess → model → postprocess → saves PNG
        │
        ▼
  routes/remove_bg.py (continued)
  5. Write job metadata to MongoDB via services/database.py
  6. Return { output_filename, download_url }
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- MongoDB running on `localhost:27017`
- The AI submodule present (`AI-Background-Remover-AI/`)

### Setup
```bash
# From the repo root (parent), activate the virtual environment
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install all dependencies (from parent root)
pip install -r requirements.txt

# Copy the backend-specific environment config
cp backend/.env.example backend/.env
# Edit backend/.env — set MONGO_URI if your MongoDB is not on localhost
```

### Run
```bash
cd backend
uvicorn app:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`  
Alternative docs (ReDoc): `http://localhost:8000/redoc`

---

## Environment Variables

All variables are loaded from `backend/.env` (next to `app.py`) via `python-dotenv`.
Copy `backend/.env.example` to `backend/.env` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `rembg` | AI backend: `rembg`, `onnx`, or `torch` |
| `ONNX_MODEL_PATH` | `ai/models/model.onnx` | Path to ONNX weights |
| `TORCH_MODEL_PATH` | `ai/models/model.pth` | Path to PyTorch weights |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `ai_bg_remover` | Database name |

See `.env.example` in the root for the full list.

---

## Database (MongoDB)

**Database:** `ai_bg_remover`  
**Collection:** `history`

Each document:
```json
{
  "_id":             "ObjectId (auto)",
  "upload_id":       "uuid string",
  "original_name":   "user's original filename",
  "output_filename": "uuid_result.png",
  "created_at":      "UTC datetime"
}
```

The connection is opened on FastAPI startup and closed on shutdown — both handled in `app.py`'s lifespan context manager.

---

## Adding a New Route

1. Create `routes/your_feature.py` with an `APIRouter`.
2. Write your endpoint functions in it.
3. Import and register the router in `app.py`:
   ```python
   from routes.your_feature import router as your_router
   app.include_router(your_router, prefix="/api")
   ```
4. If you need a database operation, add a helper to `services/database.py` or call `get_collection()` directly.

## Adding a New Service

1. Create `services/your_service.py`.
2. Keep it `async` — FastAPI runs on an async event loop.
3. For CPU-heavy work (like more AI calls), use `loop.run_in_executor(None, ...)` to avoid blocking.

---

## What Is Done vs What Is Next

### Done
- [x] FastAPI app with CORS configured for Vite dev server
- [x] MongoDB lifespan hooks (connect on startup, close on shutdown)
- [x] `POST /api/remove-background` — full upload, AI call, MongoDB write
- [x] `GET /api/download/{filename}` — secure file serve (path traversal protected)
- [x] `GET /api/history` — real MongoDB query, datetime serialized
- [x] `DELETE /api/image/{id}` — disk + MongoDB cleanup
- [x] Async AI wrapper (non-blocking inference)

### Next (for Web Team to pick up)
- [ ] JWT authentication — protect history and delete endpoints per user
- [ ] User registration and login endpoints
- [ ] Associate history records with authenticated user ID
- [ ] Rate limiting (one heavy inference at a time per IP)
- [ ] Background task queue (Celery or FastAPI BackgroundTasks) for slow models
- [ ] Cleanup job — auto-delete uploads older than 24 hours
- [ ] Unit tests for all routes (`pytest` + `httpx`)

---

## Contribution

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the parent repo for branch naming, commit format, and PR rules.

Your branch always goes into this submodule repo (`AI-Background-Remover-backend`), not the parent.
