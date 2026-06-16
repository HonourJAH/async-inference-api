# Async Inference API

A production-ready asynchronous text classification API that accepts inference requests instantly, processes them in the background using a Celery task queue, and lets clients poll for results. Built with FastAPI, Celery, Redis, and scikit-learn.

---

## How It Works

```
POST /jobs     →  validate text → save job to Redis → push to Celery queue → return job_id instantly
               →  Celery worker picks up job → runs sklearn pipeline → stores result in Redis

GET /jobs/{id} →  look up job in Redis → return current status and result
GET /health    →  health check
```

---

## Table of Contents

- [Why Async?](#why-async)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Training the Model](#training-the-model)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Request & Response Schemas](#request--response-schemas)
- [Job Lifecycle](#job-lifecycle)
- [Example Usage](#example-usage)
- [Docker](#docker)

---

## Why Async?

Synchronous ML APIs block the client until inference completes. Under load this causes timeouts, slow responses, and wasted server resources. This API decouples the request from the processing:

- The client gets a `job_id` **instantly**
- The model runs in a **separate worker process**
- The client **polls** for the result when ready
- Workers scale **independently** from the API server

---

## Project Structure

```
async-inference-api/
├── .github/
│   └── workflows/
│       └── ci.yml                    — GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                       — FastAPI app and route handlers
│   ├── worker.py                     — Celery app and inference task
│   ├── store.py                      — Redis job storage (save, get, update, delete)
│   ├── schema.py                     — Request and response schemas
│   └── services/
│       ├── __init__.py
│       └── classifier.py             — Loads joblib pipeline and runs inference
├── app/test/
│   ├── __init__.py
│   └── test_main.py                  — Full test suite
├── .dockerignore
├── .env                              — Environment variables
├── .gitignore
├── docker-compose.yml                — API + worker + Redis
├── Dockerfile
├── model.joblib                      — Trained sklearn pipeline
├── README.md
├── requirements.txt
└── train.py                          — Model training script
```

---

## Requirements

- Python 3.12+
- Docker and Docker Compose
- Redis (handled by Docker Compose)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HonourJAH/async-inference-api.git
cd async-inference-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Or create `.env` manually:

```
REDIS_URL=redis://localhost:6379/0
```

### 5. Train the model

```bash
python3 train.py
```

This generates `model.joblib` in the project root.

### 6. Start Redis

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 7. Start the API server

```bash
uvicorn app.main:app --reload
```

### 8. Start the Celery worker (separate terminal)

```bash
celery -A app.worker.celery_app worker --loglevel=info
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for broker and result backend |

---

## Training the Model

The classifier is trained on the 20 Newsgroups dataset across 5 categories using TF-IDF and Logistic Regression with GridSearchCV hyperparameter tuning:

```bash
python3 train.py
```

Trained categories:
- `sci.space`
- `alt.atheism`
- `talk.religion.misc`
- `soc.religion.christian`
- `sci.med`

The trained pipeline is saved to `model.joblib` and loaded once at worker startup.

---

## Running Tests

Tests use `fakeredis` and Celery's eager mode — no real Redis or worker needed.

```bash
pytest app/test/ -v
```

Run with coverage:

```bash
pytest app/test/ --cov=app --cov-report=term-missing
```

---

## API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `POST` | `/jobs` | Submit a text classification job | `202 Accepted` |
| `GET` | `/jobs/{job_id}` | Poll for job status and result | `200 OK` |
| `GET` | `/health` | Health check | `200 OK` |

---

## Request & Response Schemas

### `POST /jobs` — Submit a Job

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | `string` | ✅ | Text to classify. Minimum 10 characters |

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

---

### `GET /jobs/{job_id}` — Get Job Result

**Response:**

| Field | Type | Description |
|---|---|---|
| `job_id` | `uuid` | Unique job identifier |
| `status` | `string` | Current status: `queued`, `processing`, `completed`, `failed` |
| `category` | `string` | Predicted newsgroup category |
| `confidence` | `float` | Model confidence score between 0 and 1 |
| `result` | `string` | Human readable result e.g. `sci.space (99.84% confidence)` |
| `created_at` | `datetime` | When the job was submitted |
| `updated_at` | `datetime` | When the job was last updated |
| `completed_at` | `datetime` | When inference completed |
| `error` | `string` | Error message if the job failed |

**Example completed response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "category": "sci.space",
  "confidence": 0.9984,
  "result": "sci.space (99.84% confidence)",
  "created_at": "2026-06-07T16:04:19.064515Z",
  "updated_at": "2026-06-07T16:04:20.123456Z",
  "completed_at": "2026-06-07T16:04:20.123456Z",
  "error": null
}
```

---

### `GET /health`

```json
{ "status": "healthy" }
```

---

## Job Lifecycle

```
queued      ← job accepted, sitting in Redis queue
processing  ← worker picked it up, model is running
completed   ← result is ready, client can collect it
failed      ← something went wrong, error field is populated
```

---

## Example Usage

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new spacecraft into orbit today"}'
```

Response:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

### Poll for the result

```bash
curl http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000
```

Response:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "category": "sci.space",
  "confidence": 0.9984,
  "result": "sci.space (99.84% confidence)",
  "created_at": "2026-06-07T16:04:19.064515Z",
  "updated_at": "2026-06-07T16:04:20.123456Z",
  "completed_at": "2026-06-07T16:04:20.123456Z",
  "error": null
}
```

---

## Docker

### Run with Docker Compose

The easiest way to run everything — API, worker, and Redis together:

```bash
docker compose up --build
```

### Scale workers

```bash
docker compose up --build --scale worker=4
```

### Stop everything

```bash
docker compose down
```

### Services

| Service | Port | Description |
|---|---|---|
| `api` | `8000` | FastAPI server — accepts requests and returns job IDs |
| `worker` | — | Celery worker — runs inference in the background |
| `redis` | `6379` | Message broker and result store |

### Build image only

```bash
docker build -t async-inference-api .
```
