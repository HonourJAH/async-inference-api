import os
import json
import redis
from datetime import datetime, timezone
from app.schema import JobStatus

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

r = redis.from_url(REDIS_URL, decode_responses=True)

# TTL (Time To Live): how long to keep job data in Redis before it auto-expires
JOB_TTL = 3600


# Job Storage Functions
def save_job(job_id: str, text: str) -> dict:
    job = {
        "job_id": job_id,
        "text": text,
        "status": JobStatus.QUEUED.value,
        "result": None,
        "category": None,
        "confidence": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    r.set(job_id, json.dumps(job), ex=JOB_TTL)
    return job


def get_job(job_id: str) -> dict | None:
    """Retrieve a job by its ID.

    Returns None if the job doesn't exist or has expired.
    """
    data = r.get(job_id)

    if data is None:
        return None

    return json.loads(data)


def update_job(
    job_id: str,
    status: str,
    category: str = None,
    confidence: float = None,
    error: str = None,
) -> dict | None:
    job = get_job(job_id)

    if job is None:
        return None

    job["status"] = JobStatus(status).value
    job["updated_at"] = datetime.now(timezone.utc).isoformat()

    if category is not None and confidence is not None:
        job["category"] = category  # ← add
        job["confidence"] = confidence  # ← add
        job["result"] = f"{category} ({confidence:.2%} confidence)"

    if status == JobStatus.COMPLETED.value:
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

    if error is not None:
        job["error"] = error

    r.set(job_id, json.dumps(job), ex=JOB_TTL)
    return job


def delete_job(job_id: str) -> bool:
    """Delete a job from Redis.

    Returns True if the job existed and was deleted.
    Returns False if the job didn't exist.
    """
    deleted = r.delete(job_id)
    return deleted > 0
