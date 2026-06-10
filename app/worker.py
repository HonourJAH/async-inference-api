import os
from celery import Celery
from app.services.classifier import predict
from app.store import update_job

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("async_inference", broker=REDIS_URL, backend=REDIS_URL)

# celery Configuration
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    result_expires=3600,
    task_max_retries=3,
    task_acks_late=True,
    timezone="UTC",
)


# inference task
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
)
def run_inference(self, job_id: str, text: str) -> dict:
    """Celery task that runs text classification in the background.

    This is the consumer — it picks jobs off the Redis queue,
    runs them through the sklearn pipeline, and stores the result
    back in Redis for the API to retrieve.

    bind=True allows us to call self.retry() inside the task
    if something goes wrong.
    """
    try:
        result = predict(text)

        update_job(
            job_id=job_id,
            status="completed",
            category=result["category"],
            confidence=result["confidence"],
        )

    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))
        raise self.retry(exc=exc, countdown=5)
