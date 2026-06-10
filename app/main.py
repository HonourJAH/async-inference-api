import uuid
from app.store import save_job, get_job
from app.worker import run_inference
from fastapi import FastAPI, HTTPException, status
from app.schema import JobCreate, JobResponse

app = FastAPI()


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(job_text: JobCreate):

    job_id = str(uuid.uuid4())
    save_job(job_id, job_text.text)
    run_inference.delay(job_id, job_text.text)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_(job_id: str) -> JobResponse:

    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or expired"
        )

    return JobResponse(**job)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
