from datetime import datetime
from pydantic import BaseModel, Field
import uuid
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    text: str = Field(
        min_length=10,
        description="The text to classify must be at least 10 characters long.",
    )


class JobResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    category: str | None = None
    confidence: float | None = None
    result: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None
    model_config = {"from_attributes": True}
