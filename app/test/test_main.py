import pytest
import fakeredis
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.store import save_job, get_job

client = TestClient(app)


# Fixtures


@pytest.fixture(autouse=True)
def fake_redis():
    """Replace the real Redis connection with fakeredis for every test.

    autouse=True means this runs automatically for every test in this file
    without needing to explicitly request it.

    The same fake instance is patched in both store.py and worker.py
    so they share the same in-memory data — just like they would share
    a real Redis instance in production.
    """
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("app.store.r", fake):
        yield fake


@pytest.fixture(autouse=True)
def eager_celery():
    """Force Celery to run tasks synchronously in the same thread.

    Without this, .delay() pushes the task to Redis and returns immediately
    — the task never actually runs during the test.
    With this, .delay() runs the task immediately before returning.
    """
    from app.worker import celery_app

    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
    )


@pytest.fixture
def mock_predict():
    """Mock the predict function to return a fixed result.
    Used in endpoint tests that don't care about model accuracy —
    only about correct API behaviour.
    """
    with patch("app.worker.predict") as mock:
        mock.return_value = {
            "category": "sci.space",
            "confidence": 0.9984,
        }
        yield mock


@pytest.fixture
def queued_job(fake_redis):
    """Create a pre-existing queued job in fakeredis."""
    import uuid

    return save_job(str(uuid.uuid4()), "NASA launched a new spacecraft")


# Health Check


class TestHealthCheck:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_healthy_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


# POST /predict


class TestPredictEndpoint:
    def test_returns_202(self, mock_predict):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        assert response.status_code == 202

    def test_returns_job_id(self, mock_predict):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        assert "job_id" in response.json()

    def test_returns_queued_status(self, mock_predict):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        assert response.json()["status"] == "queued"

    def test_job_id_is_a_string(self, mock_predict):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        assert isinstance(response.json()["job_id"], str)

    def test_two_requests_get_different_job_ids(self, mock_predict):
        response1 = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        response2 = client.post("/jobs", json={"text": "Atheism is a belief system"})
        assert response1.json()["job_id"] != response2.json()["job_id"]

    def test_empty_text_returns_422(self, mock_predict):
        response = client.post("/jobs", json={"text": ""})
        assert response.status_code == 422

    def test_missing_text_returns_422(self, mock_predict):
        response = client.post("/jobs", json={})
        assert response.status_code == 422

    def test_job_is_saved_to_redis(self, mock_predict, fake_redis):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        job_id = response.json()["job_id"]
        job = get_job(job_id)
        assert job is not None

    def test_saved_job_has_correct_text(self, mock_predict, fake_redis):
        response = client.post("/jobs", json={"text": "NASA launched a spacecraft"})
        job_id = response.json()["job_id"]
        job = get_job(job_id)
        assert job["text"] == "NASA launched a spacecraft"


# GET /jobs/{job_id}


class TestJobsEndpoint:
    def test_returns_200_for_existing_job(self, queued_job):
        response = client.get(f"/jobs/{queued_job['job_id']}")
        assert response.status_code == 200

    def test_returns_404_for_nonexistent_job(self):
        response = client.get("/jobs/nonexistent-job-id")
        assert response.status_code == 404

    def test_returns_correct_error_for_missing_job(self):
        response = client.get("/jobs/nonexistent-job-id")
        assert "not found" in response.json()["detail"].lower()

    def test_returns_job_id(self, queued_job):
        response = client.get(f"/jobs/{queued_job['job_id']}")
        assert response.json()["job_id"] == queued_job["job_id"]

    def test_returns_queued_status_before_processing(self, queued_job):
        response = client.get(f"/jobs/{queued_job['job_id']}")
        assert response.json()["status"] == "queued"

    def test_returns_completed_status_after_processing(self, mock_predict, fake_redis):
        # Submit a job — with eager celery it processes immediately
        post_response = client.post(
            "/jobs", json={"text": "NASA launched a spacecraft"}
        )
        job_id = post_response.json()["job_id"]

        # Result should already be complete
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.json()["status"] == "completed"

    def test_returns_created_at_timestamp(self, queued_job):
        response = client.get(f"/jobs/{queued_job['job_id']}")
        assert "created_at" in response.json()

    def test_returns_updated_at_timestamp(self, queued_job):
        response = client.get(f"/jobs/{queued_job['job_id']}")
        assert "updated_at" in response.json()

    def test_returns_category_after_processing(self, mock_predict, fake_redis):
        post_response = client.post(
            "/jobs", json={"text": "NASA launched a spacecraft"}
        )
        job_id = post_response.json()["job_id"]
        get_response = client.get(f"/jobs/{job_id}")
        assert "sci.space" in get_response.json()["result"]

    def test_returns_confidence_after_processing(self, mock_predict, fake_redis):
        post_response = client.post(
            "/jobs", json={"text": "NASA launched a spacecraft"}
        )
        job_id = post_response.json()["job_id"]
        get_response = client.get(f"/jobs/{job_id}")
        assert "99.84%" in get_response.json()["result"]

    def test_returns_original_text(self, mock_predict, fake_redis):
        post_response = client.post(
            "/jobs", json={"text": "NASA launched a spacecraft"}
        )
        job_id = post_response.json()["job_id"]
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.json()["status"] == "completed"


# Store Unit Tests


class TestStore:
    def test_save_job_returns_dict(self, fake_redis):
        job = save_job("job-1", "some text")
        assert isinstance(job, dict)

    def test_save_job_has_correct_job_id(self, fake_redis):
        job = save_job("job-1", "some text")
        assert job["job_id"] == "job-1"

    def test_save_job_has_queued_status(self, fake_redis):
        job = save_job("job-1", "some text")
        assert job["status"] == "queued"

    def test_save_job_has_correct_text(self, fake_redis):
        job = save_job("job-1", "some text")
        assert job["text"] == "some text"

    def test_save_job_category_is_none(self, fake_redis):
        job = save_job("job-1", "some text")
        assert job["category"] is None

    def test_save_job_confidence_is_none(self, fake_redis):
        job = save_job("job-1", "some text")
        assert job["confidence"] is None

    def test_get_job_returns_saved_job(self, fake_redis):
        save_job("job-1", "some text")
        job = get_job("job-1")
        assert job["job_id"] == "job-1"

    def test_get_job_returns_none_for_missing_job(self, fake_redis):
        assert get_job("nonexistent") is None

    def test_update_job_changes_status(self, fake_redis):
        from app.store import update_job

        save_job("job-1", "some text")
        update_job("job-1", status="completed", category="sci.space", confidence=0.99)
        job = get_job("job-1")
        assert job["status"] == "completed"

    def test_update_job_saves_category(self, fake_redis):
        from app.store import update_job

        save_job("job-1", "some text")
        update_job("job-1", status="completed", category="sci.space", confidence=0.99)
        job = get_job("job-1")
        assert job["category"] == "sci.space"

    def test_update_job_saves_confidence(self, fake_redis):
        from app.store import update_job

        save_job("job-1", "some text")
        update_job("job-1", status="completed", category="sci.space", confidence=0.99)
        job = get_job("job-1")
        assert job["confidence"] == 0.99

    def test_update_job_returns_none_for_missing_job(self, fake_redis):
        from app.store import update_job

        result = update_job("nonexistent", status="completed")
        assert result is None

    def test_delete_job_returns_true(self, fake_redis):
        from app.store import delete_job

        save_job("job-1", "some text")
        assert delete_job("job-1") is True

    def test_delete_job_removes_from_store(self, fake_redis):
        from app.store import delete_job

        save_job("job-1", "some text")
        delete_job("job-1")
        assert get_job("job-1") is None

    def test_delete_job_returns_false_for_missing_job(self, fake_redis):
        from app.store import delete_job

        assert delete_job("nonexistent") is False


# Classifier Unit Tests


class TestClassifier:
    def test_predict_returns_dict(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert isinstance(result, dict)

    def test_predict_returns_category(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert "category" in result

    def test_predict_returns_confidence(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert "confidence" in result

    def test_predict_category_is_string(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert isinstance(result["category"], str)

    def test_predict_confidence_is_float(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert isinstance(result["confidence"], float)

    def test_predict_confidence_between_0_and_1(self):
        from app.services.classifier import predict

        result = predict("NASA launched a spacecraft")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_correct_category_for_space_text(self):
        from app.services.classifier import predict

        result = predict("NASA launched a new spacecraft into orbit")
        assert result["category"] == "sci.space"

    def test_predict_correct_category_for_atheism_text(self):
        from app.services.classifier import predict

        result = predict("Atheism is a belief system that rejects religion")
        assert result["category"] == "alt.atheism"
