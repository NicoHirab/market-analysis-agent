import uuid
from datetime import UTC, datetime
from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job:
    def __init__(self, request: dict) -> None:
        self.id = uuid.uuid4().hex
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now(UTC)
        self.request = request
        self.result: dict | None = None
        self.meta: dict | None = None


class JobRegistry:
    """In-memory, single-process job store.

    The Redis/Postgres upgrade path is discussed in the README (theory, step 4).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, request: dict) -> Job:
        job = Job(request)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
