import asyncio
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
        self.events: list[dict] = []


class JobRegistry:
    """In-memory, single-process job store.

    All mutating methods are synchronous (no await inside), so they are atomic
    on the event loop — no lock needed. The Redis/Postgres upgrade path is
    discussed in the README (theory, step 4).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create(self, request: dict) -> Job:
        job = Job(request)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def publish(self, job_id: str, event: dict) -> None:
        job = self._jobs[job_id]
        job.events.append(event)
        for queue in self._subscribers.get(job_id, []):
            queue.put_nowait(event)

    def subscribe(self, job_id: str) -> tuple[list[dict], asyncio.Queue]:
        job = self._jobs[job_id]
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(queue)
        return list(job.events), queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)
