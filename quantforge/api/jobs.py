"""Async job queue — backed by Redis if available, in-memory otherwise.

Long-running backtests + ML trainings shouldn't tie up a request. The client
POSTs → gets a job_id → polls GET /v1/jobs/{id} until status=='completed'.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from quantforge.api.cache import cache_backend

_log = logging.getLogger("quantforge.api.jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    owner: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, include_result: bool = True) -> dict:
        out = {
            "id": self.id, "kind": self.kind, "status": self.status.value,
            "progress": self.progress, "created_at": self.created_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "error": self.error, "owner": self.owner,
        }
        if include_result and self.status == JobStatus.COMPLETED:
            out["result"] = self.result
        return out


class JobManager:
    def __init__(self, max_workers: int = 4, max_queued: int = 100):
        self._jobs: dict[str, Job] = {}
        self._owner_counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_queued = max_queued

    def _persist(self, job: Job) -> None:
        cache = cache_backend()
        try:
            cache.set(f"qf:job:{job.id}", json.dumps(job.to_dict(), default=str).encode(),
                       ttl_seconds=3600 * 24)
        except Exception as e:
            # Persistence is best-effort: if Redis is down, in-memory state
            # still serves the local replica. But silent failure makes
            # on-call debugging impossible — log a single line per failure
            # so an SRE can grep for `qf:job persist` in Render's logs.
            _log.warning("qf:job persist failed for %s: %s", job.id, e)

    def _load(self, job_id: str) -> dict | None:
        cache = cache_backend()
        try:
            raw = cache.get(f"qf:job:{job_id}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            _log.warning("qf:job load failed for %s: %s", job_id, e)
        return None

    def submit(self, kind: str, func: Callable[[Job], Any], params: dict[str, Any],
                owner: str | None = None) -> Job:
        with self._lock:
            # Crude per-owner queue cap — anti-abuse
            if owner:
                in_flight = [j for j in self._jobs.values()
                              if j.owner == owner and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)]
                if len(in_flight) >= 10:
                    raise RuntimeError("too many in-flight jobs for this owner")
            if sum(1 for j in self._jobs.values()
                    if j.status == JobStatus.QUEUED) >= self._max_queued:
                raise RuntimeError("job queue full")

            job = Job(id=uuid.uuid4().hex[:16], kind=kind, params=params, owner=owner)
            self._jobs[job.id] = job
            self._persist(job)

        def _wrapped():
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._persist(job)
            # Field ordering matters: gc() reads `status` and
            # `completed_at` without per-job synchronisation. If we wrote
            # status=COMPLETED before completed_at, gc could see
            # status=COMPLETED with completed_at still None and treat the
            # job as ancient (timestamp 0), evicting it instantly. Always
            # set completed_at *before* the terminal status.
            try:
                result = func(job)
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.result = None
                job.completed_at = time.time()
                job.status = JobStatus.FAILED
                self._persist(job)
                return
            if job._cancel.is_set():
                job.completed_at = time.time()
                job.status = JobStatus.CANCELLED
            else:
                job.result = result
                job.progress = 1.0
                job.completed_at = time.time()
                job.status = JobStatus.COMPLETED
            self._persist(job)

        self._executor.submit(_wrapped)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # fall back to cache
        d = self._load(job_id)
        if d is None:
            return None
        # reconstruct a lightweight Job object (no cancel handle for foreign jobs)
        return Job(
            id=d["id"], kind=d["kind"], status=JobStatus(d["status"]),
            progress=float(d.get("progress", 0)),
            created_at=float(d.get("created_at", 0) or 0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            result=d.get("result"),
            error=d.get("error"),
            owner=d.get("owner"),
        )

    def cancel(self, job_id: str, owner: str | None = None) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if owner is not None and job.owner is not None and job.owner != owner:
                return False
            if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                return False
            job._cancel.set()
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self._persist(job)
            return True

    def list_by_owner(self, owner: str, limit: int = 50) -> list[Job]:
        with self._lock:
            owned = [j for j in self._jobs.values() if j.owner == owner]
        owned.sort(key=lambda j: j.created_at, reverse=True)
        return owned[:limit]

    def gc(self, older_than_s: int = 3600) -> int:
        cutoff = time.time() - older_than_s
        removed = 0
        with self._lock:
            for jid in list(self._jobs.keys()):
                job = self._jobs[jid]
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) \
                   and (job.completed_at or 0) < cutoff:
                    self._jobs.pop(jid, None)
                    removed += 1
        return removed


# Singleton manager — guarded by a module-level lock so concurrent
# initial requests can't both construct a JobManager and abandon one
# (which would leak the abandoned executor's threads).
_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:  # re-check inside the lock (DCL pattern)
                _manager = JobManager()
    return _manager


def reset_manager() -> None:
    """For tests."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager._executor.shutdown(wait=False, cancel_futures=True)
        _manager = None
