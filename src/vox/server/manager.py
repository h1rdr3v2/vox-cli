"""Lazy model loading, a one-at-a-time work queue, and job progress."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from vox.engines import Engine, engine_class
from vox.models import InstalledModel


@dataclass
class Job:
    id: str
    kind: str  # "transcribe" | "speak"
    model: str
    state: str = "queued"  # queued, loading, running, done, failed
    done: float = 0.0
    total: float = 0.0
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def update(self, done: float, total: float) -> None:
        self.done, self.total = float(done), float(total)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "elapsed": round((self.finished or time.time()) - self.started, 2),
        }


class Jobs:
    """Progress of in-flight requests, readable by the CLI while it waits."""

    KEEP_SECONDS = 120
    MAX_JOBS = 200

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str | None, kind: str, model: str) -> Job:
        job_id = (job_id or "").strip()[:64] or uuid.uuid4().hex
        job = Job(id=job_id, kind=kind, model=model)
        with self._lock:
            self._prune()
            self._jobs[job_id] = job
        return job

    def finish(self, job: Job, ok: bool = True) -> None:
        job.state = "done" if ok else "failed"
        job.finished = time.time()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job else None

    def active(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.finished is None)

    def _prune(self) -> None:
        now = time.time()
        for key in [k for k, j in self._jobs.items() if j.finished and now - j.finished > self.KEEP_SECONDS]:
            del self._jobs[key]
        while len(self._jobs) >= self.MAX_JOBS:
            del self._jobs[next(iter(self._jobs))]


@dataclass
class Loaded:
    model: InstalledModel
    engine: Engine
    loaded_at: float
    load_seconds: float
    last_used: float
    uses: int = 0


class ModelManager:
    """Loads a model the first time it is needed and keeps it while the server lives.

    One lock serializes all loading and inference, so requests queue up,
    a model is never loaded twice, and MLX is only used from one thread
    at a time. Memory is given back by the server exiting, not by unloading.
    """

    def __init__(self) -> None:
        self._work = threading.Lock()
        self._loaded: dict[str, Loaded] = {}
        self._meta = threading.Lock()

    @contextmanager
    def use(self, model: InstalledModel, job: Job | None = None) -> Iterator[Engine]:
        if job:
            job.state = "queued"
        with self._work:
            with self._meta:
                entry = self._loaded.get(model.id)
            if entry is None or entry.model.revision != model.revision or entry.model.path != model.path:
                if job:
                    job.state = "loading"
                start = time.time()
                engine = engine_class(model.engine)(model.path)
                engine.load()
                entry = Loaded(model, engine, loaded_at=time.time(), load_seconds=time.time() - start, last_used=time.time())
                with self._meta:
                    self._loaded[model.id] = entry
            if job:
                job.state = "running"
            entry.uses += 1
            try:
                yield entry.engine
            finally:
                entry.last_used = time.time()

    def loaded(self) -> list[dict]:
        with self._meta:
            entries = list(self._loaded.values())
        return [
            {
                "id": e.model.id,
                "type": e.model.type,
                "engine": e.model.engine,
                "loaded_at": e.loaded_at,
                "load_seconds": round(e.load_seconds, 2),
                "last_used": e.last_used,
                "requests": e.uses,
            }
            for e in entries
        ]
