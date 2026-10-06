"""HTTP routes.

OpenAI-compatible:
  POST /v1/audio/transcriptions   multipart: file, model, language, prompt, response_format
  POST /v1/audio/speech           JSON: model, input, voice, speed, response_format
  GET  /v1/models
Lifecycle:
  GET  /health                    liveness, identifies the server as vox
  GET  /vox/status                pid, loaded models, memory
  GET  /vox/jobs/{id}             progress of a request (the CLI polls this)
  POST /vox/shutdown              exit now

Only /v1 requests count as activity for the idle timer, so `vox status`
never keeps the server alive.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from vox import __version__, formats, paths
from vox.config import load_config
from vox.engines import engine_class
from vox.errors import MissingError, VoxError
from vox.models import InstalledModel, installed_models, no_model_error, pick_installed
from vox.server.manager import Job, Jobs, ModelManager

log = logging.getLogger("vox.server")

ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
MAX_INPUT_CHARS = 1_000_000


@dataclass
class ServerState:
    port: int
    persistent: bool
    idle_timeout: int
    manager: ModelManager = field(default_factory=ModelManager)
    jobs: Jobs = field(default_factory=Jobs)
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.monotonic)
    in_flight: int = 0
    request_exit: Callable[[str], None] = lambda reason: None

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_for(self) -> float:
        return 0.0 if self.in_flight else time.monotonic() - self.last_activity


class ApiError(Exception):
    def __init__(self, status: int, message: str, hint: str | None = None, code: str | None = None, exit_code: int = 1):
        super().__init__(message)
        self.status, self.message, self.hint, self.code, self.exit_code = status, message, hint, code, exit_code


def error_response(status: int, message: str, hint: str | None = None, code: str | None = None, exit_code: int = 1) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    body = {"error": {"message": message, "type": kind, "param": None, "code": code, "hint": hint, "vox_exit_code": exit_code}}
    return JSONResponse(body, status_code=status)


def _resolve(model_type: str, requested: str | None) -> InstalledModel:
    model = pick_installed(model_type, (requested or "").strip() or None, load_config(), strict=False)
    if model is None:
        err = no_model_error(model_type)
        raise ApiError(400, err.message, err.hint, code="model_not_found", exit_code=2)
    return model


def _resolve_voice(model: InstalledModel, requested: str | None) -> str | None:
    voices = model.voices()
    if requested:
        names = [v.strip() for v in requested.split(",")]
        if all(n in voices for n in names) or not voices:
            return requested.strip()
        mapped = engine_class(model.engine).openai_voice(requested.strip())
        if mapped and mapped in voices:
            return mapped
        raise ApiError(400, f"Unknown voice '{requested}' for {model.id}.", "List voices with: vox voices", code="voice_not_found")
    configured = load_config().default_voice
    if configured and all(v.strip() in voices for v in configured.split(",")):
        return configured
    return model.fallback_voice()


def _rss_bytes() -> int | None:
    from vox.system import rss_bytes

    return rss_bytes(os.getpid())


# ---------------------------------------------------------------- handlers


async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "service": "vox", "version": __version__, "pid": os.getpid()})


async def list_models(request: Request) -> Response:
    data = []
    for m in installed_models():
        try:
            created = int(datetime.fromisoformat(m.installed_at).timestamp())
        except ValueError:
            created = 0
        data.append({"id": m.id, "object": "model", "created": created, "owned_by": "vox", "type": m.type, "engine": m.engine})
    return JSONResponse({"object": "list", "data": data})


def _transcribe_job(state: ServerState, model: InstalledModel, src: Path, workdir: Path, form: dict, job: Job):
    from vox.audio import load_for_whisper

    samples = load_for_whisper(src, workdir)
    with state.manager.use(model, job) as engine:
        language = None
        if form["language"]:
            try:
                language = engine.normalize_language(form["language"])
            except ValueError as exc:
                raise ApiError(400, f"Unsupported language: {form['language']} ({exc}).", "Use a code such as en, fr or de.") from None
        job.update(0, max(1.0, len(samples) / 160))
        result = engine.transcribe(
            samples,
            language=language,
            prompt=form["prompt"],
            word_timestamps=form["words"],
            progress=job.update,
        )
        return result, engine.language_name(result.language)


async def transcriptions(request: Request) -> Response:
    state: ServerState = request.app.state.vox
    async with request.form(max_files=1, max_fields=100) as form:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise ApiError(400, "Missing 'file' in the multipart form.", "Send the audio as a multipart field named file.")
        response_format = str(form.get("response_format") or "json").lower()
        if response_format not in formats.RESPONSE_FORMATS:
            raise ApiError(400, f"Unsupported response_format '{response_format}'.", "Use one of: " + ", ".join(formats.RESPONSE_FORMATS))
        granularities = form.getlist("timestamp_granularities[]") + form.getlist("timestamp_granularities")
        options = {
            "language": str(form.get("language") or "").strip() or None,
            "prompt": str(form.get("prompt") or "").strip() or None,
            "words": "word" in granularities and response_format == "verbose_json",
        }
        model = _resolve("stt", form.get("model"))
        job = state.jobs.create(request.headers.get("x-vox-job"), "transcribe", model.id)
        ok = False
        try:
            with tempfile.TemporaryDirectory(prefix="vox-") as tmp:
                suffix = Path(upload.filename or "").suffix[:16]
                src = Path(tmp) / f"upload{suffix}"
                with open(src, "wb") as fh:
                    while chunk := await upload.read(1 << 20):
                        fh.write(chunk)
                log.info("transcribe %s (%s, %d bytes) with %s", upload.filename, response_format, src.stat().st_size, model.id)
                result, language_name = await run_in_threadpool(_transcribe_job, state, model, src, Path(tmp), options, job)
            ok = True
        finally:
            state.jobs.finish(job, ok)

    body = formats.render(result, response_format, language_name, options["words"])
    headers = {
        "X-Vox-Model": model.id,
        "X-Vox-Language": result.language or "",
        "X-Vox-Duration": f"{result.duration:.3f}",
    }
    return Response(body, media_type=formats.MEDIA_TYPES[response_format], headers=headers)


def _speech_job(state: ServerState, model: InstalledModel, text: str, voice: str | None, speed: float, job: Job):
    import numpy as np

    from vox.audio import silence
    from vox.text import chunk_text

    chunks = chunk_text(text)
    if not chunks:
        raise ApiError(400, "Nothing to say: 'input' has no speakable text.", "Send words, not only punctuation or markup.")
    with state.manager.use(model, job) as engine:
        job.update(0, len(chunks))
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(engine.synthesize(chunk.text, voice=voice, speed=speed))
            if chunk.pause_after:
                parts.append(silence(chunk.pause_after / speed, engine.sample_rate))
            job.update(i, len(chunks))
        return np.concatenate(parts), engine.sample_rate


async def speech(request: Request) -> Response:
    from vox.audio import SPEECH_FORMATS, encode

    state: ServerState = request.app.state.vox
    try:
        body = await request.json()
    except ValueError:
        raise ApiError(400, "The request body must be JSON.", 'For example: {"input": "Hello", "voice": "af_heart"}') from None
    if not isinstance(body, dict):
        raise ApiError(400, "The request body must be a JSON object.", 'For example: {"input": "Hello", "voice": "af_heart"}')
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "'input' is required and must be non-empty text.", 'For example: {"input": "Hello"}')
    if len(text) > MAX_INPUT_CHARS:
        raise ApiError(400, f"'input' is longer than {MAX_INPUT_CHARS} characters.", "Split the text into several requests.")
    fmt = str(body.get("response_format") or "mp3").lower()
    if fmt not in SPEECH_FORMATS:
        raise ApiError(400, f"Unsupported response_format '{fmt}'.", "Use one of: " + ", ".join(SPEECH_FORMATS))
    try:
        speed = float(body.get("speed", 1.0) or 1.0)
    except (TypeError, ValueError):
        raise ApiError(400, "'speed' must be a number.", "1.0 is normal speed.") from None
    if not 0.25 <= speed <= 4.0:
        raise ApiError(400, "'speed' must be between 0.25 and 4.0.", "1.0 is normal speed.")
    model = _resolve("tts", body.get("model"))
    voice = _resolve_voice(model, str(body["voice"]) if body.get("voice") else None)

    job = state.jobs.create(request.headers.get("x-vox-job"), "speak", model.id)
    ok = False
    try:
        log.info("speak %d chars with %s, voice %s, %s", len(text), model.id, voice, fmt)
        samples, sample_rate = await run_in_threadpool(_speech_job, state, model, text, voice, speed, job)
        data = await run_in_threadpool(encode, samples, sample_rate, fmt)
        ok = True
    finally:
        state.jobs.finish(job, ok)
    headers = {
        "X-Vox-Model": model.id,
        "X-Vox-Voice": voice or "",
        "X-Vox-Duration": f"{len(samples) / sample_rate:.3f}",
    }
    return Response(data, media_type=SPEECH_FORMATS[fmt], headers=headers)


async def status(request: Request) -> Response:
    state: ServerState = request.app.state.vox
    idle = state.idle_for()
    exits_in = None
    if not state.persistent and not state.in_flight:
        exits_in = max(0.0, state.idle_timeout - idle)
    rss = await run_in_threadpool(_rss_bytes)
    return JSONResponse(
        {
            "service": "vox",
            "version": __version__,
            "pid": os.getpid(),
            "port": state.port,
            "persistent": state.persistent,
            "idle_timeout": state.idle_timeout,
            "uptime": round(time.time() - state.started_at, 1),
            "idle_for": round(idle, 1),
            "exits_in": None if exits_in is None else round(exits_in, 1),
            "in_flight": state.in_flight,
            "loaded": state.manager.loaded(),
            "rss_bytes": rss,
            "log": str(paths.log_file()),
        }
    )


async def job_status(request: Request) -> Response:
    state: ServerState = request.app.state.vox
    job = state.jobs.get(request.path_params["job_id"])
    if job is None:
        return error_response(404, "No such job.")
    return JSONResponse(job)


async def shutdown(request: Request) -> Response:
    state: ServerState = request.app.state.vox
    if request.headers.get("x-vox-client") is None:
        # A custom header forces a CORS preflight, so web pages cannot stop the server.
        return error_response(403, "Use `vox stop` to stop the server.")
    return JSONResponse({"status": "stopping"}, background=BackgroundTask(state.request_exit, "vox stop"))


# ------------------------------------------------------------- middleware


class ActivityMiddleware:
    """Tracks in-flight /v1 requests for the idle timer and rejects foreign Host headers."""

    def __init__(self, app, state: ServerState):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        host = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"host"), "")
        hostname = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        if host and hostname not in ALLOWED_HOSTS:
            # Blocks DNS rebinding: a web page cannot reach the server through its own domain.
            response = error_response(403, "Requests must be addressed to 127.0.0.1 or localhost.")
            return await response(scope, receive, send)
        if not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        self.state.in_flight += 1
        self.state.touch()
        try:
            await self.app(scope, receive, send)
        finally:
            self.state.in_flight -= 1
            self.state.touch()


async def _handle_api_error(request: Request, exc: ApiError) -> Response:
    return error_response(exc.status, exc.message, exc.hint, exc.code, exc.exit_code)


async def _handle_vox_error(request: Request, exc: VoxError) -> Response:
    code = "missing_dependency" if isinstance(exc, MissingError) else None
    return error_response(400, exc.message, exc.hint, code, exc.exit_code)


async def _handle_unexpected(request: Request, exc: Exception) -> Response:
    # Starlette re-raises after this response is sent, so uvicorn logs the traceback.
    return error_response(
        500,
        f"Internal error: {type(exc).__name__}: {exc}",
        f"See the server log: {paths.pretty(paths.log_file())}",
    )


async def _watch_idle(state: ServerState) -> None:
    interval = min(1.0, state.idle_timeout / 4)
    while True:
        await asyncio.sleep(interval)
        if state.in_flight == 0 and time.monotonic() - state.last_activity >= state.idle_timeout:
            state.request_exit(f"idle for {state.idle_timeout}s")
            return


def create_app(state: ServerState) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = None if state.persistent else asyncio.create_task(_watch_idle(state))
        try:
            yield
        finally:
            if task:
                task.cancel()

    routes = [
        Route("/health", health),
        Route("/v1/models", list_models),
        Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
        Route("/v1/audio/speech", speech, methods=["POST"]),
        Route("/vox/status", status),
        Route("/vox/jobs/{job_id}", job_status),
        Route("/vox/shutdown", shutdown, methods=["POST"]),
    ]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={ApiError: _handle_api_error, VoxError: _handle_vox_error, Exception: _handle_unexpected},
    )
    app.state.vox = state
    app.add_middleware(ActivityMiddleware, state=state)
    return app
