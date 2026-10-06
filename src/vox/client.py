"""Finding or starting the server, and talking to it over HTTP."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from vox import paths
from vox.config import Config
from vox.errors import ServerError, VoxError

HOST = "127.0.0.1"
START_TIMEOUT = 60.0
LOG_ROTATE_BYTES = 5_000_000


@dataclass
class ServerInfo:
    pid: int
    port: int
    persistent: bool

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"


def _pid_alive(pid: int) -> bool:
    try:
        # Reap the server if this process spawned it and it already exited,
        # so it is not mistaken for alive as a zombie.
        if os.waitpid(pid, os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def health(port: int, timeout: float = 1.0) -> dict | None:
    """The /health payload if a vox server answers on the port, else None."""
    try:
        response = httpx.get(f"http://{HOST}:{port}/health", timeout=timeout)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("service") == "vox" else None


def read_state() -> dict | None:
    try:
        data = json.loads(paths.state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("pid"), int) else None


def find_server() -> ServerInfo | None:
    state = read_state()
    if not state or not _pid_alive(state["pid"]):
        return None
    port = int(state.get("port", 0))
    info = health(port)
    if info and info.get("pid") == state["pid"]:
        return ServerInfo(pid=state["pid"], port=port, persistent=bool(state.get("persistent")))
    return None


@contextmanager
def _spawn_lock() -> Iterator[None]:
    paths.run_dir().mkdir(parents=True, exist_ok=True)
    fd = os.open(paths.spawn_lock_file(), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _rotate_log() -> None:
    log = paths.log_file()
    try:
        if log.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log, log.with_suffix(".log.1"))
    except OSError:
        pass


def _log_tail(lines: int = 3) -> str:
    try:
        text = paths.log_file().read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(line.strip() for line in text[-lines:])


def spawn_server(cfg: Config) -> ServerInfo:
    """Start an on-demand server in the background and wait until it answers."""
    _rotate_log()
    if health(cfg.port):
        # Another vox (perhaps with a different cache folder) owns the port.
        raise ServerError(
            f"Another vox server is already using port {cfg.port}.",
            f"Stop it, or pick another port: vox config set port {cfg.port + 1}",
        )
    cmd = [
        sys.executable, "-m", "vox", "serve",
        "--port", str(cfg.port), "--idle-timeout", str(cfg.idle_timeout), "--spawned",
    ]  # fmt: skip
    with open(paths.log_file(), "ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
            cwd="/",
        )
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            detail = _log_tail()
            raise ServerError(
                "The vox server failed to start" + (f": {detail}" if detail else "."),
                f"See the log: {paths.pretty(paths.log_file())}",
            )
        info = health(cfg.port, timeout=0.5)
        if info and info.get("pid") == proc.pid:
            return ServerInfo(pid=proc.pid, port=cfg.port, persistent=False)
        time.sleep(0.1)
    raise ServerError(
        f"The vox server did not start within {START_TIMEOUT:.0f} seconds.",
        f"See the log: {paths.pretty(paths.log_file())}",
    )


def ensure_server(cfg: Config, on_spawn: Callable[[], None] | None = None) -> ServerInfo:
    with _spawn_lock():
        info = find_server()
        if info:
            return info
        if on_spawn:
            on_spawn()
        return spawn_server(cfg)


def stop_server(timeout: float = 10.0) -> ServerInfo | None:
    """Stop the running server. Returns its info, or None if none was running."""
    state = read_state()
    if not state or not _pid_alive(state["pid"]):
        return None
    pid, port = state["pid"], int(state.get("port", 0))
    info = ServerInfo(pid=pid, port=port, persistent=bool(state.get("persistent")))
    try:
        httpx.post(f"http://{HOST}:{port}/vox/shutdown", headers={"X-Vox-Client": "cli"}, timeout=2)
    except httpx.HTTPError:
        pass
    if _wait_exit(pid, timeout):
        return info
    if health(port) is None and not _is_vox_process(pid):
        return info  # pid was reused by something else
    os.kill(pid, signal.SIGTERM)
    if _wait_exit(pid, 5):
        return info
    os.kill(pid, signal.SIGKILL)
    _wait_exit(pid, 2)
    return info


def _is_vox_process(pid: int) -> bool:
    from vox.system import process_command

    command = process_command(pid)
    return "vox" in command and "serve" in command


def _wait_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


# ------------------------------------------------------------------ API


def raise_for_error(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        error = response.json().get("error", {})
        message = error.get("message") or f"HTTP {response.status_code}"
        hint = error.get("hint")
        exit_code = int(error.get("vox_exit_code") or 1)
    except (ValueError, AttributeError, TypeError):
        message, hint, exit_code = f"The server answered HTTP {response.status_code}.", None, 1
    raise VoxError(message, hint, exit_code=exit_code)


JobCallback = Callable[[dict], None]


class _ServerGone(Exception):
    """Nothing is listening: the server exited before we connected."""


def _lost_connection(kind: str) -> ServerError:
    return ServerError(
        f"Lost the connection to the vox server ({kind}).",
        f"See the log: {paths.pretty(paths.log_file())}",
    )


class Client:
    def __init__(self, server: ServerInfo, reconnect: Callable[[], ServerInfo] | None = None):
        """reconnect() is called once if the server is gone (it may have just
        exited on idle between our health check and the request)."""
        self.server = server
        self.base = server.url
        self._reconnect = reconnect

    def _poll(self, job_id: str, on_job: JobCallback, stop: threading.Event) -> None:
        with httpx.Client(base_url=self.base, timeout=2) as http:
            while not stop.wait(0.25):
                try:
                    response = http.get(f"/vox/jobs/{job_id}")
                except httpx.HTTPError:
                    continue
                if response.status_code == 200:
                    on_job(response.json())

    def _request(
        self,
        method: str,
        url: str,
        on_job: JobCallback | None,
        make_kwargs: Callable[[], dict],
        out: Path | None = None,
    ) -> httpx.Response:
        try:
            with make_kwargs() as kwargs:
                return self._send(method, url, on_job, out, kwargs)
        except _ServerGone:
            if not self._reconnect:
                raise _lost_connection("ConnectError") from None
            self.server = self._reconnect()
            self.base = self.server.url
            try:
                with make_kwargs() as kwargs:
                    return self._send(method, url, on_job, out, kwargs)
            except _ServerGone:
                raise _lost_connection("ConnectError") from None

    def _send(self, method: str, url: str, on_job: JobCallback | None, out: Path | None, kwargs: dict) -> httpx.Response:
        job_id = uuid.uuid4().hex
        headers = {"X-Vox-Job": job_id, "X-Vox-Client": "cli"}
        stop = threading.Event()
        poller = None
        if on_job:
            poller = threading.Thread(target=self._poll, args=(job_id, on_job, stop), daemon=True)
            poller.start()
        try:
            timeout = httpx.Timeout(10.0, read=None)
            with httpx.Client(base_url=self.base, timeout=timeout) as http:
                with http.stream(method, url, headers=headers, **kwargs) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise_for_error(response)
                    if out is None:
                        response.read()
                    else:
                        with open(out, "wb") as fh:
                            for chunk in response.iter_bytes(1 << 16):
                                fh.write(chunk)
                    return response
        except httpx.ConnectError:
            raise _ServerGone() from None
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as exc:
            raise _lost_connection(type(exc).__name__) from None
        finally:
            stop.set()
            if poller:
                poller.join(timeout=1)

    def transcribe(
        self,
        wav: Path,
        *,
        model: str,
        response_format: str,
        language: str | None,
        on_job: JobCallback | None = None,
    ) -> httpx.Response:
        data = {"model": model, "response_format": response_format}
        if language:
            data["language"] = language

        @contextmanager
        def make_kwargs():
            with open(wav, "rb") as fh:
                yield {"data": data, "files": {"file": (wav.name, fh, "audio/wav")}}

        return self._request("POST", "/v1/audio/transcriptions", on_job, make_kwargs)

    def speak(
        self,
        text: str,
        *,
        model: str,
        voice: str | None,
        speed: float,
        response_format: str,
        out: Path,
        on_job: JobCallback | None = None,
    ) -> httpx.Response:
        payload = {"model": model, "input": text, "speed": speed, "response_format": response_format}
        if voice:
            payload["voice"] = voice

        @contextmanager
        def make_kwargs():
            yield {"json": payload}

        return self._request("POST", "/v1/audio/speech", on_job, make_kwargs, out=out)

    def status(self) -> dict:
        response = httpx.get(f"{self.base}/vox/status", timeout=5)
        raise_for_error(response)
        return response.json()
