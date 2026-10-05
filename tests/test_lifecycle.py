"""The on-demand server as a real background process: spawn, idle exit, stop."""

import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

from vox import client, paths
from vox.config import Config
from vox.errors import ServerError


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gone(pid: int) -> bool:
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)  # reap our own child if it exited
        if reaped:
            return True
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gone(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def cfg():
    config = Config(port=free_port(), idle_timeout=2)
    yield config
    client.stop_server(timeout=3)


def test_server_exits_after_idle_timeout(cfg):
    info = client.ensure_server(cfg)
    assert client.find_server().pid == info.pid
    assert paths.state_file().exists()

    # Requests keep it alive past the timeout...
    for _ in range(3):
        time.sleep(1.0)
        assert httpx.get(f"{info.url}/v1/models").status_code == 200
    assert not gone(info.pid)

    # ...monitoring does not, and then the process exits by itself.
    httpx.get(f"{info.url}/vox/status")
    assert wait_gone(info.pid, timeout=10), "server should exit when idle"
    assert not paths.state_file().exists()
    assert client.find_server() is None
    assert "exiting: idle for 2s" in paths.log_file().read_text()


def test_stop(cfg):
    info = client.ensure_server(cfg)
    stopped = client.stop_server()
    assert stopped.pid == info.pid
    assert wait_gone(info.pid, timeout=5)
    assert client.stop_server() is None


def test_concurrent_clients_share_one_server(cfg):
    pids = []

    def worker():
        pids.append(client.ensure_server(cfg).pid)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(pids)) == 1


def test_second_serve_is_refused(cfg):
    client.ensure_server(cfg)
    proc = subprocess.run(
        [sys.executable, "-m", "vox", "serve", "--port", str(free_port())],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "already running" in proc.stderr


def test_port_taken_by_another_program(cfg):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", cfg.port))
        blocker.listen()
        with pytest.raises(ServerError, match="already in use"):
            client.ensure_server(cfg)
