"""Running the server process: single-instance lock, state file, idle exit."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import socket
import sys
import time

from vox import __version__, paths
from vox.errors import UserError

log = logging.getLogger("vox.server")
HOST = "127.0.0.1"


def _setup_logging(spawned: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Spawned servers write stderr straight into server.log. A foreground
    # `vox serve` logs to the terminal and to server.log, so `vox status`
    # always points at a useful file.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)
    if not spawned:
        file_handler = logging.FileHandler(paths.log_file(), encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    logging.captureWarnings(True)


def _bind(port: int) -> socket.socket:
    # Refuse to share the port with anything already listening on it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((HOST, port)) == 0:
            raise _port_in_use(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Lets a respawned server bind while old connections sit in TIME_WAIT.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((HOST, port))
    except OSError as exc:
        sock.close()
        if exc.errno == errno.EADDRINUSE:
            raise _port_in_use(port) from None
        raise
    return sock


def _port_in_use(port: int) -> UserError:
    return UserError(
        f"Port {port} is already in use by another program.",
        f"Pick another port: vox config set port {port + 1 if port < 65535 else 8881}",
    )


def _write_state(port: int, persistent: bool, idle_timeout: int) -> None:
    data = {
        "pid": os.getpid(),
        "port": port,
        "persistent": persistent,
        "idle_timeout": idle_timeout,
        "started_at": time.time(),
        "version": __version__,
        "log": str(paths.log_file()),
    }
    tmp = paths.state_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, paths.state_file())


def _remove_state() -> None:
    try:
        data = json.loads(paths.state_file().read_text(encoding="utf-8"))
        if data.get("pid") == os.getpid():
            paths.state_file().unlink()
    except (OSError, ValueError):
        pass


def run_server(port: int, persistent: bool, idle_timeout: int, spawned: bool = False) -> None:
    """Serve until idle (or forever when persistent), then exit the process."""
    paths.run_dir().mkdir(parents=True, exist_ok=True)

    lock_fd = os.open(paths.server_lock_file(), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise UserError("A vox server is already running.", "See it with vox status, or stop it with vox stop") from None

    # The server never touches the network: models are local folders.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    sock = _bind(port)
    _setup_logging(spawned)

    import uvicorn

    from vox.server.app import ServerState, create_app

    state = ServerState(port=port, persistent=persistent, idle_timeout=idle_timeout)
    app = create_app(state)
    config = uvicorn.Config(app, log_config=None, access_log=False, lifespan="on", timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)

    def request_exit(reason: str) -> None:
        if not server.should_exit:
            log.info("exiting: %s", reason)
            server.should_exit = True

    state.request_exit = request_exit
    mode = "persistent" if persistent else f"on-demand, exits after {idle_timeout}s idle"
    log.info("vox %s serving on http://%s:%d (pid %d, %s)", __version__, HOST, port, os.getpid(), mode)
    _write_state(port, persistent, idle_timeout)
    try:
        server.run(sockets=[sock])
    finally:
        _remove_state()
        log.info("stopped")
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os.close(lock_fd)
    # Exit hard so no library thread can keep the process (and its memory) alive.
    os._exit(0)
