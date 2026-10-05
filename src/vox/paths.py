"""Where vox keeps its files.

Config lives in ~/.config/vox, models and runtime files in ~/.cache/vox.
VOX_CONFIG_DIR, VOX_CACHE_DIR and VOX_MODELS_DIR override these (used by
the tests).
"""

from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("VOX_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / "vox"


def cache_dir() -> Path:
    override = os.environ.get("VOX_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "vox"


def config_file() -> Path:
    return config_dir() / "config.toml"


def models_dir() -> Path:
    override = os.environ.get("VOX_MODELS_DIR")
    return Path(override) if override else cache_dir() / "models"


def run_dir() -> Path:
    return cache_dir() / "run"


def log_file() -> Path:
    return run_dir() / "server.log"


def state_file() -> Path:
    """JSON file describing the running server (pid, port, mode)."""
    return run_dir() / "server.json"


def server_lock_file() -> Path:
    """Held with flock by the running server for its whole life."""
    return run_dir() / "server.lock"


def spawn_lock_file() -> Path:
    """Held briefly by clients so two commands never spawn two servers."""
    return run_dir() / "spawn.lock"


def pretty(path: Path | str) -> str:
    """Show a path with ~ for the home folder."""
    path = str(path)
    home = str(Path.home())
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path
