"""`vox uninstall`: remove everything vox put on this Mac.

Order matters: the launch agent goes first (it would restart the server),
then the server, then files, and the program itself last.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vox import desktop, finder, models, paths

LAUNCH_AGENT_LABEL = "local.vox.server"  # the label docs/launchd.md uses
SYSTEMD_UNIT = "vox.service"  # the unit name docs/linux.md uses


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def systemd_unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".config"
    return root / "systemd" / "user" / SYSTEMD_UNIT


@dataclass
class Item:
    what: str
    path: Path | None
    size: int = 0


def _espeak_links() -> list[Path]:
    links = [paths.cache_dir() / "espeak-ng-data", Path(tempfile.gettempdir()) / f"vox-espeak-{os.getuid()}"]
    return [link for link in links if link.is_symlink()]


def _model_folders() -> list[Path]:
    root = paths.models_dir()
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _model_name(folder: Path) -> str:
    model = models.get_installed_at(folder)
    return model.id if model else f"{folder.name} (partial download)"


def program_command() -> list[str] | None:
    """The command that removes the vox program, based on how it was installed
    (uv tool, pipx or Homebrew).

    None when vox runs from somewhere else (for example a development checkout).
    """
    prefix = Path(sys.prefix)
    parts = prefix.parts
    if len(parts) >= 3 and parts[-3:-1] == ("uv", "tools"):  # ~/.local/share/uv/tools/vox-cli
        uv = shutil.which("uv") or next(
            (p for p in ("/opt/homebrew/bin/uv", "/usr/local/bin/uv", str(Path.home() / ".local/bin/uv")) if os.access(p, os.X_OK)),
            "uv",
        )
        return [uv, "tool", "uninstall", prefix.name]
    if len(parts) >= 3 and parts[-2] == "venvs" and any("pipx" in p for p in parts):  # ~/.local/pipx/venvs/vox-cli
        return [shutil.which("pipx") or "pipx", "uninstall", prefix.name]
    if len(parts) >= 5 and parts[-4] == "Cellar" and parts[-1] == "libexec":  # <brew>/Cellar/vox/0.2.0/libexec
        brew = prefix.parents[3] / "bin" / "brew"
        return [str(brew) if brew.exists() else (shutil.which("brew") or "brew"), "uninstall", parts[-3]]
    return None


def plan(keep_models: bool) -> tuple[list[Item], list[Item]]:
    """(items to remove, items kept)."""
    remove: list[Item] = []
    keep: list[Item] = []
    if launch_agent_path().exists():
        remove.append(Item("Launch agent (always-on server)", launch_agent_path()))
    if systemd_unit_path().exists():
        remove.append(Item("systemd user service (always-on server)", systemd_unit_path()))
    for action in finder.QUICK_ACTIONS:
        if finder.workflow_path(action).exists():
            remove.append(Item(f'Finder Quick Action "{action.title}"', finder.workflow_path(action)))
    for path in desktop.installed():
        if path != desktop.helper_path():  # the helper goes with the settings folder
            remove.append(Item("File-manager action", path))
    if paths.config_dir().exists():
        remove.append(Item("Settings and helper scripts", paths.config_dir(), models.dir_size(paths.config_dir())))

    folders = _model_folders()
    if folders:
        size = sum(models.dir_size(f) for f in folders)
        names = ", ".join(_model_name(f) for f in folders)
        item = Item(f"Models: {names}", paths.models_dir(), size)
        (keep if keep_models else remove).append(item)
    if paths.run_dir().exists():
        remove.append(Item("Server state and logs", paths.run_dir(), models.dir_size(paths.run_dir())))
    for link in _espeak_links():
        remove.append(Item("espeak data link", link))
    return remove, keep


def remove_files(keep_models: bool) -> None:
    from vox import client

    agent = launch_agent_path()
    if agent.exists():
        launchctl = shutil.which("launchctl")
        if launchctl:
            subprocess.run([launchctl, "bootout", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"], capture_output=True)
        agent.unlink()
    unit = systemd_unit_path()
    if unit.exists():
        _systemctl("disable", "--now", SYSTEMD_UNIT)
        unit.unlink()
        _systemctl("daemon-reload")
    client.stop_server()

    finder.remove_quick_actions()
    desktop.remove()
    shutil.rmtree(paths.config_dir(), ignore_errors=True)
    if not keep_models:
        for folder in _model_folders():
            shutil.rmtree(folder, ignore_errors=True)
        _rmdir_if_empty(paths.models_dir())
    shutil.rmtree(paths.run_dir(), ignore_errors=True)
    for link in _espeak_links():
        link.unlink(missing_ok=True)
    _rmdir_if_empty(paths.cache_dir())


def _systemctl(*args: str) -> None:
    systemctl = shutil.which("systemctl")
    if systemctl:
        subprocess.run([systemctl, "--user", *args], capture_output=True)


def _rmdir_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass  # missing, or still holds kept models
