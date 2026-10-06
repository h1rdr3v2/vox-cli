"""`vox setup files`: right-click actions for Linux file managers.

GNOME Files (Nautilus) and Caja run executable scripts from a scripts
folder; Nemo reads .nemo_action files; Dolphin reads .desktop service
menus. All of them call one helper script with the selected files.
"""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from vox import paths
from vox.finder import vox_command
from vox.media import find_ffmpeg


@dataclass(frozen=True)
class Action:
    title: str
    mode: str  # argument to the helper script
    slug: str
    mimetypes: tuple[str, ...]
    icon: str


ACTIONS = (
    Action("Transcribe with vox", "transcribe", "vox-transcribe", ("audio/*", "video/*"), "audio-x-generic"),
    Action("Speak with vox", "speak", "vox-speak", ("text/plain", "text/markdown"), "text-x-generic"),
)
FILE_MANAGERS = {
    "nautilus": "GNOME Files",
    "nemo": "Nemo",
    "caja": "Caja",
    "dolphin": "Dolphin",
}


def _data_home() -> Path:
    value = os.environ.get("XDG_DATA_HOME", "")
    return Path(value) if os.path.isabs(value) else Path.home() / ".local" / "share"


def _config_home() -> Path:
    value = os.environ.get("XDG_CONFIG_HOME", "")
    return Path(value) if os.path.isabs(value) else Path.home() / ".config"


def helper_path() -> Path:
    return paths.config_dir() / "files" / "vox-files.sh"


def action_paths(manager: str, action: Action) -> Path:
    if manager == "nautilus":
        return _data_home() / "nautilus" / "scripts" / action.title
    if manager == "caja":
        return _config_home() / "caja" / "scripts" / action.title
    if manager == "nemo":
        return _data_home() / "nemo" / "actions" / f"{action.slug}.nemo_action"
    if manager == "dolphin":
        return _data_home() / "kio" / "servicemenus" / f"{action.slug}.desktop"
    raise KeyError(manager)


def detect() -> list[str]:
    """File managers installed on this machine."""
    return [name for name in FILE_MANAGERS if shutil.which(name)]


def write_helper() -> Path:
    template = resources.files("vox").joinpath("resources/vox-files.sh").read_text(encoding="utf-8")
    command = vox_command()
    ffmpeg = find_ffmpeg()
    path_dirs = []
    for directory in [Path(ffmpeg).parent if ffmpeg else None, Path(command[0]).parent]:
        if directory and str(directory) not in path_dirs:
            path_dirs.append(str(directory))
    script = template.replace("__VOX__", shlex.join(command))
    script = script.replace("__PATH__", ":".join(path_dirs) or "/usr/local/bin")
    target = helper_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    target.chmod(0o755)
    return target


def _desktop_quote(arg: str) -> str:
    """Quote an argument for an Exec= line in a .desktop file."""
    if not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def render(manager: str, action: Action, helper: Path) -> str:
    if manager in ("nautilus", "caja"):
        # Both pass the selected files as arguments.
        return f'#!/bin/sh\nexec {shlex.quote(str(helper))} {action.mode} "$@"\n'
    if manager == "nemo":
        return (
            "[Nemo Action]\n"
            f"Name={action.title}\n"
            f"Exec={_desktop_quote(str(helper))} {action.mode} %F\n"
            f"Icon-Name={action.icon}\n"
            "Selection=notnone\n"
            f"Mimetypes={';'.join(action.mimetypes)};\n"
        )
    if manager == "dolphin":
        return (
            "[Desktop Entry]\n"
            "Type=Service\n"
            f"MimeType={';'.join(action.mimetypes)};\n"
            "Actions=run;\n"
            "X-KDE-ServiceTypes=KonqPopupMenu/Plugin\n"
            "\n"
            "[Desktop Action run]\n"
            f"Name={action.title}\n"
            f"Icon={action.icon}\n"
            f"Exec={_desktop_quote(str(helper))} {action.mode} %F\n"
        )
    raise KeyError(manager)


def install(managers: list[str], helper: Path) -> list[Path]:
    written = []
    for manager in managers:
        for action in ACTIONS:
            target = action_paths(manager, action)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render(manager, action, helper), encoding="utf-8")
            # Scripts must be executable; Dolphin (KDE 6) also requires it of service menus.
            target.chmod(0o755)
            written.append(target)
    return written


def installed() -> list[Path]:
    found = [action_paths(m, a) for m in FILE_MANAGERS for a in ACTIONS if action_paths(m, a).exists()]
    if helper_path().exists():
        found.append(helper_path())
    return found


def remove() -> list[Path]:
    """Remove every action vox may have written, and the helper script."""
    removed = []
    for path in installed():
        path.unlink()
        removed.append(path)
    try:
        helper_path().parent.rmdir()
    except OSError:
        pass
    return removed
