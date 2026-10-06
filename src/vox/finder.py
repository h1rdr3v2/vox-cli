"""`vox setup finder`: the helper script and Finder Quick Actions."""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from vox import paths
from vox.media import find_ffmpeg

SERVICES_DIR = Path.home() / "Library" / "Services"
PBS = "/System/Library/CoreServices/pbs"
SHELL_ACTION = "/System/Library/Automator/Run Shell Script.action"


@dataclass(frozen=True)
class QuickAction:
    title: str
    mode: str  # argument to the helper script
    file_types: tuple[str, ...]  # UTIs Finder offers the action for


QUICK_ACTIONS = (
    QuickAction("Transcribe with vox", "transcribe", ("public.audio", "public.movie")),
    QuickAction("Speak with vox", "speak", ("public.plain-text", "net.daringfireball.markdown")),
)


def vox_command() -> list[str]:
    """Absolute command that runs this vox, independent of PATH."""
    found = shutil.which("vox")
    if found:
        return [str(Path(found).absolute())]
    argv0 = Path(sys.argv[0])
    if argv0.name == "vox" and argv0.exists():
        return [str(argv0.resolve())]
    return [sys.executable, "-m", "vox"]


def helper_path() -> Path:
    return paths.config_dir() / "finder" / "vox-finder.sh"


def write_helper() -> Path:
    template = resources.files("vox").joinpath("resources/vox-finder.sh").read_text(encoding="utf-8")
    ffmpeg = find_ffmpeg()
    path_dirs = []
    for directory in [Path(ffmpeg).parent if ffmpeg else None, Path(vox_command()[0]).parent]:
        if directory and str(directory) not in path_dirs:
            path_dirs.append(str(directory))
    script = template.replace("__VOX__", " ".join(shlex.quote(p) for p in vox_command()))
    script = script.replace("__PATH__", ":".join(path_dirs) or "/opt/homebrew/bin")
    target = helper_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    target.chmod(0o755)
    return target


# macOS shows a Quick Action's progress by counting finished steps, so each
# Quick Action has this many steps; each returns at the next tenth of the work.
PROGRESS_STEPS = 10


def shell_command(action: QuickAction, helper: Path) -> str:
    """One command that does the whole job (for a Shortcut, or by hand)."""
    return f'{shlex.quote(str(helper))} {action.mode} "$@"'


def step_commands(action: QuickAction, helper: Path) -> list[str]:
    """The Quick Action's steps: the first starts the job, each passes it on."""
    helper_arg = shlex.quote(str(helper))
    first = f'{helper_arg} step 1 {action.mode} "$@"'
    return [first] + [f'{helper_arg} step {n} "$@"' for n in range(2, PROGRESS_STEPS + 1)]


def _uuid() -> str:
    return str(uuid.uuid4()).upper()


def _shell_action(command: str, index: int) -> dict:
    """A Run Shell Script action that takes its input as arguments."""
    return {
        "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.path"]},
        "AMActionVersion": "2.0.3",
        "AMApplication": ["Automator"],
        "AMParameterProperties": {k: {} for k in ("COMMAND_STRING", "CheckedForUserDefaultShell", "inputMethod", "shell", "source")},
        "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
        "ActionBundlePath": SHELL_ACTION,
        "ActionName": "Run Shell Script",
        "ActionParameters": {
            "COMMAND_STRING": command,
            "CheckedForUserDefaultShell": True,
            "inputMethod": 1,
            "shell": "/bin/zsh",
            "source": "",
        },
        "BundleIdentifier": "com.apple.RunShellScript",
        "CFBundleVersion": "2.0.3",
        "CanShowSelectedItemsWhenRun": False,
        "CanShowWhenRun": True,
        "Category": ["AMCategoryUtilities"],
        "Class Name": "RunShellScriptAction",
        "InputUUID": _uuid(),
        "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
        "OutputUUID": _uuid(),
        "UUID": _uuid(),
        "UnlocalizedApplications": ["Automator"],
        "arguments": {
            str(i): {"default value": default, "name": name, "required": "0", "type": "0", "uuid": str(i)}
            for i, (name, default) in enumerate(
                [("inputMethod", 0), ("CheckedForUserDefaultShell", False), ("source", ""), ("COMMAND_STRING", ""), ("shell", "/bin/sh")]
            )
        },
        "conversionLabel": 0,
        "isViewVisible": 1,
        "location": f"309.000000:{305 + 120 * index}.000000",
        "nibPath": f"{SHELL_ACTION}/Contents/Resources/Base.lproj/main.nib",
    }


def _document(commands: list[str]) -> dict:
    """An Automator Quick Action running shell commands in order, each step's
    output (one line per argument) becoming the next step's arguments."""
    actions = [_shell_action(command, i) for i, command in enumerate(commands)]
    connectors = {}
    for before, after in zip(actions, actions[1:], strict=False):
        connectors[_uuid()] = {
            "from": f"{before['UUID']} - {before['UUID']}",
            "to": f"{after['UUID']} - {after['UUID']}",
        }
    return {
        "AMApplicationBuild": "534",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [{"action": action, "isViewVisible": 1} for action in actions],
        "connectors": connectors,
        "workflowMetaData": {
            "applicationBundleID": "com.apple.finder",
            "applicationBundleIDsByPath": {"/System/Library/CoreServices/Finder.app": "com.apple.finder"},
            "applicationPath": "/System/Library/CoreServices/Finder.app",
            "applicationPaths": ["/System/Library/CoreServices/Finder.app"],
            "inputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "outputTypeIdentifier": "com.apple.Automator.nothing",
            "presentationMode": 15,
            "processesInput": False,
            "serviceApplicationBundleID": "com.apple.finder",
            "serviceApplicationPath": "/System/Library/CoreServices/Finder.app",
            "serviceInputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceProcessesInput": False,
            "systemImageName": "NSActionTemplate",
            "useAutomaticInputType": False,
            "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
        },
    }


def _info(action: QuickAction) -> dict:
    return {
        "NSServices": [
            {
                "NSBackgroundColorName": "background",
                "NSIconName": "NSActionTemplate",
                "NSMenuItem": {"default": action.title},
                "NSMessage": "runWorkflowAsService",
                "NSRequiredContext": {"NSApplicationIdentifier": "com.apple.finder"},
                "NSSendFileTypes": list(action.file_types),
            }
        ]
    }


def workflow_path(action: QuickAction) -> Path:
    return SERVICES_DIR / f"{action.title}.workflow"


def install_quick_actions(helper: Path) -> list[Path]:
    """Write the Automator Quick Actions into ~/Library/Services."""
    if not Path(SHELL_ACTION).exists():
        return []
    written = []
    for action in QUICK_ACTIONS:
        contents = workflow_path(action) / "Contents"
        contents.mkdir(parents=True, exist_ok=True)
        with open(contents / "Info.plist", "wb") as fh:
            plistlib.dump(_info(action), fh)
        with open(contents / "document.wflow", "wb") as fh:
            plistlib.dump(_document(step_commands(action, helper)), fh)
        written.append(workflow_path(action))
    if os.access(PBS, os.X_OK):
        subprocess.run([PBS, "-update"], capture_output=True, timeout=30)
    return written


def remove_quick_actions() -> list[Path]:
    """Remove the Quick Actions and the helper script. Returns what was removed."""
    removed = []
    for action in QUICK_ACTIONS:
        path = workflow_path(action)
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    if removed and os.access(PBS, os.X_OK):
        subprocess.run([PBS, "-update"], capture_output=True, timeout=30)
    helper = helper_path()
    if helper.exists():
        helper.unlink()
        removed.append(helper)
        try:
            helper.parent.rmdir()
        except OSError:
            pass
    return removed
