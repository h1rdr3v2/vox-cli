"""Platform differences in one place: which engines to use, playing audio,
reading process memory, and install hints."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from vox.errors import MissingError

# Engine families. "mlx" runs on Apple Silicon (Metal). "portable" runs on
# Linux with CTranslate2 and ONNX Runtime, on the CPU or an NVIDIA GPU.
BACKENDS = ("mlx", "portable")


def is_mac() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def backend() -> str:
    """The engine family for this machine. VOX_BACKEND overrides it."""
    override = os.environ.get("VOX_BACKEND", "").strip().lower()
    if override:
        if override not in BACKENDS:
            raise MissingError(f"Unknown VOX_BACKEND: {override}.", "Use mlx or portable, or unset it.")
        return override
    if is_mac() and platform.machine() == "arm64":
        return "mlx"
    if is_linux():
        return "portable"
    raise MissingError(
        f"vox does not support this system yet ({platform.system()} {platform.machine()}).",
        "vox runs on Apple Silicon Macs and on Linux.",
    )


def install_hint(package: str) -> str:
    if is_mac():
        return f"Run: brew install {package}"
    return f"Install it with your package manager, for example: sudo apt install {package}"


# ------------------------------------------------------------------ audio


def audio_player(path: Path) -> list[str]:
    """A command that plays an audio file on this machine."""
    if is_mac():
        return ["/usr/bin/afplay", str(path)]
    wav = path.suffix.lower() == ".wav"
    candidates = [
        ("pw-play", []),  # PipeWire
        ("paplay", []),  # PulseAudio (also works on PipeWire)
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("aplay", ["-q"]),  # ALSA: WAV only
    ]
    if not wav:  # compressed audio: ffplay first, aplay cannot play it
        candidates = [candidates[2], candidates[0], candidates[1]]
    for name, args in candidates:
        found = shutil.which(name)
        if found:
            return [found, *args, str(path)]
    raise MissingError(
        "No audio player found to play the result.",
        "Install one, for example: sudo apt install pulseaudio-utils (or ffmpeg, or alsa-utils)",
    )


def playback_error(returncode: int, stderr: str) -> str | None:
    """A one-line problem description if playing failed, else None.

    ffplay exits 0 even when there is no sound device, so look at its output too.
    """
    text = stderr.lower()
    if "cannot find card" in text or "unknown pcm" in text or "no such device" in text or "connection refused" in text:
        return "No audio output device found."
    if returncode != 0:
        lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        return f"Could not play the audio ({lines[-1] if lines else f'player exited with {returncode}'})."
    return None


# -------------------------------------------------------------- processes


def rss_bytes(pid: int) -> int | None:
    """Resident memory of a process, or None if unknown."""
    if is_linux():
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def process_command(pid: int) -> str:
    """The command line of a process, or "" if unknown."""
    if is_linux():
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            return ""
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
