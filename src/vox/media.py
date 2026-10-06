"""ffmpeg helpers shared by the CLI and the server. No numpy here."""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

from vox.errors import MissingError, UserError

# Shortcuts, Quick Actions and file-manager scripts run with a minimal PATH,
# so look in the usual Homebrew and Linux locations too.
FFMPEG_CANDIDATES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/home/linuxbrew/.linuxbrew/bin/ffmpeg", "/usr/bin/ffmpeg")


def find_ffmpeg() -> str | None:
    override = os.environ.get("VOX_FFMPEG")
    if override:
        return override if os.access(override, os.X_OK) else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    return next((p for p in FFMPEG_CANDIDATES if os.access(p, os.X_OK)), None)


def require_ffmpeg() -> str:
    path = find_ffmpeg()
    if not path:
        from vox.system import install_hint

        raise MissingError("ffmpeg is not installed.", install_hint("ffmpeg"))
    return path


def _ffmpeg_error(src: Path, stderr: str) -> UserError:
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    text = "\n".join(lines).lower()
    if "does not contain any stream" in text or "output file #0 does not contain" in text or "matches no streams" in text:
        return UserError(f"{src.name} has no audio track.", "Pick a file with sound, or check it plays in QuickTime.")
    if "invalid data found" in text or "could not find codec" in text or "unknown format" in text:
        return UserError(f"{src.name} is not an audio or video file ffmpeg can read.", "Check the file is not damaged, or convert it with ffmpeg first.")
    detail = lines[-1] if lines else "unknown error"
    return UserError(f"ffmpeg could not read {src.name}: {detail}", "Check the file plays in QuickTime or VLC.")


def to_wav16k(src: Path, dst: Path) -> float:
    """Convert any audio or video file to 16 kHz mono 16-bit WAV. Returns duration in seconds."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-f", "wav", str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise _ffmpeg_error(src, proc.stderr)
    return wav_duration(dst)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)


def is_wav16k_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnchannels() == 1 and wav.getframerate() == 16000 and wav.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def pcm16_to(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """Encode raw 16-bit mono PCM with ffmpeg (mp3, flac, opus, aac)."""
    ffmpeg = require_ffmpeg()
    codec = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "flac": ["-c:a", "flac", "-f", "flac"],
        "opus": ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
        "aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    }[fmt]
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *codec, "pipe:1",
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise UserError(f"ffmpeg could not encode {fmt}: {detail[-1] if detail else 'unknown error'}", "Use .wav output, or reinstall ffmpeg: brew reinstall ffmpeg")
    return proc.stdout


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
