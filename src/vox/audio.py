"""Sample-level audio helpers for the server (uses numpy)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from vox import media

SPEECH_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}


def load_for_whisper(path: Path, workdir: Path) -> np.ndarray:
    """Read any audio or video file as 16 kHz mono float32, converting with ffmpeg if needed."""
    if not media.is_wav16k_mono(path):
        converted = workdir / "input-16k.wav"
        media.to_wav16k(path, converted)
        path = converted
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def encode(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    pcm = to_pcm16(samples)
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return buf.getvalue()
    return media.pcm16_to(pcm, sample_rate, fmt)
