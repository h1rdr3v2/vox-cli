from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from vox import engines
from vox.engines.base import Segment, STTEngine, Transcription, TTSEngine
from vox.models import MANIFEST, model_dir


class FakeSTT(STTEngine):
    name = "fake-stt"
    loads = 0

    @classmethod
    def select_files(cls, repo_files):
        return ["config.json"]

    def load(self):
        type(self).loads += 1

    def normalize_language(self, language):
        if language not in ("en", "fr"):
            raise ValueError(f"unknown language '{language}'")
        return language

    def language_name(self, code):
        return {"en": "english", "fr": "french"}.get(code)

    def transcribe(self, audio, *, language=None, prompt=None, word_timestamps=False, progress=None):
        if progress:
            progress(50, 100)
            progress(100, 100)
        return Transcription(
            text="Hello world.",
            language=language or "en",
            duration=len(audio) / 16000,
            segments=[Segment(id=0, start=0.0, end=1.5, text=" Hello world.")],
        )


class FakeTTS(TTSEngine):
    name = "fake-tts"
    sample_rate = 24000
    loads = 0
    calls: list = []

    @classmethod
    def select_files(cls, repo_files):
        return ["config.json"]

    @classmethod
    def list_voices(cls, model_dir):
        voices = Path(model_dir) / "voices"
        return sorted(p.stem for p in voices.glob("*.safetensors")) if voices.is_dir() else []

    @classmethod
    def openai_voice(cls, name):
        return {"alloy": "af_test"}.get(name)

    def load(self):
        type(self).loads += 1

    def synthesize(self, text, *, voice, speed=1.0):
        type(self).calls.append((text, voice, speed))
        t = np.arange(int(0.25 * self.sample_rate)) / self.sample_rate
        return (0.1 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)


@pytest.fixture(autouse=True)
def vox_home(tmp_path, monkeypatch):
    """Every test gets empty config and cache folders."""
    monkeypatch.setenv("VOX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("VOX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VOX_MODELS_DIR", raising=False)
    # Same catalog on every machine running the tests; portable tests opt in.
    monkeypatch.setenv("VOX_BACKEND", "mlx")
    monkeypatch.delenv("VOX_DEBUG", raising=False)
    engines.register("fake-stt", FakeSTT)
    engines.register("fake-tts", FakeTTS)
    FakeSTT.loads = 0
    FakeTTS.loads = 0
    FakeTTS.calls = []
    return tmp_path


def install_fake(model_id: str, model_type: str, voices=("af_test", "am_test")) -> Path:
    """Create a model folder with a manifest, as `vox models pull` would."""
    folder = model_dir(model_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}")
    if model_type == "tts":
        (folder / "voices").mkdir(exist_ok=True)
        for voice in voices:
            (folder / "voices" / f"{voice}.safetensors").write_bytes(b"")
    manifest = {
        "id": model_id,
        "type": model_type,
        "engine": "fake-stt" if model_type == "stt" else "fake-tts",
        "repo": f"test/{model_id}",
        "revision": "abc123",
        "installed_at": "2026-01-01T00:00:00+00:00",
        "files": ["config.json"],
    }
    (folder / MANIFEST).write_text(json.dumps(manifest))
    return folder


def write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    samples = (np.sin(np.arange(int(seconds * rate)) / 10) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return path


class CliResult:
    def __init__(self, code, out, err):
        self.code, self.out, self.err = code, out, err


@pytest.fixture
def run_cli(monkeypatch, capsys):
    """Run vox's real entry point (with its exit-code mapping) in-process."""
    from vox import cli

    def run(*args: str, stdin_tty: bool = False) -> CliResult:
        monkeypatch.setattr(sys, "argv", ["vox", *args])
        monkeypatch.setattr(cli, "interactive", lambda: stdin_tty)
        try:
            cli.main()
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        captured = capsys.readouterr()
        return CliResult(code, captured.out, captured.err)

    return run
