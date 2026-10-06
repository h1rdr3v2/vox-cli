"""Engine registry. Engines are looked up by the name stored with each model."""

from __future__ import annotations

import importlib

from vox.engines.base import Engine, STTEngine, TTSEngine

_BUILTIN = {
    "mlx-whisper": ("vox.engines.mlx_whisper", "MLXWhisperEngine"),
    "kokoro": ("vox.engines.kokoro", "KokoroEngine"),
    "mlx-audio": ("vox.engines.mlx_audio_tts", "MLXAudioTTSEngine"),
    "faster-whisper": ("vox.engines.faster_whisper", "FasterWhisperEngine"),
    "kokoro-onnx": ("vox.engines.kokoro_onnx", "KokoroOnnxEngine"),
}
_registered: dict[str, type[Engine]] = {}


def register(name: str, cls: type[Engine]) -> None:
    """Add an engine at runtime (used by tests and future backends)."""
    _registered[name] = cls


def engine_class(name: str) -> type[Engine]:
    if name in _registered:
        return _registered[name]
    if name not in _BUILTIN:
        raise KeyError(f"unknown engine '{name}'")
    module, attr = _BUILTIN[name]
    return getattr(importlib.import_module(module), attr)


__all__ = ["Engine", "STTEngine", "TTSEngine", "engine_class", "register"]
