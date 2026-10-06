"""Text-to-speech with Kokoro through mlx-audio (Apple Silicon)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vox.engines.base import TTSEngine
from vox.engines.kokoro_text import (
    OPENAI_VOICES,
    PREFERRED_VOICE,
    SPACY_EN,
    describe_voice,  # noqa: F401  (re-exported for `vox voices`)
    prepare_text_frontend,
    require_extras,
    voice_language,
    voice_names,
)
from vox.errors import MissingError

if TYPE_CHECKING:
    import numpy as np


class KokoroEngine(TTSEngine):
    name = "kokoro"
    support_assets = (SPACY_EN,)

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._model = None

    @classmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        if "config.json" not in repo_files:
            raise ValueError("the repo has no config.json")
        weights = [f for f in repo_files if "/" not in f and f.endswith(".safetensors")]
        if not weights:
            raise ValueError(
                "the repo has no .safetensors weights; use an MLX conversion such as "
                "mlx-community/Kokoro-82M-bf16"
            )
        voices = [f for f in repo_files if f.startswith("voices/") and f.endswith(".safetensors")]
        if not voices:
            raise ValueError("the repo has no voices/*.safetensors files")
        return ["config.json", *weights, *sorted(voices)]

    @classmethod
    def check_config(cls, config: dict) -> None:
        if not ("istftnet" in config and "plbert" in config):
            raise ValueError("config.json does not describe a Kokoro model")

    @classmethod
    def list_voices(cls, model_dir: Path) -> list[str]:
        voices_dir = Path(model_dir) / "voices"
        if not voices_dir.is_dir():
            return []
        return sorted(p.stem for p in voices_dir.glob("*.safetensors"))

    @classmethod
    def fallback_voice(cls, model_dir: Path) -> str | None:
        voices = cls.list_voices(model_dir)
        if PREFERRED_VOICE in voices:
            return PREFERRED_VOICE
        return voices[0] if voices else None

    @classmethod
    def openai_voice(cls, name: str) -> str | None:
        return OPENAI_VOICES.get(name.lower())

    def load(self) -> None:
        prepare_text_frontend(self.model_dir)

        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.model_dir)
        self.sample_rate = int(self._model.sample_rate)

    def synthesize(self, text: str, *, voice: str | None, speed: float = 1.0) -> np.ndarray:
        import numpy as np

        if self._model is None:
            self.load()
        voice = voice or self.fallback_voice(self.model_dir)
        if not voice:
            raise MissingError("This Kokoro model has no voices.", "Pull it again: vox models pull <id>")
        names = voice_names(voice, self.list_voices(self.model_dir))
        lang = voice_language(voice)
        require_extras(lang)

        # A comma-separated list of voice files is blended by mlx-audio.
        voice_arg = ",".join(str(self.model_dir / "voices" / f"{name}.safetensors") for name in names)
        pieces = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(
                text, voice=voice_arg, speed=speed, lang_code=lang, split_pattern=r"\n+"
            )
        ]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)
