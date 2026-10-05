"""Best-effort support for other mlx-audio TTS models pulled with hf:<org>/<repo>."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vox.engines.base import TTSEngine

if TYPE_CHECKING:
    import numpy as np

SKIP_SUFFIXES = (".md", ".wav", ".mp3", ".flac", ".png", ".jpg", ".jpeg", ".gif", ".gitattributes")
TORCH_SUFFIXES = (".pt", ".pth", ".ckpt")


class MLXAudioTTSEngine(TTSEngine):
    name = "mlx-audio"

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._model = None

    @classmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        if "config.json" not in repo_files:
            raise ValueError("the repo has no config.json")
        if not any(f.endswith(".safetensors") for f in repo_files):
            raise ValueError("the repo has no .safetensors weights (it may not be an MLX model)")
        return [
            f
            for f in repo_files
            if not f.endswith(SKIP_SUFFIXES)
            and not f.endswith(TORCH_SUFFIXES)
            and not f.startswith(("samples/", "examples/", "."))
        ]

    @classmethod
    def list_voices(cls, model_dir: Path) -> list[str]:
        voices_dir = Path(model_dir) / "voices"
        if not voices_dir.is_dir():
            return []
        return sorted({p.stem for p in voices_dir.iterdir() if p.is_file()})

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.model_dir)
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def synthesize(self, text: str, *, voice: str | None, speed: float = 1.0) -> np.ndarray:
        import numpy as np

        if self._model is None:
            self.load()
        kwargs = {"speed": speed}
        if voice:
            kwargs["voice"] = voice
        pieces = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(text=text, **kwargs)
        ]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)
