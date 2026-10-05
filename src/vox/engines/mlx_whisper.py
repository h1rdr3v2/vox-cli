"""Speech-to-text with mlx-whisper."""

from __future__ import annotations

import importlib
import types
from typing import TYPE_CHECKING

from vox.engines.base import ProgressFn, Segment, STTEngine, Transcription, Word

if TYPE_CHECKING:
    import numpy as np

WEIGHT_FILES = ("weights.safetensors", "weights.npz")
CONFIG_KEYS = ("n_mels", "n_audio_ctx", "n_audio_state", "n_text_ctx", "n_vocab")


class _ProgressBar:
    """Stands in for tqdm.tqdm inside mlx_whisper.transcribe to report progress."""

    def __init__(self, callback: ProgressFn | None, total: float = 0, **_):
        self.callback = callback
        self.total = float(total or 0)
        self.n = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n: float = 1) -> None:
        self.n += n
        if self.callback and self.total:
            self.callback(min(self.n, self.total), self.total)

    def close(self) -> None:
        pass


class MLXWhisperEngine(STTEngine):
    name = "mlx-whisper"

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._model = None

    @classmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        if "config.json" not in repo_files:
            raise ValueError("the repo has no config.json")
        weights = next((w for w in WEIGHT_FILES if w in repo_files), None)
        if weights is None:
            raise ValueError("the repo has no MLX Whisper weights (weights.safetensors or weights.npz)")
        return ["config.json", weights]

    @classmethod
    def check_config(cls, config: dict) -> None:
        missing = [k for k in CONFIG_KEYS if k not in config]
        if missing:
            raise ValueError("config.json does not describe an MLX Whisper model")

    def load(self) -> None:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        self._model = load_model(str(self.model_dir), dtype=mx.float16)

    def normalize_language(self, language: str) -> str:
        from mlx_whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

        value = language.strip().lower()
        if value in LANGUAGES:
            return value
        if value in TO_LANGUAGE_CODE:
            return TO_LANGUAGE_CODE[value]
        raise ValueError(f"unknown language '{language}'")

    def language_name(self, code: str | None) -> str | None:
        from mlx_whisper.tokenizer import LANGUAGES

        return LANGUAGES.get(code or "")

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        word_timestamps: bool = False,
        progress: ProgressFn | None = None,
    ) -> Transcription:
        # importlib, because mlx_whisper re-exports a transcribe() function
        # that shadows the submodule of the same name.
        mt = importlib.import_module("mlx_whisper.transcribe")

        if self._model is None:
            self.load()

        # mlx_whisper caches one model in ModelHolder, keyed by path. Point it at
        # ours so transcribe() reuses it instead of loading a second copy.
        path = str(self.model_dir)
        mt.ModelHolder.model = self._model
        mt.ModelHolder.model_path = path

        original_tqdm = mt.tqdm
        mt.tqdm = types.SimpleNamespace(tqdm=lambda **kw: _ProgressBar(progress, **kw))
        try:
            result = mt.transcribe(
                audio,
                path_or_hf_repo=path,
                verbose=None,
                language=language,
                initial_prompt=prompt or None,
                word_timestamps=word_timestamps,
            )
        finally:
            mt.tqdm = original_tqdm

        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            words = None
            if word_timestamps:
                words = [
                    Word(
                        word=w["word"],
                        start=float(w["start"]),
                        end=float(w["end"]),
                        probability=float(w["probability"]) if "probability" in w else None,
                    )
                    for w in seg.get("words", [])
                ]
            segments.append(
                Segment(
                    id=i,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"],
                    tokens=list(seg.get("tokens", [])),
                    temperature=float(seg.get("temperature", 0.0)),
                    avg_logprob=float(seg.get("avg_logprob", 0.0)),
                    compression_ratio=float(seg.get("compression_ratio", 0.0)),
                    no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                    words=words,
                )
            )
        return Transcription(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            duration=len(audio) / 16000,
            segments=segments,
        )
