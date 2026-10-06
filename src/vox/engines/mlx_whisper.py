"""Speech-to-text with Whisper on MLX (Apple Silicon), using mlx-audio's implementation.

The engine keeps the name "mlx-whisper" (stored in model manifests) from when
it ran on the mlx-whisper package, which needs torch to install. It loads the
same model folders: config.json plus weights.safetensors or weights.npz
(mlx-community/whisper-*-mlx), or mlx-audio's own conversions
(mlx-community/whisper-*-asr-*) with model.safetensors.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from typing import TYPE_CHECKING

from vox import languages
from vox.engines.base import ProgressFn, Segment, STTEngine, Transcription, Word

if TYPE_CHECKING:
    import numpy as np

WEIGHT_FILES = ("weights.safetensors", "weights.npz", "model.safetensors")
CONFIG_KEYS = ("n_mels", "n_audio_ctx", "n_audio_state", "n_text_ctx", "n_vocab")
# mlx-audio conversions keep the alignment heads (used for word timestamps) here.
GENERATION_CONFIG = "generation_config.json"


def _whisper_module() -> types.ModuleType:
    """mlx_audio.stt.models.whisper.whisper, without mlx-audio's other speech-to-text models.

    The mlx_audio.stt.models package imports every model it has (and
    transformers with them) when any one of them is imported, which adds
    over a second to each cold start. Registering the package as a plain
    namespace first imports only the whisper subpackage.
    """
    name = "mlx_audio.stt.models"
    if name not in sys.modules:
        spec = importlib.util.find_spec(name)
        package = types.ModuleType(name)
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        sys.modules[name] = package
    return importlib.import_module("mlx_audio.stt.models.whisper.whisper")


class _ProgressBar:
    """Stands in for tqdm.tqdm inside mlx-audio's Whisper generate() to report progress."""

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
            raise ValueError("the repo has no MLX Whisper weights (weights.safetensors, weights.npz or model.safetensors)")
        extra = [GENERATION_CONFIG] if GENERATION_CONFIG in repo_files else []
        return ["config.json", weights, *extra]

    @classmethod
    def check_config(cls, config: dict) -> None:
        if all(k in config for k in CONFIG_KEYS):
            return
        # mlx-audio conversions use the Hugging Face transformers layout.
        if config.get("model_type") == "whisper" and "d_model" in config:
            return
        raise ValueError("config.json does not describe an MLX Whisper model")

    def load(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        from vox.engines import whisper_tokenizer

        whisper = _whisper_module()
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        quantization = config.pop("quantization", None)
        model = whisper.Model(whisper.ModelDimensions.from_dict(config), dtype=mx.float16)

        weights_file = next(self.model_dir / w for w in WEIGHT_FILES if (self.model_dir / w).exists())
        weights = mx.load(str(weights_file))
        # mlx-whisper conversions store the alignment heads with the weights.
        heads = weights.pop("alignment_heads", None)
        generation_config = self.model_dir / GENERATION_CONFIG
        if heads is None and generation_config.exists():
            heads = json.loads(generation_config.read_text(encoding="utf-8")).get("alignment_heads")
        if heads is not None:
            model.set_alignment_heads(heads.tolist() if isinstance(heads, mx.array) else heads)

        weights = model.sanitize(weights)
        if quantization is not None:
            nn.quantize(
                model,
                **{k: v for k, v in quantization.items() if k in ("group_size", "bits", "mode")},
                class_predicate=lambda path, module: isinstance(module, (nn.Linear, nn.Embedding)) and f"{path}.scales" in weights,
            )
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        # mlx-audio reads the tokenizer from Hugging Face tokenizer files, which
        # mlx-whisper conversions lack. vox ships Whisper's vocabularies instead.
        def get_tokenizer(language: str | None = None, task: str = "transcribe"):
            return whisper_tokenizer.get_tokenizer(
                model.is_multilingual, num_languages=model.num_languages, language=language, task=task
            )

        model.get_tokenizer = get_tokenizer
        self._model = model

    def normalize_language(self, language: str) -> str:
        return languages.normalize(language)

    def language_name(self, code: str | None) -> str | None:
        return languages.LANGUAGES.get(code or "")

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        word_timestamps: bool = False,
        progress: ProgressFn | None = None,
    ) -> Transcription:
        if self._model is None:
            self.load()

        whisper = _whisper_module()
        original_tqdm = whisper.tqdm
        whisper.tqdm = types.SimpleNamespace(tqdm=lambda **kw: _ProgressBar(progress, **kw))
        try:
            result = self._model.generate(
                audio,
                verbose=None,
                language=language,
                initial_prompt=prompt or None,
                word_timestamps=word_timestamps,
            )
        finally:
            whisper.tqdm = original_tqdm

        segments = []
        for i, seg in enumerate(result.segments or []):
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
            text=result.text.strip(),
            language=result.language,
            duration=len(audio) / 16000,
            segments=segments,
        )
