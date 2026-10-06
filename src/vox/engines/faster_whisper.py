"""Speech-to-text with faster-whisper (CTranslate2): Linux, CPU or NVIDIA GPU."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from vox import languages
from vox.engines.base import ProgressFn, Segment, STTEngine, Transcription, Word

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("vox.engines")

CONFIG_KEYS = ("alignment_heads", "lang_ids", "suppress_ids")
SKIP = (".gitattributes", "README.md")


def pick_device(setting: str, cuda_devices: int) -> tuple[str, str]:
    """(device, compute_type) for a `device` setting of auto, cpu or cuda."""
    if setting == "cuda" or (setting == "auto" and cuda_devices > 0):
        return "cuda", "float16"
    # int8 on the CPU: much faster than float32, with near identical accuracy.
    return "cpu", "int8"


class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._model = None

    @classmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        if "config.json" not in repo_files or "model.bin" not in repo_files:
            raise ValueError("the repo is not a CTranslate2 Whisper model (config.json and model.bin)")
        return [f for f in repo_files if f not in SKIP and "/" not in f]

    @classmethod
    def check_config(cls, config: dict) -> None:
        if not any(k in config for k in CONFIG_KEYS):
            raise ValueError("config.json does not describe a faster-whisper (CTranslate2) model")

    def load(self) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        from vox.config import load_config

        setting = load_config().device
        device, compute_type = pick_device(setting, ctranslate2.get_cuda_device_count())
        threads = min(os.cpu_count() or 4, 16)
        try:
            self._model = WhisperModel(
                str(self.model_dir), device=device, compute_type=compute_type, cpu_threads=threads, local_files_only=True
            )
        except (RuntimeError, ValueError) as exc:
            if device != "cuda":
                raise
            # CUDA was found but its libraries (cuBLAS, cuDNN) are missing or incompatible.
            log.warning("could not use the GPU (%s); using the CPU", exc)
            self._model = WhisperModel(
                str(self.model_dir), device="cpu", compute_type="int8", cpu_threads=threads, local_files_only=True
            )
        log.info("faster-whisper on %s (%s)", self._model.model.device, compute_type if device == "cpu" else "float16")

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
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt or None,
            word_timestamps=word_timestamps,
        )
        total = float(info.duration or len(audio) / 16000)
        segments = []
        for i, seg in enumerate(segments_iter):  # decoding happens while iterating
            words = None
            if word_timestamps:
                words = [Word(word=w.word, start=float(w.start), end=float(w.end), probability=float(w.probability)) for w in seg.words or []]
            segments.append(
                Segment(
                    id=i,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                    tokens=list(seg.tokens),
                    temperature=float(seg.temperature or 0.0),
                    avg_logprob=float(seg.avg_logprob),
                    compression_ratio=float(seg.compression_ratio),
                    no_speech_prob=float(seg.no_speech_prob),
                    words=words,
                )
            )
            if progress and total:
                progress(min(float(seg.end), total), total)
        return Transcription(
            text="".join(s.text for s in segments).strip(),
            language=info.language,
            duration=len(audio) / 16000,
            segments=segments,
        )
