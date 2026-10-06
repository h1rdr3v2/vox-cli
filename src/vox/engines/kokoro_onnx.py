"""Text-to-speech with Kokoro on ONNX Runtime (Linux, or anywhere).

The text side is the same as on Apple Silicon (misaki with spaCy and espeak),
so pronunciation matches. The ONNX model takes phoneme ids, a style vector
picked from the voice by input length, and a speed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vox.engines.base import TTSEngine
from vox.engines.kokoro_text import (
    OPENAI_VOICES,
    PREFERRED_VOICE,
    SPACY_EN,
    make_g2p,
    phonemize,
    prepare_text_frontend,
    voice_language,
    voice_names,
)
from vox.errors import MissingError

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("vox.engines")

MODEL_FILES = ("onnx/model.onnx", "model.onnx", "kokoro-v1.0.onnx")  # full precision first
MAX_TOKENS = 510  # the model's context is 512 including a pad token at each end
STYLE_DIM = 256


def split_phonemes(phonemes: str, limit: int = MAX_TOKENS) -> list[str]:
    """Split a phoneme string into pieces of at most `limit` characters,
    preferring sentence punctuation, then commas, then spaces."""
    pieces = []
    rest = phonemes.strip()
    while len(rest) > limit:
        window = rest[: limit + 1]
        cut = max(window.rfind(p) for p in ".!?;:")
        if cut < limit // 3:
            cut = window.rfind(",")
        if cut < limit // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit - 1
        pieces.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def tokenize(phonemes: str, vocab: dict[str, int]) -> list[int]:
    """Phoneme characters to ids; characters the model does not know are dropped."""
    return [vocab[ch] for ch in phonemes if ch in vocab and vocab[ch] != 0]


def _read_vocab(model_dir: Path) -> dict[str, int]:
    tokenizer = model_dir / "tokenizer.json"
    if tokenizer.exists():
        return dict(json.loads(tokenizer.read_text(encoding="utf-8"))["model"]["vocab"])
    return dict(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["vocab"])


class KokoroOnnxEngine(TTSEngine):
    name = "kokoro-onnx"
    support_assets = (SPACY_EN,)
    sample_rate = 24000

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._session = None
        self._inputs: dict[str, str] = {}
        self._vocab: dict[str, int] = {}
        self._g2p: dict[str, object] = {}
        self._packs: dict[str, np.ndarray] = {}

    @classmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        model = next((f for f in MODEL_FILES if f in repo_files), None)
        if model is None:
            raise ValueError("the repo has no full-precision Kokoro ONNX model (onnx/model.onnx)")
        vocab = [f for f in ("tokenizer.json", "config.json") if f in repo_files]
        if not vocab:
            raise ValueError("the repo has no tokenizer.json or config.json with the phoneme vocabulary")
        voices = sorted(f for f in repo_files if f.startswith("voices/") and f.endswith(".bin"))
        if not voices:
            raise ValueError("the repo has no voices/*.bin files")
        return sorted({"config.json", *vocab} & set(repo_files)) + [model, *voices]

    @classmethod
    def list_voices(cls, model_dir: Path) -> list[str]:
        voices_dir = Path(model_dir) / "voices"
        if not voices_dir.is_dir():
            return []
        return sorted(p.stem for p in voices_dir.glob("*.bin"))

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
        import onnxruntime as ort

        from vox.config import load_config

        prepare_text_frontend(self.model_dir)
        self._vocab = _read_vocab(self.model_dir)
        model = next((self.model_dir / f for f in MODEL_FILES if (self.model_dir / f).exists()), None)
        if model is None:
            raise MissingError("The Kokoro ONNX model file is missing.", "Pull the model again: vox models pull <id>")

        device = load_config().device
        providers = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        elif device == "cuda":
            log.warning("device = cuda, but this onnxruntime has no CUDA support; using the CPU")
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self._session = ort.InferenceSession(str(model), options, providers=providers)

        # Input names differ between exports: find them by type and shape.
        for item in self._session.get_inputs():
            name = item.name.lower()
            if "int64" in item.type:
                self._inputs["ids"] = item.name
            elif "speed" in name:
                self._inputs["speed"] = item.name
            else:
                self._inputs["style"] = item.name
        if set(self._inputs) != {"ids", "style", "speed"}:
            raise MissingError(
                f"Unexpected inputs in the Kokoro ONNX model: {[i.name for i in self._session.get_inputs()]}.",
                "Use onnx-community/Kokoro-82M-v1.0-ONNX (the kokoro-82m catalog model).",
            )

    def _voice_pack(self, voice: str) -> np.ndarray:
        import numpy as np

        if voice not in self._packs:
            names = voice_names(voice, self.list_voices(self.model_dir))
            packs = [
                np.fromfile(self.model_dir / "voices" / f"{name}.bin", dtype=np.float32).reshape(-1, 1, STYLE_DIM)
                for name in names
            ]
            # A blend of voices is their average, as in Kokoro itself.
            self._packs[voice] = np.mean(np.stack(packs), axis=0) if len(packs) > 1 else packs[0]
        return self._packs[voice]

    def synthesize(self, text: str, *, voice: str | None, speed: float = 1.0) -> np.ndarray:
        import numpy as np

        if self._session is None:
            self.load()
        voice = voice or self.fallback_voice(self.model_dir)
        if not voice:
            raise MissingError("This Kokoro model has no voices.", "Pull it again: vox models pull <id>")
        pack = self._voice_pack(voice)
        lang = voice_language(voice)
        if lang not in self._g2p:
            self._g2p[lang] = make_g2p(lang)

        pieces = []
        for chunk in split_phonemes(phonemize(self._g2p[lang], text)):
            ids = tokenize(chunk, self._vocab)
            if not ids:
                continue
            style = pack[min(len(ids), len(pack) - 1)]
            audio = self._session.run(
                None,
                {
                    self._inputs["ids"]: np.array([[0, *ids, 0]], dtype=np.int64),
                    self._inputs["style"]: style.astype(np.float32),
                    self._inputs["speed"]: np.array([speed], dtype=np.float32),
                },
            )[0]
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)
