"""Engine interfaces.

An engine wraps one inference backend (mlx-whisper, mlx-audio, and later
perhaps whisper.cpp). The CLI never touches engines; only the server does.

Engine modules must stay cheap to import: heavy libraries (mlx, torch,
spaCy) are imported inside load(). Class-level helpers such as
select_files() and list_voices() are used by `vox models pull` and
`vox voices` without loading anything.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

# progress(done, total): called from the inference thread.
ProgressFn = Callable[[float, float], None]


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    tokens: list[int] = field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: list[Word] | None = None


@dataclass
class Transcription:
    text: str
    language: str | None  # ISO code such as "en"
    duration: float  # seconds of audio
    segments: list[Segment]


@dataclass(frozen=True)
class SupportAsset:
    """A file an engine needs besides the model repo, fetched by `vox models pull`.

    Wheels are unpacked into <model dir>/<target> so they can be imported
    from there without installing anything into the Python environment.
    """

    name: str
    url: str
    sha256: str
    size: int
    target: str


class Engine(ABC):
    name: ClassVar[str]
    type: ClassVar[str]
    support_assets: ClassVar[tuple[SupportAsset, ...]] = ()

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

    @classmethod
    @abstractmethod
    def select_files(cls, repo_files: list[str]) -> list[str]:
        """Pick which files of a Hugging Face repo to download.

        Raise ValueError with a readable reason if the repo is unusable.
        """

    @classmethod
    def check_config(cls, config: dict) -> None:
        """Raise ValueError if config.json does not belong to this engine."""

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once, from the server."""


class STTEngine(Engine):
    type = "stt"

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        word_timestamps: bool = False,
        progress: ProgressFn | None = None,
    ) -> Transcription:
        """Transcribe 16 kHz mono float32 audio."""

    def normalize_language(self, language: str) -> str:
        """Map a user-supplied language to the code the engine expects.

        Raise ValueError if it is not supported.
        """
        return language

    def language_name(self, code: str | None) -> str | None:
        """Full lowercase name for a language code ("en" -> "english")."""
        return None


class TTSEngine(Engine):
    type = "tts"
    sample_rate: int = 24000

    @classmethod
    def list_voices(cls, model_dir: Path) -> list[str]:
        return []

    @classmethod
    def fallback_voice(cls, model_dir: Path) -> str | None:
        voices = cls.list_voices(model_dir)
        return voices[0] if voices else None

    @classmethod
    def openai_voice(cls, name: str) -> str | None:
        """This engine's stand-in for an OpenAI voice name such as "alloy"."""
        return None

    @abstractmethod
    def synthesize(self, text: str, *, voice: str | None, speed: float = 1.0) -> np.ndarray:
        """Return float32 mono samples at self.sample_rate for one chunk of text."""
