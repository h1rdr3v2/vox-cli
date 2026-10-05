"""Text-to-speech with Kokoro through mlx-audio."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vox import paths
from vox.engines.base import SupportAsset, TTSEngine
from vox.errors import MissingError, UserError

if TYPE_CHECKING:
    import numpy as np

# Kokoro's English front end (misaki) needs spaCy's small English pipeline.
# Left alone, misaki pip-installs it on first use, which fails inside a uv
# tool environment and would be a network call outside `vox models pull`.
# vox fetches it with the model and imports it from the model folder.
SPACY_EN = SupportAsset(
    name="spaCy English pipeline (pronunciation)",
    url=(
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    ),
    sha256="1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85",
    size=12_806_118,
    target="_vox/site-packages",
)

# First letter of a Kokoro voice name is its language, second its gender.
LANGUAGES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}
GENDERS = {"f": "female", "m": "male"}
# Languages whose text front end needs an optional misaki extra.
EXTRAS = {"j": ("misaki.ja", "misaki[ja]"), "z": ("misaki.zh", "misaki[zh]")}

# OpenAI voice names, so OpenAI clients get a sensible Kokoro voice.
OPENAI_VOICES = {
    "alloy": "af_alloy",
    "ash": "am_adam",
    "ballad": "bm_george",
    "cedar": "am_liam",
    "coral": "af_sarah",
    "echo": "am_echo",
    "fable": "bm_fable",
    "marin": "af_heart",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "sage": "af_river",
    "shimmer": "af_bella",
    "verse": "am_michael",
}

PREFERRED_VOICE = "af_heart"
# espeak-ng copies its data path into a fixed 160 byte buffer, silently
# truncates longer paths, then calls exit() when it cannot find its files.
ESPEAK_SAFE_PATH_LENGTH = 140


def describe_voice(name: str) -> tuple[str, str]:
    """(language, gender) for a Kokoro voice name, or ("", "") if unknown."""
    if len(name) >= 3 and name[2] == "_":
        return LANGUAGES.get(name[0], ""), GENDERS.get(name[1], "")
    return "", ""


def _use_short_espeak_path() -> None:
    import espeakng_loader

    real = espeakng_loader.get_data_path()
    if len(real.encode()) < ESPEAK_SAFE_PATH_LENGTH:
        return
    link = paths.cache_dir() / "espeak-ng-data"
    if len(str(link).encode()) >= ESPEAK_SAFE_PATH_LENGTH:
        link = Path(tempfile.gettempdir()) / f"vox-espeak-{os.getuid()}"
    if not (link.is_symlink() and os.readlink(link) == real):
        link.parent.mkdir(parents=True, exist_ok=True)
        link.unlink(missing_ok=True)
        link.symlink_to(real)
    os.environ["ESPEAK_DATA_PATH"] = str(link)

    import misaki.espeak  # noqa: F401  (sets phonemizer's data path to the long one)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # With no explicit path, espeak reads ESPEAK_DATA_PATH. phonemizer would
    # resolve the symlink back to the long path, so it must not be given one.
    EspeakWrapper.set_data_path(None)


def _import_spacy_without_torch() -> None:
    """Import spaCy without letting thinc import torch.

    thinc treats torch as optional, Kokoro never uses it, and importing it
    adds about 2 seconds to every cold start. torch is blocked only while
    spaCy imports; anything that needs it later can still import it.
    """
    if "spacy" in sys.modules or "torch" in sys.modules:
        return
    sys.modules["torch"] = None  # makes `import torch` raise ImportError
    try:
        import spacy  # noqa: F401
        import thinc.compat  # noqa: F401
    except ImportError:
        for name in [n for n in sys.modules if n.split(".")[0] in ("spacy", "thinc")]:
            del sys.modules[name]
    finally:
        if sys.modules.get("torch", 0) is None:
            del sys.modules["torch"]
    import spacy  # noqa: F401,F811  (no-op if the guarded import worked)


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
        site = self.model_dir / SPACY_EN.target
        if not (site / "en_core_web_sm").is_dir():
            raise MissingError(
                "Kokoro's English pronunciation files are missing from the model folder.",
                "Pull the model again: vox models pull <id>",
            )
        if str(site) not in sys.path:
            sys.path.insert(0, str(site))
            importlib.invalidate_caches()
        _import_spacy_without_torch()
        _use_short_espeak_path()

        from mlx_audio.tts.utils import load_model

        self._model = load_model(self.model_dir)
        self.sample_rate = int(self._model.sample_rate)

    def _voice_paths(self, voice: str) -> list[Path]:
        available = set(self.list_voices(self.model_dir))
        result = []
        for name in (v.strip() for v in voice.split(",")):
            if name not in available:
                raise UserError(f"Unknown voice '{name}'.", "List voices with: vox voices")
            result.append(self.model_dir / "voices" / f"{name}.safetensors")
        return result

    def synthesize(self, text: str, *, voice: str | None, speed: float = 1.0) -> np.ndarray:
        import numpy as np

        if self._model is None:
            self.load()
        voice = voice or self.fallback_voice(self.model_dir)
        if not voice:
            raise MissingError("This Kokoro model has no voices.", "Pull it again: vox models pull <id>")
        voice_files = self._voice_paths(voice)
        lang = voice_files[0].stem[0]
        if lang not in LANGUAGES:
            lang = "a"
        if lang in EXTRAS:
            module, extra = EXTRAS[lang]
            try:
                importlib.import_module(module)
            except ImportError:
                raise MissingError(
                    f"{LANGUAGES[lang]} voices need the optional {extra} package.",
                    f'Reinstall vox with it: uv tool install --reinstall vox-cli --with "{extra}"',
                ) from None

        # A comma-separated list of voice files is blended by mlx-audio.
        voice_arg = ",".join(str(p) for p in voice_files)
        pieces = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(
                text, voice=voice_arg, speed=speed, lang_code=lang, split_pattern=r"\n+"
            )
        ]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)
