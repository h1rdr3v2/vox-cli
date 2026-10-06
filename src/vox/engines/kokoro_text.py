"""Kokoro pieces shared by the MLX and ONNX engines: voices, the spaCy
pipeline, espeak setup, and turning text into phonemes with misaki."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

from vox import paths
from vox.engines.base import SupportAsset
from vox.errors import MissingError, UserError

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
# espeak-ng language for the languages misaki hands to espeak.
ESPEAK_LANGUAGES = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}
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


def voice_names(voice: str, available: list[str]) -> list[str]:
    """Split a voice or a blend ("af_heart,af_bella") and check each name."""
    names = [v.strip() for v in voice.split(",") if v.strip()]
    for name in names:
        if name not in available:
            raise UserError(f"Unknown voice '{name}'.", "List voices with: vox voices")
    return names


def voice_language(voice: str) -> str:
    """Kokoro language code for a voice (the first one of a blend)."""
    first = voice.split(",")[0].strip()
    return first[0] if first and first[0] in LANGUAGES else "a"


def require_extras(lang: str) -> None:
    if lang not in EXTRAS:
        return
    module, extra = EXTRAS[lang]
    try:
        importlib.import_module(module)
    except ImportError:
        raise MissingError(
            f"{LANGUAGES[lang]} voices need the optional {extra} package.",
            f'Reinstall vox with it: uv tool install --reinstall vox-cli --with "{extra}"',
        ) from None


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


def prepare_text_frontend(model_dir: Path) -> None:
    """Make the model's bundled spaCy pipeline importable and set up espeak."""
    site = Path(model_dir) / SPACY_EN.target
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


def make_g2p(lang: str):
    """A misaki grapheme-to-phoneme converter for a Kokoro language code."""
    require_extras(lang)
    if lang in ("a", "b"):
        from misaki import en, espeak

        try:
            fallback = espeak.EspeakFallback(british=lang == "b")
        except Exception:  # noqa: BLE001  (without espeak, unknown words are skipped)
            fallback = None
        return en.G2P(trf=False, british=lang == "b", fallback=fallback, unk="")
    if lang == "j":
        from misaki import ja

        return ja.JAG2P()
    if lang == "z":
        from misaki import zh

        return zh.ZHG2P()
    from misaki import espeak

    return espeak.EspeakG2P(language=ESPEAK_LANGUAGES[lang])


def phonemize(g2p, text: str) -> str:
    result = g2p(text)
    phonemes = result[0] if isinstance(result, tuple) else result
    return (phonemes or "").strip()
