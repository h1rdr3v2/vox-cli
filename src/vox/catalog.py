"""The curated model catalog shipped with vox (catalog.toml).

A model id names the same model on every platform; each engine family
(mlx on Apple Silicon, portable on Linux) has its own download source.
Loading the catalog resolves every entry to this machine's source.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from vox.config import MODEL_TYPES

SUPPORTED_CATALOG_VERSION = 2
ENGINES = {
    "mlx": {"stt": ("mlx-whisper",), "tts": ("kokoro", "mlx-audio")},
    "portable": {"stt": ("faster-whisper",), "tts": ("kokoro-onnx",)},
}
MODEL_KEYS = {"id", "type", "note", "languages", "recommended", "default_voice", *ENGINES}
SOURCE_KEYS = {"engine", "repo", "size_mb", "revision"}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    type: str
    engine: str
    repo: str
    size_mb: int
    note: str = ""
    languages: str = ""
    recommended: bool = False
    revision: str | None = None
    default_voice: str | None = None


@dataclass(frozen=True)
class Catalog:
    version: int
    backend: str
    entries: tuple[CatalogEntry, ...]

    def get(self, model_id: str) -> CatalogEntry | None:
        return next((e for e in self.entries if e.id == model_id), None)

    def of_type(self, model_type: str) -> list[CatalogEntry]:
        """Entries of one type, recommended first, otherwise in catalog order."""
        entries = [e for e in self.entries if e.type == model_type]
        return sorted(entries, key=lambda e: not e.recommended)

    def recommended(self, model_type: str) -> CatalogEntry | None:
        entries = self.of_type(model_type)
        return entries[0] if entries else None

    def rank(self, model_id: str) -> int:
        """Sort key: recommended entries, then catalog order, then everything else."""
        for i, entry in enumerate(self.entries):
            if entry.id == model_id:
                return i - (1000 if entry.recommended else 0)
        return 10_000


def parse_catalog(text: str, backend: str) -> Catalog:
    """Parse catalog.toml, keeping the entries that have a source for `backend`."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"catalog is not valid TOML: {exc}") from None

    version = data.get("version")
    if not isinstance(version, int):
        raise CatalogError("catalog has no integer 'version'")
    if version > SUPPORTED_CATALOG_VERSION:
        raise CatalogError(f"catalog version {version} is newer than this vox supports")

    entries = []
    seen = set()
    for i, raw in enumerate(data.get("model", [])):
        where = f"model #{i + 1}"
        for key in ("id", "type"):
            if key not in raw:
                raise CatalogError(f"{where} is missing '{key}'")
        model_id = str(raw["id"])
        if model_id in seen:
            raise CatalogError(f"duplicate model id '{model_id}'")
        if model_id.startswith("hf:") or "/" in model_id:
            raise CatalogError(f"{where}: id '{model_id}' must be a short name")
        if raw["type"] not in MODEL_TYPES:
            raise CatalogError(f"{where}: type must be stt or tts, not '{raw['type']}'")
        unknown = set(raw) - MODEL_KEYS
        if unknown:
            raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")
        sources = {b: raw[b] for b in ENGINES if b in raw}
        if not sources:
            raise CatalogError(f"{where}: no source for any engine family ({', '.join(ENGINES)})")
        for family, source in sources.items():
            _check_source(f"{where} ({family})", family, raw["type"], source)
        seen.add(model_id)

        source = sources.get(backend)
        if source is None:
            continue  # not available on this platform
        fields = {k: v for k, v in raw.items() if k not in ENGINES}
        entries.append(CatalogEntry(**fields, **source))
    return Catalog(version=version, backend=backend, entries=tuple(entries))


def _check_source(where: str, family: str, model_type: str, source: object) -> None:
    if not isinstance(source, dict):
        raise CatalogError(f"{where}: must be a table with engine, repo and size_mb")
    for key in ("engine", "repo", "size_mb"):
        if key not in source:
            raise CatalogError(f"{where} is missing '{key}'")
    if source["engine"] not in ENGINES[family][model_type]:
        raise CatalogError(f"{where}: unknown {model_type} engine '{source['engine']}'")
    if str(source["repo"]).count("/") != 1:
        raise CatalogError(f"{where}: repo must look like org/name")
    if not isinstance(source["size_mb"], int) or source["size_mb"] <= 0:
        raise CatalogError(f"{where}: size_mb must be a positive integer")
    unknown = set(source) - SOURCE_KEYS
    if unknown:
        raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")


def load_catalog(backend: str | None = None) -> Catalog:
    """The shipped catalog for `backend` (default: this machine's)."""
    if backend is None:
        from vox import system

        backend = system.backend()
    return _load(backend)


@lru_cache(maxsize=4)
def _load(backend: str) -> Catalog:
    text = resources.files("vox").joinpath("catalog.toml").read_text(encoding="utf-8")
    return parse_catalog(text, backend)
