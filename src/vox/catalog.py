"""The curated model catalog shipped with vox (catalog.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from vox.config import MODEL_TYPES

SUPPORTED_CATALOG_VERSION = 1
ENGINES = {"stt": ("mlx-whisper",), "tts": ("kokoro", "mlx-audio")}


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


def parse_catalog(text: str) -> Catalog:
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
        for key in ("id", "type", "engine", "repo", "size_mb"):
            if key not in raw:
                raise CatalogError(f"{where} is missing '{key}'")
        model_id = str(raw["id"])
        if model_id in seen:
            raise CatalogError(f"duplicate model id '{model_id}'")
        if model_id.startswith("hf:") or "/" in model_id:
            raise CatalogError(f"{where}: id '{model_id}' must be a short name")
        if raw["type"] not in MODEL_TYPES:
            raise CatalogError(f"{where}: type must be stt or tts, not '{raw['type']}'")
        if raw["engine"] not in ENGINES[raw["type"]]:
            raise CatalogError(f"{where}: unknown {raw['type']} engine '{raw['engine']}'")
        if not isinstance(raw["size_mb"], int) or raw["size_mb"] <= 0:
            raise CatalogError(f"{where}: size_mb must be a positive integer")
        unknown = set(raw) - set(CatalogEntry.__dataclass_fields__)
        if unknown:
            raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")
        seen.add(model_id)
        entries.append(CatalogEntry(**{**raw, "id": model_id}))
    return Catalog(version=version, entries=tuple(entries))


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    text = resources.files("vox").joinpath("catalog.toml").read_text(encoding="utf-8")
    return parse_catalog(text)
