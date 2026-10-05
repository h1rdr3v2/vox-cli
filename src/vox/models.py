"""Installed models: the on-disk index, pulling, removing and picking defaults.

Each model lives in ~/.cache/vox/models/<slug>/ with a vox-model.json
manifest. The manifest is written last, so a folder without one is an
interrupted download: it is not listed as installed, and pulling again
resumes it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vox import paths
from vox.catalog import CatalogEntry, load_catalog
from vox.config import MODEL_TYPES, Config, load_config, save_config
from vox.engines import engine_class
from vox.engines.base import SupportAsset
from vox.errors import MissingError, UserError

MANIFEST = "vox-model.json"
HF_PREFIX = "hf:"


@dataclass
class ModelSpec:
    """What to download for a model id: a catalog entry or a raw hf: repo."""

    id: str
    type: str
    repo: str
    engine: str | None  # None: decide from config.json (hf: TTS repos)
    revision: str | None = None
    size_mb: int | None = None
    note: str = ""
    languages: str = ""
    default_voice: str | None = None
    from_catalog: bool = True


@dataclass
class InstalledModel:
    id: str
    type: str
    engine: str
    repo: str
    revision: str
    installed_at: str
    files: list[str] = field(default_factory=list)
    languages: str = ""
    note: str = ""
    default_voice: str | None = None
    path: Path = Path()

    @property
    def size_bytes(self) -> int:
        return dir_size(self.path)

    def voices(self) -> list[str]:
        if self.type != "tts":
            return []
        return engine_class(self.engine).list_voices(self.path)

    def fallback_voice(self) -> str | None:
        voices = self.voices()
        if self.default_voice and self.default_voice in voices:
            return self.default_voice
        return engine_class(self.engine).fallback_voice(self.path)


# ---------------------------------------------------------------- lookup


def slug(model_id: str) -> str:
    """Folder name for a model id. hf:org/repo becomes hf--org--repo."""
    if model_id.startswith(HF_PREFIX):
        return "hf--" + model_id[len(HF_PREFIX):].replace("/", "--")
    return model_id


def model_dir(model_id: str) -> Path:
    return paths.models_dir() / slug(model_id)


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _read_manifest(folder: Path) -> InstalledModel | None:
    try:
        data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        fields = set(InstalledModel.__dataclass_fields__) - {"path"}
        model = InstalledModel(**{k: v for k, v in data.items() if k in fields})
    except (OSError, ValueError, TypeError):
        return None
    if model.type not in MODEL_TYPES:
        return None
    model.path = folder
    return model


def installed_models(model_type: str | None = None) -> list[InstalledModel]:
    root = paths.models_dir()
    if not root.is_dir():
        return []
    catalog = load_catalog()
    models = []
    for folder in root.iterdir():
        if folder.is_dir() and (folder / MANIFEST).is_file():
            model = _read_manifest(folder)
            if model and (model_type is None or model.type == model_type):
                models.append(model)
    return sorted(models, key=lambda m: (m.type, catalog.rank(m.id), m.id))


def get_installed_at(folder: Path) -> InstalledModel | None:
    """The model in a folder, or None if it is not a complete install."""
    return _read_manifest(folder) if (folder / MANIFEST).is_file() else None


def get_installed(model_id: str) -> InstalledModel | None:
    folder = model_dir(model_id)
    if not (folder / MANIFEST).is_file():
        return None
    return _read_manifest(folder)


def parse_ref(ref: str, model_type: str | None = None) -> ModelSpec:
    """Turn `whisper-small` or `hf:org/repo` (with a type) into a ModelSpec."""
    ref = ref.strip()
    if ref.startswith(HF_PREFIX):
        repo = ref[len(HF_PREFIX):].strip("/")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise UserError(f"Invalid Hugging Face reference: {ref}.", "Use the form hf:<org>/<repo>")
        if model_type not in MODEL_TYPES:
            raise UserError(
                f"Say what kind of model {ref} is.",
                f"Add --type stt (speech to text) or --type tts (text to speech): vox models pull {ref} --type stt",
            )
        return ModelSpec(
            id=HF_PREFIX + repo,
            type=model_type,
            repo=repo,
            engine="mlx-whisper" if model_type == "stt" else None,
            from_catalog=False,
        )
    entry = load_catalog().get(ref)
    if entry is None:
        raise UserError(
            f"Unknown model: {ref}.",
            "See the catalog with: vox models list --available (or pull any repo as hf:<org>/<repo> --type stt|tts)",
        )
    if model_type and model_type != entry.type:
        raise UserError(f"{ref} is a {entry.type} model, not {model_type}.", f"Drop --type, or use --type {entry.type}.")
    return spec_from_entry(entry)


def spec_from_entry(entry: CatalogEntry) -> ModelSpec:
    return ModelSpec(
        id=entry.id,
        type=entry.type,
        repo=entry.repo,
        engine=entry.engine,
        revision=entry.revision,
        size_mb=entry.size_mb,
        note=entry.note,
        languages=entry.languages,
        default_voice=entry.default_voice,
    )


# ------------------------------------------------------------- resolution


def type_label(model_type: str) -> str:
    return "STT" if model_type == "stt" else "TTS"


def pick_installed(model_type: str, requested: str | None, cfg: Config | None = None, *, strict: bool) -> InstalledModel | None:
    """Choose the model for a request.

    With strict=True (the CLI), a requested id that is not installed is an
    error. With strict=False (the HTTP API), it falls back to the default,
    as the OpenAI-style API is called with names like "whisper-1".
    Otherwise: the configured default, then the best installed model of the
    type. Returns None only when nothing of that type is installed.
    """
    cfg = cfg or load_config()
    if requested:
        model = get_installed(requested)
        if model and model.type == model_type:
            return model
        if strict:
            if model:
                raise UserError(f"{requested} is a {model.type} model, but this needs a {model_type} model.", f"See {model_type} models with: vox models list --type {model_type}")
            catalog_entry = load_catalog().get(requested)
            if catalog_entry or requested.startswith(HF_PREFIX):
                type_flag = f" --type {model_type}" if requested.startswith(HF_PREFIX) else ""
                raise MissingError(f"Model {requested} is not installed.", f"Run: vox models pull {requested}{type_flag}")
            raise UserError(f"Unknown model: {requested}.", "See installed models with: vox models list --installed")
    default_id = cfg.default_for(model_type)
    if default_id:
        model = get_installed(default_id)
        if model and model.type == model_type:
            return model
    installed = installed_models(model_type)
    return installed[0] if installed else None


def no_model_error(model_type: str) -> MissingError:
    recommended = load_catalog().recommended(model_type)
    target = recommended.id if recommended else "<id>"
    return MissingError(f"No {type_label(model_type)} model installed.", f"Run: vox models pull {target}")


# ------------------------------------------------------------------ pull


class PullProgress:
    """Callbacks used while pulling. The CLI renders them as a progress bar."""

    def start(self, total_bytes: int, already: int) -> None: ...

    def update(self, done_bytes: int) -> None: ...

    def done(self) -> None: ...


class _DiskWatcher(threading.Thread):
    """Reports download progress by measuring bytes on disk.

    Works the same for plain HTTP and Xet downloads, and for resumed ones:
    finished files plus the partial files huggingface_hub and vox write.
    """

    def __init__(self, measure: Callable[[], int], progress: PullProgress):
        super().__init__(daemon=True)
        self.measure, self.progress = measure, progress
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(0.2):
            try:
                self.progress.update(self.measure())
            except OSError:
                pass

    def stop(self) -> None:
        self.stopped.set()
        self.join(timeout=1)


def _network_error(spec: ModelSpec, exc: Exception) -> MissingError:
    return MissingError(
        f"Could not download {spec.id} from Hugging Face ({type(exc).__name__}: {exc}).",
        f"Check your connection and run vox models pull {spec.id} again; it resumes where it stopped.",
    )


def _repo_listing(spec: ModelSpec) -> tuple[str, dict[str, int]]:
    """(commit sha, {filename: size}) for the repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        info = HfApi().model_info(spec.repo, revision=spec.revision, files_metadata=True)
    except GatedRepoError:
        raise UserError(
            f"{spec.repo} is gated on Hugging Face and needs an access token.",
            "Accept its terms on huggingface.co, then run: hf auth login",
        ) from None
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise UserError(f"Hugging Face repo not found: {spec.repo}.", "Check the spelling of <org>/<repo>.") from None
    files = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
    return info.sha, files


def _support_part(asset: SupportAsset, folder: Path) -> Path:
    return folder / "_vox" / "downloads" / (Path(asset.url).name + ".part")


def _support_done(asset: SupportAsset, folder: Path) -> bool:
    return (folder / asset.target / ".vox-complete").exists()


def _download_support_asset(asset: SupportAsset, folder: Path) -> None:
    import httpx

    if _support_done(asset, folder):
        return
    part = _support_part(asset, folder)
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > asset.size:
        part.unlink()
        have = 0
    if have < asset.size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with httpx.stream("GET", asset.url, headers=headers, follow_redirects=True, timeout=30) as response:
            if response.status_code == 200 and have:
                part.unlink()  # the server ignored the range request: start over
            elif response.status_code not in (200, 206):
                response.raise_for_status()
            with open(part, "ab") as fh:
                for chunk in response.iter_bytes(1 << 16):
                    fh.write(chunk)

    digest = hashlib.sha256()
    with open(part, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        part.unlink()
        raise MissingError(f"The download of {asset.name} was corrupted.", "Run the same vox models pull command again.")
    target = folder / asset.target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(part) as wheel:
        wheel.extractall(target)
    (target / ".vox-complete").touch()
    part.unlink()


def _measure(folder: Path, files: dict[str, int], assets: tuple[SupportAsset, ...]) -> int:
    done = 0
    for name, size in files.items():
        local = folder / name
        if local.exists():
            done += min(local.stat().st_size, size)
    partial = folder / ".cache" / "huggingface" / "download"
    if partial.is_dir():
        for incomplete in partial.rglob("*.incomplete"):
            try:
                done += incomplete.stat().st_size
            except OSError:
                pass
    for asset in assets:
        if _support_done(asset, folder):
            done += asset.size
        else:
            part = _support_part(asset, folder)
            if part.exists():
                done += min(part.stat().st_size, asset.size)
    return done


def _write_json_atomic(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def pull(spec: ModelSpec, progress: PullProgress | None = None) -> InstalledModel:
    """Download a model (resuming a partial download) and write its manifest."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hf_logging

    disable_progress_bars()  # vox draws its own
    hf_logging.set_verbosity_error()  # no "set HF_TOKEN" nags; errors still raise
    progress = progress or PullProgress()
    folder = model_dir(spec.id)
    folder.mkdir(parents=True, exist_ok=True)
    watcher = None

    try:
        sha, repo_files = _repo_listing(spec)
        if "config.json" not in repo_files:
            raise UserError(f"{spec.repo} has no config.json, so it is not a model vox can run.", "Pick an MLX conversion, for example from huggingface.co/mlx-community")

        # Find the engine and check the repo is a model it can run.
        config_path = Path(hf_hub_download(spec.repo, "config.json", revision=sha, local_dir=folder))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        engine_name = spec.engine
        if engine_name is None:
            try:
                engine_class("kokoro").check_config(config)
                engine_name = "kokoro"
            except ValueError:
                engine_name = "mlx-audio"
        engine = engine_class(engine_name)
        try:
            engine.check_config(config)
            files = engine.select_files(sorted(repo_files))
        except ValueError as exc:
            raise UserError(f"{spec.repo} cannot be used: {exc}.", "Pick an MLX conversion, for example from huggingface.co/mlx-community") from None

        sizes = {name: repo_files[name] for name in files}
        assets = engine.support_assets
        total = sum(sizes.values()) + sum(a.size for a in assets)
        progress.start(total, _measure(folder, sizes, assets))
        watcher = _DiskWatcher(lambda: _measure(folder, sizes, assets), progress)
        watcher.start()

        for name in files:
            local = folder / name
            if not (local.exists() and local.stat().st_size == sizes[name]):
                hf_hub_download(spec.repo, name, revision=sha, local_dir=folder)
        for asset in assets:
            _download_support_asset(asset, folder)
        progress.update(_measure(folder, sizes, assets))
    except (UserError, MissingError):
        raise
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            need = human_size(spec.size_mb * 1e6) if spec.size_mb else "more space"
            raise UserError(
                f"Not enough disk space to download {spec.id}.",
                f"Free up {need} and run vox models pull {spec.id} again; it resumes where it stopped.",
            ) from None
        if _is_network_error(exc):
            raise _network_error(spec, exc) from None
        raise
    except Exception as exc:
        if _is_network_error(exc):
            raise _network_error(spec, exc) from None
        raise
    finally:
        if watcher:
            watcher.stop()

    # Leave only what vox uses.
    shutil.rmtree(folder / ".cache", ignore_errors=True)
    shutil.rmtree(folder / "_vox" / "downloads", ignore_errors=True)

    installed = InstalledModel(
        id=spec.id,
        type=spec.type,
        engine=engine_name,
        repo=spec.repo,
        revision=sha,
        installed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        files=files,
        languages=spec.languages,
        note=spec.note,
        default_voice=spec.default_voice,
    )
    data = asdict(installed)
    data.pop("path")
    _write_json_atomic(folder / MANIFEST, data)
    installed.path = folder
    progress.done()
    return installed


def _is_network_error(exc: Exception) -> bool:
    import socket

    import httpx
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    network = (httpx.HTTPError, HfHubHTTPError, LocalEntryNotFoundError, ConnectionError, TimeoutError, socket.gaierror)
    return isinstance(exc, network)


# -------------------------------------------------------------- remove/default


def remove(model_id: str) -> tuple[InstalledModel | None, bool]:
    """Delete a model folder. Returns (model, was_default). Also clears partial downloads."""
    folder = model_dir(model_id)
    model = get_installed(model_id)
    if model is None and not folder.exists():
        raise UserError(f"Model {model_id} is not installed.", "See installed models with: vox models list --installed")
    shutil.rmtree(folder)
    was_default = False
    if model is not None:
        cfg = load_config()
        if cfg.default_for(model.type) == model.id:
            cfg.set_default_for(model.type, None)
            save_config(cfg)
            was_default = True
    return model, was_default


def set_default(model_id: str) -> InstalledModel:
    model = get_installed(model_id)
    if model is None:
        if load_catalog().get(model_id) or model_id.startswith(HF_PREFIX):
            raise MissingError(f"Model {model_id} is not installed.", f"Run: vox models pull {model_id} first.")
        raise UserError(f"Unknown model: {model_id}.", "See installed models with: vox models list --installed")
    cfg = load_config()
    cfg.set_default_for(model.type, model.id)
    save_config(cfg)
    return model


def ensure_default_after_pull(model: InstalledModel) -> bool:
    """Make a freshly pulled model the default if its type has none. Returns True if set."""
    cfg = load_config()
    current = cfg.default_for(model.type)
    if current and get_installed(current):
        return False
    cfg.set_default_for(model.type, model.id)
    save_config(cfg)
    return True


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB", "MB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"

