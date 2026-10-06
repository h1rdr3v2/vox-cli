"""The `vox` command line. A thin client over the local HTTP server."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Annotated, Optional

import click
import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from vox import __version__, models, paths
from vox.catalog import load_catalog
from vox.config import (
    FORMATS,
    KEYS,
    MODEL_TYPES,
    Config,
    load_config,
    save_config,
    set_value,
)
from vox.errors import MissingError, UserError, VoxError
from vox.media import format_duration, require_ffmpeg, to_wav16k
from vox.models import InstalledModel, human_size, type_label

# Piped output is not wrapped at 80 columns.
out = Console(highlight=False, width=None if sys.stdout.isatty() else 200)
err = Console(stderr=True, highlight=False, width=None if sys.stderr.isatty() else 200)

_debug = "--debug" in sys.argv or os.environ.get("VOX_DEBUG") == "1"

FORMAT_EXT = {"txt": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}
API_FORMAT = {"txt": "text", "srt": "srt", "vtt": "vtt", "json": "verbose_json"}
AUDIO_EXT = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".opus": "opus", ".ogg": "opus", ".aac": "aac", ".pcm": "pcm"}
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text")
KOKORO_ENGINES = ("kokoro", "kokoro-onnx")

app = typer.Typer(
    name="vox",
    help="Local transcription (Whisper) and speech (Kokoro) for Apple Silicon Macs and Linux.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
models_app = typer.Typer(help="Browse, download, remove and choose models.", no_args_is_help=True)
setup_app = typer.Typer(help="Integrations with macOS.", no_args_is_help=True)
config_app = typer.Typer(help="Show or change settings in ~/.config/vox/config.toml.", invoke_without_command=True)
app.add_typer(models_app, name="models")
app.add_typer(setup_app, name="setup")
app.add_typer(config_app, name="config")


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def say(message: str) -> None:
    err.print(message)


def ok(message: str) -> None:
    err.print(f"[green]✓[/green] {message}")


# ------------------------------------------------------------------ root


def _version(value: bool) -> None:
    if value:
        out.print(f"vox {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on errors.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    global _debug
    _debug = _debug or debug


# ------------------------------------------------------- model selection


class RichPullProgress(models.PullProgress):
    def __init__(self, label: str):
        self.label = label
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=err,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = None

    def start(self, total_bytes: int, already: int) -> None:
        if already > 1_000_000:
            say(f"Resuming {self.label}: {human_size(already)} of {human_size(total_bytes)} already downloaded")
        self.task = self.progress.add_task(self.label, total=total_bytes, completed=already)
        self.progress.start()

    def update(self, done_bytes: int) -> None:
        if self.task is not None:
            self.progress.update(self.task, completed=done_bytes)

    def done(self) -> None:
        self.progress.stop()


class PlainPullProgress(models.PullProgress):
    """For non-terminals (scripts, Quick Actions): a line at start, none per chunk."""

    def __init__(self, label: str):
        self.label = label

    def start(self, total_bytes: int, already: int) -> None:
        resumed = f", {human_size(already)} already here" if already > 1_000_000 else ""
        say(f"Downloading {self.label} ({human_size(total_bytes)}{resumed})")


def pull_model(spec: models.ModelSpec) -> InstalledModel:
    label = spec.id
    progress = RichPullProgress(label) if err.is_terminal else PlainPullProgress(label)
    try:
        model = models.pull(spec, progress)
    except KeyboardInterrupt:
        progress.done()
        raise UserError(
            f"Download of {spec.id} stopped.",
            f"Run vox models pull {spec.id} again to resume where it stopped.",
            exit_code=130,
        ) from None
    except BaseException:
        progress.done()
        raise
    ok(f"Installed {model.id} ({human_size(model.size_bytes)})")
    return model


def first_run_pick(model_type: str) -> InstalledModel:
    """Interactive picker shown when a command needs a model and none is installed."""
    entries = load_catalog().of_type(model_type)
    kind = "speech-to-text" if model_type == "stt" else "text-to-speech"
    say(f"No {type_label(model_type)} ({kind}) model is installed yet. Pick one to download:\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("#", "MODEL", "SIZE", "NOTE"):
        table.add_column(column)
    for i, entry in enumerate(entries, start=1):
        table.add_row(str(i), entry.id, "~" + human_size(entry.size_mb * 1e6), entry.note)
    err.print(table)
    while True:
        answer = typer.prompt("\nModel number or id", default="1", err=True).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(entries):
            entry = entries[int(answer) - 1]
            break
        entry = next((e for e in entries if e.id == answer), None)
        if entry:
            break
        say(f"Enter a number from 1 to {len(entries)}.")
    model = pull_model(models.spec_from_entry(entry))
    models.set_default(model.id)
    ok(f"{model.id} is now the default {type_label(model_type)} model.")
    return model


def require_model(model_type: str, requested: str | None, cfg: Config) -> InstalledModel:
    model = models.pick_installed(model_type, requested, cfg, strict=True)
    if model:
        return model
    if not interactive():
        raise models.no_model_error(model_type)
    return first_run_pick(model_type)


def connect(cfg: Config, progress: Progress | None = None, task=None):
    from vox import client

    def on_spawn():
        if progress is not None and task is not None:
            progress.update(task, description="Starting the vox server")

    def reconnect():
        return client.ensure_server(cfg, on_spawn)

    return client.Client(client.ensure_server(cfg, on_spawn), reconnect=reconnect)


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
        disable=not err.is_terminal,
        # Keep stdout untouched: transcripts and audio may be piped from it.
        redirect_stdout=False,
        redirect_stderr=False,
    )


class ProgressFile:
    """Keeps a file updated with the fraction of the work done, 0 to 1.

    For scripts and the Finder Quick Actions, which read it to show progress.
    Each write replaces the file whole, so a reader never sees half a number.
    """

    def __init__(self, path: Path | None, parts: int = 1):
        self.path = path
        self.parts = max(parts, 1)
        self.part = 0
        self._last = -1.0
        self._lock = threading.Lock()
        self.update(0.0)

    def start_part(self, index: int) -> None:
        self.part = index
        self.update(0.0)

    def update(self, fraction: float) -> None:
        """Progress within the current part (one file of several)."""
        if self.path is None:
            return
        value = min(1.0, (self.part + min(max(fraction, 0.0), 1.0)) / self.parts)
        with self._lock:
            if 0.0 < value < 1.0 and abs(value - self._last) < 0.002:
                return
            self._last = value
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            try:
                tmp.write_text(f"{value:.4f}\n", encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass  # progress is a convenience; never fail the job over it

    def finish(self) -> None:
        self.part = self.parts
        self.update(0.0)


def _job_updater(progress: Progress, task, model_id: str, working: str, report: ProgressFile | None = None):
    def on_job(job: dict) -> None:
        state = job.get("state")
        if state == "queued":
            progress.update(task, description="Waiting for another vox job to finish")
        elif state == "loading":
            progress.update(task, description=f"Loading {model_id}")
        elif state == "running":
            total = job.get("total") or None
            progress.update(task, description=working, total=total, completed=job.get("done", 0))
            if report and total:
                report.update(job.get("done", 0) / total)

    return on_job


# ------------------------------------------------------------ transcribe


def _transcript_target(src: Path, fmt: str, out_opt: str | None, many: bool) -> Path | None:
    if out_opt == "-":
        return None
    if out_opt is None:
        target = src.with_suffix(FORMAT_EXT[fmt])
    else:
        given = Path(out_opt).expanduser()
        if given.is_dir() or out_opt.endswith("/") or many:
            given.mkdir(parents=True, exist_ok=True)
            target = given / (src.stem + FORMAT_EXT[fmt])
        else:
            target = given
    if target.resolve() == src.resolve():
        raise UserError(f"Writing the transcript to {target.name} would overwrite the input.", "Choose another --out path or --format.")
    return target


@app.command()
def transcribe(
    files: Annotated[list[Path], typer.Argument(help="Audio or video files.", show_default=False)],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model id (default: your default STT model).")] = None,
    fmt: Annotated[Optional[str], typer.Option("--format", "-f", help="txt, srt, vtt or json.", show_default=False)] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Language code such as en or fr (default: detect).")] = None,
    out_opt: Annotated[
        Optional[str], typer.Option("--out", "-o", help="Output file or folder. Use - for stdout. Default: next to the input.")
    ] = None,
    progress_file: Annotated[
        Optional[Path], typer.Option("--progress-file", help="Keep this file updated with the progress (0 to 1), for scripts.")
    ] = None,
) -> None:
    """Transcribe audio or video to text."""
    cfg = load_config()
    if fmt is None and out_opt not in (None, "-") and Path(out_opt).suffix.lower() in {".txt", ".srt", ".vtt", ".json"}:
        fmt = Path(out_opt).suffix.lower()[1:]
    fmt = (fmt or cfg.default_format).lower()
    if fmt not in FORMATS:
        raise UserError(f"Unknown format: {fmt}.", "Use one of: " + ", ".join(FORMATS))
    for f in files:
        if not f.exists():
            raise UserError(f"File not found: {f}", "Check the path; quote names that contain spaces.")
        if f.is_dir():
            raise UserError(f"{f} is a folder.", "Pass audio or video files, for example: vox transcribe folder/*.mp4")
    many = len(files) > 1
    if many and out_opt not in (None, "-") and Path(out_opt).expanduser().is_file():
        raise UserError("--out must be a folder when transcribing several files.", "Pass a folder, for example: --out transcripts/")
    require_ffmpeg()
    stt = require_model("stt", model, cfg)

    failures = 0
    report = ProgressFile(progress_file, parts=len(files))
    with _progress() as progress:
        task = progress.add_task("Connecting to vox", total=None)
        client = connect(cfg, progress, task)
        for index, src in enumerate(files, start=1):
            prefix = f"[{index}/{len(files)}] " if many else ""
            started = time.monotonic()
            report.start_part(index - 1)
            try:
                target = _transcript_target(src, fmt, out_opt, many)
                progress.update(task, description=f"{prefix}Converting {src.name}", total=None, completed=0)
                with tempfile.TemporaryDirectory(prefix="vox-") as tmp:
                    wav = Path(tmp) / (src.stem + ".wav")
                    duration = to_wav16k(src, wav)
                    working = f"{prefix}Transcribing {src.name} ({format_duration(duration)})"
                    progress.update(task, description=working)
                    response = client.transcribe(
                        wav,
                        model=stt.id,
                        response_format=API_FORMAT[fmt],
                        language=language,
                        on_job=_job_updater(progress, task, stt.id, working, report),
                    )
                text = response.text
                if target is None:
                    progress.stop()
                    sys.stdout.write(text if text.endswith("\n") else text + "\n")
                    sys.stdout.flush()
                    progress.start()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
                elapsed = time.monotonic() - started
                lang = response.headers.get("x-vox-language")
                where = "stdout" if target is None else paths.pretty(target)
                progress.console.print(
                    f"[green]✓[/green] {prefix}{escape(src.name)} -> {escape(where)}"
                    f" [dim]({format_duration(duration)} of audio in {elapsed:.1f}s"
                    f"{', ' + lang if lang else ''}, {stt.id})[/dim]"
                )
            except VoxError as exc:
                if not many:
                    raise
                failures += 1
                progress.console.print(f"[red]✗[/red] {prefix}{escape(src.name)}: {escape(exc.message)}")
    report.finish()
    if failures:
        raise UserError(f"{failures} of {len(files)} files failed.", "See the errors above; the other files were transcribed.")


# ----------------------------------------------------------------- speak


def _read_speak_input(value: str) -> tuple[str, Path | None]:
    from vox.text import clean_markdown

    if value == "-":
        if sys.stdin.isatty():
            say("Type the text, then press Ctrl-D.")
        return sys.stdin.read(), None
    candidate = Path(value).expanduser()
    if "\n" not in value and len(value) < 1024 and candidate.is_file():
        data = candidate.read_bytes()
        if b"\x00" in data[:4096]:
            raise UserError(f"{candidate.name} is not a text file.", "Pass a .txt or .md file, or the text itself in quotes.")
        text = data.decode("utf-8", errors="replace")
        if candidate.suffix.lower() in (".md", ".markdown"):
            text = clean_markdown(text)
        return text, candidate
    if "\n" not in value and " " not in value and candidate.suffix.lower() in TEXT_SUFFIXES:
        raise UserError(f"File not found: {value}", "Check the path, or quote the text you want spoken.")
    return value, None


def _speech_target(source: Path | None, out_opt: str | None, play: bool) -> tuple[Path | None, str, bool]:
    """(path or None for stdout, format, is_temporary)."""
    if out_opt == "-":
        return None, "wav", False
    if out_opt:
        target = Path(out_opt).expanduser()
        if target.is_dir():
            target = target / ((source.stem if source else "speech") + ".wav")
        if not target.suffix:
            target = target.with_suffix(".wav")
        fmt = AUDIO_EXT.get(target.suffix.lower())
        if fmt is None:
            raise UserError(f"Unsupported audio format: {target.suffix}.", "Use .wav or .mp3 (also .flac, .opus, .aac).")
        return target, fmt, False
    if play:
        handle, tmp = tempfile.mkstemp(prefix="vox-", suffix=".wav")
        os.close(handle)
        return Path(tmp), "wav", True
    if source:
        return source.with_suffix(".wav"), "wav", False
    return Path.cwd() / "speech.wav", "wav", False


@app.command()
def speak(
    text: Annotated[str, typer.Argument(metavar="TEXT|FILE|-", help="Text to say, a .txt or .md file, or - for stdin.")],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model id (default: your default TTS model).")] = None,
    voice: Annotated[Optional[str], typer.Option("--voice", "-v", help="Voice name. See: vox voices.")] = None,
    speed: Annotated[float, typer.Option("--speed", "-s", help="Speaking rate, 0.5 to 2.0.")] = 1.0,
    out_opt: Annotated[
        Optional[str], typer.Option("--out", "-o", help="Output .wav or .mp3. Use - for stdout.", show_default=False)
    ] = None,
    play: Annotated[bool, typer.Option("--play", "-p", help="Play the audio when done.")] = False,
    progress_file: Annotated[
        Optional[Path], typer.Option("--progress-file", help="Keep this file updated with the progress (0 to 1), for scripts.")
    ] = None,
) -> None:
    """Turn text into speech.

    Writes a .wav next to an input file, or speech.wav in this folder for
    literal text. With --play and no --out, nothing is saved.
    """
    cfg = load_config()
    if not 0.5 <= speed <= 2.0:
        raise UserError("--speed must be between 0.5 and 2.0.", "For example: --speed 1.2")
    content, source = _read_speak_input(text)
    if not content.strip():
        raise UserError("Nothing to say: the text is empty.", 'Pass some text, for example: vox speak "Good morning"')
    target, fmt, temporary = _speech_target(source, out_opt, play)
    if fmt not in ("wav", "pcm"):
        require_ffmpeg()
    tts = require_model("tts", model, cfg)

    write_to = target or Path(tempfile.mkstemp(prefix="vox-", suffix=".wav")[1])
    report = ProgressFile(progress_file)
    try:
        with _progress() as progress:
            task = progress.add_task("Connecting to vox", total=None)
            client = connect(cfg, progress, task)
            working = f"Speaking {len(content.split())} words"
            progress.update(task, description=working)
            response = client.speak(
                content,
                model=tts.id,
                voice=voice,
                speed=speed,
                response_format=fmt,
                out=write_to,
                on_job=_job_updater(progress, task, tts.id, working, report),
            )
        report.finish()
        seconds = float(response.headers.get("x-vox-duration") or 0)
        used_voice = response.headers.get("x-vox-voice") or voice or ""
        details = f"[dim]({seconds:.1f}s of audio, {tts.id}{', ' + used_voice if used_voice else ''})[/dim]"
        if target is None:
            sys.stdout.buffer.write(write_to.read_bytes())
            sys.stdout.flush()
        elif not temporary:
            ok(f"Saved {escape(paths.pretty(target))} {details}")
        if play:
            if temporary:
                say(f"Playing {details}")
            _play(write_to)
    finally:
        if temporary or target is None:
            write_to.unlink(missing_ok=True)


def _play(path: Path) -> None:
    from vox.system import audio_player, playback_error

    proc = subprocess.Popen(audio_player(path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, stderr = proc.communicate()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    problem = playback_error(proc.returncode, stderr.decode(errors="replace"))
    if problem:
        raise UserError(problem, "Save the audio instead with --out speech.wav, or check your sound output.")


@app.command()
def voices(
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="TTS model id (default: your default TTS model).")] = None,
) -> None:
    """List the voices of a TTS model."""
    from vox.engines.kokoro_text import describe_voice

    cfg = load_config()
    tts = require_model("tts", model, cfg)
    names = tts.voices()
    if not names:
        say(f"{tts.id} has no preset voices.")
        return
    default = cfg.default_voice if cfg.default_voice in names else tts.fallback_voice()
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in ("VOICE", "LANGUAGE", "GENDER", "DEFAULT"):
        table.add_column(column)
    for name in names:
        language, gender = describe_voice(name) if tts.engine in KOKORO_ENGINES else ("", "")
        table.add_row(name, language, gender, "*" if name == default else "")
    out.print(table)
    say(f"[dim]{len(names)} voices in {tts.id}. Use one with: vox speak \"Hello\" --voice {default or names[0]}[/dim]")


# ---------------------------------------------------------------- models


def _check_type(value: str | None) -> str | None:
    if value is not None and value not in MODEL_TYPES:
        raise UserError(f"Unknown model type: {value}.", "Use stt (speech to text) or tts (text to speech).")
    return value


@models_app.command("list")
def models_list(
    installed: Annotated[bool, typer.Option("--installed", help="Only models on disk.")] = False,
    available: Annotated[bool, typer.Option("--available", help="The full catalog.")] = False,
    model_type: Annotated[Optional[str], typer.Option("--type", "-t", help="stt or tts.")] = None,
) -> None:
    """List models: the catalog plus anything installed."""
    _check_type(model_type)
    if installed and available:
        raise UserError("Use either --installed or --available, not both.", "Or neither, to see the catalog and your models together.")
    cfg = load_config()
    catalog = load_catalog()
    on_disk = {m.id: m for m in models.installed_models(model_type)}
    partial = {p.name for p in paths.models_dir().iterdir() if p.is_dir() and not (p / models.MANIFEST).exists()} if paths.models_dir().is_dir() else set()

    rows = []
    if installed:
        rows = [(m.id, m.type, m, None) for m in on_disk.values()]
    else:
        for entry in catalog.entries:
            if model_type in (None, entry.type):
                rows.append((entry.id, entry.type, on_disk.get(entry.id), entry))
        if not available:
            rows += [(m.id, m.type, m, None) for m in on_disk.values() if catalog.get(m.id) is None]

    if not rows:
        what = f"{type_label(model_type)} models" if model_type else "models"
        say(f"No {what} installed.")
        say("[dim]Browse the catalog with: vox models list --available[/dim]")
        return

    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in ("ID", "TYPE", "SIZE", "INSTALLED", "DEFAULT", "NOTE"):
        table.add_column(column)
    for model_id, mtype, model, entry in rows:
        if model:
            size = human_size(model.size_bytes)
            state = "yes"
        else:
            size = "~" + human_size(entry.size_mb * 1e6) if entry else ""
            state = "partial" if models.slug(model_id) in partial else "no"
        is_default = "*" if cfg.default_for(mtype) == model_id and model else ""
        note = entry.note if entry else (model.note if model else "")
        table.add_row(model_id, mtype, size, state, is_default, note)
    out.print(table)


@models_app.command("pull")
def models_pull(
    model_id: Annotated[str, typer.Argument(metavar="ID", help="Catalog id, or hf:<org>/<repo> with --type.")],
    model_type: Annotated[Optional[str], typer.Option("--type", "-t", help="stt or tts (needed for hf: repos).")] = None,
) -> None:
    """Download a model. Resumes if a previous download was interrupted."""
    _check_type(model_type)
    spec = models.parse_ref(model_id, model_type)
    existing = models.get_installed(spec.id)
    if existing:
        ok(f"{spec.id} is already installed ({human_size(existing.size_bytes)}).")
        return
    model = pull_model(spec)
    if models.ensure_default_after_pull(model):
        ok(f"{model.id} is now the default {type_label(model.type)} model.")


@models_app.command("rm")
def models_rm(model_id: Annotated[str, typer.Argument(metavar="ID")]) -> None:
    """Delete a model from disk."""
    from vox import client

    model, was_default = models.remove(model_id)
    if model is None:
        ok(f"Removed the partial download of {model_id}.")
        return
    ok(f"Removed {model_id}.")
    if was_default:
        say(f"It was the default {type_label(model.type)} model, so there is no default now.")
        remaining = models.installed_models(model.type)
        if remaining:
            say(f"[dim]Set one with: vox models default {remaining[0].id}[/dim]")
    server = client.find_server()
    if server:
        say("[dim]The running server may still hold it in memory until it exits (vox stop).[/dim]")


@models_app.command("default")
def models_default(model_id: Annotated[str, typer.Argument(metavar="ID")]) -> None:
    """Set the default model for that model's type (STT and TTS each have one)."""
    model = models.set_default(model_id)
    ok(f"Default {type_label(model.type)} model: {model.id}")


@models_app.command("info")
def models_info(model_id: Annotated[str, typer.Argument(metavar="ID")]) -> None:
    """Show a model's source, size, languages and voices."""
    from vox.engines.kokoro_text import describe_voice

    cfg = load_config()
    model = models.get_installed(model_id)
    entry = load_catalog().get(model_id)
    if model is None and entry is None:
        if model_id.startswith(models.HF_PREFIX):
            raise MissingError(f"Model {model_id} is not installed.", f"Run: vox models pull {model_id} --type stt|tts")
        raise UserError(f"Unknown model: {model_id}.", "See the catalog with: vox models list --available")

    rows: list[tuple[str, str]] = [("id", model_id)]
    mtype = model.type if model else entry.type
    rows.append(("type", f"{mtype} ({'speech to text' if mtype == 'stt' else 'text to speech'})"))
    repo = model.repo if model else entry.repo
    rows.append(("source", f"https://huggingface.co/{repo}"))
    if model:
        rows.append(("revision", model.revision[:12]))
        rows.append(("engine", model.engine))
        rows.append(("size on disk", human_size(model.size_bytes)))
        rows.append(("installed", f"yes, {model.installed_at}"))
        rows.append(("path", paths.pretty(model.path)))
    else:
        rows.append(("download size", "~" + human_size(entry.size_mb * 1e6)))
        rows.append(("installed", f"no (vox models pull {model_id})"))
    rows.append(("default", "yes" if model and cfg.default_for(mtype) == model_id else "no"))
    languages = (model.languages if model else "") or (entry.languages if entry else "")
    if languages:
        rows.append(("languages", languages))
    note = (entry.note if entry else "") or (model.note if model else "")
    if note:
        rows.append(("note", note))
    if mtype == "tts":
        if model:
            names = model.voices()
            if names and model.engine in KOKORO_ENGINES:
                grouped: dict[str, list[str]] = {}
                for name in names:
                    grouped.setdefault(describe_voice(name)[0] or "Other", []).append(name)
                rows.append(("voices", f"{len(names)} (default {model.fallback_voice()})"))
                for language, group in grouped.items():
                    rows.append(("", f"{language}: {', '.join(group)}"))
            else:
                rows.append(("voices", ", ".join(names) or "none"))
        else:
            rows.append(("voices", "listed after install"))
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows:
        table.add_row(key, escape(value))
    out.print(table)


# -------------------------------------------------------------- server


@app.command()
def start() -> None:
    """Start the server in the background and return, for apps that call the HTTP API.

    Like the server transcribe and speak start, it exits after the idle timeout.
    """
    from vox import client

    cfg = load_config()
    info = client.ensure_server(cfg)
    mode = "persistent" if info.persistent else f"exits after {format_duration(cfg.idle_timeout)} without requests"
    ok(f"vox server running on http://127.0.0.1:{info.port}/v1 ({mode})")


@app.command()
def serve(
    persistent: Annotated[bool, typer.Option("--persistent", help="Never exit on idle (models still load lazily).")] = False,
    port: Annotated[Optional[int], typer.Option("--port", help="Port on 127.0.0.1 (default from config: 8880).")] = None,
    idle_timeout: Annotated[
        Optional[str], typer.Option("--idle-timeout", help="Exit after this long without requests, e.g. 300, 5m.")
    ] = None,
    spawned: Annotated[bool, typer.Option("--spawned", hidden=True)] = False,
) -> None:
    """Run the server in the foreground (for launchd, systemd or debugging).

    To start it in the background, use vox start. transcribe and speak start
    it on their own. Use --persistent for an always-on server.
    """
    from vox import client
    from vox.config import parse_duration, parse_port
    from vox.server.main import run_server

    if not spawned and client.find_server():
        raise UserError("A vox server is already running.", "See it with vox status, or stop it with vox stop")
    cfg = load_config()
    port = parse_port(port, "--port") if port is not None else cfg.port
    timeout = parse_duration(idle_timeout, "--idle-timeout") if idle_timeout is not None else cfg.idle_timeout
    if not spawned:
        mode = "persistent" if persistent else f"exits after {format_duration(timeout)} idle"
        say(f"vox server on http://127.0.0.1:{port}/v1 ({mode}). Press Ctrl-C to stop.")
    run_server(port=port, persistent=persistent, idle_timeout=timeout, spawned=spawned)


@app.command()
def status() -> None:
    """Show whether the server is running, its loaded models and memory use."""
    from vox import client

    info = client.find_server()
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    if info is None:
        out.print("vox server: [bold]not running[/bold] (using no memory)")
        table.add_row("log", paths.pretty(paths.log_file()))
        out.print(table)
        return
    data = client.Client(info).status()
    mode = "persistent" if data["persistent"] else f"on-demand, exits after {format_duration(data['idle_timeout'])} idle"
    out.print(f"vox server: [bold green]running[/bold green] ({mode})")
    table.add_row("pid", str(data["pid"]))
    table.add_row("port", f"{data['port']}  (http://127.0.0.1:{data['port']}/v1)")
    table.add_row("uptime", format_duration(data["uptime"]))
    if data.get("in_flight"):
        table.add_row("activity", f"{data['in_flight']} request(s) in progress")
    elif data.get("exits_in") is not None:
        table.add_row("idle", f"{format_duration(data['idle_for'])}, exits in {format_duration(data['exits_in'])}")
    loaded = data.get("loaded") or []
    if loaded:
        for i, item in enumerate(loaded):
            table.add_row("models" if i == 0 else "", f"{item['id']} ({item['type']}, {item['requests']} requests)")
    else:
        table.add_row("models", "none loaded")
    if data.get("rss_bytes") is not None:
        table.add_row("memory", f"{human_size(data['rss_bytes'])} RSS")
    table.add_row("log", paths.pretty(data.get("log") or paths.log_file()))
    out.print(table)


@app.command()
def stop() -> None:
    """Stop the server now. Its memory goes back to the OS."""
    from vox import client

    info = client.stop_server()
    if info is None:
        say("The vox server is not running.")
    else:
        ok(f"Stopped the vox server (pid {info.pid}).")


@app.command("uninstall")
def uninstall_cmd(
    keep_models: Annotated[
        bool, typer.Option("--keep-models", help="Leave downloaded models on disk for a later reinstall.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Remove vox completely: Quick Actions, settings, logs, models and the program itself."""
    from vox import uninstall

    remove, keep = uninstall.plan(keep_models)
    program = uninstall.program_command()
    program_text = " ".join([Path(program[0]).name, *program[1:]]) if program else ""

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column()
    table.add_column(style="dim")
    table.add_column(justify="right")
    for item in remove:
        table.add_row(escape(item.what), escape(paths.pretty(item.path)) if item.path else "", human_size(item.size) if item.size else "")
    if program:
        table.add_row("The vox program", program_text, "")
    if not remove and not program:
        say("Nothing to remove: vox left no files behind.")
        say(f"[dim]This vox runs from {escape(paths.pretty(sys.prefix))}; delete that environment yourself if you no longer need it.[/dim]")
        return

    say("[bold]vox uninstall will remove:[/bold]")
    err.print(table)
    for item in keep:
        say(f"[bold]Keeping[/bold] {escape(item.what)} in {escape(paths.pretty(item.path))} ({human_size(item.size)})")
    if not program:
        say(
            f"[dim]The program itself (in {escape(paths.pretty(sys.prefix))}) was not installed with uv, pipx or Homebrew, so it stays."
            " Remove it the way you installed it, for example: pip uninstall vox-cli[/dim]"
        )

    if not yes:
        if not interactive():
            raise UserError("Uninstalling needs confirmation.", "Run: vox uninstall --yes")
        if not typer.confirm("\nContinue?", default=False, err=True):
            say("Nothing was removed.")
            return

    from vox import finder

    had_finder = finder.helper_path().exists() or any(finder.workflow_path(a).exists() for a in finder.QUICK_ACTIONS)
    had_finder = had_finder and sys.platform == "darwin"
    uninstall.remove_files(keep_models)
    for item in remove:
        ok(f"Removed {escape(item.what)}")
    if had_finder:
        say("[dim]Shortcuts you built yourself in the Shortcuts app are not touched; delete those there.[/dim]")
    if keep and keep_models:
        say(f"[dim]Models stay in {escape(paths.pretty(paths.models_dir()))}. Reinstall vox to use them again, or delete that folder.[/dim]")

    if program:
        # Last step: the running program deletes itself. Use plain writes from
        # here on, since its modules may no longer be on disk.
        sys.stderr.write(f"Removing the vox program: {program_text}\n")
        sys.stderr.flush()
        try:
            code = subprocess.run(program).returncode
        except OSError:
            code = 127
        if code == 0:
            sys.stderr.write("vox is uninstalled.\n")
        else:
            sys.stderr.write(f"Could not remove the program. Run this yourself: {program_text}\n")
        sys.stderr.flush()
        os._exit(0 if code == 0 else 1)


# ----------------------------------------------------------------- setup


@setup_app.command("finder")
def setup_finder(
    print_only: Annotated[bool, typer.Option("--print-only", help="Only print the paths, change nothing.")] = False,
    uninstall: Annotated[bool, typer.Option("--uninstall", help="Remove the Quick Actions and the helper script.")] = False,
) -> None:
    """macOS: add "Transcribe with vox" and "Speak with vox" to Finder's Quick Actions."""
    from vox import finder, system
    from vox.media import find_ffmpeg

    if not system.is_mac():
        raise UserError("vox setup finder is for the macOS Finder.", "On Linux use: vox setup files")
    if uninstall:
        removed = finder.remove_quick_actions()
        for path in removed:
            ok(f"Removed {paths.pretty(path)}")
        if not removed:
            say("No vox Quick Actions were installed.")
        say("[dim]Shortcuts you built yourself in the Shortcuts app are not touched; delete those there.[/dim]")
        return

    ffmpeg = find_ffmpeg()
    vox_cmd = " ".join(finder.vox_command())
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("vox", vox_cmd)
    table.add_row("ffmpeg", ffmpeg or "[red]not found[/red] (brew install ffmpeg)")
    if print_only:
        table.add_row("helper", f"{paths.pretty(finder.helper_path())} (written by vox setup finder)")
        out.print(table)
        return

    helper = finder.write_helper()
    table.add_row("helper", str(helper))
    out.print(table)
    written = finder.install_quick_actions(helper)
    out.print()
    if written:
        for path in written:
            ok(f"Installed Quick Action: {paths.pretty(path)}")
        say(
            "\nRight-click a file in Finder, then Quick Actions (or Services). If the actions do not\n"
            "show up, enable them in System Settings > General > Login Items & Extensions > Finder."
        )
    else:
        say("Automator is not available, so no Quick Actions were generated.")
    say("\nTo build them in Shortcuts instead, add a Run Shell Script action (zsh, input as arguments) with:")
    for action in finder.QUICK_ACTIONS:
        out.print(f"  {action.title}:  {finder.shell_command(action, helper)}")
    say("[dim]Full steps: docs/finder.md[/dim]")


@setup_app.command("files")
def setup_files(
    print_only: Annotated[bool, typer.Option("--print-only", help="Only show what would be set up, change nothing.")] = False,
    uninstall: Annotated[bool, typer.Option("--uninstall", help="Remove the right-click actions and the helper script.")] = False,
) -> None:
    """Linux: add "Transcribe with vox" and "Speak with vox" to your file manager's right-click menu.

    Supports GNOME Files (Nautilus), Nemo, Caja and Dolphin.
    """
    from vox import desktop, finder, system
    from vox.media import find_ffmpeg

    if not system.is_linux():
        raise UserError("vox setup files is for Linux file managers.", "On a Mac use: vox setup finder")
    if uninstall:
        removed = desktop.remove()
        for path in removed:
            ok(f"Removed {escape(paths.pretty(path))}")
        if not removed:
            say("No vox file-manager actions were installed.")
        return

    detected = desktop.detect()
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("vox", " ".join(finder.vox_command()))
    table.add_row("ffmpeg", find_ffmpeg() or f"[red]not found[/red] ({system.install_hint('ffmpeg')})")
    table.add_row("found", ", ".join(desktop.FILE_MANAGERS[m] for m in detected) or "no supported file manager")
    if print_only:
        table.add_row("helper", f"{paths.pretty(desktop.helper_path())} (written by vox setup files)")
        out.print(table)
        return

    helper = desktop.write_helper()
    table.add_row("helper", str(helper))
    out.print(table)
    out.print()
    if not detected:
        say("No supported file manager was found (GNOME Files, Nemo, Caja, Dolphin).")
        say("You can still call the helper from your own tools:")
        out.print(f"  {helper} transcribe FILE...")
        return
    for path in desktop.install(detected, helper):
        ok(f"Installed {escape(paths.pretty(path))}")
    say(
        "\nRight-click a file: GNOME Files and Caja list the actions under Scripts; Nemo and Dolphin show them"
        " in the menu (Dolphin: Actions). Restart the file manager if they do not appear yet."
    )


# ---------------------------------------------------------------- config


@config_app.callback()
def config_show(ctx: typer.Context) -> None:
    """Show the config file and its values."""
    if ctx.invoked_subcommand:
        return
    cfg = load_config()
    out.print(f"[bold]{paths.pretty(paths.config_file())}[/bold]" + ("" if paths.config_file().exists() else " (not created yet)"))
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_column(style="dim")
    for key, help_text in KEYS.items():
        value = getattr(cfg, key)
        table.add_row(key, "" if value is None else str(value), help_text)
    out.print(table)
    say(f"[dim]Change a value with: vox config set KEY VALUE. Models: {paths.pretty(paths.models_dir())}[/dim]")


@config_app.command("get")
def config_get(key: Annotated[str, typer.Argument(help="Setting name.")]) -> None:
    """Print one setting's value (empty if unset), for scripts."""
    if key not in KEYS:
        raise UserError(f"Unknown config key: {key}.", "Known keys: " + ", ".join(KEYS))
    value = getattr(load_config(), key)
    out.print("" if value is None else str(value), markup=False, highlight=False)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Setting name.")],
    value: Annotated[str, typer.Argument(help="New value.")],
) -> None:
    """Change a setting."""
    cfg = load_config()
    if key in ("default_stt", "default_tts"):
        expected = key.removeprefix("default_")
        model = models.get_installed(value)
        if model is not None and model.type != expected:
            raise UserError(f"{value} is a {model.type} model, not {expected}.", f"Set it with: vox config set default_{model.type} {value}")
        models.set_default(value)
        ok(f"{key} = {value}")
        return
    if key == "default_voice":
        tts = models.pick_installed("tts", None, cfg, strict=False)
        if tts and value not in tts.voices() and not all(v in tts.voices() for v in value.split(",")):
            raise UserError(f"{tts.id} has no voice named {value}.", "List voices with: vox voices")
    set_value(cfg, key, value)
    save_config(cfg)
    ok(f"{key} = {getattr(cfg, key)}")


@config_app.command("unset")
def config_unset(key: Annotated[str, typer.Argument(help="Setting name.")]) -> None:
    """Reset a setting to its default."""
    cfg = load_config()
    set_value(cfg, key, None)
    save_config(cfg)
    ok(f"{key} reset")


# ------------------------------------------------------------------ main


def _report(exc: VoxError) -> None:
    err.print(f"[bold red]Error:[/bold red] {escape(exc.message)}")
    if exc.hint:
        err.print(f"  {escape(exc.hint)}")


# Typer 0.2x ships its own copy of Click; older versions use Click itself.
_CLICK_EXCEPTIONS = [click.exceptions]
try:
    from typer._click import exceptions as _typer_click_exceptions

    _CLICK_EXCEPTIONS.append(_typer_click_exceptions)
except ImportError:  # pragma: no cover
    pass
_USAGE_ERRORS = tuple(m.UsageError for m in _CLICK_EXCEPTIONS)
_NO_ARGS_IS_HELP = tuple(m.NoArgsIsHelpError for m in _CLICK_EXCEPTIONS if hasattr(m, "NoArgsIsHelpError"))
_ABORT = tuple({click.exceptions.Abort, typer.Abort})
_EXIT = tuple({click.exceptions.Exit, typer.Exit})


def main() -> None:
    if sys.platform == "win32":
        err.print("[bold red]Error:[/bold red] vox does not run on Windows yet.")
        err.print("  vox runs on Apple Silicon Macs and on Linux.")
        sys.exit(2)
    try:
        result = app(standalone_mode=False)
    except _NO_ARGS_IS_HELP as exc:
        # Typer's rich help prints while the exception is built; plain Click needs show().
        if type(exc).__module__.startswith("click"):
            exc.show()
        sys.exit(0)
    except _USAGE_ERRORS as exc:
        exc.show()
        sys.exit(1)
    except _ABORT:
        err.print("Aborted.")
        sys.exit(130)
    except _EXIT as exc:
        sys.exit(exc.exit_code)
    except VoxError as exc:
        if _debug:
            traceback.print_exc()
        _report(exc)
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        err.print("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        if _debug:
            traceback.print_exc()
        err.print(f"[bold red]Error:[/bold red] {escape(type(exc).__name__)}: {escape(str(exc))}")
        err.print("  Run the command again with --debug for details.")
        sys.exit(1)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
