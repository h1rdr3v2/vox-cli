# vox

Local transcription and speech for Apple Silicon Macs and Linux.

AI coding agents reinstall Whisper and torch in every project. vox is one local install with an OpenAI-compatible server they can all share, and the same commands work for you in the terminal and from a right-click in your file manager. Free meeting transcription; nothing leaves your computer.

- **Transcribe**: audio or video in, text out (Whisper).
- **Speak**: text in, audio out (Kokoro).
- **Drop-in for OpenAI audio code**: change the base URL, keep `whisper-1`, `tts-1` and `voice="nova"`.

| | Apple Silicon Mac | Linux (x86_64, arm64) |
|---|---|---|
| Transcription | Whisper on `mlx-audio` (GPU via Metal) | `faster-whisper` (CPU, or NVIDIA GPU) |
| Speech | Kokoro on `mlx-audio` | Kokoro on ONNX Runtime |
| Right-click menu | Finder Quick Actions | GNOME Files, Nemo, Caja, Dolphin |

The commands, model names, settings and HTTP API are the same everywhere. Linux details: [docs/linux.md](docs/linux.md).

vox installs no models: you pick the ones you want. It sends no telemetry, and it touches the network only when you download a model.

**Memory:** vox starts a small background server only when a command needs it, loads only the model that command uses, and keeps it warm for back-to-back calls. After 5 idle minutes the server exits, so an idle vox uses zero memory.

## Install

Requirements: macOS 14 or later on Apple Silicon, or Linux on x86_64 or arm64; Python 3.11 or later; ffmpeg.

With Homebrew (Mac or Linux), which also installs Python and ffmpeg:

```bash
brew tap h1rdr3v2/vox-cli https://github.com/h1rdr3v2/vox-cli
brew install h1rdr3v2/vox-cli/vox
```

Or with uv on a Mac:

```bash
brew install uv ffmpeg
uv tool install --compile-bytecode git+https://github.com/h1rdr3v2/vox-cli
```

Or with pipx on Linux (Debian or Ubuntu shown; use your package manager):

```bash
sudo apt install ffmpeg pipx git
pipx install git+https://github.com/h1rdr3v2/vox-cli
```

With uv or pipx, `vox` lands on your PATH (in `~/.local/bin`). If your shell cannot find it, run `uv tool update-shell` (or `pipx ensurepath`) and open a new terminal. uv and pipx are interchangeable here; plain `pip install` into a virtual environment also works. To remove vox and everything it created, run `vox uninstall` (see [Uninstall](#uninstall)).

The very first transcription or speech after installing on a Mac can take up to a minute while macOS checks the newly installed libraries (MLX, spaCy) once. After that, a cold start takes a few seconds.

The PyPI distribution name is `vox-cli`. The command is `vox`.

## Quick start

```bash
vox transcribe meeting.mp4
```

The first time, vox sees that no speech-to-text model is installed and asks you to pick one:

```
No STT (speech-to-text) model is installed yet. Pick one to download:

#  MODEL                   SIZE     NOTE
1  whisper-large-v3-turbo  ~1.6 GB  recommended: near large-v3 accuracy, much faster
2  whisper-tiny            ~74 MB   fastest, lowest accuracy
...
Model number or id [1]:
```

Press Enter for the recommended model. vox downloads it, makes it your default, and writes `meeting.txt` next to the video.

The same happens the first time you speak:

```bash
vox speak "Good morning" --play
```

You can also pull models ahead of time:

```bash
vox models list --available
vox models pull whisper-large-v3-turbo
vox models pull kokoro-82m
```

In scripts (no terminal), vox does not prompt. It exits with code 2 and a hint such as `Run: vox models pull whisper-large-v3-turbo`.

## Use it from apps and AI agents

### Existing OpenAI audio code

vox speaks OpenAI's audio API on `127.0.0.1:8880`, so code written for OpenAI works by changing one URL. Model and voice names vox does not have (`whisper-1`, `tts-1`, `nova`, `alloy`, ...) fall back to your default models and similar Kokoro voices.

```bash
vox start   # starts the server in the background; it exits after 5 idle minutes
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8880/v1", api_key="not-needed")

with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)

client.audio.speech.create(model="tts-1", voice="nova", input="Good morning").write_to_file("morning.mp3")
```

Or with curl:

```bash
curl http://127.0.0.1:8880/v1/audio/transcriptions \
  -F file=@meeting.m4a -F model=whisper-large-v3-turbo -F response_format=text
```

```bash
curl http://127.0.0.1:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-82m", "input": "Good morning", "voice": "af_heart", "response_format": "wav"}' \
  -o morning.wav
```

| Endpoint | |
|---|---|
| `POST /v1/audio/transcriptions` | multipart `file`, `model`, optional `language`, `prompt`, `response_format` (`json`, `text`, `srt`, `vtt`, `verbose_json`) |
| `POST /v1/audio/speech` | JSON `model`, `input`, `voice`, optional `speed`, `response_format` (`mp3` default, `wav`, `flac`, `opus`, `aac`, `pcm`) |
| `GET /v1/models` | installed models |
| `GET /health` | liveness |

If `model` is missing or not an installed id (for example `whisper-1`), the default model of that type is used. If no model of that type is installed, the API returns 400. OpenAI voice names (`alloy`, `nova`, `onyx`, ...) map to similar Kokoro voices.

The server listens on `127.0.0.1` only. For an always-on server, use `vox serve --persistent` (see [Server](#server)).

### AI coding agents

Coding agents don't know vox is installed, so they tend to `pip install` Whisper or a TTS package into each project. Add this to your agent's global instructions (`~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex; more in [docs/agents.md](docs/agents.md)):

```markdown
## Speech: use vox (installed on this computer)

For speech-to-text or text-to-speech, prefer vox over installing Whisper, Kokoro, torch or similar packages:
- Transcribe: `vox transcribe FILE --out -` prints the text (`--format srt|vtt|json` for timestamps).
- Speak: `vox speak "text" --out speech.wav` (`vox voices` lists voices).
- Installed models: `vox models list --installed`.
- Code that needs an API: vox serves OpenAI's audio API at http://127.0.0.1:8880/v1 (any api_key).
  Start the server first with `vox start` (it returns at once and stops after 5 idle minutes).

vox is only on this computer: for apps other people will run, ask before depending on it.
```

## Commands

### Transcribe

```
vox transcribe <file...> [--model ID] [--format txt|srt|vtt|json] [--language CODE] [--out PATH]
```

- Accepts any audio or video ffmpeg can read. Each file is converted to 16 kHz mono WAV in a temporary folder first.
- Output goes next to the input with the same name: `clip.mp4` gives `clip.txt`. `--out -` prints to stdout. With several files, `--out` must be a folder.
- `--format json` writes segments with timestamps, the detected language and the duration. If you omit `--format`, an `--out` ending in `.srt`, `.vtt` or `.json` picks the format; otherwise the `default_format` setting (txt) applies.
- The language is detected automatically unless you pass `--language` (`en`, `fr`, `german`, ...).
- `--progress-file PATH` keeps PATH updated with how far the job is, from `0` to `1`, for scripts and other apps (`vox speak` has it too).

### Speak

```
vox speak <text | file.txt | file.md | -> [--model ID] [--voice NAME] [--speed N] [--out PATH] [--play]
vox voices [--model ID]
```

- Input is a literal string, a `.txt` or `.md` file (Markdown syntax is stripped), or `-` for stdin.
- Output: `notes.txt` gives `notes.wav` next to it; literal text gives `speech.wav` in the current folder. `--out file.mp3` writes MP3 (via ffmpeg); `.flac`, `.opus` and `.aac` work too. `--out -` writes WAV to stdout.
- `--play` plays the result (with `afplay` on a Mac; `pw-play`, `paplay`, `ffplay` or `aplay` on Linux). With `--play` and no `--out`, nothing is saved.
- Long text is split into sentence-sized chunks and the audio is joined, with a short pause between paragraphs.
- `--speed` ranges from 0.5 to 2.0. `vox voices` lists the voices of the TTS model; Kokoro has 54 across American and British English, Spanish, French, Hindi, Italian, Brazilian Portuguese, Japanese and Mandarin. Voices can be blended: `--voice af_heart,af_bella`.

Japanese and Mandarin voices need an extra text package in vox's environment, for example `uv tool install --reinstall --with "misaki[ja]" git+https://github.com/h1rdr3v2/vox-cli` (or `misaki[zh]`).

### Models

```
vox models list [--installed | --available] [--type stt|tts]
vox models pull <id>                       # catalog id
vox models pull hf:<org>/<repo> --type stt|tts   # any compatible model on Hugging Face
vox models rm <id>
vox models default <id>
vox models info <id>
```

- `list` shows id, type, size on disk, whether it is installed, which one is the default, and a note. Without flags it shows the catalog plus anything else you installed.
- `pull` shows a progress bar and resumes if interrupted: run the same command again. The first model of each type becomes the default.
- `rm` deletes the model folder. If it was the default, the default is cleared and vox says so.
- `default` sets the default for that model's type. STT and TTS each have their own default.
- `info` shows the source repo, size, languages and (for TTS) voices.

Catalog:

| id | type | size | note |
|---|---|---|---|
| whisper-tiny | stt | ~74 MB | fastest, lowest accuracy |
| whisper-base | stt | ~144 MB | fast, fair accuracy |
| whisper-small | stt | ~481 MB | balanced speed and accuracy |
| whisper-medium | stt | ~1.5 GB | accurate, slower |
| whisper-large-v3-turbo | stt | ~1.6 GB | recommended: near large-v3 accuracy, much faster |
| whisper-large-v3 | stt | ~3.1 GB | most accurate, slowest |
| kokoro-82m | tts | ~370 MB | recommended TTS; best in English |

Each id downloads the right files for your platform: MLX conversions on a Mac, faster-whisper (CTranslate2) and ONNX versions on Linux. The Kokoro download includes spaCy's small English pipeline (12 MB), which Kokoro uses for pronunciation on both. `hf:` repos must match the platform: MLX Whisper or mlx-audio TTS repos on a Mac (TTS other than Kokoro is best effort), CTranslate2 Whisper or Kokoro ONNX repos on Linux.

### Server

```
vox serve [--persistent] [--port 8880] [--idle-timeout 5m]
vox start
vox status
vox stop
```

You do not need to start anything: `transcribe` and `speak` start the server when needed. `vox start` starts it in the background and returns, for apps that call the HTTP API; it still exits after the idle timeout. `vox status` shows whether it is running, its PID and port, the loaded models, its memory use (RSS) and the log path. `vox stop` shuts it down now.

`vox serve` runs a server in the foreground. With `--persistent` it never exits on idle, but still loads models only when first used. To run it at login, see [docs/launchd.md](docs/launchd.md) (Mac) or the systemd section of [docs/linux.md](docs/linux.md).

### Settings and right-click actions

```
vox config                      # show settings
vox config set KEY VALUE        # default_voice, idle_timeout, port, default_format, device, ...
vox setup finder                # Mac: add Finder Quick Actions
vox setup files                 # Linux: add file-manager actions
vox setup finder --uninstall    # remove them again (or: vox setup files --uninstall)
vox uninstall [--keep-models]   # remove vox completely
```

Both setup commands add **Transcribe with vox** and **Speak with vox** to the right-click menu. See [docs/finder.md](docs/finder.md) for the Mac and [docs/linux.md](docs/linux.md) for Linux. On Linux, `device` picks `auto` (an NVIDIA GPU when usable), `cpu` or `cuda`.

## How the memory behavior works

The CLI is a thin client. `vox transcribe` and `vox speak` look for a running server (a state file plus a health check). If there is none, they start one in the background and wait for it. The server loads a model the first time a request needs it, so a transcription never loads Kokoro. Requests queue: the same model is never loaded twice. After `idle_timeout` seconds without requests (default 300), the server process exits and its memory goes back to the operating system. It does not try to unload models inside a long-lived process, because MLX and Python may not return freed memory.

A cold start (process start plus model load) takes a few seconds. Later calls reuse the warm model.

## Files

| What | Where |
|---|---|
| Settings | `~/.config/vox/config.toml` |
| Models | `~/.cache/vox/models/<id>/` (each with a `vox-model.json` manifest) |
| Server state, lock and log | `~/.cache/vox/run/` (`server.log`) |

On Linux, `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` move these folders as usual.

## Uninstall

```bash
vox uninstall                 # asks first, then removes everything
vox uninstall --keep-models   # same, but leaves ~/.cache/vox/models for a later reinstall
```

It lists what it will remove, then: stops the server, removes the right-click actions (Finder or Linux file managers) and their helper, the launch agent or systemd user service from the docs if you made one, `~/.config/vox`, the server logs, the models (unless `--keep-models`), and finally the program itself (`brew uninstall vox`, `uv tool uninstall vox-cli` or `pipx uninstall vox-cli`, matching how you installed it). Add `--yes` to skip the question in scripts. Shortcuts you built by hand in the Shortcuts app are not touched. If you installed vox another way (for example plain pip), it says so and leaves the program for you to remove the same way.

To remove only the right-click actions: `vox setup finder --uninstall` (Mac) or `vox setup files --uninstall` (Linux).

If you reinstall after `--keep-models`, vox finds the kept models again. `brew uninstall vox` or `uv tool uninstall vox-cli` on its own removes only the program and leaves settings and models in place.

## Errors and exit codes

Errors are one line plus a hint, for example (on a Mac):

```
Error: ffmpeg is not installed.
  Run: brew install ffmpeg
```

Add `--debug` to any command for a full traceback. Exit codes: `0` success, `1` user error, `2` missing dependency or model.

## Development

See [docs/development.md](docs/development.md) for running the tests (including on Linux in Docker), the engine interfaces, and publishing a release.

## License

MIT. See [LICENSE](LICENSE).
