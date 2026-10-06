# vox on Linux

vox runs on Linux on x86_64 and arm64 with the same commands, model ids, settings and HTTP API as on a Mac. Only the engines underneath differ:

- **Transcription**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), on the CPU or an NVIDIA GPU.
- **Speech**: Kokoro on [ONNX Runtime](https://onnxruntime.ai). The text side (misaki, spaCy, espeak) is the same as on a Mac, so pronunciation matches.

`vox models pull whisper-large-v3-turbo` downloads the CTranslate2 conversion on Linux and the MLX one on a Mac, so the same instructions work on both.

## Install

You need Python 3.11 or later and ffmpeg. On Debian or Ubuntu:

```bash
sudo apt install ffmpeg pipx git
pipx ensurepath
pipx install git+https://github.com/h1rdr3v2/vox-cli
```

With uv instead: `uv tool install git+https://github.com/h1rdr3v2/vox-cli`. With [Homebrew on Linux](https://docs.brew.sh/Homebrew-on-Linux), which brings its own Python and ffmpeg: `brew tap h1rdr3v2/vox-cli https://github.com/h1rdr3v2/vox-cli && brew install h1rdr3v2/vox-cli/vox`. On Fedora, use `sudo dnf install ffmpeg pipx` (ffmpeg comes from RPM Fusion); on Arch, `sudo pacman -S ffmpeg python-pipx`.

Then follow the quick start in the [README](../README.md): the first `vox transcribe` or `vox speak` in a terminal offers to download a model.

## GPU (NVIDIA)

The `device` setting picks where transcription runs:

```bash
vox config set device auto   # default: the GPU when usable, else the CPU
vox config set device cpu    # always the CPU
vox config set device cuda   # the GPU
```

faster-whisper uses the GPU through CTranslate2, which needs the NVIDIA driver plus the CUDA 12 cuBLAS and cuDNN 9 libraries. If they are missing, vox logs a warning in `~/.cache/vox/run/server.log` and uses the CPU. See faster-whisper's [GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu) for installing those libraries.

On the CPU, vox runs Whisper with 8-bit weights (`int8`), which is several times faster than full precision with nearly the same accuracy. `whisper-small` or `whisper-large-v3-turbo` are good choices; `whisper-large-v3` is slow without a GPU.

Kokoro is small enough that the CPU is fast; it uses the GPU only if you replace `onnxruntime` with `onnxruntime-gpu` in vox's environment.

## Playing audio

`vox speak --play` uses the first player it finds: `pw-play` (PipeWire), `paplay` (PulseAudio), `ffplay` (comes with ffmpeg on most distributions) or `aplay` (ALSA, WAV only). On a machine with no sound output, vox says so instead of failing silently; save to a file with `--out` instead.

## Right-click actions in your file manager

```bash
vox setup files
```

This adds **Transcribe with vox** (audio and video) and **Speak with vox** (text files) for each file manager it finds:

| File manager | Where the action appears | Files written |
|---|---|---|
| GNOME Files (Nautilus) | right-click > Scripts | `~/.local/share/nautilus/scripts/` |
| Caja (MATE) | right-click > Scripts | `~/.config/caja/scripts/` |
| Nemo (Cinnamon) | right-click menu | `~/.local/share/nemo/actions/` |
| Dolphin (KDE) | right-click > Actions | `~/.local/share/kio/servicemenus/` |

Each action calls a helper script, `~/.config/vox/files/vox-files.sh`, which runs vox on every selected file, writes the result next to it, and shows a desktop notification (`notify-send`) when it starts and when it is done. Restart the file manager if the actions do not show up right away (`nautilus -q`, or log out and back in).

Install the models first; actions cannot show the interactive model picker. To remove the actions: `vox setup files --uninstall`. To see what would be set up: `vox setup files --print-only`.

## Always-on server (systemd)

By default vox starts its server when needed and stops it after 5 idle minutes. If other apps call the OpenAI-compatible API at `http://127.0.0.1:8880/v1`, you can keep a persistent server running as a systemd user service.

Save this as `~/.config/systemd/user/vox.service`, with the path from `which vox`:

```ini
[Unit]
Description=vox local transcription and speech server

[Service]
ExecStart=/home/you/.local/bin/vox serve --persistent
Restart=on-failure

[Install]
WantedBy=default.target
```

Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now vox.service
vox status
```

To stop it for good: `systemctl --user disable --now vox.service`. `vox uninstall` does this for you if the file is at that path. A persistent server keeps every model it has loaded until it restarts: `systemctl --user restart vox.service` frees that memory.

## Folders

vox follows the XDG base directories: settings in `$XDG_CONFIG_HOME/vox` (default `~/.config/vox`), models and logs in `$XDG_CACHE_HOME/vox` (default `~/.cache/vox`).

## Uninstall

```bash
vox uninstall                 # everything, after asking
vox uninstall --keep-models   # keep ~/.cache/vox/models
```

This also removes the file-manager actions and the systemd service above.
