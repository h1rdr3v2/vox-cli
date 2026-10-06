# Developing vox

```bash
uv sync
uv run pytest
```

uv is only a convenience here; `python3 -m venv .venv && .venv/bin/pip install -e . pytest && .venv/bin/pytest` works too. The dev environment also installs the Linux engines, so on a Mac you can try them with `VOX_BACKEND=portable`.

To run the suite on Linux from a Mac (needs Docker; arm64 by default, `--platform linux/amd64` for x86_64):

```bash
scripts/test-linux.sh
```

To publish a release (the Homebrew formula in `Formula/vox.rb` installs the tagged version):

```bash
scripts/release.sh 0.3.0
git push origin main v0.3.0
```

Homebrew users then get it with `brew update && brew upgrade vox`. `brew install --HEAD h1rdr3v2/vox-cli/vox` builds from `main` instead.

The tests use fake engines and temporary folders. `tests/test_integration.py` runs a real speak-then-transcribe round trip when an STT and a TTS model are installed (in `~/.cache/vox/models`, or the folder in `VOX_INTEGRATION_MODELS`).

Engines live behind small interfaces (`STTEngine`, `TTSEngine` in `src/vox/engines/base.py`), so another backend such as whisper.cpp can be added without touching the CLI.
