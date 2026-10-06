"""End to end with real models: speak a sentence, then transcribe it back.

Runs only when an STT and a TTS model are installed, in ~/.cache/vox/models
or in the folder named by VOX_INTEGRATION_MODELS. Config and runtime files
stay in a temporary folder.
"""

import os
import socket
import time
from pathlib import Path

import pytest

MODELS = Path(os.environ.get("VOX_INTEGRATION_MODELS") or Path.home() / ".cache" / "vox" / "models")


def _installed(model_type: str) -> bool:
    import json

    if not MODELS.is_dir():
        return False
    for manifest in MODELS.glob("*/vox-model.json"):
        try:
            if json.loads(manifest.read_text()).get("type") == model_type:
                return True
        except ValueError:
            pass
    return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (_installed("stt") and _installed("tts")), reason=f"needs an STT and a TTS model in {MODELS}"),
]


def test_speak_then_transcribe(monkeypatch, tmp_path):
    from vox import client
    from vox.config import Config

    monkeypatch.setenv("VOX_MODELS_DIR", str(MODELS))
    monkeypatch.delenv("VOX_BACKEND", raising=False)  # this machine's real engines
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = Config(port=port, idle_timeout=30)
    try:
        api = client.Client(client.ensure_server(cfg))
        wav = tmp_path / "speech.wav"
        started = time.monotonic()
        api.speak("Good morning. The weather is lovely today.", model="", voice=None, speed=1.0, response_format="wav", out=wav)
        assert wav.stat().st_size > 10_000
        response = api.transcribe(wav, model="", response_format="text", language=None)
        text = response.text.lower()
        assert "morning" in text and "weather" in text
        status = api.status()
        assert {m["type"] for m in status["loaded"]} == {"stt", "tts"}
        assert time.monotonic() - started < 120
    finally:
        client.stop_server()
