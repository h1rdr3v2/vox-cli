"""HTTP API tests with fake engines (no MLX, no models)."""

import io
import threading
import time
import wave

import pytest
from conftest import FakeSTT, FakeTTS, install_fake, write_wav
from starlette.testclient import TestClient

from vox.config import Config, save_config
from vox.server.app import ServerState, create_app


@pytest.fixture
def state():
    return ServerState(port=8880, persistent=True, idle_timeout=300)


@pytest.fixture
def api(state):
    with TestClient(create_app(state), base_url="http://127.0.0.1:8880") as client:
        yield client


@pytest.fixture
def wav_bytes(tmp_path):
    return write_wav(tmp_path / "speech.wav", seconds=1.0).read_bytes()


def transcribe(api, wav_bytes, **data):
    return api.post("/v1/audio/transcriptions", files={"file": ("speech.wav", wav_bytes, "audio/wav")}, data=data)


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok" and body["service"] == "vox"


def test_models_endpoint(api):
    assert api.get("/v1/models").json() == {"object": "list", "data": []}
    install_fake("whisper-tiny", "stt")
    data = api.get("/v1/models").json()["data"]
    assert [(m["id"], m["object"], m["type"]) for m in data] == [("whisper-tiny", "model", "stt")]


def test_transcription_without_models_is_400(api, wav_bytes):
    response = transcribe(api, wav_bytes)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"] == "No STT model installed."
    assert error["hint"] == "Run: vox models pull whisper-large-v3-turbo"
    assert error["code"] == "model_not_found"


def test_transcription_formats(api, wav_bytes):
    install_fake("whisper-tiny", "stt")
    response = transcribe(api, wav_bytes)
    assert response.status_code == 200
    assert response.json() == {"text": "Hello world."}
    assert response.headers["x-vox-model"] == "whisper-tiny"

    assert transcribe(api, wav_bytes, response_format="text").text == "Hello world.\n"
    assert "00:00:00,000 --> 00:00:01,500" in transcribe(api, wav_bytes, response_format="srt").text
    assert transcribe(api, wav_bytes, response_format="vtt").text.startswith("WEBVTT")
    verbose = transcribe(api, wav_bytes, response_format="verbose_json", language="fr").json()
    assert verbose["language"] == "french"
    assert verbose["duration"] == 1.0
    assert transcribe(api, wav_bytes, response_format="docx").status_code == 400
    assert transcribe(api, wav_bytes, language="klingon").status_code == 400


def test_openai_model_names_use_the_default(api, wav_bytes):
    install_fake("whisper-tiny", "stt")
    install_fake("whisper-base", "stt")
    save_config(Config(default_stt="whisper-base"))
    response = transcribe(api, wav_bytes, model="whisper-1")
    assert response.headers["x-vox-model"] == "whisper-base"
    response = transcribe(api, wav_bytes, model="whisper-tiny")
    assert response.headers["x-vox-model"] == "whisper-tiny"


def test_missing_file_is_400(api):
    response = api.post("/v1/audio/transcriptions", data={"model": "x"}, files={"other": ("a", b"x")})
    assert response.status_code == 400


def test_speech_wav(api):
    install_fake("kokoro-82m", "tts")
    response = api.post("/v1/audio/speech", json={"input": "Good morning. How are you?", "response_format": "wav"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == 24000 and wav.getnchannels() == 1
    assert response.headers["x-vox-voice"] == "af_test"
    assert FakeTTS.calls[0][0] == "Good morning. How are you?"


def test_speech_pcm_and_voices(api):
    install_fake("kokoro-82m", "tts")
    pcm = api.post("/v1/audio/speech", json={"input": "Hi.", "voice": "am_test", "response_format": "pcm"})
    assert pcm.headers["content-type"] == "audio/pcm"
    assert len(pcm.content) == int(0.25 * 24000) * 2
    assert FakeTTS.calls[-1][1] == "am_test"
    # OpenAI voice names map to a local voice.
    mapped = api.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy", "response_format": "wav"})
    assert mapped.headers["x-vox-voice"] == "af_test"
    unknown = api.post("/v1/audio/speech", json={"input": "Hi.", "voice": "zz_nobody", "response_format": "wav"})
    assert unknown.status_code == 400 and unknown.json()["error"]["code"] == "voice_not_found"
    save_config(Config(default_voice="am_test"))
    configured = api.post("/v1/audio/speech", json={"input": "Hi.", "response_format": "wav"})
    assert configured.headers["x-vox-voice"] == "am_test"


@pytest.mark.parametrize(
    "body",
    [
        {"input": ""},
        {"input": "hi", "speed": 9},
        {"input": "hi", "speed": "fast"},
        {"input": "hi", "response_format": "midi"},
        {"voice": "x"},
    ],
)
def test_speech_validation(api, body):
    install_fake("kokoro-82m", "tts")
    assert api.post("/v1/audio/speech", json=body).status_code == 400


def test_speech_without_tts_model(api):
    install_fake("whisper-tiny", "stt")
    response = api.post("/v1/audio/speech", json={"input": "hi"})
    assert response.status_code == 400
    assert response.json()["error"]["hint"] == "Run: vox models pull kokoro-82m"


def test_long_text_is_chunked(api):
    install_fake("kokoro-82m", "tts")
    text = " ".join(f"This is sentence number {i} of a long text." for i in range(40))
    response = api.post("/v1/audio/speech", json={"input": text, "response_format": "wav"}, headers={"X-Vox-Job": "job1"})
    assert response.status_code == 200
    assert len(FakeTTS.calls) > 1
    job = api.get("/vox/jobs/job1").json()
    assert job["state"] == "done" and job["done"] == job["total"] == len(FakeTTS.calls)


def test_transcribe_loads_only_stt(api, state, wav_bytes):
    install_fake("whisper-tiny", "stt")
    install_fake("kokoro-82m", "tts")
    transcribe(api, wav_bytes)
    loaded = api.get("/vox/status").json()["loaded"]
    assert [m["id"] for m in loaded] == ["whisper-tiny"]
    assert FakeTTS.loads == 0


def test_requests_share_one_loaded_model(state, wav_bytes):
    install_fake("whisper-tiny", "stt")
    original = FakeSTT.load

    def slow_load(self):
        time.sleep(0.3)
        original(self)

    FakeSTT.load = slow_load
    try:
        app = create_app(state)
        results = []

        def worker():
            with TestClient(app, base_url="http://127.0.0.1") as client:
                results.append(transcribe(client, wav_bytes).status_code)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        FakeSTT.load = original
    assert results == [200] * 4
    assert FakeSTT.loads == 1


def test_status_and_activity(api, state):
    status = api.get("/vox/status").json()
    assert status["persistent"] is True and status["exits_in"] is None
    before = state.last_activity
    time.sleep(0.01)
    api.get("/health")
    api.get("/vox/status")
    assert state.last_activity == before, "monitoring must not keep the server alive"
    api.get("/v1/models")
    assert state.last_activity > before


def test_foreign_host_is_rejected(api):
    assert api.get("/health", headers={"Host": "evil.example:8880"}).status_code == 403
    assert api.get("/health", headers={"Host": "localhost:8880"}).status_code == 200


def test_shutdown_needs_vox_header(api, state):
    calls = []
    state.request_exit = calls.append
    assert api.post("/vox/shutdown").status_code == 403
    assert api.post("/vox/shutdown", headers={"X-Vox-Client": "test"}).status_code == 200
    assert calls == ["vox stop"]
