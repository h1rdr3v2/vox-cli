"""CLI argument handling and exit codes (0 ok, 1 user error, 2 missing model/dependency)."""

import pytest
from conftest import install_fake

from vox import cli, models, system
from vox.config import load_config


def test_help_and_version(run_cli):
    assert run_cli("--help").code == 0
    result = run_cli("--version")
    assert result.code == 0 and "vox " in result.out


def test_no_args_shows_help(run_cli):
    result = run_cli()
    assert result.code == 0
    assert "transcribe" in result.out + result.err


def test_usage_errors_exit_1(run_cli):
    assert run_cli("transcribe").code == 1  # missing FILE
    assert run_cli("frobnicate").code == 1
    assert run_cli("speak", "hi", "--speed", "fast").code == 1


def test_fresh_install_lists_nothing(run_cli):
    result = run_cli("models", "list", "--installed")
    assert result.code == 0
    assert result.out.strip() == ""
    assert "No models installed" in result.err


def test_models_list_marks_installed_and_default(run_cli):
    install_fake("whisper-tiny", "stt")
    models.set_default("whisper-tiny")
    result = run_cli("models", "list", "--type", "stt")
    line = next(line for line in result.out.splitlines() if line.startswith("whisper-tiny "))
    assert " yes " in line and " * " in line
    assert "kokoro" not in result.out
    assert run_cli("models", "list", "--type", "xyz").code == 1
    assert run_cli("models", "list", "--installed", "--available").code == 1


def test_transcribe_without_model_non_tty_gives_hint(run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: "/usr/bin/true")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    result = run_cli("transcribe", str(clip))
    assert result.code == 2
    assert "No STT model installed." in result.err
    assert "Run: vox models pull whisper-large-v3-turbo" in result.err


def test_transcribe_without_model_tty_shows_picker(run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: "/usr/bin/true")
    picked = []

    def fake_pull(spec):
        picked.append(spec.id)
        return install_fake_model(spec.id)

    def install_fake_model(model_id):
        install_fake(model_id, "stt")
        return models.get_installed(model_id)

    monkeypatch.setattr(cli, "pull_model", fake_pull)
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "1")
    monkeypatch.setattr(cli, "connect", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    result = run_cli("transcribe", str(clip), stdin_tty=True)
    assert picked == ["whisper-large-v3-turbo"]  # recommended model is option 1
    assert "Pick one to download" in result.err
    assert load_config().default_stt == "whisper-large-v3-turbo"


def test_transcribe_argument_errors(run_cli, tmp_path, monkeypatch):
    assert run_cli("transcribe", str(tmp_path / "missing.mp3")).code == 1
    assert run_cli("transcribe", str(tmp_path)).code == 1  # a folder
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x")
    result = run_cli("transcribe", str(clip), "--format", "docx")
    assert result.code == 1 and "Unknown format" in result.err


def test_missing_ffmpeg_exits_2(run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr("vox.media.find_ffmpeg", lambda: None)
    clip = tmp_path / "a.mp3"
    clip.write_bytes(b"x")
    result = run_cli("transcribe", str(clip))
    assert result.code == 2
    assert "ffmpeg is not installed." in result.err
    assert system.install_hint("ffmpeg") in result.err


def test_transcript_target(tmp_path):
    src = tmp_path / "clip.mp4"
    assert cli._transcript_target(src, "txt", None, False) == tmp_path / "clip.txt"
    assert cli._transcript_target(src, "srt", None, False) == tmp_path / "clip.srt"
    assert cli._transcript_target(src, "txt", "-", False) is None
    assert cli._transcript_target(src, "txt", str(tmp_path / "out.txt"), False) == tmp_path / "out.txt"
    folder = tmp_path / "subs"
    assert cli._transcript_target(src, "vtt", str(folder) + "/", True) == folder / "clip.vtt"
    with pytest.raises(cli.UserError, match="overwrite the input"):
        cli._transcript_target(tmp_path / "notes.txt", "txt", None, False)


def test_speak_input_detection(tmp_path, monkeypatch):
    assert cli._read_speak_input("Good morning") == ("Good morning", None)
    note = tmp_path / "note.md"
    note.write_text("# Hello\n\nThis is **bold**.")
    text, source = cli._read_speak_input(str(note))
    assert source == note and "Hello." in text and "**" not in text
    with pytest.raises(cli.UserError, match="File not found"):
        cli._read_speak_input(str(tmp_path / "missing.txt"))
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("from stdin"))
    assert cli._read_speak_input("-") == ("from stdin", None)


def test_speech_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.txt"
    assert cli._speech_target(None, None, False) == (tmp_path / "speech.wav", "wav", False)
    assert cli._speech_target(source, None, False) == (tmp_path / "notes.wav", "wav", False)
    assert cli._speech_target(None, "out.mp3", False) == (tmp_path.joinpath("out.mp3").relative_to(tmp_path), "mp3", False)
    path, fmt, temporary = cli._speech_target(None, None, True)
    assert temporary and fmt == "wav" and path.suffix == ".wav"
    path.unlink()
    assert cli._speech_target(None, "-", False) == (None, "wav", False)
    with pytest.raises(cli.UserError, match="Unsupported audio format"):
        cli._speech_target(None, "out.xyz", False)


def test_speak_errors(run_cli):
    assert run_cli("speak", "   ").code == 1
    assert run_cli("speak", "hi", "--speed", "5").code == 1
    result = run_cli("speak", "hello there")
    assert result.code == 2 and "No TTS model installed." in result.err


def test_models_commands(run_cli):
    result = run_cli("models", "default", "whisper-tiny")
    assert result.code == 2 and "vox models pull whisper-tiny" in result.err
    assert run_cli("models", "pull", "whisper-gigantic").code == 1
    assert run_cli("models", "pull", "hf:org/repo").code == 1  # needs --type
    assert run_cli("models", "rm", "whisper-tiny").code == 1

    install_fake("whisper-tiny", "stt")
    assert run_cli("models", "default", "whisper-tiny").code == 0
    assert load_config().default_stt == "whisper-tiny"
    info = run_cli("models", "info", "whisper-tiny")
    assert info.code == 0 and "huggingface.co/test/whisper-tiny" in info.out
    assert run_cli("models", "info", "kokoro-82m").code == 0
    already = run_cli("models", "pull", "whisper-tiny")
    assert already.code == 0 and "already installed" in already.err
    removed = run_cli("models", "rm", "whisper-tiny")
    assert removed.code == 0 and "no default now" in removed.err
    assert load_config().default_stt is None


def test_voices(run_cli):
    install_fake("kokoro-82m", "tts", voices=("af_heart", "bm_george"))
    result = run_cli("voices")
    assert result.code == 0
    assert "af_heart" in result.out and "bm_george" in result.out


def test_config_commands(run_cli):
    assert run_cli("config").code == 0
    assert run_cli("config", "set", "idle_timeout", "10m").code == 0
    assert load_config().idle_timeout == 600
    assert run_cli("config", "set", "port", "abc").code == 1
    assert run_cli("config", "set", "nope", "1").code == 1
    got = run_cli("config", "get", "idle_timeout")
    assert got.code == 0 and got.out.strip() == "600"
    assert run_cli("config", "get", "default_voice").out.strip() == ""
    assert run_cli("config", "get", "nope").code == 1
    assert run_cli("config", "unset", "idle_timeout").code == 0
    assert load_config().idle_timeout == 300


def test_status_when_stopped(run_cli):
    result = run_cli("status")
    assert result.code == 0 and "not running" in result.out
    assert run_cli("stop").code == 0


def test_errors_are_one_line_plus_hint(run_cli, tmp_path):
    result = run_cli("speak", "hello")
    lines = [line for line in result.err.splitlines() if line.strip()]
    assert lines == ["Error: No TTS model installed.", "  Run: vox models pull kokoro-82m"]
    assert "Traceback" not in result.err
