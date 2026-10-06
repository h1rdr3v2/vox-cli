"""Linux support: engine choice, platform helpers, portable engines, file managers.

These run on any OS: platform checks are patched where needed.
"""

import stat
import sys
from pathlib import Path

import pytest
from conftest import install_fake

from vox import desktop, languages, models, paths, system, uninstall
from vox.config import Config, load_config, save_config, set_value
from vox.engines import engine_class
from vox.engines.faster_whisper import FasterWhisperEngine, pick_device
from vox.engines.kokoro_onnx import KokoroOnnxEngine, split_phonemes, tokenize
from vox.engines.kokoro_text import voice_language, voice_names
from vox.errors import MissingError, UserError

# ----------------------------------------------------------- engine family


@pytest.mark.parametrize(
    "platform, machine, expected",
    [("darwin", "arm64", "mlx"), ("linux", "x86_64", "portable"), ("linux", "aarch64", "portable")],
)
def test_backend_auto(monkeypatch, platform, machine, expected):
    monkeypatch.delenv("VOX_BACKEND")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(system.platform, "machine", lambda: machine)
    assert system.backend() == expected


@pytest.mark.parametrize("platform, machine", [("darwin", "x86_64"), ("win32", "AMD64")])
def test_backend_unsupported(monkeypatch, platform, machine):
    monkeypatch.delenv("VOX_BACKEND")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(system.platform, "machine", lambda: machine)
    with pytest.raises(MissingError, match="does not support this system"):
        system.backend()


def test_backend_override(monkeypatch):
    monkeypatch.setenv("VOX_BACKEND", "portable")
    assert system.backend() == "portable"
    monkeypatch.setenv("VOX_BACKEND", "cuda-magic")
    with pytest.raises(MissingError):
        system.backend()


def test_hf_models_use_the_platform_engine(monkeypatch):
    monkeypatch.setenv("VOX_BACKEND", "portable")
    assert models.parse_ref("hf:org/whisper-ct2", "stt").engine == "faster-whisper"
    assert models.parse_ref("hf:org/kokoro-onnx", "tts").engine == "kokoro-onnx"
    monkeypatch.setenv("VOX_BACKEND", "mlx")
    assert models.parse_ref("hf:org/whisper-mlx", "stt").engine == "mlx-whisper"
    assert models.parse_ref("hf:org/some-tts", "tts").engine is None  # decided from config.json


def test_portable_catalog_pull_spec(monkeypatch):
    monkeypatch.setenv("VOX_BACKEND", "portable")
    spec = models.parse_ref("kokoro-82m")
    assert (spec.engine, spec.repo) == ("kokoro-onnx", "onnx-community/Kokoro-82M-v1.0-ONNX")
    assert models.no_model_error("stt").hint == "Run: vox models pull whisper-large-v3-turbo"


# ---------------------------------------------------------- platform helpers


def test_install_hint(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert system.install_hint("ffmpeg") == "Run: brew install ffmpeg"
    monkeypatch.setattr(sys, "platform", "linux")
    assert "sudo apt install ffmpeg" in system.install_hint("ffmpeg")


def test_audio_player_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    available = {}
    monkeypatch.setattr(system.shutil, "which", lambda name: available.get(name))
    wav, mp3 = tmp_path / "a.wav", tmp_path / "a.mp3"

    with pytest.raises(MissingError, match="No audio player"):
        system.audio_player(wav)
    available["aplay"] = "/usr/bin/aplay"
    assert system.audio_player(wav) == ["/usr/bin/aplay", "-q", str(wav)]
    with pytest.raises(MissingError):
        system.audio_player(mp3)  # aplay cannot play MP3
    available["ffplay"] = "/usr/bin/ffplay"
    assert system.audio_player(mp3)[0] == "/usr/bin/ffplay"
    available["paplay"] = "/usr/bin/paplay"
    assert system.audio_player(wav) == ["/usr/bin/paplay", str(wav)]


def test_playback_error():
    assert system.playback_error(0, "") is None
    alsa = "ALSA lib confmisc.c:855:(parse_card) cannot find card '0'\nALSA lib pcm.c:2666:(snd_pcm_open_noupdate) Unknown PCM default"
    assert system.playback_error(0, alsa) == "No audio output device found."
    assert system.playback_error(1, "Connection failure: Connection refused\n") == "No audio output device found."
    assert system.playback_error(2, "bad file\n") == "Could not play the audio (bad file)."


def test_audio_player_mac(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert system.audio_player(tmp_path / "a.mp3")[0] == "/usr/bin/afplay"


def test_rss_and_command_of_this_process():
    import os

    rss = system.rss_bytes(os.getpid())
    assert rss is None or rss > 1_000_000
    command = system.process_command(os.getpid()).lower()
    # "python ... pytest" normally; just "pytest" under emulation (binfmt).
    assert command == "" or "py" in command


def test_xdg_folders_on_linux(monkeypatch, tmp_path):
    monkeypatch.delenv("VOX_CONFIG_DIR")
    monkeypatch.delenv("VOX_CACHE_DIR")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xc"))
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/ignored")
    assert paths.config_dir() == tmp_path / "xc" / "vox"
    assert paths.cache_dir() == Path.home() / ".cache" / "vox"
    monkeypatch.setattr(sys, "platform", "darwin")  # macOS keeps ~/.config
    assert paths.config_dir() == Path.home() / ".config" / "vox"


def test_device_setting():
    cfg = Config()
    assert cfg.device == "auto"
    set_value(cfg, "device", "CUDA")
    save_config(cfg)
    assert load_config().device == "cuda"
    with pytest.raises(UserError):
        set_value(cfg, "device", "tpu")


def test_languages():
    assert languages.normalize("fr") == "fr"
    assert languages.normalize("French") == "fr"
    assert languages.normalize("castilian") == "es"
    assert len(languages.LANGUAGES) == 100
    with pytest.raises(ValueError):
        languages.normalize("klingon")


# --------------------------------------------------------- portable engines


def test_faster_whisper_files_and_device():
    files = ["README.md", ".gitattributes", "config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
    assert FasterWhisperEngine.select_files(files) == ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
    with pytest.raises(ValueError):
        FasterWhisperEngine.select_files(["config.json", "weights.npz"])
    FasterWhisperEngine.check_config({"alignment_heads": [], "lang_ids": []})
    with pytest.raises(ValueError):
        FasterWhisperEngine.check_config({"n_mels": 80})
    assert pick_device("auto", 0) == ("cpu", "int8")
    assert pick_device("auto", 1) == ("cuda", "float16")
    assert pick_device("cpu", 2) == ("cpu", "int8")
    assert pick_device("cuda", 0) == ("cuda", "float16")


def test_kokoro_onnx_files():
    files = ["config.json", "tokenizer.json", "onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_quantized.onnx", "voices/af_heart.bin", "voices/bm_george.bin", "README.md"]
    assert KokoroOnnxEngine.select_files(files) == ["config.json", "tokenizer.json", "onnx/model.onnx", "voices/af_heart.bin", "voices/bm_george.bin"]
    with pytest.raises(ValueError, match="full-precision"):
        KokoroOnnxEngine.select_files(["config.json", "onnx/model_fp16.onnx", "voices/a.bin"])


def test_kokoro_onnx_voices(tmp_path):
    (tmp_path / "voices").mkdir()
    for name in ("bm_george", "af_heart"):
        (tmp_path / "voices" / f"{name}.bin").write_bytes(b"")
    assert KokoroOnnxEngine.list_voices(tmp_path) == ["af_heart", "bm_george"]
    assert KokoroOnnxEngine.fallback_voice(tmp_path) == "af_heart"
    assert KokoroOnnxEngine.openai_voice("Nova") == "af_nova"


def test_split_phonemes_and_tokenize():
    text = "ðə kwˈɪk bɹˈWn fˈɑks. " * 60
    pieces = split_phonemes(text, limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")
    assert all(p.endswith(".") for p in pieces[:-1])
    assert split_phonemes("short") == ["short"]
    assert tokenize("ab?", {"$": 0, "a": 43, "b": 44, "?": 6}) == [43, 44, 6]
    assert tokenize("a#b", {"a": 1, "b": 2}) == [1, 2]  # unknown characters are dropped


def test_voice_helpers():
    assert voice_language("bf_emma,af_heart") == "b"
    assert voice_language("xx_unknown") == "a"
    assert voice_names("af_heart, bm_george", ["af_heart", "bm_george"]) == ["af_heart", "bm_george"]
    with pytest.raises(UserError, match="Unknown voice"):
        voice_names("af_nobody", ["af_heart"])


def test_portable_engines_are_registered():
    assert engine_class("faster-whisper") is FasterWhisperEngine
    assert engine_class("kokoro-onnx") is KokoroOnnxEngine


# -------------------------------------------------------- file managers


@pytest.fixture
def linux_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgconfig"))
    return tmp_path


def test_desktop_actions(linux_home):
    helper = desktop.write_helper()
    assert helper.stat().st_mode & stat.S_IXUSR
    script = helper.read_text()
    assert "__VOX__" not in script and "__PATH__" not in script and "run_vox()" in script

    written = desktop.install(["nautilus", "nemo", "caja", "dolphin"], helper)
    assert len(written) == 8
    nautilus = linux_home / "data" / "nautilus" / "scripts" / "Transcribe with vox"
    assert nautilus in written and nautilus.stat().st_mode & stat.S_IXUSR
    assert 'transcribe "$@"' in nautilus.read_text()
    assert (linux_home / "xdgconfig" / "caja" / "scripts" / "Speak with vox").exists()
    nemo = (linux_home / "data" / "nemo" / "actions" / "vox-transcribe.nemo_action").read_text()
    assert "Mimetypes=audio/*;video/*;" in nemo and "transcribe %F" in nemo
    dolphin = linux_home / "data" / "kio" / "servicemenus" / "vox-speak.desktop"
    assert "MimeType=text/plain;text/markdown;" in dolphin.read_text()
    assert dolphin.stat().st_mode & stat.S_IXUSR

    removed = desktop.remove()
    assert len(removed) == 9  # 8 actions + helper
    assert desktop.installed() == []


def test_desktop_quote():
    assert desktop._desktop_quote("/usr/bin/x") == "/usr/bin/x"
    assert desktop._desktop_quote('/home/a b/$x"y') == '"/home/a b/\\$x\\"y"'


def test_setup_commands_point_to_the_right_platform(run_cli, monkeypatch, linux_home):
    monkeypatch.setattr(sys, "platform", "linux")
    result = run_cli("setup", "finder")
    assert result.code == 1 and "vox setup files" in result.err
    monkeypatch.setattr(desktop, "detect", lambda: ["nautilus"])
    result = run_cli("setup", "files")
    assert result.code == 0, result.err
    assert (linux_home / "data" / "nautilus" / "scripts" / "Speak with vox").exists()
    assert run_cli("setup", "files", "--uninstall").code == 0
    assert desktop.installed() == []
    monkeypatch.setattr(sys, "platform", "darwin")
    result = run_cli("setup", "files")
    assert result.code == 1 and "vox setup finder" in result.err


def test_uninstall_removes_linux_pieces(run_cli, monkeypatch, linux_home):
    monkeypatch.setattr(uninstall, "program_command", lambda: None)
    monkeypatch.setattr(uninstall, "_systemctl", lambda *a: None)
    helper = desktop.write_helper()
    desktop.install(["nautilus", "dolphin"], helper)
    unit = uninstall.systemd_unit_path()
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n")
    install_fake("whisper-tiny", "stt")

    result = run_cli("uninstall", "--yes")
    assert result.code == 0, result.err
    assert "systemd user service" in result.err and "File-manager action" in result.err
    assert not unit.exists()
    assert desktop.installed() == []
    assert not paths.config_dir().exists() and not paths.cache_dir().exists()
