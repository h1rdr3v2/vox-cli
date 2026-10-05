import pytest
from conftest import install_fake

from vox import models
from vox.config import Config, load_config, save_config
from vox.engines.kokoro import KokoroEngine
from vox.engines.mlx_whisper import MLXWhisperEngine
from vox.errors import MissingError, UserError


def test_nothing_installed_on_fresh_install():
    assert models.installed_models() == []
    assert not models.paths.models_dir().exists()


def test_installed_listing_ignores_partial_downloads():
    install_fake("whisper-tiny", "stt")
    partial = models.model_dir("whisper-base")
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")
    assert [m.id for m in models.installed_models()] == ["whisper-tiny"]
    assert models.get_installed("whisper-base") is None


def test_slug_and_hf_ids():
    assert models.slug("whisper-tiny") == "whisper-tiny"
    assert models.slug("hf:org/name.en") == "hf--org--name.en"
    install_fake("hf:org/custom", "stt")
    assert models.get_installed("hf:org/custom").id == "hf:org/custom"


def test_parse_ref():
    spec = models.parse_ref("whisper-small")
    assert (spec.type, spec.repo, spec.engine) == ("stt", "mlx-community/whisper-small-mlx", "mlx-whisper")
    hf = models.parse_ref("hf:mlx-community/whisper-small.en-mlx", "stt")
    assert (hf.id, hf.repo, hf.engine, hf.from_catalog) == ("hf:mlx-community/whisper-small.en-mlx", "mlx-community/whisper-small.en-mlx", "mlx-whisper", False)
    assert models.parse_ref("hf:a/b", "tts").engine is None  # decided from config.json
    with pytest.raises(UserError, match="Say what kind of model"):
        models.parse_ref("hf:a/b")
    with pytest.raises(UserError, match="Invalid Hugging Face reference"):
        models.parse_ref("hf:just-a-name", "stt")
    with pytest.raises(UserError, match="Unknown model"):
        models.parse_ref("whisper-gigantic")
    with pytest.raises(UserError, match="is a stt model"):
        models.parse_ref("whisper-tiny", "tts")


class TestDefaultResolution:
    def test_none_installed(self):
        assert models.pick_installed("stt", None, Config(), strict=True) is None
        err = models.no_model_error("stt")
        assert isinstance(err, MissingError) and err.exit_code == 2
        assert err.message == "No STT model installed."
        assert err.hint == "Run: vox models pull whisper-large-v3-turbo"
        assert models.no_model_error("tts").hint == "Run: vox models pull kokoro-82m"

    def test_configured_default_wins(self):
        install_fake("whisper-tiny", "stt")
        install_fake("whisper-large-v3-turbo", "stt")
        cfg = Config(default_stt="whisper-tiny")
        assert models.pick_installed("stt", None, cfg, strict=True).id == "whisper-tiny"

    def test_falls_back_to_best_installed(self):
        install_fake("whisper-tiny", "stt")
        install_fake("whisper-large-v3-turbo", "stt")
        cfg = Config(default_stt="whisper-medium")  # default no longer installed
        assert models.pick_installed("stt", None, cfg, strict=True).id == "whisper-large-v3-turbo"

    def test_types_are_separate(self):
        install_fake("whisper-tiny", "stt")
        assert models.pick_installed("tts", None, Config(), strict=True) is None

    def test_requested_model(self):
        install_fake("whisper-tiny", "stt")
        install_fake("whisper-base", "stt")
        cfg = Config(default_stt="whisper-tiny")
        assert models.pick_installed("stt", "whisper-base", cfg, strict=True).id == "whisper-base"

    def test_cli_is_strict_about_requested_models(self):
        install_fake("whisper-tiny", "stt")
        install_fake("kokoro-82m", "tts")
        with pytest.raises(MissingError, match="not installed") as info:
            models.pick_installed("stt", "whisper-small", Config(), strict=True)
        assert info.value.hint == "Run: vox models pull whisper-small"
        with pytest.raises(UserError, match="Unknown model"):
            models.pick_installed("stt", "nope", Config(), strict=True)
        with pytest.raises(UserError, match="is a tts model"):
            models.pick_installed("stt", "kokoro-82m", Config(), strict=True)

    def test_api_falls_back_for_unknown_names(self):
        install_fake("whisper-tiny", "stt")
        assert models.pick_installed("stt", "whisper-1", Config(), strict=False).id == "whisper-tiny"


def test_remove_clears_default():
    install_fake("whisper-tiny", "stt")
    save_config(Config(default_stt="whisper-tiny", default_tts="kokoro-82m"))
    model, was_default = models.remove("whisper-tiny")
    assert model.id == "whisper-tiny" and was_default
    cfg = load_config()
    assert cfg.default_stt is None and cfg.default_tts == "kokoro-82m"
    assert not models.model_dir("whisper-tiny").exists()
    with pytest.raises(UserError, match="not installed"):
        models.remove("whisper-tiny")


def test_set_default_and_ensure_default_after_pull():
    with pytest.raises(MissingError):
        models.set_default("whisper-tiny")
    install_fake("whisper-tiny", "stt")
    install_fake("whisper-base", "stt")
    assert models.ensure_default_after_pull(models.get_installed("whisper-tiny")) is True
    assert models.ensure_default_after_pull(models.get_installed("whisper-base")) is False
    assert load_config().default_stt == "whisper-tiny"
    models.set_default("whisper-base")
    assert load_config().default_stt == "whisper-base"


def test_whisper_file_selection():
    assert MLXWhisperEngine.select_files(["README.md", "config.json", "weights.npz"]) == ["config.json", "weights.npz"]
    both = ["config.json", "weights.npz", "weights.safetensors"]
    assert MLXWhisperEngine.select_files(both) == ["config.json", "weights.safetensors"]
    with pytest.raises(ValueError):
        MLXWhisperEngine.select_files(["config.json", "model.bin"])
    with pytest.raises(ValueError):
        MLXWhisperEngine.check_config({"model_type": "llama"})


def test_kokoro_file_selection():
    files = ["config.json", "kokoro-v1_0.safetensors", "kokoro-v1_0.pth", "samples/a.wav", "voices/af_heart.safetensors", "voices/af_heart.pt", "README.md"]
    assert KokoroEngine.select_files(files) == ["config.json", "kokoro-v1_0.safetensors", "voices/af_heart.safetensors"]
    with pytest.raises(ValueError, match="safetensors"):
        KokoroEngine.select_files(["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"])


def test_human_size():
    assert models.human_size(74_418_540) == "74 MB"
    assert models.human_size(1_614_000_000) == "1.6 GB"
