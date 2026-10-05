import tomllib

import pytest

from vox import paths
from vox.config import (
    Config,
    dumps,
    load_config,
    parse_duration,
    save_config,
    set_value,
)
from vox.errors import UserError


def test_defaults_when_missing():
    cfg = load_config()
    assert cfg.default_stt is None and cfg.default_tts is None
    assert cfg.port == 8880
    assert cfg.idle_timeout == 300
    assert cfg.default_format == "txt"
    assert not paths.config_file().exists()


def test_roundtrip():
    cfg = Config(default_stt="whisper-tiny", default_tts="kokoro-82m", default_voice="af_heart", port=9000, idle_timeout=60, default_format="srt")
    save_config(cfg)
    loaded = load_config()
    assert loaded == cfg
    # The file is valid TOML a person can read.
    data = tomllib.loads(paths.config_file().read_text())
    assert data["default_stt"] == "whisper-tiny"
    assert data["port"] == 9000


def test_unset_defaults_are_omitted():
    save_config(Config())
    data = tomllib.loads(paths.config_file().read_text())
    assert "default_stt" not in data and "default_voice" not in data


def test_unknown_keys_survive_a_rewrite():
    paths.config_file().parent.mkdir(parents=True)
    paths.config_file().write_text('port = 8890\ncustom = "keep me"\n\n[extra]\nflag = true\n')
    cfg = load_config()
    cfg.default_stt = "whisper-base"
    save_config(cfg)
    data = tomllib.loads(paths.config_file().read_text())
    assert data["custom"] == "keep me"
    assert data["extra"] == {"flag": True}
    assert data["port"] == 8890


def test_strings_are_escaped():
    cfg = Config(default_voice='odd "voice"\\name')
    assert tomllib.loads(dumps(cfg))["default_voice"] == 'odd "voice"\\name'


@pytest.mark.parametrize("value, seconds", [(300, 300), ("300", 300), ("45s", 45), ("5m", 300), ("1h", 3600), (" 2M ", 120)])
def test_parse_duration(value, seconds):
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["0", 0, "-5", "5 minutes", "abc", True])
def test_parse_duration_rejects(value):
    with pytest.raises(UserError):
        parse_duration(value)


@pytest.mark.parametrize("text", ["port = 0", "port = 'x'", "idle_timeout = 'soon'", "default_format = 'docx'", "port = = 1"])
def test_invalid_file_values(text):
    paths.config_file().parent.mkdir(parents=True)
    paths.config_file().write_text(text)
    with pytest.raises(UserError) as info:
        load_config()
    assert info.value.hint


def test_set_value():
    cfg = Config()
    set_value(cfg, "idle_timeout", "10m")
    set_value(cfg, "port", "9999")
    set_value(cfg, "default_format", "VTT")
    assert (cfg.idle_timeout, cfg.port, cfg.default_format) == (600, 9999, "vtt")
    set_value(cfg, "port", None)
    assert cfg.port == 8880
    with pytest.raises(UserError, match="Unknown config key"):
        set_value(cfg, "colour", "blue")
