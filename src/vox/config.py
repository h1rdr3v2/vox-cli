"""Read and write ~/.config/vox/config.toml."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vox import paths
from vox.errors import UserError

FORMATS = ("txt", "srt", "vtt", "json")
MODEL_TYPES = ("stt", "tts")

DEFAULT_PORT = 8880
DEFAULT_IDLE_TIMEOUT = 300  # seconds
DEFAULT_FORMAT = "txt"

# Keys `vox config set` understands, with a short description each.
KEYS = {
    "default_stt": "default speech-to-text model id",
    "default_tts": "default text-to-speech model id",
    "default_voice": "default voice for vox speak",
    "port": "server port (127.0.0.1 only)",
    "idle_timeout": "seconds of inactivity before the server exits (also 30s, 5m, 1h)",
    "default_format": "default transcript format: txt, srt, vtt or json",
}


@dataclass
class Config:
    default_stt: str | None = None
    default_tts: str | None = None
    default_voice: str | None = None
    port: int = DEFAULT_PORT
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT
    default_format: str = DEFAULT_FORMAT
    extra: dict[str, Any] = field(default_factory=dict)

    def default_for(self, model_type: str) -> str | None:
        return self.default_stt if model_type == "stt" else self.default_tts

    def set_default_for(self, model_type: str, model_id: str | None) -> None:
        if model_type == "stt":
            self.default_stt = model_id
        else:
            self.default_tts = model_id


def parse_duration(value: Any, key: str = "idle_timeout") -> int:
    """Seconds from an int or a string like "300", "45s", "5m", "1h"."""
    if isinstance(value, bool):
        raise _invalid(key, value)
    if isinstance(value, (int, float)):
        seconds = int(value)
    else:
        match = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", str(value).lower())
        if not match:
            raise _invalid(key, value)
        seconds = int(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    if seconds < 1:
        raise _invalid(key, value)
    return seconds


def parse_port(value: Any, key: str = "port") -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise _invalid(key, value) from None
    if isinstance(value, bool) or not 1 <= port <= 65535:
        raise _invalid(key, value)
    return port


def parse_format(value: Any, key: str = "default_format") -> str:
    fmt = str(value).lower()
    if fmt not in FORMATS:
        raise _invalid(key, value)
    return fmt


def _invalid(key: str, value: Any) -> UserError:
    return UserError(
        f"Invalid value for {key} in {paths.pretty(paths.config_file())}: {value!r}.",
        f"Fix it with: vox config set {key} <value>  ({KEYS.get(key, '')})",
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def load_config(path: Path | None = None) -> Config:
    path = path or paths.config_file()
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise UserError(
            f"Could not read {paths.pretty(path)}: {exc}.",
            "Fix the file or delete it to start from defaults.",
        ) from None

    cfg = Config()
    cfg.default_stt = _optional_str(data.pop("default_stt", None))
    cfg.default_tts = _optional_str(data.pop("default_tts", None))
    cfg.default_voice = _optional_str(data.pop("default_voice", None))
    if "port" in data:
        cfg.port = parse_port(data.pop("port"))
    if "idle_timeout" in data:
        cfg.idle_timeout = parse_duration(data.pop("idle_timeout"))
    if "default_format" in data:
        cfg.default_format = parse_format(data.pop("default_format"))
    cfg.extra = data
    return cfg


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Cannot write {type(value).__name__} to TOML")


def dumps(cfg: Config) -> str:
    lines = ["# vox configuration. See `vox config` for the keys.", ""]
    values: dict[str, Any] = {
        "default_stt": cfg.default_stt,
        "default_tts": cfg.default_tts,
        "default_voice": cfg.default_voice,
        "port": cfg.port,
        "idle_timeout": cfg.idle_timeout,
        "default_format": cfg.default_format,
    }
    tables = {}
    for key, value in cfg.extra.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            values[key] = value
    for key, value in values.items():
        if value is not None:
            lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        lines += ["", f"[{name}]"]
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config.", suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(cfg))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def set_value(cfg: Config, key: str, value: str | None) -> None:
    """Set (or with value None, reset) a key, validating it."""
    if key not in KEYS:
        raise UserError(f"Unknown config key: {key}.", "Known keys: " + ", ".join(KEYS))
    if key in ("default_stt", "default_tts", "default_voice"):
        setattr(cfg, key, _optional_str(value))
    elif key == "port":
        cfg.port = DEFAULT_PORT if value is None else parse_port(value)
    elif key == "idle_timeout":
        cfg.idle_timeout = DEFAULT_IDLE_TIMEOUT if value is None else parse_duration(value)
    elif key == "default_format":
        cfg.default_format = DEFAULT_FORMAT if value is None else parse_format(value)
