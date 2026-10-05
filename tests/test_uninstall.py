"""`vox uninstall` and `vox setup finder --uninstall`, on temporary folders."""

import sys
from pathlib import Path

import pytest
from conftest import install_fake

from vox import finder, paths, uninstall
from vox.config import Config, save_config


@pytest.fixture
def mac(tmp_path, monkeypatch):
    """Point ~/Library locations at a temp folder and never remove the real program."""
    services = tmp_path / "Library" / "Services"
    monkeypatch.setattr(finder, "SERVICES_DIR", services)
    monkeypatch.setattr(finder, "PBS", "/nonexistent")
    monkeypatch.setattr(uninstall, "launch_agent_path", lambda: tmp_path / "Library" / "LaunchAgents" / "local.vox.server.plist")
    monkeypatch.setattr(uninstall, "program_command", lambda: None)
    return tmp_path


def _setup_everything():
    helper = finder.write_helper()
    finder.install_quick_actions(helper)
    save_config(Config(default_stt="whisper-tiny"))
    install_fake("whisper-tiny", "stt")
    install_fake("kokoro-82m", "tts")
    partial = paths.models_dir() / "whisper-small"
    partial.mkdir(parents=True)
    paths.run_dir().mkdir(parents=True)
    paths.log_file().write_text("log")
    return helper


def test_setup_finder_uninstall_removes_actions_and_helper(run_cli, mac):
    helper = finder.write_helper()
    finder.install_quick_actions(helper)
    assert all(finder.workflow_path(a).exists() for a in finder.QUICK_ACTIONS)

    result = run_cli("setup", "finder", "--uninstall")
    assert result.code == 0
    assert not any(finder.workflow_path(a).exists() for a in finder.QUICK_ACTIONS)
    assert not helper.exists()
    assert "Removed" in result.err
    assert run_cli("setup", "finder", "--uninstall").err.startswith("No vox Quick Actions")


def test_uninstall_needs_confirmation_without_a_terminal(run_cli, mac):
    _setup_everything()
    result = run_cli("uninstall")
    assert result.code == 1
    assert "vox uninstall --yes" in result.err
    assert paths.config_dir().exists()


def test_uninstall_can_be_declined(run_cli, mac, monkeypatch):
    _setup_everything()
    monkeypatch.setattr("vox.cli.typer.confirm", lambda *a, **k: False)
    result = run_cli("uninstall", stdin_tty=True)
    assert result.code == 0 and "Nothing was removed" in result.err
    assert paths.config_dir().exists() and paths.models_dir().exists()


def test_uninstall_everything(run_cli, mac):
    _setup_everything()
    agent = uninstall.launch_agent_path()
    agent.parent.mkdir(parents=True)
    agent.write_text("<plist/>")

    result = run_cli("uninstall", "--yes")
    assert result.code == 0, result.err
    assert "whisper-tiny" in result.err and "whisper-small (partial download)" in result.err
    assert not paths.config_dir().exists()
    assert not paths.cache_dir().exists()
    assert not agent.exists()
    assert not any(finder.workflow_path(a).exists() for a in finder.QUICK_ACTIONS)


def test_uninstall_keep_models(run_cli, mac):
    _setup_everything()
    result = run_cli("uninstall", "--yes", "--keep-models")
    assert result.code == 0, result.err
    assert "Keeping" in result.err
    assert (paths.models_dir() / "whisper-tiny" / "vox-model.json").exists()
    assert (paths.models_dir() / "kokoro-82m").exists()
    assert not paths.config_dir().exists()
    assert not paths.run_dir().exists()


def test_uninstall_with_nothing_installed(run_cli, mac):
    result = run_cli("uninstall", "--yes")
    assert result.code == 0
    assert "Nothing to remove" in result.err


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("/Users/me/.local/share/uv/tools/vox-cli", ["tool", "uninstall", "vox-cli"]),
        ("/Users/me/.local/pipx/venvs/vox-cli", ["uninstall", "vox-cli"]),
        ("/Users/me/code/vox-cli/.venv", None),
    ],
)
def test_program_command(monkeypatch, prefix, expected):
    monkeypatch.setattr(sys, "prefix", prefix)
    command = uninstall.program_command()
    assert (command[1:] if command else None) == expected
    if command:
        assert Path(command[0]).name in ("uv", "pipx")
