"""Finder Quick Actions: the 10-step workflow, the progress file, and the helper script."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from vox import cli, finder


def test_quick_actions_have_ten_connected_steps(tmp_path):
    helper = tmp_path / "vox finder.sh"
    commands = finder.step_commands(finder.QUICK_ACTIONS[0], helper)
    assert len(commands) == finder.PROGRESS_STEPS == 10
    assert commands[0] == f"'{helper}' step 1 transcribe \"$@\""
    assert commands[9] == f"'{helper}' step 10 \"$@\""

    doc = finder._document(commands)
    actions = [a["action"] for a in doc["actions"]]
    assert [a["ActionParameters"]["COMMAND_STRING"] for a in actions] == commands
    assert all(a["ActionParameters"]["inputMethod"] == 1 for a in actions)  # input as arguments
    links = {(c["from"].split(" - ")[0], c["to"].split(" - ")[0]) for c in doc["connectors"].values()}
    assert links == {(a["UUID"], b["UUID"]) for a, b in zip(actions, actions[1:], strict=False)}


def test_one_command_for_shortcuts(tmp_path):
    assert finder.shell_command(finder.QUICK_ACTIONS[1], tmp_path / "h.sh") == f"{tmp_path}/h.sh speak \"$@\""


def test_progress_file(tmp_path):
    path = tmp_path / "progress"
    report = cli.ProgressFile(path, parts=4)
    assert path.read_text() == "0.0000\n"
    report.start_part(1)
    report.update(0.5)
    assert float(path.read_text()) == pytest.approx(0.375)
    report.update(5)  # clamped to the end of this part
    assert float(path.read_text()) == pytest.approx(0.5)
    report.finish()
    assert path.read_text() == "1.0000\n"
    cli.ProgressFile(None).update(0.5)  # no file: nothing happens


FAKE_VOX = """#!/bin/sh
# Stands in for vox: writes progress, then the output file next to the input.
mode="$1"; file="$2"; progress="$4"
case "$file" in *fail*) echo "Error: could not read it." >&2; exit 1 ;; esac
echo 0.5 > "$progress"; sleep 0.2; echo 1 > "$progress"
case "$mode" in transcribe) echo text > "${file%.*}.txt" ;; speak) echo wav > "${file%.*}.wav" ;; esac
"""


@pytest.fixture
def helper(tmp_path, monkeypatch):
    if sys.platform != "darwin" or not shutil.which("zsh"):
        pytest.skip("the Finder helper is a macOS zsh script")
    fake = tmp_path / "fake-vox"
    fake.write_text(FAKE_VOX)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(finder, "vox_command", lambda: [str(fake)])
    monkeypatch.setenv("VOX_FINDER_QUIET", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    return finder.write_helper()


def _run(helper: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/zsh", str(helper), *args], capture_output=True, text=True, timeout=60, env=os.environ)


def test_helper_steps_hand_the_job_along(helper, tmp_path):
    clip = tmp_path / "my clip.m4a"
    clip.write_bytes(b"")
    first = _run(helper, "step", "1", "transcribe", str(clip))
    job = first.stdout.strip()
    assert first.returncode == 0 and "vox-job." in job
    for step in range(2, 10):
        result = _run(helper, "step", str(step), job)
        assert result.returncode == 0 and result.stdout.strip() == job
    last = _run(helper, "step", "10", job)
    assert last.returncode == 0 and last.stdout == ""
    assert (tmp_path / "my clip.txt").read_text() == "text\n"
    assert not Path(job).exists()


def test_helper_one_shot_mode(helper, tmp_path):
    good, bad = tmp_path / "a.txt", tmp_path / "fail.txt"
    good.write_text("hi")
    bad.write_text("hi")
    assert _run(helper, "speak", str(good)).returncode == 0
    assert (tmp_path / "a.wav").exists()
    assert _run(helper, "speak", str(good), str(bad)).returncode == 1


def test_helper_ignores_bad_job_paths(helper, tmp_path):
    keep = tmp_path / "not-a-job"
    keep.mkdir()
    assert _run(helper, "step", "5", str(keep)).returncode == 0
    assert keep.exists()
    assert _run(helper, "step", "x").returncode == 1
