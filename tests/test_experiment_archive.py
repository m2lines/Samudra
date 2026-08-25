# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import threading

import pytest

from scripts import experiment_archive


def test_archive_destination_appends_run_name():
    assert (
        experiment_archive.archive_destination("remote:bucket/archive/", "run-42")
        == "remote:bucket/archive/run-42"
    )


@pytest.mark.parametrize("run_name", ["", ".", "..", "nested/run", "nested\\run"])
def test_archive_destination_rejects_unsafe_run_name(run_name):
    with pytest.raises(experiment_archive.ArchiveError):
        experiment_archive.archive_destination("remote:bucket/archive", run_name)


def test_copy_command_is_non_destructive_and_excludes_sensitive_files(tmp_path):
    command = experiment_archive.copy_command(
        tmp_path / "run", "remote:bucket/archive/run"
    )

    assert command[:2] == ["rclone", "copy"]
    assert "sync" not in command
    assert "--delete" not in command
    for pattern in experiment_archive.EXCLUDE_PATTERNS:
        assert pattern in command


def test_publish_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry run must not invoke rclone")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = experiment_archive.publish_run(
        run_dir, "remote:bucket/archive", apply=False
    )

    assert result["status"] == "planned"
    output = capsys.readouterr().out
    assert "rclone copy" in output
    assert "rclone check" in output


def test_publish_copies_then_verifies(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    commands = []

    def record(command, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", record)

    result = experiment_archive.publish_run(
        run_dir, "remote:bucket/archive", apply=True
    )

    assert [command[1] for command in commands] == ["copy", "check"]
    assert result["status"] == "verified"
    assert result["verified"] is True


def test_status_tracks_run_lifecycle(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    timestamps = iter(("start", "finish"))
    monkeypatch.setattr(experiment_archive, "utc_now", lambda: next(timestamps))

    status_path = experiment_archive.write_status(run_dir, "running")
    experiment_archive.write_status(run_dir, "completed", exit_code=0)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status == {
        "schema_version": 1,
        "run_name": "run-a",
        "state": "completed",
        "started_at": "start",
        "updated_at": "finish",
        "finished_at": "finish",
        "exit_code": 0,
        "slurm_job_id": "1234",
        "slurm_restart_count": "2",
    }


def test_status_rejects_unknown_state(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    with pytest.raises(experiment_archive.ArchiveError, match="invalid archive state"):
        experiment_archive.write_status(run_dir, "unknown")


def test_watch_publishes_without_full_verification(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    calls = []

    def record_publish(run_dir, archive_base, apply, verify, rclone_bin):
        calls.append((run_dir, archive_base, apply, verify, rclone_bin))

    monkeypatch.setattr(experiment_archive, "publish_run", record_publish)

    result = experiment_archive.watch_run(
        run_dir,
        "remote:bucket/archive",
        interval_seconds=0.001,
        apply=True,
        stop_event=threading.Event(),
        max_cycles=1,
    )

    assert result == 0
    assert calls == [(run_dir, "remote:bucket/archive", True, False, "rclone")]


def test_watch_reports_copy_failures(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def fail_publish(*args, **kwargs):
        command = ["rclone", "copy", str(run_dir), "remote:archive/run-a"]
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(experiment_archive, "publish_run", fail_publish)

    result = experiment_archive.watch_run(
        run_dir,
        "remote:bucket/archive",
        interval_seconds=0.001,
        apply=True,
        max_cycles=1,
    )

    assert result == 1
    assert "Periodic archive copy failed" in capsys.readouterr().err
