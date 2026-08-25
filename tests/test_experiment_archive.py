# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from scripts import experiment_archive


def _write_executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _run_harness_to_archive_failure(
    tmp_path, restart_count, failed_command, name="run-a"
):
    repository = Path(__file__).parents[1]
    harness = repository / "scripts" / "slurm_apptainer_train.sbatch"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "apptainer",
        """#!/bin/bash
if [[ "$1" == "exec" ]]; then
  printf unknown
fi
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf 'node-a\n'
""",
    )
    _write_executable(
        fake_bin / "rclone",
        """#!/bin/bash
printf '%s\n' "$*" >> "${RCLONE_LOG}"
if [[ "$1" == "${RCLONE_FAIL_COMMAND}" ]]; then
  exit 9
fi
""",
    )

    data_root = tmp_path / "data"
    output_base = tmp_path / "runs"
    scratch_dir = tmp_path / "scratch"
    run_dir = output_base / name
    data_root.mkdir()
    output_base.mkdir()
    scratch_dir.mkdir()
    image_path = tmp_path / "image.sif"
    image_path.write_text("image", encoding="utf-8")
    if restart_count:
        run_dir.mkdir()
        (run_dir / "checkpoint").write_text("existing", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "USER": "torch-user",
            "CONFIG": "config.yaml",
            "NAME": name,
            "DATA_ROOT": str(data_root),
            "OUTPUT_BASE": str(output_base),
            "SCRATCH_DIR": str(scratch_dir),
            "SIF_PATH": str(image_path),
            "PUBLISH_TO_OSN": "1",
            "ARCHIVE_BASE": "remote:bucket/archive",
            "ARCHIVE_OWNER": "owner-a",
            "ARCHIVE_INTERVAL_SECONDS": "60",
            "ARCHIVE_TOOL": str(repository / "scripts" / "experiment_archive.py"),
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NODELIST": "node-a",
            "SLURM_NNODES": "1",
            "SLURM_GPUS_ON_NODE": "1",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_RESTART_COUNT": str(restart_count),
            "RCLONE_LOG": str(tmp_path / "rclone.log"),
            "RCLONE_FAIL_COMMAND": failed_command,
        }
    )
    result = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, run_dir, tmp_path / "rclone.log"


def test_team_torch_harness_publishes_by_default():
    harness = Path(__file__).parents[1] / "scripts" / "slurm_apptainer_train.sbatch"
    contents = harness.read_text(encoding="utf-8")

    assert 'PUBLISH_TO_OSN="${PUBLISH_TO_OSN:-1}"' in contents
    assert 'ARCHIVE_OWNER="${ARCHIVE_OWNER:-${CURRENT_USER}}"' in contents
    assert "*/*|*\\\\*)" in contents
    assert 'preflight "${RUN_DIR}" --apply' in contents
    assert 'mv -- "${RUN_DIR}" "${quarantine_path}"' in contents


def test_harness_rejects_nested_run_name(tmp_path):
    result, run_dir, _ = _run_harness_to_archive_failure(
        tmp_path, restart_count=0, failed_command="none", name="nested/run"
    )

    assert result.returncode == 2
    assert "NAME must be one path segment" in result.stderr
    assert not run_dir.exists()


def test_fresh_run_preflight_failure_is_quarantined(tmp_path):
    result, run_dir, rclone_log = _run_harness_to_archive_failure(
        tmp_path, restart_count=0, failed_command="lsf"
    )

    assert result.returncode == 8
    assert not run_dir.exists()
    quarantined = list(run_dir.parent.glob("run-a.archive-preflight-failed-123-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "run-provenance.json").is_file()
    status = json.loads(
        (quarantined[0] / experiment_archive.STATUS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert status["state"] == "failed"
    assert status["exit_code"] == 8
    assert "lsf remote:bucket/archive/owner-a/run-a" in rclone_log.read_text(
        encoding="utf-8"
    )


def test_requeued_run_archive_failure_keeps_existing_directory(tmp_path):
    result, run_dir, rclone_log = _run_harness_to_archive_failure(
        tmp_path, restart_count=1, failed_command="copy"
    )

    assert result.returncode == 8
    assert (run_dir / "checkpoint").is_file()
    assert not list(run_dir.parent.glob("run-a.archive-preflight-failed-*"))
    rclone_calls = rclone_log.read_text(encoding="utf-8")
    assert "copy " in rclone_calls
    assert "lsf " not in rclone_calls


def test_default_archive_uses_public_samudra_prefix():
    assert (
        experiment_archive.DEFAULT_ARCHIVE_BASE
        == "nyu-osn:m2lines-pubs/Samudra/experiments"
    )


def test_archive_destination_appends_run_name():
    assert (
        experiment_archive.archive_destination(
            "remote:bucket/archive/", "owner-a", "run-42"
        )
        == "remote:bucket/archive/owner-a/run-42"
    )


@pytest.mark.parametrize("run_name", ["", ".", "..", "nested/run", "nested\\run"])
def test_archive_destination_rejects_unsafe_run_name(run_name):
    with pytest.raises(experiment_archive.ArchiveError):
        experiment_archive.archive_destination(
            "remote:bucket/archive", "owner-a", run_name
        )


@pytest.mark.parametrize("owner", ["", ".", "..", "team/owner", " team"])
def test_archive_destination_rejects_unsafe_owner(owner):
    with pytest.raises(experiment_archive.ArchiveError):
        experiment_archive.archive_destination("remote:bucket/archive", owner, "run-a")


def test_copy_command_is_non_destructive_and_excludes_sensitive_files(tmp_path):
    command = experiment_archive.copy_command(
        tmp_path / "run", "remote:bucket/archive/run"
    )

    assert command[:2] == ["rclone", "copy"]
    assert "sync" not in command
    assert "--delete" not in command
    for pattern in experiment_archive.EXCLUDE_PATTERNS:
        assert pattern in command


def test_copy_command_excludes_atomic_checkpoint_temporary_files(tmp_path):
    command = experiment_archive.copy_command(
        tmp_path / "run", "remote:bucket/archive/run"
    )

    assert "/saved_nets/tmp*" in command


def test_preflight_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry run must not invoke rclone")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = experiment_archive.preflight_destination(
        run_dir, "remote:bucket/archive", "owner-a", apply=False
    )

    assert result["status"] == "planned"
    assert "rclone lsf remote:bucket/archive/owner-a/run-a" in capsys.readouterr().out


def test_preflight_accepts_an_empty_destination(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    commands = []

    def return_empty(command, check, stdout):
        assert check is True
        assert stdout == subprocess.PIPE
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(subprocess, "run", return_empty)

    result = experiment_archive.preflight_destination(
        run_dir, "remote:bucket/archive", "owner-a", apply=True
    )

    assert commands == [
        [
            "rclone",
            "lsf",
            "remote:bucket/archive/owner-a/run-a",
            "--max-depth",
            "1",
        ]
    ]
    assert result["status"] == "available"


def test_preflight_rejects_an_existing_destination(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def return_existing(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="checkpoint.pt\n")

    monkeypatch.setattr(subprocess, "run", return_existing)

    with pytest.raises(
        experiment_archive.ArchiveError, match="already contains objects"
    ):
        experiment_archive.preflight_destination(
            run_dir, "remote:bucket/archive", "owner-a", apply=True
        )


def test_publish_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry run must not invoke rclone")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = experiment_archive.publish_run(
        run_dir, "remote:bucket/archive", "owner-a", apply=False
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
        run_dir, "remote:bucket/archive", "owner-a", apply=True
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

    status_path = experiment_archive.write_status(run_dir, "owner-a", "running")
    experiment_archive.write_status(run_dir, "owner-a", "completed", exit_code=0)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status == {
        "schema_version": 1,
        "owner": "owner-a",
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
        experiment_archive.write_status(run_dir, "owner-a", "unknown")


def test_status_rejects_owner_change(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    experiment_archive.write_status(run_dir, "owner-a", "running")

    with pytest.raises(experiment_archive.ArchiveError, match="owner changed"):
        experiment_archive.write_status(run_dir, "owner-b", "running")


def test_watch_publishes_without_full_verification(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    calls = []

    def record_publish(run_dir, archive_base, owner, apply, verify, rclone_bin):
        calls.append((run_dir, archive_base, owner, apply, verify, rclone_bin))

    monkeypatch.setattr(experiment_archive, "publish_run", record_publish)

    result = experiment_archive.watch_run(
        run_dir,
        "remote:bucket/archive",
        "owner-a",
        interval_seconds=0.001,
        apply=True,
        stop_event=threading.Event(),
        max_cycles=1,
    )

    assert result == 0
    assert calls == [
        (run_dir, "remote:bucket/archive", "owner-a", True, False, "rclone")
    ]


def test_watch_reports_copy_failures(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    def fail_publish(*args, **kwargs):
        command = [
            "rclone",
            "copy",
            str(run_dir),
            "remote:archive/owner-a/run-a",
        ]
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(experiment_archive, "publish_run", fail_publish)

    result = experiment_archive.watch_run(
        run_dir,
        "remote:bucket/archive",
        "owner-a",
        interval_seconds=0.001,
        apply=True,
        max_cycles=1,
    )

    assert result == 1
    assert "Periodic archive copy failed" in capsys.readouterr().err
