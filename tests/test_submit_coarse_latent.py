# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[1]


@pytest.fixture
def submission_environment(tmp_path: Path) -> tuple[dict[str, str], list[Path]]:
    scratch = tmp_path / "scratch"
    data = scratch / "data"
    output = scratch / "runs"
    logs = scratch / "logs"
    fake_bin = tmp_path / "bin"
    for directory in (data, output, logs, fake_bin):
        directory.mkdir(parents=True)

    required = [
        tmp_path / "inverse.pt",
        tmp_path / "code.img",
        tmp_path / "code.img.sha256",
        tmp_path / "code.img.json",
        tmp_path / "container.sif",
        tmp_path / "train.sbatch",
        tmp_path / "audit.sbatch",
        tmp_path / "audit.py",
        tmp_path / "inverse_audit.py",
    ]
    for path in required:
        path.write_text("fixture\n")

    counter = tmp_path / "sbatch-counter"
    counter.write_text("9000\n")
    calls = tmp_path / "sbatch-calls"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
read -r job_id < "${FAKE_SBATCH_COUNTER}"
job_id="$((job_id + 1))"
printf '%s\\n' "${job_id}" > "${FAKE_SBATCH_COUNTER}"
printf 'SBATCH_ARGS=%q CONFIG=%q NAME=%q TRAIN_ARGS=%q CHECKPOINT=%q NCCL_P2P_DISABLE=%q\\n' \
  "$*" "${CONFIG:-}" "${NAME:-}" "${ARGS:-}" "${CHECKPOINT:-}" \
  "${NCCL_P2P_DISABLE:-}" \
  >> "${FAKE_SBATCH_CALLS}"
printf '%s\\n' "${job_id}"
"""
    )
    fake_sbatch.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SCRATCH_DIR": str(scratch),
        "DATA_ROOT": str(data),
        "OUTPUT_BASE": str(output),
        "LOG_DIR": str(logs),
        "SBATCH_SCRIPT": str(required[5]),
        "AUDIT_SBATCH_SCRIPT": str(required[6]),
        "AUDIT_SCRIPT": str(required[7]),
        "INVERSE_AUDIT_SCRIPT": str(required[8]),
        "WANDB_API_KEY": "test-only",  # pragma: allowlist secret
        "FAKE_SBATCH_COUNTER": str(counter),
        "FAKE_SBATCH_CALLS": str(calls),
        "DATE_TAG": "test",
    }
    return environment, required


def test_s2_submission_wires_matched_best_checkpoint_evaluations(
    submission_environment: tuple[dict[str, str], list[Path]],
) -> None:
    environment, required = submission_environment
    result = subprocess.run(
        [
            REPOSITORY / "scripts/submit_coarse_latent_s2.sh",
            required[0],
            required[1],
            required[4],
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    rows = [line.split("\t") for line in result.stdout.splitlines()]
    assert rows[0][:4] == [
        "arm",
        "train_job_id",
        "validation_job_id",
        "audit_job_id",
    ]
    assert [row[:4] for row in rows[1:]] == [
        ["physical-only", "9001", "9002", "9003"],
        ["latent-only", "9004", "9005", "9006"],
        ["combined-001", "9007", "9008", "9009"],
        ["combined-01", "9010", "9011", "9012"],
    ]
    calls = Path(environment["FAKE_SBATCH_CALLS"]).read_text().splitlines()
    assert len(calls) == 12
    assert "--dependency=afterok:9001" in calls[1]
    assert "best_validation_ckpt.pt" in calls[1]
    assert "--dependency=afterok:9001" in calls[2]
    assert "best_validation_ckpt.pt" in calls[2]


def test_s3_submission_supports_two_node_four_gpu_layout(
    submission_environment: tuple[dict[str, str], list[Path]],
) -> None:
    environment, required = submission_environment
    environment = {
        **environment,
        "TRAIN_NODES": "2",
        "TRAIN_GPUS_PER_NODE": "4",
    }
    subprocess.run(
        [
            REPOSITORY / "scripts/submit_coarse_latent_s3.sh",
            required[0],
            "1",
            "0.1",
            required[1],
            required[4],
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    calls = Path(environment["FAKE_SBATCH_CALLS"]).read_text().splitlines()
    assert "--nodes=2" in calls[0]
    assert "--gres=gpu:4" in calls[0]
    assert "--cpus-per-task=32" in calls[0]
    assert "--mem=256G" in calls[0]


def test_s3_submission_requests_eight_gpus_and_audits_best_checkpoint(
    submission_environment: tuple[dict[str, str], list[Path]],
) -> None:
    environment, required = submission_environment
    result = subprocess.run(
        [
            REPOSITORY / "scripts/submit_coarse_latent_s3.sh",
            required[0],
            "1",
            "0.01",
            required[1],
            required[4],
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    rows = [line.split("\t") for line in result.stdout.splitlines()]
    assert rows[0][:3] == [
        "train_job_id",
        "validation_job_id",
        "audit_job_id",
    ]
    assert rows[1][:3] == ["9001", "9002", "9003"]
    calls = Path(environment["FAKE_SBATCH_CALLS"]).read_text().splitlines()
    assert len(calls) == 3
    assert "--gres=gpu:8" in calls[0]
    assert "--dependency=afterok:9001" in calls[1]
    assert "best_validation_ckpt.pt" in calls[1]
    assert "--dependency=afterok:9001" in calls[2]
    assert "best_validation_ckpt.pt" in calls[2]


def test_s3_submission_sizes_rtx6000_and_disables_nccl_p2p(
    submission_environment: tuple[dict[str, str], list[Path]],
) -> None:
    environment, required = submission_environment
    environment = {
        **environment,
        "TRAIN_GPU_FAMILY": "rtx6000",
    }
    result = subprocess.run(
        [
            REPOSITORY / "scripts/submit_coarse_latent_s3.sh",
            required[0],
            "1",
            "0.1",
            required[1],
            required[4],
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    rows = [line.split("\t") for line in result.stdout.splitlines()]
    assert rows[0][-1] == "gpu_family"
    assert rows[1][-1] == "rtx6000"
    calls = Path(environment["FAKE_SBATCH_CALLS"]).read_text().splitlines()
    assert "--constraint=rtx6000" in calls[0]
    assert "--cpus-per-task=128" in calls[0]
    assert "--mem=1400G" in calls[0]
    assert "NCCL_P2P_DISABLE=1" in calls[0]
    assert "--constraint=rtx6000" in calls[1]
    assert "--cpus-per-task=16" in calls[1]
    assert "--mem=175G" in calls[1]
    assert "NCCL_P2P_DISABLE=1" in calls[1]
