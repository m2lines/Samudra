# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

VARIANT_SCRIPT = Path(__file__).parents[2] / "scripts" / "om4_data_variant.sh"
SLURM_HARNESSES = [
    VARIANT_SCRIPT.parent / "slurm_preprocess_om4.sbatch",
    VARIANT_SCRIPT.parent / "slurm_make_norm_om4.sbatch",
]
NORM_HARNESS = SLURM_HARNESSES[1]


def _resolve_variant(variant: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"DATA_VARIANT": variant}
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; resolve_om4_data_variant || exit $?; '
            'printf "%s|%s|%s|%s" "$DATA_VARIANT" "$OM4_SOURCE_STORE" '
            '"$OM4_OUTPUT_SUFFIX" "$OM4_WFO_SOURCE_STORE"',
            "bash",
            str(VARIANT_SCRIPT),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (
            "averaged",
            "averaged|om4_5daily.zarr||om4_5daily_snapshots.zarr",
        ),
        ("snapshots", "snapshots|om4_5daily_snapshots.zarr|_snapshots|"),
    ],
)
def test_om4_data_variant(variant, expected):
    result = _resolve_variant(variant)

    assert result.returncode == 0
    assert result.stdout == expected


def test_om4_data_variant_rejects_unknown_value():
    result = _resolve_variant("instantaneous")

    assert result.returncode == 2
    assert "unknown DATA_VARIANT='instantaneous'" in result.stderr


@pytest.mark.parametrize("harness", SLURM_HARNESSES)
def test_slurm_harness_resolves_sidecar_from_repo_checkout(harness):
    script = harness.read_text()

    # sbatch executes a copy under Slurm's spool directory, so BASH_SOURCE
    # cannot locate a helper stored beside the original script.
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in script
    assert 'VARIANT_SCRIPT="${REPO_DIR%/}/scripts/om4_data_variant.sh"' in script
    assert script.index('REPO_DIR="${REPO_DIR:-') < script.index(
        'source "${VARIANT_SCRIPT}"'
    )


def test_preprocessing_harness_runs_selected_data_subproject():
    script = SLURM_HARNESSES[0].read_text()

    # An editable conda install may reference another checkout. Running from
    # REPO_DIR/data ensures Python imports the revision selected for this job.
    assert 'cd "${REPO_DIR}/data"' in script
    assert script.index('cd "${REPO_DIR}/data"') < script.index(
        "\npython -m ocean_preprocessing om4"
    )


def _run_norm_harness(tmp_path, *, max_attempts, succeed_on_attempt):
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { :; }\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        "count=0\n"
        '[[ -f "$NORM_TEST_STATE" ]] && read -r count < "$NORM_TEST_STATE"\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "$count" > "$NORM_TEST_STATE"\n'
        "(( count >= NORM_SUCCEED_ON_ATTEMPT ))\n"
    )
    fake_python.chmod(0o755)

    state = tmp_path / "attempts"
    env = os.environ | {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",  # pragma: allowlist secret
        "CONDA_SH": str(conda_sh),
        "NORM_MAX_ATTEMPTS": str(max_attempts),
        "NORM_RETRY_DELAY_SECONDS": "0",
        "NORM_SUCCEED_ON_ATTEMPT": str(succeed_on_attempt),
        "NORM_TEST_STATE": str(state),
        "OUTPUT_PATH": "s3://test/input/OM4.zarr",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "REPO_DIR": str(VARIANT_SCRIPT.parents[1]),
    }
    result = subprocess.run(
        ["bash", str(NORM_HARNESS)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    return result, int(state.read_text())


def test_norm_harness_retries_then_succeeds(tmp_path):
    result, attempts = _run_norm_harness(tmp_path, max_attempts=3, succeed_on_attempt=3)

    assert result.returncode == 0
    assert attempts == 3
    assert "Normalization attempt 3/3" in result.stdout


def test_norm_harness_stops_after_bounded_attempts(tmp_path):
    result, attempts = _run_norm_harness(
        tmp_path, max_attempts=2, succeed_on_attempt=99
    )

    assert result.returncode != 0
    assert attempts == 2
    assert "normalization failed after 2 attempts" in result.stderr


def test_norm_harness_is_requeueable():
    assert "#SBATCH --requeue" in NORM_HARNESS.read_text()
