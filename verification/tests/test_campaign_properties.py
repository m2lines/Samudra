# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the campaign CLI on generated suites: empty or all-skipped runs can't pass."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[2]


@settings(max_examples=8, deadline=None)
@given(st.lists(st.sampled_from(["pass", "skip", "fail"]), max_size=3))
def test_shared_runner_records_real_outcomes_and_replay_inputs(outcomes):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        tests = root / "checks"
        tests.mkdir()
        config = root / "pytest.ini"
        config.write_text("[pytest]\n")
        lines = ["import pytest"]
        for index, outcome in enumerate(outcomes):
            if outcome == "skip":
                lines.append("@pytest.mark.skip(reason='generated skip')")
            lines.append(
                f"def test_generated_{index}():\n    assert {outcome != 'fail'}"
            )
        (tests / "test_generated.py").write_text("\n".join(lines) + "\n")
        output = root / "artifacts"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "samudra_verification.campaign",
                "--project",
                "harness",
                "--lane",
                "pinned",
                "--tests",
                str(tests),
                "--dependency-file",
                str(config),
                "--output",
                str(output),
                "--seed",
                "42",
                "--examples",
                "1",
                "--timeout",
                "20",
                "--",
                "-c",
                str(config),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        # 1: a counterexample; 0: passed; 2: invalid, since nothing ran.
        expected = 1 if "fail" in outcomes else (0 if "pass" in outcomes else 2)
        assert process.returncode == expected, process.stdout + process.stderr
        result = json.loads((output / "result.json").read_text())
        assert (result["status"] == "passed") == (expected == 0)
        replay = json.loads((output / "replay.json").read_text())
        assert replay["project"] == "harness" and replay["seed"] == 42
        assert (output / "dependency-input.txt").read_bytes() == config.read_bytes()
        assert (output / "pytest.log").is_file() and (output / "pytest.jsonl").is_file()
