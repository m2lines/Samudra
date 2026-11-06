# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared CLI in isolation: empty/skipped checks cannot look green."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[3]


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
                str(ROOT / "scripts" / "run_property_suite.py"),
                "--suite",
                "harness",
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
        expected = 1 if "fail" in outcomes else (0 if "pass" in outcomes else 5)
        assert process.returncode == expected, process.stdout + process.stderr
        result = json.loads((output / "result.json").read_text())
        assert (result["status"] == "passed") == (expected == 0)
        replay = json.loads((output / "replay.json").read_text())
        assert replay["suite"] == "harness" and replay["seed"] == 42
        assert (output / "dependency-input.txt").read_bytes() == config.read_bytes()
        assert (output / "pytest.log").is_file() and (output / "junit.xml").is_file()
