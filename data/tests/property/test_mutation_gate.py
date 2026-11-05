# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A broken test process or incomplete baseline must never produce a mutation kill."""

import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given
from hypothesis import strategies as st

outcome = runpy.run_path(
    Path(__file__).resolve().parents[1] / "mutation" / "virtual_archive_mutants.py"
)["outcome"]


@given(st.sampled_from([-9, -1, 0, 2, 3, 4, 5]))
def test_assertion_text_cannot_turn_a_failed_runner_into_a_kill(returncode):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite = ET.Element("testsuite")
        ET.SubElement(
            ET.SubElement(suite, "testcase"), "failure"
        ).text = "assertion failure"
        ET.ElementTree(suite).write(path)
        assert outcome(returncode, path, 1) == "invalid"
        path.unlink()
        assert outcome(1, path, 1) == "invalid"


@given(st.sampled_from(["error", "skipped"]))
def test_setup_errors_and_skips_invalidate_an_apparent_kill(interruption):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite = ET.Element("testsuite")
        case = ET.SubElement(suite, "testcase")
        ET.SubElement(case, "failure")
        ET.ElementTree(suite).write(path)
        assert outcome(1, path, 1) == "killed"
        ET.SubElement(case, interruption)
        ET.ElementTree(suite).write(path)
        assert outcome(1, path, 1) == "invalid"


@given(st.integers(1, 8))
def test_removing_a_selected_test_invalidates_a_clean_baseline(count):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite = ET.Element("testsuite")
        for _ in range(count):
            ET.SubElement(suite, "testcase")
        ET.ElementTree(suite).write(path)
        assert outcome(0, path, count) == "survived"
        suite.remove(suite[-1])
        ET.ElementTree(suite).write(path)
        assert outcome(0, path, count) == "invalid"
