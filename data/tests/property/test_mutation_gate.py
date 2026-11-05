# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A broken test process or incomplete baseline must never produce a mutation kill."""

import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

gate = runpy.run_path(
    Path(__file__).resolve().parents[1] / "mutation" / "virtual_archive_mutants.py"
)
outcome = gate["outcome"]
# Only construct inner test wrappers; they are never executed by these properties.
integer_test = given(st.integers())


def evidence(path, nodes):
    expected = {
        "nodeids": sorted(nodes),
        "profile": gate["EXPECTED_PROFILE"],
        "target_sha256": "a" * 64,
        "mutant_sha256": None,
    }
    (path.parent / "execution.json").write_text(json.dumps(expected))
    return expected


def add_case(suite, node="tests/property/test_subject.py::property"):
    case = ET.SubElement(suite, "testcase")
    ET.SubElement(
        ET.SubElement(case, "properties"),
        "property",
        name="mutation_nodeid",
        value=node,
    )
    return case


def failure_report(path):
    suite = ET.Element("testsuite")
    ET.SubElement(add_case(suite), "failure").text = "assertion failure"
    ET.ElementTree(suite).write(path)
    expected = evidence(path, ["tests/property/test_subject.py::property"])
    return suite, expected


@given(st.sampled_from([-9, -1, 0, 2, 3, 4, 5]))
def test_assertion_text_cannot_turn_a_failed_runner_into_a_kill(returncode):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        _, expected = failure_report(path)
        assert outcome(returncode, path, expected) == "invalid"
        path.unlink()
        assert outcome(1, path, expected) == "invalid"


@given(st.sampled_from(["error", "skipped"]))
def test_setup_errors_and_skips_invalidate_an_apparent_kill(interruption):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite, expected = failure_report(path)
        assert outcome(1, path, expected) == "killed"
        ET.SubElement(suite[0], interruption)
        ET.ElementTree(suite).write(path)
        assert outcome(1, path, expected) == "invalid"


@given(st.integers(1, 8))
def test_removing_a_selected_test_invalidates_a_clean_baseline(count):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite = ET.Element("testsuite")
        nodes = [
            f"tests/property/test_subject.py::property_{index}"
            for index in range(count)
        ]
        for node in nodes:
            add_case(suite, node)
        expected = evidence(path, nodes)
        ET.ElementTree(suite).write(path)
        assert outcome(0, path, expected) == "survived"
        suite.remove(suite[-1])
        ET.ElementTree(suite).write(path)
        assert outcome(0, path, expected) == "invalid"


@given(st.text(alphabet="abc012", min_size=1, max_size=20))
def test_same_count_of_different_tests_does_not_count_as_a_kill(other):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        suite, expected = failure_report(path)
        assert outcome(1, path, expected) == "killed"
        suite.find("./testcase/properties/property").set("value", other)
        ET.ElementTree(suite).write(path)
        assert outcome(1, path, expected) == "invalid"


@given(
    st.sampled_from(
        [
            "max_examples",
            "phases",
            "database_disabled",
            "deadline_disabled",
            "target_sha256",
            "mutant_sha256",
            "nodeids",
        ]
    )
)
def test_profile_source_or_collection_drift_invalidates_a_kill(field):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        _, expected = failure_report(path)
        assert outcome(1, path, expected) == "killed"
        actual = json.loads(json.dumps(expected))
        changes = {
            "max_examples": 1,
            "phases": ["explicit"],
            "database_disabled": False,
            "deadline_disabled": False,
            "target_sha256": "b" * 64,
            "mutant_sha256": "b" * 64,
            "nodeids": [],
        }
        if field in actual["profile"]:
            actual["profile"][field] = changes[field]
        else:
            actual[field] = changes[field]
        (path.parent / "execution.json").write_text(json.dumps(actual))
        assert outcome(1, path, expected) == "invalid"


@given(st.integers(1, 100))
def test_per_test_settings_cannot_override_the_mutation_profile(budget):
    function = integer_test(lambda value: None)
    function = settings(max_examples=budget)(function)
    with pytest.raises(InvalidArgument):
        gate["bind_profile"]([function], settings.get_profile("mutation"))


@given(st.integers(1, 8))
def test_parametrized_targets_bind_the_same_function_only_once(count):
    function = integer_test(lambda value: None)
    gate["bind_profile"]([function] * count, settings.get_profile("mutation"))


@given(st.sampled_from(["xml", "json", "utf8", "missing"]))
def test_corrupt_or_missing_execution_evidence_cannot_produce_a_kill(corruption):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        _, expected = failure_report(path)
        actual = path.parent / "execution.json"
        if corruption == "xml":
            path.write_text("<testsuite>")
        elif corruption == "json":
            actual.write_text("{")
        elif corruption == "utf8":
            actual.write_bytes(b"\xff")
        else:
            actual.unlink()
        assert outcome(1, path, expected) == "invalid"
