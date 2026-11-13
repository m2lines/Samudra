# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A broken test process or incomplete baseline must never produce a mutation kill."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

from samudra.test_utils import mutations as gate
from tests.report_log_fixtures import events, write_log

outcome = gate.outcome
# Only construct inner test wrappers; they are never executed by these properties.
integer_test = given(st.integers())


def evidence(path, nodes):
    expected = {
        "nodeids": sorted(nodes),
        "profile": gate.EXPECTED_PROFILE,
        "target_sha256": "a" * 64,
        "mutant_sha256": None,
    }
    (path.parent / "execution.json").write_text(json.dumps(expected))
    return expected


def failure_report(path, failures=(("builtins.AssertionError", "oracle"),)):
    trace = events()
    trace[1]["user_properties"] = [
        *trace[1]["user_properties"],
        ("failures", json.dumps([list(failure) for failure in failures])),
    ]
    write_log(path, trace)
    expected = evidence(path, [trace[0]["nodeid"]])
    return trace, expected


@given(st.sampled_from([-9, -1, 0, 2, 3, 4, 5]))
def test_assertion_text_cannot_turn_a_failed_runner_into_a_kill(returncode):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        _, expected = failure_report(path)
        assert outcome(returncode, path, expected).status == "invalid"
        path.unlink()
        assert outcome(1, path, expected).status == "invalid"


@given(
    st.lists(
        st.sampled_from(
            [
                ("builtins.AssertionError", "oracle"),
                ("hypothesis.errors.FailedHealthCheck", "hypothesis"),
                ("builtins.NameError", "broken_mutant"),
            ]
        ),
        min_size=1,
        max_size=3,
    )
)
def test_only_oracle_failures_kill_a_mutant(failures):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        _, expected = failure_report(path, failures)
        oracle_only = all(kind == "oracle" for _, kind in failures)
        assert outcome(1, path, expected).status == (
            "killed" if oracle_only else "invalid"
        )


@given(st.sampled_from(["error", "skipped"]))
def test_setup_errors_and_skips_invalidate_an_apparent_kill(interruption):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        suite, expected = failure_report(path)
        assert outcome(1, path, expected).status == "killed"
        suite[0]["outcome"] = "failed" if interruption == "error" else "skipped"
        write_log(path, suite)
        assert outcome(1, path, expected).status == "invalid"


@given(st.integers(1, 8))
def test_removing_a_selected_test_invalidates_a_clean_baseline(count):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        nodes = [f"test_subject.py::property_{index}" for index in range(count)]
        trace = [event for node in nodes for event in events(node, "passed")]
        expected = evidence(path, nodes)
        write_log(path, trace, exitcode=0)
        assert outcome(0, path, expected).status == "survived"
        write_log(path, trace[:-3], exitcode=0)
        assert outcome(0, path, expected).status == "invalid"


@given(st.text(alphabet="abc012", min_size=1, max_size=20))
def test_same_count_of_different_tests_does_not_count_as_a_kill(other):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        suite, expected = failure_report(path)
        assert outcome(1, path, expected).status == "killed"
        for event in suite:
            event["nodeid"] = other
        write_log(path, suite)
        assert outcome(1, path, expected).status == "invalid"


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
        path = Path(directory) / "pytest.jsonl"
        _, expected = failure_report(path)
        assert outcome(1, path, expected).status == "killed"
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
        assert outcome(1, path, expected).status == "invalid"


@given(st.integers(1, 100))
def test_per_test_settings_cannot_override_the_mutation_profile(budget):
    function = integer_test(lambda value: None)
    function = settings(max_examples=budget)(function)
    with pytest.raises(InvalidArgument):
        gate.bind_profile([function], settings.get_profile("mutation"))


@given(st.integers(1, 8))
def test_parametrized_targets_bind_the_same_function_only_once(count):
    function = integer_test(lambda value: None)
    gate.bind_profile([function] * count, settings.get_profile("mutation"))


@given(st.sampled_from(["events", "json", "utf8", "missing"]))
def test_corrupt_or_missing_execution_evidence_cannot_produce_a_kill(corruption):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        _, expected = failure_report(path)
        actual = path.parent / "execution.json"
        if corruption == "events":
            path.write_text("{")
        elif corruption == "json":
            actual.write_text("{")
        elif corruption == "utf8":
            actual.write_bytes(b"\xff")
        else:
            actual.unlink()
        assert outcome(1, path, expected).status == "invalid"


@given(
    elapsed=st.integers(0, 2000),
    per_case=st.integers(1, 120),
    budget=st.integers(1, 1000),
)
def test_shared_budget_caps_each_case_and_stops_scheduling(elapsed, per_case, budget):
    allowed = gate.remaining_timeout(budget, per_case, elapsed)
    assert 0 <= allowed <= per_case
    assert allowed + elapsed <= budget or allowed == 0
    assert gate.remaining_timeout(budget, per_case, elapsed + 1) <= allowed
    if elapsed >= budget:
        assert allowed == 0
        result = gate.run_case(
            None, [], 0, allowed, target_sha256="unused", catalog_path=None
        )
        assert result["status"] == "budget_exhausted"


@settings(max_examples=3, deadline=None)
@given(limit=st.integers(1, 9), seed=st.integers(0, 1000))
def test_real_subprocess_gate_distinguishes_baseline_from_inserted_fault(limit, seed):
    """The same generated oracle passes clean code and kills an in-memory fault."""
    import time

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pytest.ini").write_text("[pytest]\n")
        (root / "conftest.py").write_text(
            "from samudra.test_utils.profiles import configure_profiles\nconfigure_profiles()\n"
        )
        target = root / "verification_subject.py"
        target.write_text("def identity(value):\n    return value\n")
        properties = root / "test_subject.py"
        properties.write_text(
            "from hypothesis import given, strategies as st\n"
            "from verification_subject import identity\n"
            f"@given(st.integers(0, {limit}))\n"
            "def test_identity(value):\n    assert identity(value) == value\n"
        )
        catalog = root / "identity_mutations.py"
        catalog.write_text(
            "from pathlib import Path\n"
            "from samudra.test_utils.catalogs import Catalog, Mutation\n"
            "root = Path(__file__).parent\n"
            "CATALOG = Catalog(root, 'verification_subject', root/'verification_subject.py', "
            "root/'test_subject.py', ("
            "Mutation('offset', 'return value', 'return value + 1', 'test_identity'), "
            "Mutation('broken', 'return value', 'return undefined_name', 'test_identity')))\n"
        )
        # The broken mutant fails with NameError, which is not an oracle verdict.
        assert not gate.run_catalog(
            catalog, root / "output", [seed], 15, time.monotonic() + 60
        )
        results = json.loads((root / "output/results.json").read_text())
        assert results[f"baseline-{seed}"]["status"] == "passed"
        assert results[f"offset-{seed}"]["status"] == "killed"
        assert results[f"broken-{seed}"]["status"] == "invalid"
        assert "NameError" in results[f"broken-{seed}"]["reason"]
        assert target.read_text() == "def identity(value):\n    return value\n"
