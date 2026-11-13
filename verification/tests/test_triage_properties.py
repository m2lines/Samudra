# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Failure content cannot alter stable identities or create issue fanout."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st
from report_log_fixtures import events, write_log
from samudra_verification import triage
from samudra_verification.evidence import Outcome, read_run
from samudra_verification.spec import CampaignSpec, Lane

reports = triage.reports
failure_text = st.text(max_size=128)


@given(st.lists(failure_text, min_size=1, max_size=10))
def test_different_counterexamples_for_one_property_create_one_issue(messages):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        identities = []
        for message in messages:
            write_log(path, events(message=message))
            actual = reports(path, "https://example.com/run/1")
            assert len(actual) == 1
            identities.append(set(actual))
            assert actual.keys() == reports(path, "https://example.com/run/2").keys()
            assert message == "\n".join(
                line[4:]
                for line in next(iter(actual.values()))["body"].split("\n")
                if line.startswith("    ")
            )
        assert all(identity == identities[0] for identity in identities)


@given(st.text(alphabet="abc012", min_size=1, max_size=12))
def test_project_namespaces_prevent_collisions(name):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        write_log(path, events(nodeid=name, project="data"))
        data = reports(path, "https://example.com/run")
        write_log(path, events(nodeid=name, project="samudra"))
        assert reports(path, "https://example.com/run").keys().isdisjoint(data.keys())


@given(
    st.sampled_from(["setup", "call", "teardown"]),
    st.sampled_from(["passed", "failed", "skipped"]),
    st.booleans(),
)
def test_phase_failures_have_one_shared_classification(phase, outcome, proposed):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        trace = events(outcome="passed", proposed=proposed)
        next(event for event in trace if event["when"] == phase)["outcome"] = outcome
        write_log(path, trace, exitcode=int(outcome == "failed"))
        run = read_run(path)
        expected = (
            Outcome.ERROR
            if outcome == "failed" and phase != "call"
            else Outcome(outcome)
        )
        assert run.cases[0].outcome == expected
        assert bool(reports(path, "https://example.com/run")) == (
            expected == Outcome.FAILED and not proposed
        )


@given(st.integers(0, 3))
def test_interrupted_streams_never_create_product_reports(completed_phases):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pytest.jsonl"
        write_log(path, events()[:completed_phases], exitcode=None)
        assert not read_run(path).complete
        assert reports(path, "https://example.com/run") == {}


@given(
    st.lists(st.sampled_from(["absent", "open", "closed", "pull_request"]), max_size=8),
    failure_text,
)
def test_issue_filing_is_idempotent_across_paginated_open_and_closed_issues(
    states, message
):
    """Simulate the CLI boundary: only new property failures create issues."""
    candidates = {
        f"<!-- property-failure:{index} -->": {
            "title": f"Property {index}",
            "body": message,
        }
        for index in range(len(states))
    }
    # Real reports contain their marker; retain arbitrary generated failure text.
    for marker, report in candidates.items():
        report["body"] = marker + "\n" + report["body"]
    issues = []
    for (marker, _), state in zip(candidates.items(), states):
        if state != "absent":
            issue = {"body": marker, "state": state}
            if state == "pull_request":
                issue["pull_request"] = {}
            issues.append(issue)
    created = []

    def fake_gh(*args, input=None):
        if args[0] == "api":
            assert "--paginate" in args and "--slurp" in args
            return json.dumps([[issue] for issue in issues])
        assert args[:2] == ("issue", "create")
        assert args[-2:] == ("--body-file", "-")
        title = args[args.index("--title") + 1]
        created.append({"title": title, "body": input})
        issues.append({"body": input, "state": "open"})
        return "https://example.com/new-issue"

    triage.file_new_reports(candidates, "owner/repo", invoke=fake_gh)
    assert created == [
        report
        for report, state in zip(candidates.values(), states)
        if state in ("absent", "pull_request")
    ]
    original_count = len(created)
    triage.file_new_reports(candidates, "owner/repo", invoke=fake_gh)
    assert len(created) == original_count


@given(
    st.lists(
        st.sampled_from(["new", "open", "closed", "pull_request"]),
        min_size=1,
        max_size=8,
    )
)
def test_read_only_triage_keeps_every_recurrence_without_issue_fanout(states):
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        path = root / "pytest.jsonl"
        CampaignSpec(
            project="data",
            lane=Lane.PINNED,
            string_storage=None,
            profile="soak",
            examples=1,
            seed=0,
            timeout_seconds=1,
            command=[],
            cwd=str(root),
            revision="0" * 40,
            python="3.12",
            platform="test",
            dependency_file="unused",
            dependency_sha256="0" * 64,
        ).write(root)
        write_log(
            path,
            [
                event
                for index in range(len(states))
                for event in events(f"test_subject.py::property_{index}")
            ],
        )
        candidates = reports(path, "https://example.com/run")
        existing = []
        for index, (marker, state) in enumerate(zip(candidates, states), start=1):
            if state == "new":
                continue
            issue = {
                "body": marker,
                "number": index,
                "state": state,
                "html_url": f"https://example.com/issues/{index}",
            }
            if state == "pull_request":
                issue["pull_request"] = {}
            existing.append(issue)

        def fake_cli(command, **kwargs):
            assert command[:2] == ["gh", "api"]  # No issue create/edit/comment calls.
            return json.dumps([[issue] for issue in existing])

        patch.setattr(subprocess, "check_output", fake_cli)
        patch.setattr(
            sys,
            "argv",
            [
                "reporter",
                str(root),
                "--repo",
                "owner/repo",
                "--run-url",
                "https://example.com/run",
                "--triage-output",
                str(root / "triage"),
            ],
        )
        triage.main()
        summary = json.loads((root / "triage" / "summary.json").read_text())
        assert summary["total"] == len(states)
        assert summary["new"] == sum(
            state in ("new", "pull_request") for state in states
        )
        assert summary["recurrences"] == sum(
            state in ("open", "closed") for state in states
        )
        assert summary["issues_created"] == 0
        for failure, state in zip(summary["failures"], states):
            assert len(failure["existing_issues"]) == (state in ("open", "closed"))
            for issue in failure["existing_issues"]:
                assert issue["url"] in (root / "triage" / "summary.md").read_text()
