# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Generated failure messages must not break issue identity or XML extraction."""

import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given
from hypothesis import strategies as st

reporter = runpy.run_path(
    Path(__file__).resolve().parents[3] / "scripts" / "file_property_failures.py"
)
reports = reporter["reports"]

xml_text = st.text(
    alphabet=st.characters(exclude_categories=("Cs", "Cc", "Cn")), max_size=128
)
xml_text = st.one_of(xml_text, st.sampled_from(["\u2028", "\u2029", "```\n$()`", ""]))


@given(st.lists(xml_text, min_size=1, max_size=10))
def test_different_counterexamples_for_one_property_create_one_issue(messages):
    suite = ET.Element("testsuite")
    for message in messages:
        case = ET.SubElement(suite, "testcase", classname="suite", name="property")
        ET.SubElement(case, "failure").text = message
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        ET.ElementTree(suite).write(path, encoding="utf-8")
        actual = reports(path, "https://example.com/run/1")
        assert len(actual) == 1
        # Failure content and run URL must not change the deduplication identity.
        assert actual.keys() == reports(path, "https://example.com/run/2").keys()
        assert messages[-1] == "\n".join(
            line[4:]
            for line in next(iter(actual.values()))["body"].split("\n")
            if line.startswith("    ")
        )


@given(
    st.lists(
        st.sampled_from(["error", "skipped", "pass", "contract_probe"]), max_size=10
    )
)
def test_environment_errors_and_skips_do_not_file_product_bugs(outcomes):
    suite = ET.Element("testsuite")
    for index, outcome in enumerate(outcomes):
        case = ET.SubElement(suite, "testcase", classname="suite", name=str(index))
        if outcome == "contract_probe":
            properties = ET.SubElement(case, "properties")
            ET.SubElement(properties, "property", name="contract_probe", value="true")
            ET.SubElement(case, "failure").text = "unsettled contract"
        elif outcome != "pass":
            ET.SubElement(case, outcome).text = "runner or collection failure"
    with TemporaryDirectory() as directory:
        path = Path(directory) / "junit.xml"
        ET.ElementTree(suite).write(path, encoding="utf-8")
        assert reports(path, "https://example.com/run/1") == {}


@given(
    st.lists(st.sampled_from(["absent", "open", "closed", "pull_request"]), max_size=8),
    xml_text,
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

    reporter["file_new_reports"](candidates, "owner/repo", invoke=fake_gh)
    assert created == [
        report
        for report, state in zip(candidates.values(), states)
        if state in ("absent", "pull_request")
    ]
    original_count = len(created)
    reporter["file_new_reports"](candidates, "owner/repo", invoke=fake_gh)
    assert len(created) == original_count
