# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic public pytest events for adversarial report-consumer tests."""

import json


def events(
    nodeid="test_subject.py::test_property",
    outcome="failed",
    *,
    message="assertion failed",
    project="data",
    proposed=False,
):
    properties = [("project", project)]
    if proposed:
        properties.append(("invariant_status", "proposed"))
    return [
        {
            "$report_type": "TestReport",
            "nodeid": nodeid,
            "when": phase,
            "outcome": outcome if phase == "call" else "passed",
            "user_properties": properties,
            "longrepr": message if phase == "call" and outcome == "failed" else None,
        }
        for phase in ("setup", "call", "teardown")
    ]


def write_log(path, reports, exitcode=1):
    stream = [{"$report_type": "SessionStart"}, *reports]
    if exitcode is not None:
        stream.append({"$report_type": "SessionFinish", "exitstatus": exitcode})
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=True) + "\n" for event in stream)
    )
