# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""What happened in a pytest session, read from pytest-reportlog's JSONL events.

This module records evidence only. Deciding whether a session can be trusted, and
what it means, belongs to `verdicts`.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class Case:
    nodeid: str
    outcome: Outcome
    properties: dict
    details: str

    @property
    def project(self) -> str:
        """The project whose conftest tagged this test (see `profiles.tag_project_items`)."""
        if "project" not in self.properties:
            raise ValueError(
                f"{self.nodeid} has no project tag; its conftest must call tag_project_items"
            )
        return self.properties["project"]

    @property
    def proposed(self) -> bool:
        """Whether the test checks an invariant the team hasn't agreed on yet."""
        return self.properties.get("invariant_status") == "proposed"


@dataclass(frozen=True)
class PytestRun:
    cases: tuple[Case, ...]
    exitcode: int | None
    collection_errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.exitcode is not None

    def counts(self) -> dict[str, int]:
        return {
            outcome.value: sum(case.outcome == outcome for case in self.cases)
            for outcome in Outcome
        }


def read_run(path: Path) -> PytestRun:
    """Read public pytest events; reject contradictory or malformed evidence."""
    reports: dict[str, dict[str, dict]] = {}
    collection_errors = []
    exitcode = None
    started = False
    for line in path.read_text().split("\n"):
        if not line:
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("Expected a pytest event object")
        kind = event.get("$report_type")
        if exitcode is not None:
            raise ValueError("Events after SessionFinish")
        if kind == "SessionStart":
            if started:
                raise ValueError("Duplicate SessionStart")
            started = True
        elif kind == "SessionFinish":
            if not started or type(event.get("exitstatus")) is not int:
                raise ValueError("Invalid SessionFinish")
            exitcode = event["exitstatus"]
        elif kind == "CollectReport" and event.get("outcome") == "failed":
            collection_errors.append(str(event.get("longrepr", "Collection failed")))
        elif kind == "TestReport":
            nodeid, phase, outcome = event["nodeid"], event["when"], event["outcome"]
            if (
                not started
                or not isinstance(nodeid, str)
                or phase not in ("setup", "call", "teardown")
            ):
                raise ValueError("Invalid test identity or phase")
            if outcome not in ("passed", "failed", "skipped"):
                raise ValueError(f"Unknown pytest outcome: {outcome}")
            phases = reports.setdefault(nodeid, {})
            if phase in phases:
                raise ValueError(
                    f"Duplicate {phase} for {nodeid}; rerun plugins need an explicit policy"
                )
            phases[phase] = event
    cases = []
    for nodeid, phases in reports.items():
        properties = dict(
            pair
            for event in phases.values()
            for pair in event.get("user_properties", [])
        )
        errors = [
            event
            for phase, event in phases.items()
            if phase != "call" and event["outcome"] == "failed"
        ]
        if errors:
            outcome = Outcome.ERROR
        elif any(event["outcome"] == "skipped" for event in phases.values()):
            outcome = Outcome.SKIPPED
        elif set(phases) != {"setup", "call", "teardown"}:
            outcome = Outcome.ERROR
        else:
            outcome = Outcome(phases["call"]["outcome"])
        details = []
        for event in phases.values():
            if event["outcome"] == "failed":
                value = event.get("longrepr", "")
                details.append(
                    value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False, indent=2)
                )
                details.extend(
                    f"{title}\n{text}" for title, text in event.get("sections", [])
                )
        cases.append(Case(nodeid, outcome, properties, "\n".join(details)))
    return PytestRun(tuple(cases), exitcode, tuple(collection_errors))
