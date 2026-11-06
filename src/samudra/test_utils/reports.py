# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Domain results over pytest-reportlog's native JSONL event stream.

A call failure is a counterexample. A setup/teardown/collection failure is an
execution error. An interrupted stream is an incomplete run. These distinctions
are shared by ordinary campaigns, mutation checks, and human triage.
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
    def contract_probe(self) -> bool:
        return self.properties.get("contract_probe") == "true"


@dataclass(frozen=True)
class Run:
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


def read_run(path: Path) -> Run:
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
    return Run(tuple(cases), exitcode, tuple(collection_errors))


def campaign_status(returncode: int, run: Run) -> str:
    """A green process is insufficient without complete, nonempty pytest evidence."""
    if not run.complete or run.exitcode != returncode:
        return "incomplete"
    if run.collection_errors or any(
        case.outcome == Outcome.ERROR for case in run.cases
    ):
        return "error"
    if returncode == 5 and not run.cases:
        return "no_tests"
    if returncode != 0:
        return "failed"
    if any(case.outcome == Outcome.FAILED for case in run.cases):
        return "invalid"
    return (
        "passed"
        if any(case.outcome == Outcome.PASSED for case in run.cases)
        else "no_tests"
    )
