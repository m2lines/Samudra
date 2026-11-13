# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a pytest session can be trusted, and what a trusted one means.

Every experiment asks the same first question, answered by `untrustworthy`: did the
session finish, collect cleanly, and run what we asked for? Only then does each
experiment interpret the failures:

* A campaign either passes or finds counterexamples (`CampaignStatus`).
* A mutation-analysis trial runs either the unmodified code (a control) or a
  mutant. A control is clean or dirty; a mutant is killed or survives
  (`TrialStatus`).
"""

from enum import StrEnum
from typing import NamedTuple

from .evidence import Outcome, PytestRun

# pytest's exit codes for "tests ran, none failed" and "tests ran, some failed".
PYTEST_OK, PYTEST_FAILED = 0, 1


class CampaignStatus(StrEnum):
    PASSED = "passed"
    COUNTEREXAMPLE = "counterexample"
    INVALID = "invalid"
    TIMEOUT = "timeout"


class TrialStatus(StrEnum):
    CLEAN = "clean"  # Control: the properties pass on unmodified code.
    DIRTY = "dirty"  # Control: a property fails on unmodified code.
    KILLED = "killed"  # Mutant: a property detected the fault.
    SURVIVED = "survived"  # Mutant: every property still passed.
    INVALID = "invalid"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Verdict(NamedTuple):
    status: StrEnum  # A CampaignStatus or TrialStatus.
    reason: str = ""


def untrustworthy(
    returncode: int,
    run: PytestRun,
    *,
    allow_skips: bool,
    expected_nodeids: tuple[str, ...] | None = None,
) -> str | None:
    """Why the session can't be trusted, or None if it can.

    With `expected_nodeids`, every test that ran must be one we asked for. A session
    may stop early after a failure, so a subset is allowed here. Interpretations that
    claim "everything passed" must check that everything ran.
    """
    if not run.complete:
        return "the pytest session did not finish"
    if run.exitcode != returncode:
        return f"the process exited {returncode} but pytest reported {run.exitcode}"
    if run.collection_errors:
        return f"collection errors: {list(run.collection_errors)}"
    for case in run.cases:
        if case.outcome == Outcome.ERROR:
            return f"{case.nodeid}: setup or teardown error"
        if case.outcome == Outcome.SKIPPED and not allow_skips:
            return f"{case.nodeid}: skipped"
    ran = [
        case for case in run.cases if case.outcome in (Outcome.PASSED, Outcome.FAILED)
    ]
    if not ran:
        return "no tests ran"
    if expected_nodeids is not None:
        unexpected = sorted({case.nodeid for case in run.cases} - set(expected_nodeids))
        if unexpected:
            return f"ran tests that weren't requested: {unexpected}"
    failed = any(case.outcome == Outcome.FAILED for case in run.cases)
    if returncode not in (PYTEST_OK, PYTEST_FAILED):
        return f"pytest exited {returncode}"
    if (returncode == PYTEST_FAILED) != failed:
        return (
            f"pytest exited {returncode}, but {'some' if failed else 'no'} tests failed"
        )
    return None


def campaign_verdict(returncode: int, run: PytestRun) -> Verdict:
    # Campaigns may deselect or skip tests that need a GPU, so skips are allowed.
    if problem := untrustworthy(returncode, run, allow_skips=True):
        return Verdict(CampaignStatus.INVALID, problem)
    failed = sorted(case.nodeid for case in run.cases if case.outcome == Outcome.FAILED)
    if failed:
        return Verdict(CampaignStatus.COUNTEREXAMPLE, ", ".join(failed))
    return Verdict(CampaignStatus.PASSED)
