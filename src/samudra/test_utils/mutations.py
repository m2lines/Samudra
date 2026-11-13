# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded named-fault catalogs over real property tests, not arbitrary code mutation."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from hypothesis import settings
from hypothesis.errors import HypothesisException

from .artifacts import git_revision, write_environment
from .catalogs import load_catalog
from .profiles import MUTATION_SETTINGS, profile_evidence
from .reports import Outcome, read_run

EXPECTED_PROFILE = profile_evidence(MUTATION_SETTINGS)
EXECUTION: dict[str, Any] = {}
# Errors that mean the mutant itself is broken code, not that the oracle saw a fault.
BROKEN_MUTANT = (NameError, ImportError, SyntaxError)


class Verdict(NamedTuple):
    status: str
    reason: str = ""


def pytest_sessionstart(session):
    catalog = load_catalog(Path(os.environ["SAMUDRA_MUTATION_CATALOG"]))
    subject = importlib.import_module(catalog.module)
    if (
        subject.__file__ is None
        or Path(subject.__file__).resolve() != catalog.target.resolve()
    ):
        raise RuntimeError(f"Unexpected import location: {subject.__file__}")
    EXECUTION["target_sha256"] = hashlib.sha256(catalog.target.read_bytes()).hexdigest()
    EXECUTION["mutant_sha256"] = None
    if source := os.environ.get("SAMUDRA_MUTATION_SOURCE"):
        contents = Path(source).read_bytes()
        exec(compile(contents, source, "exec"), subject.__dict__)
        EXECUTION["mutant_sha256"] = hashlib.sha256(contents).hexdigest()


def bind_profile(functions, profile):
    from hypothesis import is_hypothesis_test  # type: ignore[attr-defined]

    for function in set(functions):
        if not is_hypothesis_test(function):
            raise ValueError(f"Expected a Hypothesis test: {function.__name__}")
        settings(profile)(function)  # Public API rejects conflicting explicit settings.


def pytest_collection_finish(session):
    active = settings()
    bind_profile((item.obj for item in session.items), active)
    EXECUTION["profile"] = profile_evidence(active)
    EXECUTION["nodeids"] = sorted(item.nodeid for item in session.items)
    Path(os.environ["SAMUDRA_MUTATION_EVIDENCE"]).write_text(
        json.dumps(EXECUTION, indent=2) + "\n"
    )


def failure_kind(error: BaseException) -> str:
    if isinstance(error, HypothesisException):
        return (
            "hypothesis"  # Health check, Unsatisfiable, Flaky: not an oracle verdict.
        )
    if isinstance(error, BROKEN_MUTANT):
        return "broken_mutant"
    return "oracle"


def leaf_failures(error: BaseException) -> list[tuple[str, str]]:
    """Each underlying exception's qualified name and kind, flattening groups."""
    if isinstance(error, BaseExceptionGroup) and not isinstance(
        error, HypothesisException
    ):
        return [leaf for inner in error.exceptions for leaf in leaf_failures(inner)]
    kind = type(error)
    return [(f"{kind.__module__}.{kind.__qualname__}", failure_kind(error))]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if call.when == "call" and call.excinfo is not None:
        report.user_properties.append(
            ("failures", json.dumps(leaf_failures(call.excinfo.value)))
        )
    return report


def outcome(returncode, report: Path, expected) -> Verdict:
    """A mutant is killed only by an oracle failure in the intended, complete run.

    The oracle must have failed: an assertion, `pytest.raises` not raising, or the
    subject crashing. Hypothesis's own errors (health checks, Unsatisfiable, Flaky)
    and errors showing the mutant is broken code (NameError, ImportError,
    SyntaxError) make the run invalid instead.
    """
    try:
        executed = json.loads((report.parent / "execution.json").read_text())
        if executed != expected:
            drift = sorted(
                key
                for key in expected.keys() | executed.keys()
                if expected.get(key) != executed.get(key)
            )
            return Verdict("invalid", f"execution evidence differs: {drift}")
        run = read_run(report)
    except (OSError, ValueError, KeyError) as error:
        return Verdict(
            "invalid", f"unreadable evidence: {type(error).__name__}: {error}"
        )
    if not run.complete:
        return Verdict("invalid", "pytest session did not finish")
    if run.exitcode != returncode:
        return Verdict("invalid", f"exit {returncode} but report says {run.exitcode}")
    if run.collection_errors:
        return Verdict("invalid", f"collection errors: {run.collection_errors}")
    ran = sorted(case.nodeid for case in run.cases)
    if not ran or ran != expected["nodeids"]:
        return Verdict("invalid", f"ran {ran}, expected {expected['nodeids']}")
    for case in run.cases:
        if case.outcome in (Outcome.ERROR, Outcome.SKIPPED):
            return Verdict("invalid", f"{case.nodeid}: {case.outcome}")
    failed = [case for case in run.cases if case.outcome == Outcome.FAILED]
    if returncode == 0 and not failed:
        return Verdict("survived")
    if returncode != 1 or not failed:
        return Verdict("invalid", f"exit {returncode} with {len(failed)} failed tests")
    for case in failed:
        if "failures" not in case.properties:
            return Verdict("invalid", f"{case.nodeid}: no recorded failure type")
        for name, kind in json.loads(case.properties["failures"]):
            if kind != "oracle":
                return Verdict("invalid", f"{case.nodeid}: {kind} error {name}")
    names = sorted(
        {name for case in failed for name, _ in json.loads(case.properties["failures"])}
    )
    return Verdict("killed", ", ".join(names))


def remaining_timeout(deadline, per_case, now):
    return min(per_case, max(0, deadline - now))


def run_case(
    directory, nodes, seed, timeout, mutant_path=None, *, target_sha256, catalog_path
):
    if timeout <= 0:
        return {
            "status": "budget_exhausted",
            "reason": "the shared budget ran out before this case started",
            "seconds": 0,
            "seed": seed,
            "returncode": None,
        }
    catalog = load_catalog(catalog_path)
    directory.mkdir(parents=True)
    env = dict(
        os.environ,
        SAMUDRA_HYPOTHESIS_PROFILE="mutation",
        PYTHONUNBUFFERED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        SAMUDRA_MUTATION_CATALOG=str(catalog_path),
        SAMUDRA_MUTATION_EVIDENCE=str(directory / "execution.json"),
    )
    for key in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTHONPATH",
        "PYTHONOPTIMIZE",
        "SAMUDRA_MUTATION_SOURCE",
    ):
        env.pop(key, None)
    if mutant_path:
        env["SAMUDRA_MUTATION_SOURCE"] = str(mutant_path)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *nodes,
        "-q",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=1",
        f"--hypothesis-seed={seed}",
        "--hypothesis-show-statistics",
        f"--report-log={directory / 'pytest.jsonl'}",
        "-p",
        "hypothesispytest",
        "-p",
        "pytest_reportlog",
        "-p",
        "samudra.test_utils.mutations",
    ]
    expected = {
        "profile": EXPECTED_PROFILE,
        "target_sha256": target_sha256,
        "mutant_sha256": hashlib.sha256(mutant_path.read_bytes()).hexdigest()
        if mutant_path
        else None,
        "nodeids": sorted(
            Path(node.split("::", 1)[0]).relative_to(catalog.project).as_posix()
            + "::"
            + node.split("::", 1)[1]
            for node in nodes
        ),
    }
    (directory / "expected-execution.json").write_text(
        json.dumps(expected, indent=2) + "\n"
    )
    started = time.monotonic()
    with (directory / "pytest.log").open("w") as log:
        try:
            process = subprocess.run(
                command,
                cwd=catalog.project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            verdict = outcome(process.returncode, directory / "pytest.jsonl", expected)
            code = process.returncode
        except subprocess.TimeoutExpired:
            verdict, code = Verdict("timeout", f"exceeded {timeout:.0f}s"), None
    return {
        "status": verdict.status,
        "reason": verdict.reason,
        "returncode": code,
        "seconds": round(time.monotonic() - started, 3),
        "seed": seed,
        "command": command,
    }


def run_catalog(path, output, seeds, timeout, deadline):
    catalog = load_catalog(path)
    output.mkdir()
    original = catalog.target.read_bytes()
    for mutant in catalog.mutations:
        if original.count(mutant.before.encode()) != 1:
            raise ValueError(
                f"{path.name}/{mutant.name}: expected exactly one source anchor; review after refactoring"
            )
    target_hash = hashlib.sha256(original).hexdigest()
    (output / "catalog.py").write_bytes(path.read_bytes())
    (output / "target.py").write_bytes(original)
    (output / "properties.py").write_bytes(catalog.properties.read_bytes())
    results = {}

    def record(name, result):
        results[name] = result
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        reason = f": {result['reason']}" if result["reason"] else ""
        print(
            f"{path.stem}/{name}: {result['status']} ({result['seconds']}s){reason}",
            flush=True,
        )

    nodes = list(
        dict.fromkeys(
            f"{catalog.properties}::{mutant.property}" for mutant in catalog.mutations
        )
    )
    # One process per seed runs every assigned property on unmodified code. Each
    # property keeps its own `timeout`, so the batch gets one per property.
    for seed in seeds:
        result = run_case(
            output / f"baseline-{seed}",
            nodes,
            seed,
            remaining_timeout(deadline, timeout * len(nodes), time.monotonic()),
            target_sha256=target_hash,
            catalog_path=path,
        )
        result["status"] = {"survived": "passed", "killed": "failed"}.get(
            result["status"], result["status"]
        )
        record(f"baseline-{seed}", result)
        if result["status"] != "passed":
            return False
    for mutant in catalog.mutations:
        directory = output / mutant.name
        directory.mkdir()
        source = directory / "mutated_source.py"
        source.write_bytes(
            original.replace(mutant.before.encode(), mutant.after.encode(), 1)
        )
        compile(source.read_bytes(), str(source), "exec")
        for seed in seeds:
            record(
                f"{mutant.name}-{seed}",
                run_case(
                    directory / str(seed),
                    [f"{catalog.properties}::{mutant.property}"],
                    seed,
                    remaining_timeout(deadline, timeout, time.monotonic()),
                    source,
                    target_sha256=target_hash,
                    catalog_path=path,
                ),
            )
    return all(result["status"] in ("passed", "killed") for result in results.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalogs",
        nargs="+",
        type=Path,
        help="Flat *_mutations.py files or their directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 95920261105])
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--budget", type=float, default=600)
    args = parser.parse_args()
    if not (0 < args.timeout < float("inf") and 0 < args.budget < float("inf")):
        parser.error("timeout and budget must be positive and finite")
    paths = sorted(
        {
            p.resolve()
            for candidate in args.catalogs
            for p in (
                candidate.glob("*_mutations.py") if candidate.is_dir() else [candidate]
            )
        }
    )
    if not paths or any(
        not path.is_file() or not path.name.endswith("_mutations.py") for path in paths
    ):
        parser.error("Expected at least one existing *_mutations.py catalog")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.budget
    (output / "replay.json").write_text(
        json.dumps(
            {
                "revision": git_revision(),
                "catalogs": list(map(str, paths)),
                "seeds": args.seeds,
                "timeout": args.timeout,
                "budget": args.budget,
                "profile": EXPECTED_PROFILE,
                "python": sys.version,
            },
            indent=2,
        )
        + "\n"
    )
    write_environment(output / "environment.txt")
    shared = output / "engine"
    shared.mkdir()
    for module in Path(__file__).parent.glob("*.py"):
        (shared / module.name).write_bytes(module.read_bytes())
    summaries = {}
    try:
        for path in paths:
            summaries[path.stem] = (
                "passed"
                if run_catalog(
                    path, output / path.stem, args.seeds, args.timeout, deadline
                )
                else "failed"
            )
    finally:
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        (output / "summary.md").write_text(
            "# Named mutation campaigns\n\n"
            + "\n".join(
                f"- `{name}`: **{status}** (details in `{name}/results.json`)"
                for name, status in summaries.items()
            )
            + "\n\nA failing baseline prevents mutation claims. Download the run artifact for JSONL events and replay inputs.\n"
        )
    raise SystemExit(
        0 if all(status == "passed" for status in summaries.values()) else 1
    )


if __name__ == "__main__":
    main()
