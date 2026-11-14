# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mutation analysis over hand-written fault catalogs: do the properties catch them?

Each trial runs a project's properties in a fresh pytest process under one
condition. A *control* trial runs every detector on unmodified code; a *mutant*
trial plants one fault in memory and runs that fault's detectors. The controls for
all seeds run first, and any dirty control stops the gate, so a bug that already
exists is never credited as a kill.

Every trial is capped at `--timeout` seconds per property it runs. Mutant trials
also share a `--budget` per catalog, which starts once the controls finish, so slow
controls can't starve the mutants.

This module is also the pytest plugin each trial loads (`-p samudra_verification.gate`).
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pytest
from hypothesis.errors import HypothesisException

from .artifacts import git_revision, write_environment
from .evidence import Outcome, read_run
from .faults import SUFFIX, Catalog, Fault, load_catalog, module_path
from .profiles import MUTATION_SETTINGS, Profile
from .records import from_json, to_json
from .verdicts import TrialStatus, Verdict, untrustworthy

EXPECTED_PROFILE = Profile.of(MUTATION_SETTINGS)
# Errors that mean the mutant itself is broken code, not that the oracle saw a fault.
BROKEN_MUTANT = (NameError, ImportError, SyntaxError)


@dataclass(frozen=True)
class Control:
    label = "control"


@dataclass(frozen=True)
class Mutant:
    fault: Fault
    source: Path  # The fault's module with the substitution applied.

    @property
    def label(self) -> str:
        return self.fault.name


@dataclass(frozen=True)
class Trial:
    condition: Control | Mutant
    seed: int
    nodeids: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.condition.label}-{self.seed}"


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str

    @classmethod
    def of(cls, path: Path, contents: bytes) -> "SourceFile":
        return cls(str(path), hashlib.sha256(contents).hexdigest())


@dataclass(frozen=True)
class PlantedMutant:
    module: str
    sha256: str


@dataclass(frozen=True)
class Execution:
    """What a trial process ran. The gate writes the expected one before each
    trial, the trial process writes what it actually ran, and any difference makes
    the trial invalid.
    """

    targets: dict[str, SourceFile]  # Each fault module, as the trial imported it.
    mutant: PlantedMutant | None
    profile: Profile
    nodeids: tuple[str, ...]

    def drift(self, other: "Execution") -> list[str]:
        return [
            field.name
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        ]


# --- Inside each trial's pytest process ----------------------------------------

# What `pytest_sessionstart` found, completed by `pytest_collection_finish`.
STARTED: dict[str, Any] = {}


def pytest_sessionstart(session):
    catalog = load_catalog(Path(os.environ["SAMUDRA_FAULT_CATALOG"]))
    targets = {}
    for module in catalog.modules:
        subject = importlib.import_module(module)
        if subject.__file__ is None:
            raise RuntimeError(f"{module} has no source file")
        path = Path(subject.__file__).resolve()
        targets[module] = SourceFile.of(path, path.read_bytes())
    mutant = None
    if source := os.environ.get("SAMUDRA_MUTANT_SOURCE"):
        module = os.environ["SAMUDRA_MUTANT_MODULE"]
        contents = Path(source).read_bytes()
        exec(compile(contents, source, "exec"), sys.modules[module].__dict__)
        mutant = PlantedMutant(module, hashlib.sha256(contents).hexdigest())
    STARTED.update(targets=targets, mutant=mutant)


def bind_profile(functions, profile):
    from hypothesis import is_hypothesis_test, settings  # type: ignore[attr-defined]

    for function in set(functions):
        if not is_hypothesis_test(function):
            raise ValueError(f"Expected a Hypothesis test: {function.__name__}")
        settings(profile)(function)  # Public API rejects conflicting explicit settings.


def pytest_collection_finish(session):
    from hypothesis import settings

    active = settings()
    bind_profile((item.obj for item in session.items), active)
    executed = Execution(
        **STARTED,
        profile=Profile.of(active),
        nodeids=tuple(sorted(item.nodeid for item in session.items)),
    )
    Path(os.environ["SAMUDRA_TRIAL_EVIDENCE"]).write_text(to_json(executed))


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


# --- In the gate process ---------------------------------------------------------


def log_tail(path: Path, lines: int = 3) -> str:
    """The last few nonblank lines of a trial's log, for its verdict."""
    try:
        text = path.read_text(errors="replace")
    except OSError as error:
        return f"no log ({error})"
    return " | ".join([line for line in text.splitlines() if line.strip()][-lines:])


def interpret(
    trial: Trial, returncode: int, report: Path, expected: Execution
) -> Verdict:
    """Classify a finished trial.

    A detection must be an oracle failure: an assertion, `pytest.raises` not
    raising, or the subject crashing. Hypothesis's own errors (health checks,
    Unsatisfiable, Flaky) and errors showing the mutant is broken code (NameError,
    ImportError, SyntaxError) make the trial invalid instead.
    """
    if not (report.parent / "execution.json").exists():
        return Verdict(
            TrialStatus.INVALID,
            f"pytest stopped before collecting the trial (exit code {returncode}): "
            f"{log_tail(report.parent / 'pytest.log')}",
        )
    try:
        executed = from_json(Execution, (report.parent / "execution.json").read_text())
        if drift := expected.drift(executed):
            return Verdict(TrialStatus.INVALID, f"execution evidence differs: {drift}")
        run = read_run(report)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return Verdict(
            TrialStatus.INVALID, f"unreadable evidence: {type(error).__name__}: {error}"
        )
    if problem := untrustworthy(
        returncode, run, allow_skips=False, expected_nodeids=trial.nodeids
    ):
        return Verdict(TrialStatus.INVALID, problem)
    control = isinstance(trial.condition, Control)
    failed = [case for case in run.cases if case.outcome == Outcome.FAILED]
    if not failed:
        ran = sorted(case.nodeid for case in run.cases)
        if ran != sorted(trial.nodeids):
            return Verdict(TrialStatus.INVALID, f"only ran {ran}")
        return Verdict(TrialStatus.CLEAN if control else TrialStatus.SURVIVED)
    causes = set()
    for case in failed:
        if "failures" not in case.properties:
            return Verdict(TrialStatus.INVALID, f"{case.nodeid}: no recorded failure")
        for name, kind in json.loads(case.properties["failures"]):
            if kind != "oracle":
                return Verdict(
                    TrialStatus.INVALID, f"{case.nodeid}: {kind} error {name}"
                )
            causes.add(f"{case.nodeid}: {name}")
    status = TrialStatus.DIRTY if control else TrialStatus.KILLED
    return Verdict(status, "; ".join(sorted(causes)))


def remaining_timeout(deadline, per_trial, now):
    return min(per_trial, max(0, deadline - now))


def run_trial(
    trial: Trial,
    directory: Path,
    timeout,
    *,
    catalog_path,
    targets: dict[str, SourceFile],
):
    """Run one trial in a fresh pytest process and record its verdict."""
    base = {"trial": trial.label, "seed": trial.seed}
    if timeout <= 0:
        reason = "the mutant budget ran out before this trial started"
        return {**base, "status": TrialStatus.BUDGET_EXHAUSTED, "reason": reason}
    catalog = load_catalog(catalog_path)
    directory.mkdir(parents=True)
    env = dict(
        os.environ,
        SAMUDRA_HYPOTHESIS_PROFILE="mutation",
        PYTHONUNBUFFERED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        SAMUDRA_FAULT_CATALOG=str(catalog_path),
        SAMUDRA_TRIAL_EVIDENCE=str(directory / "execution.json"),
    )
    for key in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTHONPATH",
        "PYTHONOPTIMIZE",
        "SAMUDRA_MUTANT_SOURCE",
        "SAMUDRA_MUTANT_MODULE",
    ):
        env.pop(key, None)
    mutant = None
    if isinstance(trial.condition, Mutant):
        env["SAMUDRA_MUTANT_SOURCE"] = str(trial.condition.source)
        env["SAMUDRA_MUTANT_MODULE"] = trial.condition.fault.module
        mutant = PlantedMutant(
            trial.condition.fault.module,
            hashlib.sha256(trial.condition.source.read_bytes()).hexdigest(),
        )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *trial.nodeids,
        "-q",
        # The project's ini addopts may name options from plugins that trials
        # don't autoload (the root project's jaxtyping), so the gate sets them all.
        "-o",
        "addopts=",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=1",
        f"--hypothesis-seed={trial.seed}",
        "--hypothesis-show-statistics",
        f"--report-log={directory / 'pytest.jsonl'}",
        "-p",
        "hypothesispytest",
        "-p",
        "pytest_reportlog",
        "-p",
        "samudra_verification.gate",
    ]
    expected = Execution(
        targets, mutant, EXPECTED_PROFILE, tuple(sorted(trial.nodeids))
    )
    (directory / "expected-execution.json").write_text(to_json(expected))
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
        except subprocess.TimeoutExpired:
            verdict, code = (
                Verdict(TrialStatus.TIMEOUT, f"exceeded {timeout:.0f}s"),
                None,
            )
        else:
            code = process.returncode
            verdict = interpret(trial, code, directory / "pytest.jsonl", expected)
    return {
        **base,
        "status": verdict.status,
        "reason": verdict.reason,
        "returncode": code,
        "seconds": round(time.monotonic() - started, 3),
        "command": command,
    }


def target_sources(catalog: Catalog) -> dict[str, tuple[Path, bytes]]:
    """Each fault module's file and bytes, resolved as the trial processes will.

    Trials run from the catalog's project root, so modules there take precedence.
    The execution evidence check catches any trial that resolves them differently.
    """
    search_path = [str(catalog.project), *sys.path]
    paths = {module: module_path(module, search_path) for module in catalog.modules}
    return {module: (path, path.read_bytes()) for module, path in paths.items()}


def run_catalog(path: Path, output: Path, seeds, timeout, budget) -> bool:
    catalog = load_catalog(path)
    output.mkdir()
    sources = target_sources(catalog)
    for fault in catalog.faults:
        if sources[fault.module][1].count(fault.before.encode()) != 1:
            raise ValueError(
                f"{path.name}/{fault.name}: expected exactly one source anchor in "
                f"{fault.module}; review after refactoring"
            )
    targets = {
        module: SourceFile.of(file, contents)
        for module, (file, contents) in sources.items()
    }
    (output / "catalog.py").write_bytes(path.read_bytes())
    for module, (_, contents) in sources.items():
        (output / f"target-{module}.py").write_bytes(contents)
    results: dict[str, dict] = {}

    def record(result):
        results[result["trial"]] = result
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        reason = f": {result['reason']}" if result["reason"] else ""
        print(
            f"{path.stem}/{result['trial']}: {result['status']} "
            f"({result.get('seconds', 0)}s){reason}",
            flush=True,
        )

    every_detector = tuple(
        sorted({node for fault in catalog.faults for node in fault.detectors})
    )
    for seed in seeds:
        trial = Trial(Control(), seed, every_detector)
        record(
            run_trial(
                trial,
                output / trial.label,
                timeout * len(trial.nodeids),
                catalog_path=path,
                targets=targets,
            )
        )
        if results[trial.label]["status"] != TrialStatus.CLEAN:
            return False
    deadline = time.monotonic() + budget
    for fault in catalog.faults:
        directory = output / fault.name
        directory.mkdir()
        source = directory / "mutant.py"
        source.write_bytes(
            sources[fault.module][1].replace(
                fault.before.encode(), fault.after.encode(), 1
            )
        )
        compile(source.read_bytes(), str(source), "exec")
        for seed in seeds:
            trial = Trial(Mutant(fault, source), seed, fault.detectors)
            record(
                run_trial(
                    trial,
                    directory / str(seed),
                    remaining_timeout(
                        deadline, timeout * len(trial.nodeids), time.monotonic()
                    ),
                    catalog_path=path,
                    targets=targets,
                )
            )
    return all(
        result["status"] in (TrialStatus.CLEAN, TrialStatus.KILLED)
        for result in results.values()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalogs",
        nargs="+",
        type=Path,
        help=f"Flat *{SUFFIX} files or their directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 95920261105])
    parser.add_argument(
        "--timeout", type=float, default=60, help="Seconds per property in a trial"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=600,
        help="Seconds shared by each catalog's mutant trials, after its controls",
    )
    args = parser.parse_args()
    if not (0 < args.timeout < float("inf") and 0 < args.budget < float("inf")):
        parser.error("timeout and budget must be positive and finite")
    paths = sorted(
        {
            p.resolve()
            for candidate in args.catalogs
            for p in (
                candidate.glob(f"*{SUFFIX}") if candidate.is_dir() else [candidate]
            )
        }
    )
    if not paths or any(
        not path.is_file() or not path.name.endswith(SUFFIX) for path in paths
    ):
        parser.error(f"Expected at least one existing *{SUFFIX} catalog")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "replay.json").write_text(
        json.dumps(
            {
                "revision": git_revision(),
                "catalogs": list(map(str, paths)),
                "seeds": args.seeds,
                "timeout": args.timeout,
                "budget": args.budget,
                "profile": asdict(EXPECTED_PROFILE),
                "python": sys.version,
            },
            indent=2,
        )
        + "\n"
    )
    write_environment(output / "environment.txt")
    summaries = {}
    try:
        for path in paths:
            passed = run_catalog(
                path, output / path.stem, args.seeds, args.timeout, args.budget
            )
            summaries[path.stem] = "passed" if passed else "failed"
    finally:
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        (output / "summary.md").write_text(
            "# Mutation analysis\n\n"
            + "\n".join(
                f"- `{name}`: **{status}** (details in `{name}/results.json`)"
                for name, status in summaries.items()
            )
            + "\n\nA dirty control stops a catalog before any mutant runs, so an "
            "existing bug is never credited as a kill. The artifact holds each "
            "trial's pytest events and replay inputs.\n"
        )
    raise SystemExit(
        0 if all(status == "passed" for status in summaries.values()) else 1
    )


if __name__ == "__main__":
    main()
