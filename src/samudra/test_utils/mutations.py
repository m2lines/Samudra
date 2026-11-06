# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded named-fault catalogs over real property tests, not arbitrary code mutation."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .catalogs import load_catalog
from .reports import Outcome, read_run

EXPECTED_PROFILE = {
    "max_examples": 40,
    "phases": ["generate"],
    "database_disabled": True,
    "deadline_disabled": True,
}
EXECUTION: dict[str, Any] = {}


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
    from hypothesis import is_hypothesis_test, settings  # type: ignore[attr-defined]

    for function in set(functions):
        if not is_hypothesis_test(function):
            raise ValueError(f"Expected a Hypothesis test: {function.__name__}")
        settings(profile)(function)  # Public API rejects conflicting explicit settings.


def pytest_collection_finish(session):
    from hypothesis import settings

    active = settings()
    bind_profile((item.obj for item in session.items), active)
    EXECUTION["profile"] = {
        "max_examples": active.max_examples,
        "phases": [phase.name for phase in active.phases],
        "database_disabled": active.database is None,
        "deadline_disabled": active.deadline is None,
    }
    EXECUTION["nodeids"] = sorted(item.nodeid for item in session.items)
    Path(os.environ["SAMUDRA_MUTATION_EVIDENCE"]).write_text(
        json.dumps(EXECUTION, indent=2) + "\n"
    )


def outcome(returncode, report: Path, expected):
    """Only a clean call failure in the intended completed run is a killed mutant."""
    try:
        if json.loads((report.parent / "execution.json").read_text()) != expected:
            return "invalid"
        run = read_run(report)
    except (OSError, ValueError, KeyError):
        return "invalid"
    if not run.complete or run.exitcode != returncode or run.collection_errors:
        return "invalid"
    if (
        not run.cases
        or sorted(case.nodeid for case in run.cases) != expected["nodeids"]
    ):
        return "invalid"
    if any(case.outcome in (Outcome.ERROR, Outcome.SKIPPED) for case in run.cases):
        return "invalid"
    if returncode == 1 and any(case.outcome == Outcome.FAILED for case in run.cases):
        return "killed"
    return (
        "survived"
        if returncode == 0 and all(case.outcome == Outcome.PASSED for case in run.cases)
        else "invalid"
    )


def remaining_timeout(deadline, per_case, now):
    return min(per_case, max(0, deadline - now))


def run_case(
    directory, nodes, seed, timeout, mutant_path=None, *, target_sha256, catalog_path
):
    if timeout <= 0:
        return {
            "status": "budget_exhausted",
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
            status, code = (
                outcome(process.returncode, directory / "pytest.jsonl", expected),
                process.returncode,
            )
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
    return {
        "status": status,
        "returncode": code,
        "seconds": round(time.monotonic() - started, 3),
        "seed": seed,
        "command": command,
    }


def run_catalog(path, output, seeds, timeout, deadline):
    catalog = load_catalog(path)
    output.mkdir()
    original = catalog.target.read_text()
    for mutant in catalog.mutations:
        if original.count(mutant.before) != 1:
            raise ValueError(
                f"{path.name}/{mutant.name}: expected exactly one source anchor; review after refactoring"
            )
    target_hash = hashlib.sha256(original.encode()).hexdigest()
    (output / "catalog.py").write_bytes(path.read_bytes())
    (output / "target.py").write_text(original)
    (output / "properties.py").write_bytes(catalog.properties.read_bytes())
    results = {}

    def record(name, result):
        results[name] = result
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"{path.stem}/{name}: {result['status']} ({result['seconds']}s)", flush=True
        )

    nodes = list(
        dict.fromkeys(
            f"{catalog.properties}::{mutant.property}" for mutant in catalog.mutations
        )
    )
    for seed in seeds:
        for index, node in enumerate(nodes):
            result = run_case(
                output / f"baseline-{seed}-{index}",
                [node],
                seed,
                remaining_timeout(deadline, timeout, time.monotonic()),
                target_sha256=target_hash,
                catalog_path=path,
            )
            result["status"] = {"survived": "passed", "killed": "failed"}.get(
                result["status"], result["status"]
            )
            record(f"baseline-{seed}-{index}", result)
            if result["status"] != "passed":
                return False
    for mutant in catalog.mutations:
        directory = output / mutant.name
        directory.mkdir()
        source = directory / "mutated_source.py"
        source.write_text(original.replace(mutant.before, mutant.after, 1))
        compile(source.read_text(), str(source), "exec")
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
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
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
    (output / "environment.txt").write_text(
        "\n".join(
            sorted(
                f"{package.metadata['Name']}=={package.version}"
                for package in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
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
