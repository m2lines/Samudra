# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded historical-fault mutations, executable as a runner and a pytest plugin.

The production file is never edited. Each mutant is compiled into the imported
module in a fresh pytest process. Exact, unique anchors fail loudly when a
refactor requires reviewing a mutant. These are named regression mutations,
not a claim of a comprehensive mutation score for the whole project.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2]
TARGET = DATA / "ocean_preprocessing" / "virtualize_om4.py"
SUITE = DATA / "tests" / "property" / "test_virtual_archive.py"
EXPECTED_PROFILE = {
    "max_examples": 40,
    "phases": ["generate"],
    "database_disabled": True,
    "deadline_disabled": True,
}
EXECUTION = {}


@dataclass(frozen=True)
class Mutation:
    name: str
    before: str
    after: str
    property: str


MUTATIONS = (
    Mutation(
        "scalar_chunk_key",
        'refs[f"{name}/0"] = refs.pop(f"{name}/")',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    Mutation(
        "storage_fill_masks_zeros",
        'metadata["fill_value"] = None',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    Mutation(
        "unescaped_registry",
        "store = LocalURIStore(store)",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "double_decode_local_key",
        "return self.store.head(unquote(path))",
        "return self.store.head(unquote(unquote(path)))",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "encoded_async_key",
        "return await self.store.get_range_async(unquote(path), **options)",
        "return await self.store.get_range_async(path, **options)",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "encoded_reference_path",
        "reference[0] = unquote(reference[0])",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "replace_competing_directory",
        "rename_directory_exclusive(directory, output)",
        "os.rename(directory, output)",
        "test_publication_schedule_never_clobbers_another_writer",
    ),
    Mutation(
        "ignore_selected_source_changes",
        "if source.snapshot() != snapshot:",
        "if False:",
        "test_source_mutations_abort_publication[modify]",
    ),
    Mutation(
        "missing_chunk_shape_diagnostic",
        "if current_chunks != chunks:",
        "if False:",
        "test_native_file_partitions_round_trip_or_report_layout_error",
    ),
    Mutation(
        "missing_partial_chunk_diagnostic",
        "index < len(datasets) - 1\n                and dataset.sizes[concat_dim] % current_chunks[axis]",
        "False",
        "test_nonfinal_partial_chunks_have_actionable_errors",
    ),
    Mutation(
        "accept_unsafe_sparse_fill",
        "if sum(1 for _ in array.manifest.iter_refs()) < grid_size:",
        "if False:",
        "test_sparse_native_chunks_require_declared_cf_fill",
    ),
)


def pytest_sessionstart(session):
    """Activate only when this module is explicitly loaded as a mutation plugin."""
    from ocean_preprocessing import virtualize_om4 as subject

    if Path(subject.__file__).resolve() != TARGET:
        raise RuntimeError(
            f"Mutation target import came from {subject.__file__}, expected {TARGET}"
        )
    EXECUTION["target_sha256"] = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    EXECUTION["mutant_sha256"] = None
    if mutant_source := os.environ.get("SAMUDRA_MUTATION_SOURCE"):
        mutant_path = Path(mutant_source)
        contents = mutant_path.read_bytes()
        exec(compile(contents, str(mutant_path), "exec"), subject.__dict__)
        EXECUTION["mutant_sha256"] = hashlib.sha256(contents).hexdigest()


def pytest_collection_finish(session):
    """Record effective settings after conftest loading, and bind JUnit to node IDs."""
    from hypothesis import settings

    active = settings.default
    bind_profile((item.obj for item in session.items), active)
    EXECUTION["profile"] = {
        "max_examples": active.max_examples,
        "phases": [phase.name for phase in active.phases],
        "database_disabled": active.database is None,
        "deadline_disabled": active.deadline is None,
    }
    EXECUTION["nodeids"] = sorted(item.nodeid for item in session.items)
    for item in session.items:
        item.user_properties.append(("mutation_nodeid", item.nodeid))
    Path(os.environ["SAMUDRA_MUTATION_EVIDENCE"]).write_text(
        json.dumps(EXECUTION, indent=2) + "\n"
    )


def bind_profile(functions, profile):
    """Bind generation-only settings through the public API; reject test overrides."""
    from hypothesis import is_hypothesis_test, settings

    for function in set(functions):
        if not is_hypothesis_test(function):
            raise ValueError(
                f"Mutation target {function.__name__} must be a Hypothesis test"
            )
        # Hypothesis rejects a second explicit @settings decorator. Conflicting
        # per-test overrides must fail the baseline, rather than bypass this gate.
        settings(profile)(function)


def outcome(returncode, junit, expected):
    """Only a completed test failure counts as a kill; setup errors are not kills."""
    evidence = junit.parent / "execution.json"
    if not junit.exists() or not evidence.exists():
        return "invalid"
    try:
        if json.loads(evidence.read_text()) != expected:
            return "invalid"
        cases = list(ET.parse(junit).iter("testcase"))
    except (json.JSONDecodeError, ET.ParseError, UnicodeError):
        return "invalid"
    identities = [
        case.findall("./properties/property[@name='mutation_nodeid']") for case in cases
    ]
    if not cases or any(len(identity) != 1 for identity in identities):
        return "invalid"
    nodeids = [identity[0].get("value") for identity in identities]
    if any(not isinstance(nodeid, str) for nodeid in nodeids):
        return "invalid"
    if sorted(nodeids) != expected["nodeids"] or any(
        case.find("error") is not None or case.find("skipped") is not None
        for case in cases
    ):
        return "invalid"
    failures = sum(case.find("failure") is not None for case in cases)
    if returncode == 1 and failures:
        return "killed"
    if returncode == 0 and not failures:
        return "survived"
    return "invalid"


def remaining_timeout(deadline, per_case, now):
    """Share a wall-clock budget without allowing a case to overrun it."""
    return min(per_case, max(0, deadline - now))


def run_case(directory, nodes, seed, timeout, mutant_path=None, *, target_sha256):
    if timeout <= 0:
        return {
            "status": "budget_exhausted",
            "seconds": 0,
            "seed": seed,
            "returncode": None,
            "command": [],
        }
    directory.mkdir(parents=True)
    env = dict(os.environ, SAMUDRA_HYPOTHESIS_PROFILE="mutation", PYTHONUNBUFFERED="1")
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONOPTIMIZE"):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(DATA)])
    env["SAMUDRA_MUTATION_EVIDENCE"] = str(directory / "execution.json")
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
        f"--junitxml={directory / 'junit.xml'}",
        "-p",
        "hypothesispytest",
        "-p",
        "virtual_archive_mutants",
    ]
    if mutant_path is not None:
        env["SAMUDRA_MUTATION_SOURCE"] = str(mutant_path)
    else:
        env.pop("SAMUDRA_MUTATION_SOURCE", None)
    expected = {
        "nodeids": sorted(
            str(Path(node.split("::", 1)[0]).relative_to(DATA).as_posix())
            + "::"
            + node.split("::", 1)[1]
            for node in nodes
        ),
        "profile": EXPECTED_PROFILE,
        "target_sha256": target_sha256,
        "mutant_sha256": hashlib.sha256(mutant_path.read_bytes()).hexdigest()
        if mutant_path
        else None,
    }
    (directory / "expected-execution.json").write_text(
        json.dumps(expected, indent=2) + "\n"
    )
    started = time.monotonic()
    with (directory / "pytest.log").open("w") as log:
        try:
            process = subprocess.run(
                command,
                cwd=DATA,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            status = outcome(process.returncode, directory / "junit.xml", expected)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            status, returncode = "timeout", None
    return {
        "status": status,
        "returncode": returncode,
        "seconds": round(time.monotonic() - started, 3),
        "seed": seed,
        "command": command,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New directory for logs and replay inputs",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 95920261105])
    parser.add_argument(
        "--budget",
        type=float,
        default=600,
        help="Shared seconds for the entire gate, including all seeds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Seconds allowed per pytest subprocess",
    )
    args = parser.parse_args()
    if not (0 < args.budget < float("inf") and 0 < args.timeout < float("inf")):
        parser.error("budget and timeout must be positive")
    deadline = time.monotonic() + args.budget
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = TARGET.read_text()
    # Validate all anchors before starting; stale mutants must never silently skip.
    for mutant in MUTATIONS:
        if original.count(mutant.before) != 1:
            raise RuntimeError(
                f"{mutant.name}: expected exactly one mutation anchor; review the changed production code"
            )
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DATA, text=True
        ).strip(),
        "target_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "profile": "mutation",
        "expected_profile": EXPECTED_PROFILE,
        "seeds": args.seeds,
        "timeout_seconds": args.timeout,
        "budget_seconds": args.budget,
        "planned_cases": len(args.seeds) * (len(MUTATIONS) + 1),
    }
    # Preserve the small replay surface even when run from an uncommitted tree.
    replay_sources = [
        TARGET,
        SUITE,
        SUITE.parent / "conftest.py",
        Path(__file__).resolve(),
        DATA.parent / "scripts" / "property_testing.py",
    ]
    metadata["source_sha256"] = {}
    for path in replay_sources:
        relative = path.relative_to(DATA.parent)
        contents = path.read_bytes()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        metadata["source_sha256"][str(relative)] = hashlib.sha256(contents).hexdigest()
    (output / "replay.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "environment.txt").write_text(
        "\n".join(
            sorted(
                f"{package.metadata['Name']}=={package.version}"
                for package in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
    results = {}

    def record(name, result):
        results[name] = result
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {result['status']} ({result['seconds']}s)", flush=True)

    nodes = list(dict.fromkeys(f"{SUITE}::{mutant.property}" for mutant in MUTATIONS))
    for seed in args.seeds:
        result = run_case(
            output / f"baseline-{seed}",
            nodes,
            seed,
            remaining_timeout(deadline, args.timeout, time.monotonic()),
            target_sha256=metadata["target_sha256"],
        )
        # A failing unmodified baseline invalidates the experiment, not a mutant kill.
        result["status"] = {"survived": "passed", "killed": "failed"}.get(
            result["status"], result["status"]
        )
        record(f"baseline-{seed}", result)
        if result["status"] != "passed":
            raise SystemExit(
                "Unmodified baseline failed; inspect artifacts before attributing failures to mutations"
            )
    for mutant in MUTATIONS:
        directory = output / mutant.name
        directory.mkdir()
        source = directory / "mutated_source.py"
        source.write_text(original.replace(mutant.before, mutant.after, 1))
        compile(source.read_text(), str(source), "exec")
        for seed in args.seeds:
            result = run_case(
                directory / str(seed),
                [f"{SUITE}::{mutant.property}"],
                seed,
                remaining_timeout(deadline, args.timeout, time.monotonic()),
                source,
                target_sha256=metadata["target_sha256"],
            )
            record(f"{mutant.name}-{seed}", result)
    if any(result["status"] not in ("passed", "killed") for result in results.values()):
        raise SystemExit(
            "Mutation gate failed: survived, invalid, or timed-out runs require investigation"
        )


if __name__ == "__main__":
    main()
