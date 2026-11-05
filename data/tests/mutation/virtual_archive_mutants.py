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
    mutant_path = Path(os.environ["SAMUDRA_MUTATION_SOURCE"])
    from ocean_preprocessing import virtualize_om4 as subject

    if Path(subject.__file__).resolve() != TARGET:
        raise RuntimeError(
            f"Mutation target import came from {subject.__file__}, expected {TARGET}"
        )
    exec(compile(mutant_path.read_text(), str(mutant_path), "exec"), subject.__dict__)


def outcome(returncode, junit, expected_count):
    """Only a completed test failure counts as a kill; setup errors are not kills."""
    if not junit.exists():
        return "invalid"
    cases = list(ET.parse(junit).iter("testcase"))
    if len(cases) != expected_count or any(
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


def run_case(directory, nodes, seed, timeout, mutant_path=None):
    directory.mkdir(parents=True)
    env = dict(os.environ, SAMUDRA_HYPOTHESIS_PROFILE="mutation", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), str(DATA), env.get("PYTHONPATH", "")]
    )
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
    ]
    if mutant_path is not None:
        env["SAMUDRA_MUTATION_SOURCE"] = str(mutant_path)
        command.extend(["-p", "virtual_archive_mutants"])
    else:
        env.pop("SAMUDRA_MUTATION_SOURCE", None)
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
            status = outcome(process.returncode, directory / "junit.xml", len(nodes))
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
        "--timeout",
        type=float,
        default=60,
        help="Seconds allowed per pytest subprocess",
    )
    args = parser.parse_args()
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
        "max_examples": 40,
        "phases": ["generate"],
        "seeds": args.seeds,
        "timeout_seconds": args.timeout,
    }
    # Preserve the small replay surface even when run from an uncommitted tree.
    replay_sources = [
        TARGET,
        SUITE,
        SUITE.parent / "conftest.py",
        Path(__file__).resolve(),
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
        result = run_case(output / f"baseline-{seed}", nodes, seed, args.timeout)
        # A failing unmodified baseline invalidates the experiment, not a mutant kill.
        if result["status"] == "survived":
            result["status"] = "passed"
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
                args.timeout,
                source,
            )
            record(f"{mutant.name}-{seed}", result)
    if any(result["status"] not in ("passed", "killed") for result in results.values()):
        raise SystemExit(
            "Mutation gate failed: survived, invalid, or timed-out runs require investigation"
        )


if __name__ == "__main__":
    main()
