# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one property campaign and write the artifacts needed to replay it.

A campaign is one pytest session over property tests with a fixed seed and example
budget. Its directory holds `replay.json` (the `CampaignSpec`), the dependency file
it used, the installed packages, pytest's log and event stream, and the verdict.
"""

import argparse
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path

from .artifacts import git_revision, write_environment
from .evidence import read_run
from .spec import CampaignSpec, Lane
from .verdicts import CampaignStatus, Verdict, campaign_verdict

EXIT_CODES = {
    CampaignStatus.PASSED: 0,
    CampaignStatus.COUNTEREXAMPLE: 1,
    CampaignStatus.INVALID: 2,
    CampaignStatus.TIMEOUT: 124,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="e.g. samudra or data")
    parser.add_argument("--lane", required=True, type=Lane, choices=list(Lane))
    parser.add_argument("--tests", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--output", required=True, type=Path, help="New artifact directory"
    )
    parser.add_argument("--dependency-file", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        not all(path.exists() for path in args.tests)
        or not args.dependency_file.is_file()
    ):
        parser.error("test paths and dependency-file must exist")
    if args.examples < 1 or not 0 < args.timeout < float("inf"):
        parser.error("examples and timeout must be positive and finite")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    env = dict(
        os.environ,
        SAMUDRA_HYPOTHESIS_PROFILE="soak",
        SAMUDRA_HYPOTHESIS_EXAMPLES=str(args.examples),
        PYTHONUNBUFFERED="1",
    )
    extra = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    command = [
        sys.executable,
        "-m",
        "pytest",
        *map(str, args.tests),
        "-v",
        "--tb=short",
        "--disable-warnings",
        f"--hypothesis-seed={seed}",
        "--hypothesis-show-statistics",
        f"--report-log={output / 'pytest.jsonl'}",
        *extra,
    ]
    spec = CampaignSpec(
        project=args.project,
        lane=args.lane,
        string_storage=env.get("SAMUDRA_STRING_STORAGE"),
        profile="soak",
        examples=args.examples,
        seed=seed,
        timeout_seconds=args.timeout,
        command=command,
        cwd=str(Path.cwd()),
        revision=git_revision(),
        python=sys.version,
        platform=platform.platform(),
        dependency_file=str(args.dependency_file),
        dependency_sha256=hashlib.sha256(args.dependency_file.read_bytes()).hexdigest(),
    )
    spec.write(output)
    (output / "dependency-input.txt").write_bytes(args.dependency_file.read_bytes())
    write_environment(output / "environment.txt")
    print(
        f"Running {spec.label}, seed {seed}; log: {output / 'pytest.log'}", flush=True
    )
    with (output / "pytest.log").open("w") as log:
        try:
            process = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            verdict = Verdict(CampaignStatus.TIMEOUT, f"exceeded {args.timeout:.0f}s")
        else:
            try:
                verdict = campaign_verdict(
                    process.returncode, read_run(output / "pytest.jsonl")
                )
            except (OSError, ValueError, KeyError) as error:
                verdict = Verdict(
                    CampaignStatus.INVALID,
                    f"unreadable evidence: {type(error).__name__}: {error}",
                )
    (output / "result.json").write_text(
        json.dumps(
            {"campaign": spec.label, "status": verdict.status, "reason": verdict.reason}
        )
        + "\n"
    )
    (output / "summary.md").write_text(
        f"### {spec.label} property campaign: {verdict.status}\n\n"
        + (f"{verdict.reason}\n\n" if verdict.reason else "")
        + f"Seed: `{seed}`; requested examples per property: {args.examples}. "
        "Download the run artifact below for replay.json, pytest.jsonl, "
        "the environment and the full log.\n"
    )
    print((output / "pytest.log").read_text(errors="replace"), end="", flush=True)
    raise SystemExit(EXIT_CODES[CampaignStatus(verdict.status)])


if __name__ == "__main__":
    main()
