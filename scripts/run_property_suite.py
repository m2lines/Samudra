# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one independently installed property suite with common replay artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tests", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New artifact directory"
    )
    parser.add_argument("--dependency-file", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.tests.is_dir() or not args.dependency_file.is_file():
        parser.error("tests must be a directory and dependency-file must exist")
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
        str(args.tests),
        "-v",
        "--tb=short",
        "--disable-warnings",
        f"--hypothesis-seed={seed}",
        "--hypothesis-show-statistics",
        f"--junitxml={output / 'junit.xml'}",
        *extra,
    ]
    metadata = {
        "schema_version": 1,
        "suite": args.suite,
        "profile": "soak",
        "examples": args.examples,
        "seed": seed,
        "command": command,
        "cwd": str(Path.cwd()),
        "timeout_seconds": args.timeout,
        "python": sys.version,
        "platform": platform.platform(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dependency_file": str(args.dependency_file),
        "dependency_sha256": hashlib.sha256(
            args.dependency_file.read_bytes()
        ).hexdigest(),
        "lane": env.get("VERIFICATION_LANE", "pinned"),
        "string_storage": env.get("SAMUDRA_STRING_STORAGE"),
    }
    (output / "replay.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "dependency-input.txt").write_bytes(args.dependency_file.read_bytes())
    (output / "environment.txt").write_text(
        "\n".join(
            sorted(
                f"{package.metadata['Name']}=={package.version}"
                for package in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
    print(
        f"Running {args.suite}, seed {seed}; log: {output / 'pytest.log'}", flush=True
    )
    with (output / "pytest.log").open("w") as log:
        try:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            code = result.returncode
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            code, status = 124, "timeout"
    if code == 0:
        try:
            cases = list(ET.parse(output / "junit.xml").iter("testcase"))
            executed = [case for case in cases if case.find("skipped") is None]
            if not executed:
                code, status = 5, "no_tests"
            elif any(
                case.find("error") is not None or case.find("failure") is not None
                for case in executed
            ):
                code, status = 2, "invalid"
        except (OSError, ET.ParseError):
            code, status = 2, "invalid"
    (output / "result.json").write_text(
        json.dumps({"suite": args.suite, "status": status, "returncode": code}) + "\n"
    )
    print((output / "pytest.log").read_text(errors="replace"), end="", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
