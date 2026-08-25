#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Publish Samudra experiment run directories to an rclone archive.

The CLI is deliberately dry-run by default.  ``--apply`` is required before it
will invoke rclone, and publication only copies files: it never deletes source
or destination objects.
"""

# The Torch host may provide Python 3.6 even though Samudra itself requires a
# newer interpreter inside its container.
# ruff: noqa: UP006, UP007, UP017, UP035, UP045

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_ARCHIVE_BASE = "nyu-osn:m2lines-pubs/Samudra/experiments"
STATUS_FILENAME = "archive-status.json"
VALID_STATES = ("running", "requeued", "completed", "failed")

# A run directory is public after publication.  Keep credentials, local W&B
# state, and files that may still be in the middle of an atomic write out of it.
EXCLUDE_PATTERNS = (
    "**/.env",
    "**/.env.*",
    "**/.netrc",
    "**/*credentials*",
    "**/*.key",
    "**/*.pem",
    "**/*.part",
    "**/*.tmp",
    "**/.nfs*",
    "**/wandb/**",
)


class ArchiveError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _exclusion_args() -> List[str]:
    args = []  # type: List[str]
    for pattern in EXCLUDE_PATTERNS:
        args.extend(("--exclude", pattern))
    return args


def _validate_archive_segment(value: str, label: str) -> str:
    if (
        value in ("", ".", "..")
        or value != value.strip()
        or "/" in value
        or "\\" in value
    ):
        raise ArchiveError(f"invalid {label}: {value!r}")
    return value


def archive_destination(archive_base: str, owner: str, run_name: str) -> str:
    if not archive_base.strip():
        raise ArchiveError("archive base must not be empty")
    owner = _validate_archive_segment(owner, "archive owner")
    run_name = _validate_archive_segment(run_name, "run name")
    return f"{archive_base.rstrip('/')}/{owner}/{run_name}"


def copy_command(
    run_dir: Path, destination: str, rclone_bin: str = "rclone"
) -> List[str]:
    return [
        rclone_bin,
        "copy",
        str(run_dir),
        destination,
        "--stats=30s",
        *_exclusion_args(),
    ]


def check_command(
    run_dir: Path, destination: str, rclone_bin: str = "rclone"
) -> List[str]:
    return [
        rclone_bin,
        "check",
        str(run_dir),
        destination,
        "--one-way",
        "--size-only",
        *_exclusion_args(),
    ]


def validate_run_dir(run_dir: Path) -> Path:
    if run_dir.is_symlink():
        raise ArchiveError(f"refusing symlinked run directory: {run_dir}")
    if not run_dir.is_dir():
        raise ArchiveError(f"run directory does not exist: {run_dir}")
    resolved = run_dir.resolve()
    if resolved == Path("/"):
        raise ArchiveError("refusing to publish filesystem root")
    return resolved


def publish_run(
    run_dir: Path,
    archive_base: str,
    owner: str,
    apply: bool,
    verify: bool = True,
    rclone_bin: str = "rclone",
) -> Dict[str, object]:
    run_dir = validate_run_dir(run_dir)
    destination = archive_destination(archive_base, owner, run_dir.name)
    commands = [copy_command(run_dir, destination, rclone_bin)]
    if verify:
        commands.append(check_command(run_dir, destination, rclone_bin))

    for command in commands:
        print(_display_command(command), flush=True)
        if apply:
            subprocess.run(command, check=True)

    return {
        "source": str(run_dir),
        "destination": destination,
        "owner": owner,
        "status": "verified" if apply and verify else "copied" if apply else "planned",
        "verified": bool(apply and verify),
    }


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def write_status(
    run_dir: Path, owner: str, state: str, exit_code: Optional[int] = None
) -> Path:
    run_dir = validate_run_dir(run_dir)
    owner = _validate_archive_segment(owner, "archive owner")
    if state not in VALID_STATES:
        raise ArchiveError(f"invalid archive state: {state!r}")
    status_path = run_dir / STATUS_FILENAME
    previous = {}  # type: Dict[str, object]
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ArchiveError(
                f"could not read existing status file: {status_path}"
            ) from error
        if not isinstance(loaded, dict):
            raise ArchiveError(f"existing status file is not an object: {status_path}")
        previous = loaded
        previous_owner = previous.get("owner")
        if previous_owner is not None and previous_owner != owner:
            raise ArchiveError(
                f"archive owner changed from {previous_owner!r} to {owner!r}"
            )

    now = utc_now()
    status = {
        "schema_version": 1,
        "owner": owner,
        "run_name": run_dir.name,
        "state": state,
        "started_at": previous.get("started_at", now),
        "updated_at": now,
    }  # type: Dict[str, object]
    for environment_name, field_name in (
        ("SLURM_JOB_ID", "slurm_job_id"),
        ("SLURM_RESTART_COUNT", "slurm_restart_count"),
    ):
        value = os.environ.get(environment_name)
        if value is not None:
            status[field_name] = value
    if exit_code is not None:
        status["exit_code"] = exit_code
    if state in ("completed", "failed"):
        status["finished_at"] = now
    _write_json_atomic(status_path, status)
    return status_path


def watch_run(
    run_dir: Path,
    archive_base: str,
    owner: str,
    interval_seconds: float,
    apply: bool,
    rclone_bin: str = "rclone",
    stop_event: Optional[threading.Event] = None,
    max_cycles: Optional[int] = None,
) -> int:
    if interval_seconds <= 0:
        raise ArchiveError("archive interval must be positive")
    owner = _validate_archive_segment(owner, "archive owner")
    validate_run_dir(run_dir)
    stop_event = stop_event or threading.Event()
    failures = 0
    cycles = 0
    while not stop_event.wait(interval_seconds):
        try:
            publish_run(
                run_dir,
                archive_base,
                owner,
                apply=apply,
                verify=False,
                rclone_bin=rclone_bin,
            )
        except subprocess.CalledProcessError as error:
            failures += 1
            print(
                f"Periodic archive copy failed with exit code {error.returncode}: "
                f"{_display_command(error.cmd)}",
                file=sys.stderr,
                flush=True,
            )
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
    return 1 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-base",
        default=DEFAULT_ARCHIVE_BASE,
        help="rclone destination containing owner and experiment directories",
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="archive namespace for the user or team that owns the experiment",
    )
    parser.add_argument("--rclone-bin", default="rclone", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")

    publish_parser = subparsers.add_parser("publish", help="publish one run directory")
    publish_parser.add_argument("run_dir", type=Path)
    publish_parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the copy; without this flag only print the commands",
    )
    publish_parser.add_argument(
        "--no-verify", action="store_true", help="skip rclone's size verification"
    )

    watch_parser = subparsers.add_parser(
        "watch", help="periodically publish one active run until signaled"
    )
    watch_parser.add_argument("run_dir", type=Path)
    watch_parser.add_argument("--interval", type=float, default=900.0)
    watch_parser.add_argument(
        "--apply",
        action="store_true",
        help="perform periodic copies; without this flag only print the plans",
    )

    status_parser = subparsers.add_parser(
        "status", help="write public lifecycle metadata into a run directory"
    )
    status_parser.add_argument("run_dir", type=Path)
    status_parser.add_argument("state", choices=VALID_STATES)
    status_parser.add_argument("--exit-code", type=int)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    try:
        if args.command == "publish":
            publish_run(
                args.run_dir,
                args.archive_base,
                args.owner,
                apply=args.apply,
                verify=not args.no_verify,
                rclone_bin=args.rclone_bin,
            )
            return 0
        if args.command == "status":
            write_status(args.run_dir, args.owner, args.state, args.exit_code)
            return 0
        if args.command == "watch":
            stop_event = threading.Event()

            def request_stop(signum: int, frame: object) -> None:
                del signum, frame
                stop_event.set()

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            return watch_run(
                args.run_dir,
                args.archive_base,
                args.owner,
                interval_seconds=args.interval,
                apply=args.apply,
                rclone_bin=args.rclone_bin,
                stop_event=stop_event,
            )
    except ArchiveError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
