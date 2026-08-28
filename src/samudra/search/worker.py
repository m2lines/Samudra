# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Internal worker entry points used by compute executors."""

import argparse
import os
from pathlib import Path

from samudra.search import SearchConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("task")
    task.add_argument("config", type=Path)
    task.add_argument("state", type=Path)
    task.add_argument("rung", type=int)
    task.add_argument("task", type=int, nargs="?")
    task.add_argument("--anchor", action="store_true")
    probe = commands.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("state", type=Path)
    probe.add_argument("rung", type=int)
    probe.add_argument("task", type=int, nargs="?")
    advance = commands.add_parser("advance")
    advance.add_argument("config", type=Path)
    advance.add_argument("state", type=Path)
    advance.add_argument("rung", type=int)
    release = commands.add_parser("release-probe")
    release.add_argument("config", type=Path)
    release.add_argument("state", type=Path)
    release.add_argument("rung", type=int)
    args = parser.parse_args()

    config = SearchConfig.from_yaml_and_cli([str(args.config)])
    search = config.build()
    if search.state_path != args.state:
        raise ValueError(f"State path mismatch: {search.state_path} != {args.state}")
    if args.command in {"task", "probe"}:
        task_index = args.task
        if task_index is None:
            task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        search.train_task(
            args.rung,
            task_index,
            anchor=args.anchor if args.command == "task" else False,
            probe=args.command == "probe",
        )
    elif args.command == "advance":
        search.advance(args.rung)
    else:
        search.release_probe(args.rung)


if __name__ == "__main__":
    main()
