# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Launch one torchrun agent per node for an allocated search trial."""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.worker_args:
        raise ValueError("worker arguments are required")

    node_list = os.environ["SLURM_STEP_NODELIST"]
    hosts = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not hosts:
        raise RuntimeError(f"Slurm returned no hosts for {node_list!r}")
    node_rank = int(os.environ["SLURM_NODEID"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={args.nnodes}",
            f"--nproc-per-node={args.nproc_per_node}",
            f"--node-rank={node_rank}",
            f"--master-addr={hosts[0]}",
            f"--master-port={args.master_port}",
            "--module",
            "samudra.search.worker",
            *args.worker_args,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
