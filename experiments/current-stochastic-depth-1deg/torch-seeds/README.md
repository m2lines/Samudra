<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Additional paired seeds on Torch

Seeds 16 and 17 each compare control with constant stochastic depth 0.1. The
training source remains the original merged experiment source; only seed,
platform, per-GPU batch size, names, and runtime paths change.

Each run uses one eight-RTX6000 host, 128 CPUs, batch size 2 per GPU and gradient
accumulation 2: effective batch 32, matching the initial four-GPU batch-4 pair.
The September 2026 1° OM4 release is staged separately from Torch's older v3
copy and verified before submission. Rust loading and tau_hfds are preserved.

Use the x86 PhysicsNeMo image for original main revision 0b53202488d18724d5cf7d306230fb72c2779528,
with a checksum-verified code overlay built from this experiment branch. The
overlay builder must verify its dependency files against the container.

Final epoch-70 and EMA rollouts and OM4 metrics run after training. Observation
metrics are a separate CPU follow-up, with the original wet masks restored as
in the first pair. No visualization is requested for this wave. Results belong
on PR #915; hardware/world-size differences must be retained in provenance when
comparing against seed 15 on Beta.

## Submitted jobs

| Seed | Control | SD 0.1 |
|---|---:|---:|
| 16 | 19457054 | 19457056 |
| 17 | 19457059 | 19457061 |

Submitted 2026-10-08 with 48-hour limits and 512 GiB RAM per job. Torch rejects
`--exclusive` and `--mem=0`; the accepted request names all eight GPUs and all
128 CPUs. Startup was queued when recorded here, not yet verified training.

Preparation job 19454930 completed successfully, verified identical dependency
files for the source overlay, and imported the Rust extension. The DTN copy
and full `rclone check --download` verified 385,213 matching files, zero
differences, for the September 2026 release. All six runtime YAML files were
hash-checked against the committed copies before submission.

Runtime root: `/scratch/jr7309/runs/current-sd-seeds16-17`; logs are
`train-JOBID.out` / `.err`, outputs are under `output/RUN_NAME`. Runtime configs
are in `configs/` under that root. Source overlay is pinned to
`c3df51b5e0a8e497201377b27423eaf91d712c3e`; subsequent experiment commits change
runtime paths, launch policy, and reporting only. The initial queued launch
uses NCCL_P2P_DISABLE=1 on all four runs.

Persistent startup and per-job completion follow-ups check scheduler state,
then verify real logs and artifacts before reporting results. Observation
scoring remains a separate CPU stage; no visualization is scheduled.
