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

## Control seed 16 recovery

Job 19457054 reached epoch 2 and failed after 6m07s with
`CUDNN_STATUS_EXECUTION_FAILED_CUDART` during backward on rank 0. No more
specific CUDA cause was reported; this is not established as a data-loader or
hardware fault. Eight ranks, effective batch 32, and online W&B were confirmed.
The interrupted W&B segment is `ocean_emulators/default/vfs6v3ev`.

Retry **19460455** resumes from the saved end-of-epoch-1 `ckpt.pt` into a new
`current-1deg-control-seed16-torch-retry1` output directory. The original output
and logs are retained. Source, seed, data, loader and model settings are unchanged;
this is a recovery attempt, not a confirmed fix. The first attempt consumed
about 0.82 allocated GPU-hours. Report this resumed run separately in provenance;
no bitwise equivalence to uninterrupted training is claimed.

## Shared epoch-2 failure investigation

All four original jobs failed on **gr103**, on rank 0 during backward at the
fourth batch of epoch 2, after completing epoch 1 and its validation cycle.
Control 16 reported the cuDNN error above; the other three reported an illegal
CUDA memory access. This repeatability suggests a shared problem, but does not
establish whether it is software, CUDA libraries, or hardware.

| Original job | Elapsed | W&B segment |
|---|---|---|
| 19457054 | 6m07s | vfs6v3ev |
| 19457056 | 4m42s | ic27x334 |
| 19457059 | 4m33s | kt36r3b4 |
| 19457061 | 4m39s | 3m11xsef |

The four failed allocations total about 2.67 GPU-hours. Retry 19460455 was
canceled after 23 seconds during distributed initialization because it also
landed on gr103; it did not produce a resumed training result.

**19461486** resumes the same control-16 epoch-1 checkpoint on **gr101**, using
`resume-control16-other-node.sh`. Source, container, Rust loader (including
CUDA prefetch), NCCL settings, batch size, and model parameters are unchanged.
It writes `current-1deg-control-seed16-torch-retry2`. Torch rejects node exclusion;
the explicit gr101 request passed `sbatch --test-only`. The job was queued when
recorded here. Startup and completion callbacks are installed and verified.
The other three retries are deferred until this host comparison is inspected.
No final-epoch or observation scores exist for the additional seeds yet.
