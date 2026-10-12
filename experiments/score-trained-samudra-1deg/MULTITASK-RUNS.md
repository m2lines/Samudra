<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Structured multitask execution record

The [approved F/M comparison](MULTITASK.md) uses the user-confirmed 50/25/12.5/12.5 mixture.
This is an experiment/results record, not intended to merge.

## October 12: Torch qualification

Source `79cb5003903bb6b800bae16d562de8dbd484948a`, immutable snapshot under
`/scratch/jr7309/runs/score-multitask-1deg-20261012/sources/`.

| Stage | Torch job | Resources | Last verified state |
| --- | --- | --- | --- |
| All-task qualification | 19648755 | 1 H100, 6 CPUs, 64 GiB RAM, 40-minute ceiling | Submitted; delayed eligibility to attach callback |
| Completion notification | 19648789 | CPU-only, 1 CPU, 512 MiB | Pending `afterany:19648755` |
| F and M production | Not yet submitted | Plan: 2 H100s each, 16 CPUs, 192 GiB RAM | Await qualification |

Campaign root: `/scratch/jr7309/runs/score-multitask-1deg-20261012`.
Resolved inputs: `configs/{qualification,train}-{forecast,multitask}.yaml`.
Outputs: `outputs/qualification-multitask/`, then `outputs/train-{forecast,multitask}/`.
Logs: `logs/<job>.{out,err}`; allocation records: `allocations/`.

The qualification explicitly backpropagates all four tasks on real data, runs three optimizer updates,
checks one-step/completion and 360-day ensemble validation, and writes `qualification.json` plus a
qualification-only checkpoint. It does not count as a training epoch or initialize production weights.
Production retains batch 4/rank, accumulation 4, effective batch 32 and 70 epochs. Both arms use online W&B.

The completion callback registration is `f4d1a330-e42b-4131-88f4-6bfc25456181`, with an hourly fallback
reminder to this conversation. Login-node authentication passed; the batch script repeats the check on
the compute node. Callback credentials stay in the private remote directory and are not committed.

Torch held job 19648662 could not be released (`scontrol release` returned an unspecified error twice).
It and dependent notification 19648699 were cancelled before allocation; their callback was revoked.
The replacement uses a two-minute delayed start to attach monitoring. Beta held qualification 212521
was also cancelled without allocation after the user requested available Torch capacity.

## Environment and data verification

Container: `/scratch/jr7309/.apptainer-images/sd-seeds-base-0b5320248.sif`.
Its `uv.lock` and `pyproject.toml` match the source byte-for-byte; both are compared again at launch.
The installed x86 Rust extension imports successfully. PyTorch reports
`2.12.0a0+0291f960b6.nv26.04.48445190`. Qualification must confirm CUDA and real loader operation.

Torch data: `/scratch/jr7309/data/om4-v2026-09/om4_onedeg`.
Beta reference: `/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg`.
[Fingerprint utility](fingerprint_data.py) checks metadata, static arrays including masks/coordinates,
all normalization chunks, and first/middle/last time chunks of each ocean array. Both hosts returned
identical hashes: 504 files / 67,652,886 bytes for `OM4.zarr`, and 303 files / 66,587 bytes for each
normalization store. This sampled comparison is not a full reread of every ocean field. The machine-readable
record is `data-sample-fingerprint.json` in the Torch campaign root. Training additionally saves the
resolved normalization, masks, coordinates, variable order, time splits and upstream attributes.

Local validation: 21 model/numerical tests passed, including initial C parity, mask leakage, requested-only
gradients and checkpointed backward equivalence. Scoped pre-commit and shell syntax checks passed.
`evaluate_score.py --arm F|M` loads the new conditioning architecture for the same held-out forecast
protocol; final completion and continuous-rollout evaluation remain to be qualified before use.
