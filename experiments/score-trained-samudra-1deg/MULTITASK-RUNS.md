<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Structured multitask execution record

The [approved F/M comparison](MULTITASK.md) uses the user-confirmed 50/25/12.5/12.5 mixture.
This is an experiment/results record, not intended to merge.

## October 12: Torch qualification

Qualification source `79cb5003903bb6b800bae16d562de8dbd484948a`, immutable snapshot under
`/scratch/jr7309/runs/score-multitask-1deg-20261012/sources/`.
Production source is `480bcc7df886a808ab324c3410a055f4a8fa73b1`. Model, training, loss and common
configuration code are byte-identical to qualification; production adds the GPU-visibility guard described below.

| Stage | Torch job | Resources | Last verified state |
| --- | --- | --- | --- |
| All-task qualification | 19648755 | 1 H100, 6 CPUs, 64 GiB RAM | Completed, exit 0; 2:01 on gh001 |
| Completion notification | 19648789 | CPU-only, 1 CPU, 512 MiB | Completed; callback delivered |
| F: forecast-only | 19649331 | 2 H100s, 16 CPUs, 192 GiB RAM, 36-hour ceiling | Submitted; eligible after 13:34 Eastern |
| M: structured multitask | 19649333 | Same as F | Submitted; eligible after 13:34 Eastern |
| Production notifications | 19649371 / 19649375 | CPU-only, 1 CPU, 512 MiB each | `afterany` dependencies on F / M |

Campaign root: `/scratch/jr7309/runs/score-multitask-1deg-20261012`.
Resolved inputs: `configs/{qualification,train}-{forecast,multitask}.yaml`.
Outputs: `outputs/qualification-multitask/`, then `outputs/train-{forecast,multitask}/`.
Logs: `logs/<job>.{out,err}`; allocation records: `allocations/`.

The qualification explicitly backpropagated all four tasks on real data, ran three optimizer updates,
and passed one-step/completion and 360-day ensemble validation. [Raw qualification results](results/multitask-qualification-19648755.json)
record **42.42 GB peak allocated GPU memory** and 18.74 seconds for the six training minibatches, including
initial summary overhead. Job cost was 0.0336 allocated H100 GPU-hours. The largest recorded host RSS was
11.29 GiB. These short-run timings do not establish full epoch throughput. The randomly sampled updates
did not happen to choose a gap task, but the explicit pre-update backward probe verified it separately.
The published JSON retains every metric and represents its verified Git fingerprint as a commit URL.
The qualification-only checkpoint does not count as a training epoch or initialize production weights.
Production retains batch 4/rank, accumulation 4, effective batch 32 and 70 epochs. Both arms use online W&B.

The qualification callback `f4d1a330-e42b-4131-88f4-6bfc25456181` delivered and was retired after verification.
Authentication passed on the login and compute nodes. Callback credentials stay in the private remote
directory and are not committed. Production receives separate callbacks and hourly reminders.

F callback: `ccd71a7c-9edf-4134-86e9-e3e9ccd08321`; M callback: `6e8c72d6-35e1-4d8c-a7ed-75ae5b950b43`.
Both authenticated successfully from Torch and have hourly reminders. Production routing was verified:
account `torch_pr_347_lzanna`, partition `h100`, constraint `h100`, two GPUs per job,
`Comment=preemption=yes;requeue=true`, `Requeue=1`. Neither run is claimed to have started merely from
submission or an advisory start estimate. W&B URLs and first-epoch logging will be verified at bring-up.

Torch held job 19648662 could not be released (`scontrol release` returned an unspecified error twice).
It and dependent notification 19648699 were cancelled before allocation; their callback was revoked.
The replacement uses a two-minute delayed start to attach monitoring. Beta held qualification 212521
was also cancelled without allocation after the user requested available Torch capacity.

## Environment and data verification

Container: `/scratch/jr7309/.apptainer-images/sd-seeds-base-0b5320248.sif`.
Its `uv.lock` and `pyproject.toml` match the source byte-for-byte; both are compared again at launch.
The installed x86 Rust extension imports successfully. PyTorch reports
`2.12.0a0+0291f960b6.nv26.04.48445190`. Qualification confirmed CUDA and real Rust-loader operation.
The production harness additionally forwards Slurm's GPU visibility through the clean container environment
and asserts that the visible GPU count equals the requested process count; training/model code is unchanged.

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
