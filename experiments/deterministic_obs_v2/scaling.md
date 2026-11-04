<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Rust loading and distributed qualification

The observation trainer now supports the shared Samudra Rust I/O runtime,
one prefetched update per rank, and distributed training. CPU tests and the
one-GPU exact prefetch/resume comparison pass. GPU performance and utilization
remain unqualified. This extends the
[initial single-GPU qualification](qualification.md). No scientific screening
or 16k training run has been submitted.

Code `f3aab652e1f090e2b1385d03de92ba4e04c2a313` ran in a read-only overlay in Torch
job `21797129` on one RTX PRO 6000 Blackwell GPU. At 22:10:49 ET on 2026-11-03,
both synchronous controls matched the prefetched run's saved state exactly.
The run was stopped after that comparison because OM4 loading remained slow.
It consumed 2,629 GPU-seconds (0.7303 GPU-hours). Its dependent eight-GPU job
`21797131` was cancelled before allocation. The one-GPU request had eight CPU
cores and 96 GiB host memory; the eight-GPU request had 64 cores and 192 GiB.
Both used a 90-minute limit, preemption-only routing, requeue, and account
`torch_pr_347_lzanna`. Their
[configs](configs/scaling-v2/) use the large initializer and 50/50 OM4/observation
main training, with global batch eight in both cases. Main training alternates entire
observation and OM4 updates; every cooldown update uses observations.

The [saved records](receipts/scaling-v2/flat-om4-1gpu.jsonl) contain eight updates
per source in each comparison. Median observation time was 10.95 s with
synchronous loading and 8.89 s with prefetch; median exposed data wait fell from
2.20 s to 0.001 s. OM4 medians were 41.53 s, 45.47 s and 108.57 s in the
synchronous, prefetched and repeated synchronous runs. Loading conditions changed
too much to infer a speedup from these controls. The prefetched OM4 median still
included 40.63 s of foreground data wait.

The original utilization recorder rejected PyTorch's unprefixed GPU UUID and
wrote only `No devices were found`. A corrected recorder captured the end of the
run, mostly synchronous controls. It does not cover the required steady window.
Code `710e49603ced7c669c5e504d0b8f6ac626090438` adds the required `GPU-` prefix,
checks the selected devices before archive/model setup, and fails if telemetry
exits during training. It also avoids duplicate pinned/GPU copies of tensors
shared by forecast and reconstruction.

CPU job `21803580` is preparing a lossless packed OM4 cache with that code.
It includes all 2,812 eligible native frames and all 77 physical channels, with
one chunk per frame instead of 77 separate variable chunks. Every packed frame
must match the source through independent Python and Rust decoders. The reader
checks the pinned proof, original metadata, complete native-index list, interval
bounds, channel order and cache hashes. Normalization, masks, cohort and calendar
rules stay in the existing loader. Preparation and the replacement GPU
qualification are pending; no packed-cache performance result is claimed.

The first scaling attempt used code `aa51d5bfa4e95291333ec73f2314db5ceeef4953`.
Job `21794028` completed one observation update, then failed while pinning an
expanded OM4 validity mask whose tensor storage overlapped. Job `21794026` was
cancelled before allocation to avoid repeating that failure on eight GPUs.
The fix materializes contiguous mask values before pinning. A CPU regression
reproduces the failure and passes with the fix; an early CUDA staging check is
included in the replacement qualification. The failed attempt consumed
324 GPU-seconds (0.09 GPU-hours). Its [receipt](receipts/scaling-v1/attempt.json)
and [first-update metrics](receipts/scaling-v1/failed-1gpu.jsonl) are retained.
That single warmup update does not support a steady-throughput estimate.

The Rust backend supports both the existing flat OM4 Zarr store and the packed
OM4 cache. Observation SST, SSH/geostrophic velocities, and eight ERA5
fields are packed into a float32 interval cache read by the existing compact
Rust reader. The cached geostrophic velocities are unused by this task; surface
inputs and observed surface targets use SST and SSH. Monthly IAP profile targets retain their existing file reader;
native-interval ERA5 forcing for OM4 retains its existing memory-mapped adapter.
Each process shares one eight-thread Rust read pool across OM4 and observations.
One Python producer thread prepares the next update, including replay targets,
auxiliary initializer inputs, and observation coverage used for OM4.
Pinned CPU buffers and a CUDA copy stream overlap device transfer with compute.
There are no PyTorch DataLoader worker processes.

CPU job `21792849` prepared and read back all 8,006 five-day observation intervals
in 11m49s. Interval starts have a one-day stride from 1993-01-26 through
2014-12-27, so replay retains its original alignment across calendar months.
Replay selects five-day bins anchored at its original forecast origin and weights
bins by overlap with the target calendar month. Every interval matched the
original Python aggregation exactly through Rust.
The cache occupies 16,270,941,520 bytes in 8,011 files at
`/scratch/jr7309/deterministic-obs-v2/native-observations-v1`.
The SHA256 of its `COMPLETE.json` proof file is
`72abd11ccb5cfdea532a5f024ad07c48719daec1021ac0b88c5e7e74c9065fee`.
The preparation script executed with SHA256
`324b13199c605d145155300220f3d652a2ee07063565bf25ccfb18781db9b7d0`;
the executed copy remains at
`/scratch/jr7309/deterministic-obs-v2/frozen-data/prepare_observation_native.py`.
The committed copy adds formatting and type-checker annotations.
Training verifies the pinned proof, source inventory and cached file hashes.

The global batch remains eight monthly windows per optimizer update, normally
four fresh and four replayed. Eight ranks process one window each; one rank
accumulates eight. Rank zero owns the global sampler and source-separated replay
pools, broadcasts example metadata, distributes cached states, and gathers the
new states in global example order. Pool capacity stays two entries per source
and lead-month bucket, independent of rank count.

Planning update N+1 before computing N makes states generated during N eligible
starting at N+2. This changes replay timing from the original sequential loop.
The same rule applies to one and eight GPUs. Empty-pool startup can supply more
than four fresh examples. The pending plan, selected cached states, and per-rank
RNG states are checkpointed. Resume requires the same world size. A world-size
change would need a separately defined restart policy.

Focused CPU tests cover exact native/Python fresh and shifted replay inputs,
prefetch on/off checkpoint equality, two-rank DDP global batching and pool
capacity, exact distributed resume, and uncooled fork continuation. The broader
observation/model suite passed 85 tests before the added fork test, which also
passes. Relevant pre-commit checks pass.

CPU job `21795066` checked real data with code
`aa51d5bfa4e95291333ec73f2314db5ceeef4953` and completed
with exit `0:0` in 2m45s. All eight native/Python comparisons passed with zero
tolerance and matching NaN locations: observation origins May 1993 and January
2001, plus OM4 window indices 0 and 1 under the [training cohort rules](README.md).
Each was checked at month offset zero (fresh) and one (the following month's
targets, with five-day bins anchored at the same origin). Inputs, targets, masks, forcing and
calendar weights matched. The [receipt](receipts/scaling-v1/real-input-parity.json)
lists every tensor checked. This establishes real-data parity for those cases;
the one-GPU comparison above separately establishes GPU prefetch parity.
OM4 indices 0 and 1 select the first two eligible months in chronological order;
these cases do not establish parity at calendar gaps.

Each GPU qualification uses four warmup updates, then replays the same saved
state through 16 updates with synchronous loading, asynchronous loading, and a
second synchronous run. Model weights, optimizer, EMA, pending plan, RNG and
replay payloads must match exactly within that world size. All three controls
use the new N+2 replay-planning rule.
The comparison does not establish equivalence to the original sequential trainer.
The asynchronous
branch then continues through update 128 and two cooldown updates. Cooldown
linearly lowers the learning rate to 1e-6, retains Adam state, and restarts the
exponential moving average (EMA) from the current model weights. An uncooled fork
continues the saved main-training trajectory before that transition. Rank weights
must agree. Annual validation runs from 2013-11-01 for 365 days and must produce
finite RMSE metrics for SST, SSH, temperature depths 1–13 and salinity depths 0–13,
using the [design's depth and normalization conventions](README.md).

The replacement qualification records NVIDIA `utilization.gpu` for each assigned GPU once per second.
The release gate is a mean above 50% on every GPU over at least two minutes,
from the start of steady-stage update 25 through the last validation/checkpoint
record at update 128. That window includes its data waits, checkpoints and annual
validation. The report will also separate active training time and the deliberately
synchronous comparison stages. Successful numerical checks alone do not establish
the utilization threshold.

Runtime estimates will use steady mixed-source update times and separate
validation/checkpoint costs. Queue delay and unknown preemption frequency remain
excluded. The old serial-loader estimates are historical measurements. A real
scheduler interruption/resume check is still required after GPU bring-up.
