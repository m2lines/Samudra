<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Rust loading and distributed qualification

The optimized Rust/prefetch path passed single-GPU numerical and utilization
qualification in job `21816758`. It completed 128 main and two cooldown updates
on one RTX PRO 6000 Blackwell in 28m07s, using code
`7bef723cb1f60d4e202aeece6fd4c1b11eb2e539`. The allocation cost was 0.4686 GPU-hours.
Four-GPU retry `21926614` also passed numerical and utilization qualification,
completing 128 main and two cooldown updates in 16m06s (1.0733 GPU-hours) on `gr103`.
Real Slurm interruption/resume test `21926632` started after that success and
remains under separate monitoring. No scientific screen or 16k run has been submitted.

The four-GPU and scheduler-test requests are eligible for both
`rtx6000_lzanna` and `rtx6000`, using account `torch_pr_347_lzanna`,
`preemption=yes;requeue=true`, and `Requeue=1`. On 2026-11-04, these replaced
preemption-only jobs `21816773` and `21822548`, both cancelled before allocation.
Torch rejected in-place routing changes and release of temporary held replacements,
so the final requests were resubmitted without holds. Their dependency and
completion callbacks were rebuilt. The [routing receipt](receipts/routing-2026-11-04.json)
preserves all replaced job IDs; queue age restarted with the final submissions.

The four-GPU requests replace eight-GPU jobs `21857804` and `21857835`,
both cancelled before allocation. Each requests 32 CPUs and 128 GiB RAM.
Two examples accumulated per rank preserve global batch eight, learning rate,
and replay policy. The [replacement receipt](receipts/four-gpu-2026-11-04.json)
records the original four-GPU job and callback IDs.

Job `21909447` ran on `gr103` for 11m32s and exited `1:0`, consuming 0.7689
GPU-hours. All four ranks timed out after 600 seconds in the startup barrier
in `samudra/utils/distributed.py`. No optimizer update, training-utilization
telemetry, or `QUALIFIED.json` success receipt was produced. Its dependent
restart job `21909510` was cancelled before allocation. The
[failure receipt](receipts/scaling-v4/4gpu-first-attempt/failure.json) and
[original log](receipts/scaling-v4/4gpu-first-attempt/slurm.log.gz) preserve this attempt.

The submit scripts omitted the RTX settings documented in `docs/torch.md`:
`NCCL_P2P_DISABLE=1` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. Slurm did not
retain the job environment, so their actual values in the failed allocation
are unknown. Retry `21926614` exports both values explicitly on the host and
through `APPTAINERENV_`, with NCCL diagnostic logging. A
[preflight with a 120-second timeout](torch_nccl_preflight.py) exercises the shared
distributed startup and checks ten integer sum reductions against their expected
values before training. The retry retains the code, model, data, global batch
and replay controls. Qualifier `21926614`, dependent restart test `21926632`,
and their completion callbacks are recorded in the
[retry receipt](receipts/nccl-retry-2026-11-05.json).
The retry passed the preflight on all four GPUs. Its
[qualification receipt](receipts/scaling-v4/4gpu/qualification-summary.json)
confirms exact complete-checkpoint equality between synchronous loading,
prefetched loading and repeated synchronous loading, all restored from the same
four-update anchor. Uncooled continuation remained isolated, and final model
hashes agree across all four ranks. Annual validations at updates 128 and 130
were finite; each is one autoregressive forecast scoring 29 channels over
73 steps and 12 complete months. The checks compare runs with the same
four-rank layout. Global batch eight and fresh/replay counts match single-GPU
job `21816758`: 268/260 observation examples and 260/252 OM4 examples.
The complete receipt, input parity cases, step records, launch log and telemetry
are retained in the [four-GPU artifacts](receipts/scaling-v4/4gpu/).

The [four-GPU utilization audit](receipts/scaling-v4/4gpu/utilization.json)
covers 303.24 s from update 25's start through update 128's validation and
checkpoint work. Each GPU has 303 samples, complete coverage, and a maximum
sampling gap of 1.001 s. Mean utilization is 55.07%, 53.06%, 51.09% and 52.17%;
all pass the 50% gate. The lowest GPU has only 1.09 percentage points of margin,
and this five-minute window does not establish long-run utilization.
Across 52 updates per source, observations average 3.014 s and OM4 1.963 s.
Foreground data wait averages 2.89 ms and 0.37 ms respectively; observation
median wait is 0.41 ms, with one 129 ms wait. The
[audit script](audit_scaling_utilization.py) also reproduces the archived
single-GPU utilization result.

The [single-GPU receipt](receipts/scaling-v4/1gpu/qualification-summary.json)
records exact synchronous/prefetched comparisons, isolated uncooled checkpoint
handling, and finite annual validations at updates 128 and 130. Each validation
scored 29 channels over 73 steps and 12 complete months. The final model hash
also matches the recovered pre-optimization run exactly. The three prefetch
controls all reload the same four-update anchor; they do not establish the
pending uninterrupted-versus-scheduler-resumed GPU comparison.

The [utilization audit](receipts/scaling-v4/1gpu/utilization.json) covers 762.4 s,
from update 25's start through update 128's validation and checkpoint work.
All 762 telemetry samples average 82.18% GPU utilization, above the 50% gate;
the time-weighted mean is 82.24%, with complete window coverage and a maximum
sampling gap of 1.006 s. Across 52 updates per source, observations averaged
8.87 s and OM4 averaged 4.90 s. Foreground data wait averaged 0.88 ms and
0.83 ms respectively. The previous run measured 30.51 s per OM4 update with
25.60 s waiting for data. These are successive allocations, not an isolated
hardware experiment, but the optimized path meets the required utilization gate.
[Step records](receipts/scaling-v4/1gpu/scaling.jsonl) and
[raw telemetry](receipts/scaling-v4/1gpu/gpu-utilization.csv.gz) are retained.

The single-GPU time estimates apply to arm D: 170M parameters, mixed OM4/observations,
replay, and global batch eight on one RTX PRO 6000. They include validation every
250 updates and checkpointing every 25, using measured costs of 1.94 s per annual
validation and 6.71 s per checkpoint. The extra 30% covers execution variation;
queue delay and unknown preemption frequency are excluded. The observed-range
columns use minimum/maximum step times, not confidence intervals. The other optimized arms remain unmeasured.

| Work | Updates | Mean-based time | With 30% allowance | Range with allowance |
| --- | ---: | ---: | ---: | ---: |
| Arm D screen: 4k main + 1k cooldown | 5,000 | 10.5 h | 13.7 h | 13.1–14.5 h |
| Arm D if selected: 16k main + four 1k cooldowns, including its screen | 20,000 | 42.1 h | 54.8 h | 52.3–58.1 h |
| Remaining after its completed 5k screen | 15,000 | 31.6 h | 41.1 h | 39.2–43.6 h |

The screen contains 3,000 observation and 2,000 OM4 updates:
`3,000 × 8.866 s + 2,000 × 4.901 s`, plus validation/checkpoints.
The [estimate receipt](receipts/scaling-v4/1gpu/runtime-estimates.json) records
inputs and assumptions. These are engineering estimates, not forecast-skill results.

The [four-GPU estimates](receipts/scaling-v4/4gpu/runtime-estimates.json) use the
same [arm D configuration](configs/scaling-v4/D-mixed-large-4gpu.yaml), mixture
and global batch. A complete annual validation costs 1.983 s; the slowest
measured single-file checkpoint costs 5.613 s. The 24.36 s checkpoint block at
update 128 writes several files and is not charged as one file. Validation runs
every 250 updates and latest checkpoints every 25, with additional best,
milestone and endpoint saves. Source update timings exclude these costs. For
the screen, 223 checkpoint writes comprise 200 scheduled writes, up to 20
improved-best saves, and three milestone/main/endpoint copies. The calculation is
`3,000 × 3.014 s + 2,000 × 1.963 s + 20 × 1.983 s + 223 × 5.613 s`.
The projected range uses the minimum and maximum measured source-update times
and applies the same 30% allowance to both endpoints. The four cooldowns branch
from uncooled checkpoints at 4k, 8k, 12k and 16k; they do not alter later main
training. Queue delay and unknown preemption frequency remain excluded. These successive
allocations measure engineering performance; they are not a controlled hardware
benchmark or a comparison of forecast skill.

| Work on four GPUs | Mean-based time | With 30% allowance | Range with allowance | GPU-hours at mean plus allowance |
| --- | ---: | ---: | ---: | ---: |
| Arm D screen: 4k main + 1k cooldown | 3.96 h | 5.15 h | 4.88–5.47 h | 20.6 |
| Arm D if selected: 16k main + four 1k cooldowns, including its screen | 15.83 h | 20.58 h | 19.52–21.87 h | 82.3 |
| Remaining after its completed 5k screen | 11.87 h | 15.44 h | 14.64–16.40 h | 61.7 |

Scheduler job `21926632` will run two 16-main-plus-two-cooldown trajectories
through `samudra.observations.train` and the shared Slurm/Apptainer launcher on
four GPUs, with two accumulated examples per rank. One is uninterrupted. After the other has saved at least six main
updates, a bounded watcher sends USR1 to its batch shell, which calls
`scontrol requeue`. The restarted allocation must preserve the W&B run ID and
produce an exactly equal complete checkpoint payload: model, optimizer, EMA,
sampler, replay, pending plan and per-rank RNG state. The test uses the same
runtime code as the scaling qualification; its [orchestration and audit](scheduler_resume.py)
are frozen from commit `5bfcce5e4`. A failed prerequisite cancels the dependent
request. CPU tests reject a missing scheduler restart, changed optimizer state,
and a changed tracking identity. The real scheduler result remains pending.

Earlier attempts and data qualification follow. They extend the
[initial single-GPU qualification](qualification.md).

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

CPU job `21803580` completed the lossless packed OM4 cache in 11m52s with exit
`0:0`. It includes all 2,812 eligible native frames and all 77 physical channels,
with one chunk per frame instead of 77 separate variable chunks. Every packed
frame matched all 77 original channels exactly through both Python and Rust readbacks.
The cache occupies 26,774,925,591 bytes, versus 56,123,020,800 bytes of raw
float32 values. Its `COMPLETE.json` proof has SHA256
`fcae8e6573adad1dcfd125459ce0ee6b1152a0f32aa299bc7e70148e9965e5dc`;
the [preparation receipt](receipts/scaling-v3/packed-om4-preparation.json) records
the source hashes and interval bounds. At startup, the packed-data training adapter
checks the pinned proof, original metadata, complete native-index list, interval
bounds, channel order and cache hashes. Normalization, masks, cohort and calendar
rules stay in the existing loader.

Replacement GPU jobs `21805440` (one GPU) and `21805446` (eight GPUs) were submitted
on 2026-11-03 with code `710e49603ced7c669c5e504d0b8f6ac626090438` and
[configs that pin the packed cache proof](configs/scaling-v3/).
The one-GPU job completed all 128 main and two cooldown updates, exact
prefetch comparisons from a common restored checkpoint, and both
73-step/29-channel annual validations.
It then failed while hashing a scalar model tensor for the final receipt.
The checkpoints and metrics remain available; no `QUALIFIED.json` was written.
Code `efc58064ec70bdcc4d4264422c2e7172c90700a6` fixes the scalar byte view and passes
a CPU regression test. CPU job `21814480` completed in 3m48s and independently
verified all three comparison payloads, the uncooled 128-update checkpoint, the
130-update endpoint, cooldown EMA state, finite weights and annual validation
records. The [recovery summary](receipts/scaling-v3/recovery-summary.json) links
the saved results to their hashes without changing the failed scheduler status.
The dependent eight-GPU job
was cancelled before allocation. The failed one-GPU allocation consumed
3,960 GPU-seconds (1.10 GPU-hours).

The [steady records](receipts/scaling-v3/scaling.jsonl) cover 52 updates per
source from updates 25 through 128. Observation updates averaged 8.94 s, with
0.0008 s of exposed data wait. OM4 averaged 30.51 s, including 25.60 s of data
wait. The [utilization audit](receipts/scaling-v3/utilization.json) spans 2,096.5 s
from update 25's start through validation/checkpoint work at update 128; its
2,096 samples average 30.50%. This fails the release gate. The
[raw telemetry](receipts/scaling-v3/gpu-utilization.csv.gz) is retained.

The same CPU job [profiled two identical eight-window OM4 batches](receipts/scaling-v3/loading-profile.json).
Each contained four fresh target months and four next-month targets on replay
interval grids, including auxiliary initialization and observation coverage reads.
This was a loading-only probe without cached model states.
They took 9.63 s and 8.81 s; Rust physical reads accounted for 1.52 s and 1.02 s.
Much of the remaining time was array preparation. The GPU job's longer loading
times may also reflect contention; this CPU profile does not isolate that effect.
Code `7bef723cb1f60d4e202aeece6fd4c1b11eb2e539` reuses owned arrays during normalization
and stages only the two required dense initial states plus forecast targets.
It retains all surface history, forcing, masks and supervised channels. Exact
normalization and input-slicing tests pass. Production training and qualification
use the same strict CUDA determinism settings.

CPU job `21816667` completed the [three-pass comparison](receipts/scaling-v4/loading-profile.json)
in 5m30s on node `cs625`. All model inputs and targets matched exactly for the
same eight-window OM4 plan. Both CPU probes time the loader call, excluding
exact comparison checks and garbage collection. The newer probe changes CPU
array preparation and retains a reference batch; the probes also ran on different
nodes (`cs630` and `cs625`). The cause of the much slower original-loader timing
in the newer probe is unresolved. Original, optimized and repeated-original batch
wall times were 47.95 s, 47.21 s and 57.47 s. Rust reads took 1.71 s, 8.64 s and
1.05 s respectively. ERA5 and observation coverage reads also rose during the
optimized pass. The comparison retains the first complete CPU batch for exact
checks, so later passes have a different live memory footprint; cumulative peak
RSS was 7.71, 12.06 and 12.89 GB. These measurements do not establish an
end-to-end speedup. GPU training must measure the effect of reduced copies and
the smaller dense payload. The [audit receipt](receipts/scaling-v4/loading-audit.json)
records the CPU result and limits.

GPU job `21816758` subsequently passed, as reported above. Its four-GPU
retry `21926614` uses the same code and [configuration](configs/scaling-v4/)
and completed successfully, as reported above.

The historical estimates below apply only to arm D before optimization: the 170M-parameter mixed-source
recipe with replay and global batch eight, on one RTX PRO 6000 Blackwell GPU.
They use code `710e496` and precede the `7bef723` loading changes, whose measured GPU timings are reported above. They include validation every 250 updates and checkpoints every 25.
The conservative measured costs are 2.00 s per annual validation and 7.89 s per
single checkpoint. The range uses the observed per-source minimum and maximum
update times; it is not a confidence interval. Queue delay and unknown preemption
frequency are excluded. At that stage, four-GPU timings and timings for the other optimized
arms were unmeasured. Rows for the selected recipe assume this same
170M mixed-source configuration wins the screen.

| Work | Updates | Mean-based time | With 30% allowance | Range with allowance |
| --- | ---: | ---: | ---: | ---: |
| One screen: 4k main + 1k cooldown | 5,000 | 24.9 h | 32.4 h | 23.3–51.0 h |
| Selected recipe: 16k main + four 1k cooldowns, including its screen | 20,000 | 99.6 h | 129.4 h | 93.2–203.8 h |
| Selected recipe after its completed 5k screen | 15,000 | 74.7 h | 97.1 h | 69.9–152.8 h |

For example, the screen contains 3,000 observation updates and 2,000 OM4 updates:
`3,000 × 8.936 s + 2,000 × 30.507 s`, plus validation/checkpoint overhead.
The [estimate receipt](receipts/scaling-v3/runtime-estimates.json) preserves the
inputs and arithmetic. These measurements describe the path that failed the
utilization gate; full scientific training remains unsubmitted.

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
four fresh and four replayed. Four ranks accumulate two windows each; eight ranks process one each, and
one rank accumulates eight. Rank zero owns the global sampler and source-separated replay
pools, broadcasts example metadata, distributes cached states, and gathers the
new states in global example order. Pool capacity stays two entries per source
and lead-month bucket, independent of rank count.

Planning update N+1 before computing N makes states generated during N eligible
starting at N+2. This changes replay timing from the original sequential loop.
The same rule applies to one, four and eight GPUs. Empty-pool startup can supply more
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
All three GPU controls reload the same four-update checkpoint. This establishes
repeatable reload/continuation and prefetch parity; the current pipeline's
uninterrupted-versus-resumed GPU comparison remains part of the pending scheduler
interruption test. CPU tests already cover that uninterrupted comparison.
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

Runtime estimates use steady mixed-source update times and separate
validation/checkpoint costs. Queue delay and unknown preemption frequency remain
excluded. The old serial-loader estimates are historical measurements. A real
scheduler interruption/resume check is still required after GPU bring-up.
