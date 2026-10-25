<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Monthly diffuser performance diagnostic

This diagnostic measures training speed outside the processor. It does not change
any scientific run or promote its disposable weights. Beta_test job **217084**
uses one GPU, a two-hour cap, and producer **921025144**. All seven comparisons are complete. Compiled variants failed full-graph tracing;
job **217194**, producer **ac1d1b4c0**, completed both with explicitly permitted
graph breaks. Original failures are preserved.

## Comparisons

All variants use the original 128-channel, 45×90 latent model, BF16, deterministic
CUDA settings, AdamW, effective batch eight, and unchanged full monthly,
forecast, and reconstruction losses. Each update repeats one example eight times. Each variant uses the same fixed real OM4 window
and OBS training month. Full77 model targets, observed surface support, and
monthly interior support retain their existing masks. Reusing fixed batches
isolates compute; separately recorded cold/repeat loads are not full-loader
throughput estimates.

1. Existing checkpointed implementation.
2. Decoder activation checkpoint disabled.
3. Initializer and decoder activation checkpoints disabled.
4. Initializer activation checkpoint disabled.
5. Fused AdamW, with checkpointing unchanged.
6. `torch.compile` of the denoiser core, checkpointing unchanged.
7. Compiled denoiser core with decoder checkpoint disabled.

Compilation uses default mode and `fullgraph=True`: unsupported graph breaks
fail visibly. The script records compiler counters and recompile logs. Each
variant/source performs one cold update for loss/gradient/update agreement,
one warmup update, and three measured updates. Cold first-update time includes compilation, execution, and first optimizer-state
allocation; it is not pure compiler time. It is reported separately from warm update time. No processor settings are modified.

## Checks and interpretation

Inputs, initial parameters, random draws, masks, and losses match. The baseline
first-update reference retains clipped gradients and post-update parameters.
Alternatives report relative loss difference, gradient L2 difference, update
L2 difference, and maximum parameter difference. An approximate screening gate
requires gradient difference below 2% and update difference below 5%; passing
is not proof of long-training scientific equivalence. Cold and warm full-update
wall times and peak allocated/reserved GPU memory are recorded. EMA and
validation are excluded, so these are training-update compute timings.

Data integrity checks remain enabled. Source metadata hashes are revalidated;
OM4 packed-frame checks run before repeated-batch timing. The benchmark has no
permission or code path to write scientific checkpoint directories.

Remote artifacts:
`beta:/projects/ny/lz1955/multiscale/jrusak/runs/monthly-diffuser-performance-20261024`.

Remote host-authenticated callback:
`b75a42b9-2bad-4a87-805b-fa9743ace303`, owned by the parent conversation and
monitored by `/root/diffuser_performance`. Queue uncertainty leaves the results
ETA unset until useful runtime evidence is available.

## Initial results

Job 217084 completed in 8m51s (0.1475 GPU-hours). Median seconds per cached
effective-batch-eight update; memory is peak allocated GB (decimal).

| Variant | OM4 seconds | OBS seconds | OM4 GB | OBS GB |
|---|---:|---:|---:|---:|
| baseline | 3.590 | 3.646 | 4.25 | 4.51 |
| decoder-no-checkpoint | 3.293 | 3.392 | 7.56 | 8.52 |
| initializer-no-checkpoint | 3.350 | 3.447 | 5.39 | 5.65 |
| all-no-checkpoint | 3.188 | 3.058 | 8.66 | 9.63 |
| fused-adamw | 3.534 | 3.525 | 4.25 | 4.51 |

Removing both initializer and decoder checkpoints reduced cached update time
by **11.2% on OM4 and 16.1% on OBS**, at roughly twice the allocated memory.
Every checkpoint-removal variant matched first-update loss, clipped gradients,
and parameters **bitwise exactly** in both examples. This is the strongest
initial candidate, subject to an end-to-end loader/memory check before adoption.

Fused AdamW saved only 1.6–3.3%, within the uncertainty of three measurements;
maximum parameter difference was 5.96e-8. It is a lower-priority change.

Baseline cold/repeated OM4 loads took 4.94/0.93 seconds per example, compared
with OBS 0.49/0.003 seconds. These two observations are not an I/O distribution,
but illustrate why the compute speedup need not translate directly to full
training speed. No validation or EMA costs are included above.

Both compiled variants failed before a completed first update: deterministic
CUDA replicate padding in `F.pad` executes a dynamic import that `fullgraph=True`
refuses to trace. This is a compiler compatibility failure, not a numerical or
model-training failure. Recovery keeps deterministic settings and permits graph
breaks; it measures the actual partially compiled implementation rather than
silently claiming full compilation. Recovery callback:
`d27bc2c1-911d-4967-8d8d-3f08afbad65d`, job 217194.

## Compilation results and recommendation

Recovery job 217194 completed in 3m54s. Both sources and variants have complete
metric artifacts. Total diagnostic allocation across both jobs: **12m45s =
0.2125 GPU-hours**. Login and compute-node callback checks succeeded.

| Variant | OM4 seconds | OBS seconds | OM4 GB | OBS GB |
|---|---:|---:|---:|---:|
| Partially compiled decoder, checkpoints on | 3.224 | 3.220 | 3.70 | 4.45 |
| Partially compiled decoder, decoder checkpoint off | 2.826 | 2.902 | 5.57 | 6.77 |

The combined treatment reduced cached update time **21.3% OM4 / 20.4% OBS**
relative to the original baseline. Relative to simply disabling decoder
checkpointing, its incremental reductions were **14.2% / 14.4%**. These short
measurements ran in a different allocation from the eager baseline, without
randomized order or repetitions; treat percentages as screening estimates.

This is **partial compilation**, not a fully compiled denoiser: deterministic
replicate padding creates graph breaks, and the resumed convolution frame hit
Dynamo's eight-recompile limit and falls back to eager for further variants.
All raw graph-break/recompile logs and counters are retained. No unsupported
operation was silently removed and deterministic settings stayed enabled.

The first compiled OM4 update took **39.50s** versus the eager baseline's
5.34s. With the measured warmed savings, that extra cold cost would amortize
in roughly **90 OM4 updates**. OBS reused already compiled shapes. The second
compile variant's 9.36s cold update benefited from persistent compiler caches;
it is not an independent fresh-compilation measurement. For 50/50 mixed
updates, the first variant's rough amortization is also under 100 updates,
excluding loader/validation effects and future shape recompilations.

Compilation changed first-update numerics: loss relative errors 4.1e-8 OM4 /
1.6e-7 OBS; gradient relative L2 errors **0.171% / 0.175%**; Adam update relative
L2 errors **2.08% / 2.70%**. Maximum individual parameter differences reached
approximately **2e-4**, reflecting near-zero gradient sign differences amplified
by Adam's first step. These pass the loose screening thresholds but are **not
bitwise or scientifically demonstrated equivalent**. Do not silently apply
compilation to an in-progress matched experiment.

Recommended next implementation is optional activation-checkpoint disabling
with default unchanged: it provided exact agreement here and a useful speedup
at modest absolute GPU memory. Compilation is a plausible additional gain, but
needs clean padding graph capture and a matched short learning-curve check
before scientific adoption. Fused AdamW is low priority. Re-test memory with
the planned larger processor; these peaks describe the current small processor.
No optimizations have been applied to shared-ERA5 or other scientific runs.

## Follow-up: genuine microbatches (approved)

The user adopted checkpointing off + compiled denoising + fused AdamW as future
baseline defaults. The follow-up only selects microbatch size 1/2/4/8 with
accumulation 8/4/2/1, holding effective batch eight fixed. It uses eight distinct
real months per source, selected to have a common trajectory length. Each
retains its own calendar weights, duration, lead target, and support masks.
The benchmark-specific loss computes per-example normalized errors before
averaging, matching the single-example objective. Independent per-example/task
noise streams are identical across microbatch variants; this changes random
stream ordering relative to production, but not the noise distribution. A tiny
synthetic test checks the batched denoising loss against the original decoder
loss for different masks, durations, and forecast positions.

First-update gradients and parameters will be compared to optimized
microbatch-one. This measures numerical effects of batching, not equivalence
to the original uncompiled implementation. Data are cached before timing;
loading, EMA, and validation are excluded. Compilation cold-start time is
separate. All updates and weights remain disposable.

Submitted microbatch diagnostic **217212** on Beta_test (one GPU, two-hour cap),
producer **46ee56f5d**. Artifacts:
`beta:/projects/ny/lz1955/multiscale/jrusak/runs/monthly-microbatch-performance-20261024`.
Remote host-authenticated callback `6cc604c3-901f-4833-bac5-67b92b4453c6`
was monitored by `/root/diffuser_performance`; all four sizes completed.

## Microbatch results

Job 217212 completed in **6m56s (0.1156 GPU-hours)**. All four sizes fit; no
OOMs or failed variants. Effective batch remains eight in every row. Median
seconds/update excludes loading, EMA, validation, and cold compilation.

| Microbatch | Accumulation | OM4 seconds | OBS seconds | OM4 examples/s | OBS examples/s | Peak allocated GB (OM4 / OBS) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2.804 | 2.826 | 2.85 | 2.83 | 11.51 / 13.42 |
| 2 | 4 | 1.977 | 1.936 | 4.05 | 4.13 | 17.08 / 19.21 |
| 4 | 2 | 1.436 | 1.448 | 5.57 | 5.52 | 28.20 / 30.80 |
| 8 | 1 | 1.283 | 1.281 | 6.23 | 6.24 | 50.27 / 53.80 |

**Microbatch eight is the preferred size for the present model on Beta.** It
provides **2.18× OM4 / 2.21× OBS throughput** versus optimized microbatch one,
with no change in effective batch. Peak allocated memory is 50.3/53.8 GB;
maximum reserved across sources is 58.2 GB on a 197.6 GB device. This includes
the eight cached real examples, but excludes a production EMA model and loader
prefetch buffers. Microbatch four is a useful lower-memory fallback (~31 GB).

First-update microbatch-eight loss differs by 1.23e-7 relative on OM4 and
2.18e-8 on OBS. Gradient relative L2 differences are **0.259% / 0.239%**, and
Adam update relative L2 differences are **1.46% / 1.70%**, compared with optimized
microbatch one using matching per-example targets and noise. Maximum individual
parameter differences are 1.87e-4 / 1.93e-4. This is ordinary finite-precision
batch-kernel/reduction sensitivity at this scale, not bitwise equivalence;
it does not establish identical long-training trajectories.

Cold first OM4 updates take 38–45 seconds as new batch shapes compile. OBS
uses those shapes and begins at 1.3–2.9 seconds depending on size. Existing
compiler disk caches can affect cold timing. The known deterministic-padding
graph breaks and eight-recompile-limit fallback occur at every microbatch size.
This remains the explicitly partial-compiled baseline agreed with the user.

The cohort contains eight different OM4 windows and eight different observation
months, not eight repeats of one example. Month lengths and masks remain
individual. Forecast leads are independently seeded per example and held fixed
across this short timing screen; noise changes by update. Common trajectory
length was selected for batching, so future production batching must bucket
compatible lengths or implement explicit padding without dropping examples.
The standalone benchmark wrapper is not yet production-loader integration.

The combined defaults and microbatch eight are recommended for the next
baseline qualification. Recheck memory/throughput for the larger Samudra-like
processor and then choose its microbatch. No current scientific run was altered.
All performance diagnostics together consumed **0.3281 GPU-hours**.

## Shared-ERA5 migration qualification

The user authorized applying the optimized setup to the still-unstarted shared
ERA5 experiment, subject to avoiding large behavioral changes. Job **217213**
(one-GPU Beta_test, two-hour cap), producer **55b2939e1**, compares 16 matched
mixed updates before any production switch. Original production remains held
and its placement must respect the requirement against reserving four GPUs for
one-GPU work without explicit confirmation.

The production helper preserves the original eight sequential sample draws and
consumes every per-example sigma/noise/native-lead draw in the original order.
It groups only within that update by compatible trajectory length: no dropped
months, no resampling, no cross-update grouping, and count/8 gradient weights.
Month weights, masks, targets, and duration remain individual. Scalar-weight
BF16 monthly aggregation semantics are preserved, rather than promoting the
monthly latent by broadcasting a float32 weight array. Twenty targeted tests
pass, including original-objective/RNG and single-example gradient agreement.

Frozen engineering gates: exact sample/draw/post-update generator hashes; first
gradient relative L2 below 1%; first Adam update difference below 5%; all 16
matched loss differences below 2%; each of five fixed-noise raw and EMA
validation RMSE components within 1%; exact serialized optimizer/EMA/RNG replay
for both implementations. Failure is investigated or falls back to a smaller
optimization subset; thresholds are not loosened after looking at results.
These checks do not establish identical long-training trajectories.

Artifacts: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-optimized-20261024`.
Remote callback: `dc1369af-a989-4873-ab55-0b2c432a1b2e`.

### Four-GPU execution qualification

The user subsequently authorized all four GPUs for shared-ERA5 production with
**two examples per rank and global effective batch eight**. Job **217218** is
an actual four-GPU short qualification, dependent on reference job 217213,
using producer **a1216ce4e**. Production is still held, not launched.

Rank zero prepares the same original eight examples/draws and broadcasts tensor
payloads through NCCL. Each rank gets exactly two examples. Different trajectory
lengths are handled inside one local DDP forward/backward; each local example
has weight 1/2, and DDP averages four rank gradients to the original global1/8
weight. This avoids unequal collective counts or weighting different months by
number of groups. All four ranks' CPU/CUDA RNG states are saved; model, optimizer,
EMA, common data/diffusion RNG, and counters retain their existing checkpoint
contracts. Qualification also requires identical model hashes across ranks.

The same frozen numerical gates compare DDP against **both** original and
optimized single-GPU references, including fixed-noise raw/EMA validation and
exact serialized replay. Twenty-one targeted local synthetic tests pass.
The eventual launcher runs the main8k-plus-cooldown trajectory first and then
qualifies/runs the independent saved4k cooldown, each using all four GPUs.
No independent worker occupies GPU1 while a four-rank job is active.

Four-rank artifacts: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-ddp-20261024`.
Callback: `ab7b78fd-13d9-49f0-8375-80b0525831d6`.

### Migration gate failure: no production promotion

Reference job 217213 completed both training qualifications and exact serialized
replay, but the final comparison **failed the raw validation gate**. All 16
sample/noise/RNG traces match exactly; maximum training-loss relative difference
is 0.0058%, first gradient difference 0.217%, and first update difference 2.51%.
EMA validation differs by at most 0.0036%. Raw validation differs substantially:
SST 7.029→7.566°C (+7.6%), upper OHC +26.6%, velocity -68.5%.

The optimized raw model uses compiled inference whereas EMA validation uses
eager inference. This is a lead, not an established cause. Short evaluation job
**217228**, producer **6284f0d7e**, evaluates optimized raw weights eagerly and
baseline raw weights with compilation, using the same fixed noise/origin and
unaltered saved checkpoints. Callback: `2e49f0eb-319a-4367-a0a0-9155ba28bbbb`.

Dependent DDP qualification 217218 never started and was canceled after the
prerequisite failure. No qualification threshold was relaxed and production
remains held. Failed attempts and all outputs remain preserved. Reference
qualification consumed 15m56s = 0.2656 GPU-hours; DDP attempt consumed zero.

### Inference isolation and safe compilation scope

Job 217228 completed in 3m13s. Optimized raw weights evaluated **eagerly** give
SST 7.03268°C versus baseline 7.02903°C (+0.052%), upper OHC +0.050%, deep OHC
-0.018%, and effectively unchanged velocity. Fresh compiled baseline weights
also match their eager evaluation closely. Thus the large failed raw metric
was specific to the in-process compiled evaluation; the exact compiler-state
cause is not established, but a consistent eager evaluation removes it without
changing trained weights or loosening the gate.

Producer **6e326a55e** compiles only an explicit training-denoiser callable.
`core.forward` and both raw/EMA diffusion sampling remain eager. Twenty-two
local synthetic tests pass, including that a compiled training hook cannot
change evaluation routing. Actual four-GPU qualification **217230** uses this
implementation and validates against both original and optimized single-GPU
references. The latter uses the independently measured eager raw evaluation,
with its source checkpoint SHA checked before substitution; the failed compiled
evaluation remains unchanged in the original proof and report.

Artifacts: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-ddp-20261024/eager-evaluation`.
Remote callback: `657648d1-93d2-4326-b976-169b540dad27`.
Four-GPU job **217230 passed** in 13m18s (0.8867 GPU-hours). All 16 original
sample, corruption, lead, and post-draw RNG traces match exactly; all four rank
model hashes agree; serialized optimizer/EMA/RNG replay is exact. Comparisons
against both the original and optimized single-GPU implementations pass.

| DDP versus original | Observed difference | Frozen limit |
|---|---:|---:|
| First gradient, relative L2 | 0.198% | 1% |
| First Adam update, relative L2 | 2.55% | 5% |
| Largest of 16 training-loss differences | 0.00977% | 2% |
| Largest raw validation component difference | 0.0646% | 1% |
| Largest EMA validation component difference | 0.00104% | 1% |

Raw SST is 7.03357 versus 7.02903°C, a 0.00454°C difference. This is an
engineering continuity check after only 16 disposable updates, not a scientific
forecast score or proof of identical long-run training. The one-origin EKE
component is zero in both implementations and is not an informative gate.
Peak allocated memory across ranks is 29.35 GB per GPU. Proof files are
`COMPARISON-original.json`, `COMPARISON-optimized.json`, and
`COMPARISON-single-eager.json` in the artifact directory above.

**Performance remains unresolved:** warm observation updates took about 3–5s,
but OM4 updates took 48–68s, versus about 13–18s in the optimized single-GPU
reference. These qualification timings include exact target/noise hashing,
first-read integrity checks, data loading, and communication; they do not isolate
GPU compute or establish a four-GPU speedup. The production handoff must inspect
warm end-to-end timing rather than extrapolate the cached microbatch benchmark.
No scientific production was launched by this qualification. Production remains
held for the coordinated handoff using `launch_shared_era5_ddp.py`; that launcher
runs the 8k trajectory and independent 4k cooldown sequentially on all four GPUs.

Total performance and migration checks, including the failed raw-validation
attempt, consumed **1.5339 GPU-hours**. Failed proofs remain preserved.
