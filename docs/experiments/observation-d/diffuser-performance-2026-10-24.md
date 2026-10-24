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
