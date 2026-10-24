<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Monthly diffuser performance diagnostic

This diagnostic measures training speed outside the processor. It does not change
any scientific run or promote its disposable weights. Beta_test job **217084**
uses one GPU, a two-hour cap, and producer **921025144**. Five eager variants completed; compiled variants failed full-graph tracing and
are being retried with explicitly permitted graph breaks in job **217194**,
producer **ac1d1b4c0**. Original failures are preserved.

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
