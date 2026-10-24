<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Monthly diffuser performance diagnostic

This diagnostic measures training speed outside the processor. It does not change
any scientific run or promote its disposable weights. Beta_test job **217084**
uses one GPU, a two-hour cap, and producer **921025144**. Results are pending.

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
