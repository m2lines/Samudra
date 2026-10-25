<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Physical-state diffusion: replay qualification and performance

Beta job **217313**, code `390e901ad`, completed in 24m21s on one GB200.
Earlier job 217306 stopped on NumPy integer serialization in replay logging;
the fix converts logged lead/age values to Python integers and has a regression
test. No scientific production run has launched.

## Replay and restart

Six effective-batch-eight updates (microbatch two, two members, eight Heun levels)
completed, including checkpoint/Adam/EMA/buffer restart after update four.
Both sources consumed eight replay examples, with starts at 30 and 60 days.
This validates the implemented path, not one-year stability or replay efficacy.

Observed replay-inclusive update times were 16.8/43.4s for OM4 and 33.5/36.4s
for observations. The 43.4s OM4 update followed process restart; this small sample
does not separate data loading, warmup, and compute. Peak memory logged by the
trainer is cumulative within each process, not independently reset per update.
See [qualification receipt](physical-performance-assets/REPLAY_QUALIFIED.json).

## Cached compute screen

All rows use effective batch eight. Each task repeats one cached training month
eight times; these timings exclude loading, EMA, replay generation and validation.
Two warm optimizer updates were measured. Different microbatch sizes assign
different stochastic draws; this is a throughput screen, not matched training.

| Microbatch | OM4 seconds/update | OBS seconds/update | OBS peak GiB |
|---:|---:|---:|---:|
| 1 | 2.35 | 36.54 | 10.31 |
| 2 | 1.59 | 31.48 | 19.19 |
| 4 | 1.42 | 28.88 | 36.83 |
| 8 | 1.06 | 28.22 | 71.96 |

Microbatch eight improves cached throughput by 2.21x on OM4 and 1.30x on OBS
versus microbatch one. Four is almost as fast on OBS with half the memory.
Training-only compilation at microbatch one slowed OBS to 39.23s and incurred
a 61.9s cold OM4 update. Fused AdamW was approximately neutral (36.51s OBS).
Checkpoint removal exhausted approximately 180 GiB even at microbatch one.

**Recommended settings:** keep BF16, vectorized members and activation
checkpointing; leave compilation off. Allow microbatch eight for fresh data,
with automatic grouping by actual temporal length. With half-batch replay,
fresh groups contain at most four examples; replay examples currently execute
sequentially, with their members vectorized. The batch-eight compute gain
therefore cannot be applied wholesale to the replay-inclusive runtime.
Fused AdamW already passed the real-data replay qualification; retain it without
claiming a significant speedup.

The initial 100 OM4 + 500 updates/source + 100 OBS cooldown pilot is roughly
8–13 GPU-hours using the observed replay-inclusive ranges, before evaluation
and checkpoint overhead. This is provisional, not an allocation or ETA.
Full Beta production requires a suitable four-GPU implementation/allocation;
the current trainer is single-GPU. Do not reserve four GPUs for it silently.

## Numerical diagnostic limitation

Recorded gradient cosines exceed one, including the identical baseline.
They are invalid and must not be used as parity evidence. The raw records are
retained unchanged for audit. Gradient vectors were not saved, so those cosines
cannot be corrected retrospectively. Future profiles use float64 reductions.
CPU tests independently cover checkpoint and member-vectorization equivalence,
per-example loss weighting and replay alignment. The measured timings and OOM
outcomes do not depend on the cosine diagnostic.

[Raw profile results](physical-performance-assets/profile-results.json) and
[timing CSV](physical-performance-assets/timings.csv).
