# One-face 8xH200 deadline patch

This document records the focused launch patch applied to the
`asynchronous-train_blending` worktree for the one-face rental run. The patch
does not implement the asynchronous validator and does not change the chosen
rank-local block sampling distribution.

## 1. Rank-local loss normalization

Changed:

- `face_wide_loss_norms` now receives `face_tiles` and `rank_tiles` rather
  than `world_size`.
- The per-rank denominator is the fixed fraction of the face represented by
  one replay row:

  ```text
  face denominator * rank_tiles / face_tiles
  = face denominator * 9 / 36
  = face denominator / 4
  ```

- The `gradient_z` pair-count normalization uses the same fixed 9/36 share;
  its face-wide per-pair valid-cell counts remain unchanged.
- `gradient_z_l1_loss` no longer clamps fixed fractional pair counts to 1.
  It uses a small positive denominator floor instead. Ordinary locally
  derived pair counts are integers at least 1 and are unchanged; fractional
  counts such as 1/4 and 1/2 now retain the rank/block share they encode.
- `Trainer._install_face_loss_normalization` supplies the actual block size
  (`_block_tiles_per_row`) instead of the DDP world size.
- The collective-free path in `FaceParallelContext.global_loss_norms` also
  expresses its normalization through face and rank tile counts.

Reason:

Each rank-local replay row always contains nine tiles. With the previous
`face denominator / world_size` expression, four ranks happened to produce
the desired scale, while eight ranks doubled both the loss and its gradient.
The corrected expression is independent of how many independent rows DDP
averages. Separately, the old minimum-1 clamp had been silently underweighting
the vertical-gradient term whenever its distributed pair count was below 1.

Tests changed or added:

- A four-rank/eight-rank simulated DDP test verifies that a constant unit
  error has loss 1 and parameter gradient 2 at both world sizes.
- The end-to-end rank-local denominator test now verifies the 9/36 face share
  directly rather than relying on the previous world-size-one coincidence.

## 2. Step-level learning-rate warmup

Ported from `llc_cpu_working` without porting the unrelated curriculum epoch
offset:

- Added `lr_warmup_steps` and `lr_warmup_start_factor` to `TrainConfig`.
- Added the pure `linear_warmup_factor` schedule helper.
- Both ordinary and replay optimizer-step paths apply warmup immediately
  before `optimizer.step()`.
- Logged LR now comes from the optimizer, so it reports the value actually
  used during warmup.
- Before an epoch scheduler update, the active warmup factor is divided out.
  This is required because recursive schedulers such as cosine advance from
  the optimizer's current LR; leaving the factor in would permanently bake
  it into the next epoch's schedule.
- The active factor is stored in checkpoints so an incomplete-epoch/emergency
  resume does not apply the same factor twice.
- Resume behavior continues from checkpointed `num_batches_seen`; finetuning
  intentionally restarts at the beginning of the ramp.

Tests changed or added:

- Exact endpoints, monotonicity, post-ramp behavior, disabling, and invalid
  start-factor handling for the pure schedule.
- Trainer-level coverage that successive warmup factors replace rather than
  compound with one another.
- Trainer-level coverage that warmup composes correctly with cosine across an
  epoch boundary.

## 3. Deliberately unchanged

- The 16 possible 3x3 face-interior blocks remain uniformly sampled. This
  retains simple rank-local rows, exposes all face-interior blend edges over
  time, and remains compatible with a later move to global/nonstatic ranks.
- Asynchronous validation remains deferred. Training continues to write
  per-epoch EMA snapshots for a future validator.
- Pinned host wet-mask placement and transfer behavior are unchanged.
- The LR value, cosine target, replay curriculum, block size, and
  `tiles_per_chunk` choices in the rental script are unchanged.
- No H200 preflight or training job was submitted as part of this patch.

## 4. Preflight readiness checks

Before submission, the patch is checked for:

- acceptance of the rental script's warmup CLI arguments;
- construction of an eight-rank, rank-local, offloaded-validation config;
- four-rank/eight-rank loss and gradient scale invariance;
- focused face, replay, scheduling, checkpoint, and chunk-reader tests;
- Python compilation and shell syntax for both rental and preflight scripts.

The launch scripts remain:

- `JOBS/train_llc_1face_8xh200_rental.sh`
- `JOBS/other/test_8xh200_preflight.sh`

Verification results for this patch:

- The exact rental overrides parse successfully from the async worktree:
  rank-local blending, offloaded validation, 36 source tiles, and 3200 warmup
  steps are all present in the resulting configuration.
- 148 focused face, normalization, loss, reader, DDP synchronization, EMA,
  scheduler, and warmup tests pass. The five tests that create local Gloo
  process groups were run outside the network sandbox; all five pass.
- 28 additional replay and non-finite-checkpoint tests pass.
- Modified Python files compile, both job scripts pass `bash -n`, and both
  worktrees pass `git diff --check`.
- The H200 preflight was intentionally not submitted or executed.
