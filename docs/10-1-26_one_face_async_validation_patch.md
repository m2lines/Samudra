# One-face asynchronous validation patch

This change completes the validator half of the one-face 8xH200 plan. The
trainer remains rank-local on eight H200s and writes immutable per-epoch EMA
snapshots. A separate face-parallel job watches those snapshots and scores
them on two H100s without feeding anything back into training.

The implementation follows
`analyze-both-home-codycruz-ocean-emulato-resilient-frog.md`, adapted from its
older two-run/4xH200 proposal to the current single 8xH200 rental.

## Validator implementation

Added `src/ocean_emulators/offload_validate.py` in the
`asynchronous-train_blending` worktree.

The validator:

- accepts one or more comma-separated run directories through
  `--offload.runs`;
- scans only immutable `saved_nets/ema_ckpt_epNNNN.pt` snapshots and never
  reads the mutable `ema_ckpt.pt` name;
- takes the lowest unvalidated epoch in each run, then chooses FIFO by snapshot
  modification time across runs;
- broadcasts `job`, `idle`, and `stop` commands over a long-timeout Gloo
  control group so all ranks enter NCCL validation collectives together;
- verifies snapshot epoch/completeness, confirms it contains inference EMA
  weights rather than a second `ema_params` copy, loads it strictly, and checks
  a parameter checksum across ranks;
- runs one-step, short autoregressive, and long autoregressive validation under
  `torch.no_grad()`;
- captures loss and RMSE at every rollout step in addition to the existing
  scalar/plot metrics;
- disqualifies an epoch from best-checkpoint selection if any validation term
  due at that epoch is missing or non-finite;
- records non-finite rollout-curve points as JSON `null`, so a diverged epoch
  is durably rejected instead of making the result writer fail;
- keeps best-loss/contributor state independently for every watched run;
- releases the unused startup EMA parameter copy before validation to preserve
  H100 memory;
- supports `--offload.once` for the preflight and restart/equivalence checks;
- stops between snapshots on SIGTERM, SIGINT, or SIGUSR1.

## Durable outputs

For each training run, rank 0 writes:

- `offload_val/metrics.jsonl`: scalar metrics, rollout curves, phase timings,
  and peak reserved CUDA memory. The file is atomically rewritten and an epoch
  record is replaced rather than duplicated after a retry.
- `offload_val/state.json`: validated epochs, current best epoch/score,
  combined-loss contributors, W&B id, and rejection reasons. It is written
  last, so an interrupted epoch is repeated rather than skipped.
- `saved_nets/best_validation_ema_ckpt.pt`: an atomic hardlink (or copy
  fallback) to the best strict-score snapshot.
- A separate resumable W&B validation run when `--offload.wandb-mode` is
  `online` or `offline`. Local validation results remain authoritative if W&B
  is temporarily unavailable.

## Configuration guard

The validator compares its resolved configuration with each training run's
saved `config.yaml` before accepting snapshots.

It requires matching model numerics, loss, tile geometry, variables,
normalization inputs, validation window, seeded draws, validation horizons,
surface-snapshot mode, strides, and replay/blending semantics. It also requires
that the training run used EMA, disabled inference, enabled face parallelism,
and wrote snapshots using `validation_mode=offload` or `both`.

The guard intentionally ignores:

- rank-local versus face-parallel ownership and validator chunk size;
- model checkpointing mode;
- replay storage dtype and training/LR/curriculum-only fields;
- worker/read-thread/backend knobs;
- the packed data path, because the H100 validator deliberately reads the
  byte-equivalent disk face cache while H200 training reads the flash copy.

Every differing guarded path is included in a rejected run's state file.

## Trainer-side safety fixes

- The rental script now explicitly passes `surface_snapshot=true`. Without it,
  `Trainer` rejects face parallelism at startup even when validation is
  offloaded.
- All one-step/short/long validation settings are explicit environment knobs
  in the rental script, ensuring the saved training config is exactly what the
  companion validates.
- `save_checkpoint` now reports success. A per-epoch EMA snapshot is published
  only after a fresh EMA checkpoint was written successfully. A non-finite
  model can no longer cause the previous epoch's EMA to be relabeled as the
  current epoch.

## Slurm companions

Added:

- `JOBS/other/test_8xh200_preflight-validation.sh`
- `JOBS/train_llc_1face_8xh200_rental-validation.sh`

Both request node2905 on `pi_abodner`, 2xH100, 32 CPUs, and 400 GB host memory.
The preflight companion mirrors the 1:30 H200 preflight limit; the rental
companion mirrors the 4-day-18-hour rental limit.

The jobs are submitted explicitly after obtaining the H200 trainer job id:

```bash
sbatch --export=ALL,TRAIN_JOB_ID=<preflight-job-id> \
  JOBS/other/test_8xh200_preflight-validation.sh

sbatch --export=ALL,TRAIN_JOB_ID=<rental-job-id> \
  JOBS/train_llc_1face_8xh200_rental-validation.sh
```

If the rental used a custom `EXPERIMENT_NAME`, export that exact same value as
`TRAIN_EXPERIMENT_NAME` for its validator companion. A supplied experiment
name is used verbatim; the trainer adds the Slurm job id only to its default
name. This keeps the multi-cell preflight and its watcher on the same path.

The preflight's final H200 cell records a deliberately short but complete
validation contract: 2 one-step samples, one 4-step short rollout, and one
8-step long rollout. Its H100 companion waits for epoch 1, validates it once,
writes all durable outputs, and exits. The full rental retains 200 one-step
samples, 5x72 short rollouts, and 2x480 long rollouts from epoch 1.

## Verification

- CPU unit tests cover config acceptance/rejection, FIFO selection, strict
  non-finite handling, JSON-safe diverged curves, independent run state,
  argument separation, atomic result replacement, and stale-snapshot
  prevention. The focused validator/non-finite suite passes 20 tests.
- A two-epoch tiny-face equivalence test confirms that `both` and `offload`
  produce bitwise-identical EMA snapshots and that validating the snapshot
  later reproduces inline one-step and autoregressive scalar metrics exactly.
- The full rental and validator argument sets resolve to identical guarded
  validation signatures.
- The clean face/distributed regression selection passes 101 tests (one
  CUDA-only test deselected on the CPU login node).
- The Python modules compile and all four trainer/validator job scripts pass
  `bash -n`.
- No H100/H200 validation or training job was submitted by this change.
