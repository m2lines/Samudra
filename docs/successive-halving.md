<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Successive-halving architecture searches

`samudra search` allocates progressively larger epoch budgets to promising model
configurations. Its search models, validation, ranking, state transitions, and
CLI live in `samudra.search`. The small
`scripts/successive_halving_worker.sbatch` file is only the deployment adapter
between Slurm and the package. The system implements one synchronous,
Slurm-backed successive-halving bracket. It is a useful foundation for
Hyperband, but does not yet launch the multiple brackets with different initial
budgets that define the full Hyperband algorithm.

Each candidate is an immutable training environment: a config plus either a
commit-built Apptainer code layer or a pinned container image. Each array task
runs through the normal `slurm_apptainer_train.sbatch` harness. At the end of an
epoch, Samudra atomically writes `training_summary.json` beside the run. The
promotion job reads those summaries from the shared filesystem, ranks only
complete finite results, and submits the next array.

W&B remains the place for curves and interactive analysis. A candidate keeps
the same W&B run when promoted, because its checkpoint carries the W&B ID. The
W&B config records the search name, manifest checksum, orchestrator and
candidate commits, candidate name, rung, target budget, and parent checkpoint;
numeric rung and target-budget markers are also logged. Promotion itself reads
the local summary and therefore does not depend on W&B or network availability.

## Search lifecycle

For rungs `[1, 3, 6, 12]`, a promoted model trains to one total epoch, resumes
the complete optimizer and EMA state to three total epochs, then similarly to
six and twelve. A new output directory is used at every rung, while the
checkpoint's W&B ID resumes the candidate's existing W&B run.

Schedulers must be invariant to extending the configured final epoch. A null or
step-based scheduler is safe. A schedule computed from `epochs` at construction
can change meaning between rungs and should not be used without an explicit
promotion-aware design.

Candidates marked `fixed: true` are reference anchors. They start at the full
budget in a separate array and therefore do not delay early promotion rungs or
consume the promotion quota. The final report waits for both the promoted
finalists and anchors. Failed, incomplete, non-finite, or checkpoint-less
candidates are never eligible.

## Manifest

Start from `scripts/successive_halving.example.yaml`. Use full commit hashes
with code layers. The worker checks that the code-layer manifest contains the
declared commit before training.

Important fields:

- `rungs` are cumulative total epochs, not additional epochs.
- `promotion_fraction` applies only to non-fixed candidates.
- `metric` names a finite numeric top-level field in `training_summary.json`.
- `max_concurrent` limits simultaneous array tasks; Slurm may run fewer.
- `time_by_rung` supplies a walltime for each rung; anchors use the final one.
- `runtime.train_harness` and all layer/config paths must exist on the cluster.
- `runtime.worker_harness` is the thin Slurm adapter; it is copied into the
  immutable search bundle before submission.
- `args` are ordinary Samudra CLI overrides. The controller owns `epochs`,
  `resume_ckpt_path`, output name, and W&B group.

Validate and preview the maximum population at each rung:

```bash
samudra search plan search.yaml
```

Perform a submission dry run. This creates a self-contained state bundle and
prints the `sbatch` commands without submitting them:

```bash
samudra search start search.yaml \
  --state-root=/scratch/$USER/searches \
  --dry-run
```

Launch the search by omitting `--dry-run`. The bundle contains an immutable
manifest copy, controller snapshot and checksum, worker, `state.json`, Slurm
logs, and one CSV leaderboard per completed rung. The state records the
orchestrator Git commit, package version, dirty status, controller/worker hashes,
candidate commits, checkpoint lineage, Slurm job IDs, optimizer steps, and W&B
IDs. By default `start` refuses a controller checkout with tracked changes.
`--allow-dirty` exists for deliberate development probes, but the bundle hash
makes that weaker provenance explicit. Re-running `advance` for an already
advanced rung fails loudly so a retry cannot duplicate promotions.

The bundled controller is what delayed promotion jobs execute, so a checkout or
package update cannot silently change a running search. For a wheel or container
without Git metadata, set `SAMUDRA_CODE_COMMIT` to the immutable package commit;
otherwise `start` refuses to launch.

## Experimental use

A five-batch debug run is still appropriate before entering candidates into a
search, but it measures compatibility and memory rather than model quality. A
typical first bracket carries all candidates through one full epoch, promotes
roughly half to three epochs, and reserves rollout validation for the later
rungs. Durable multi-year observation metrics belong after finalist training,
not in the high-throughput inner loop.
