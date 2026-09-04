<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Architecture search

Training every model idea to convergence wastes compute: weak candidates often
become distinguishable after only a small amount of training. Samudra's search
runner uses successive halving to train all candidates with a small initial
budget, retain the best fraction, and give progressively larger budgets only to
the survivors. This is useful for comparing model architectures as well as
ordinary hyperparameters such as learning rate and batch size.

Successive halving is the resource-allocation primitive used by
[Hyperband](https://jmlr.org/papers/v18/16-558.html). Samudra currently runs one
successive-halving bracket. It does not yet run Hyperband's collection of
brackets with different exploration-versus-training tradeoffs.

## Run a search

Copy the bundled example and Torch executor configuration from
`src/samudra/configs/search/`, then edit the output paths, Slurm account, and
candidate list. Build the configured code layer from that same committed
revision. The runner verifies the layer's manifest before submitting work, and
completed training results must report the same commit before promotion. All
candidate models must therefore exist in one experiment branch before the
search is submitted.

Run the entire search with one command:

```bash
python -m samudra.search path/to/search.yaml
```

The runner submits the first rung and fixed baselines. On Slurm, dependent
controller jobs automatically rank completed candidates and submit each later
rung; users do not manually advance the search. For a local laptop,
workstation, or Colab notebook, include `search/local.yaml` instead of
`search/torch.yaml`. The local executor runs one isolated candidate process per
visible GPU and never interprets scheduler environment variables. With no GPU
it runs candidates sequentially in the controller process. It is also the
simplest choice inside a single-node Slurm allocation when the batch shell sees
all allocated GPUs; use `slurm_allocation` when trials must reach other nodes.
Fixed anchors and rung zero are co-scheduled, and each later rung uses up to the
same GPU capacity. Set `executor.max_concurrent` only when memory, CPU, storage,
or service limits require using fewer than the available GPUs.

By default, every candidate remains a single-GPU experiment in every rung. To
assign newly idle GPUs to survivors without changing the scientific training
trajectory, opt in to adaptive data parallelism:

```yaml
resources:
  strategy: adaptive_data_parallel
  max_gpus_per_candidate: 16
  effective_global_batch_size: 64
  allowed_world_sizes: [1, 2, 4, 8, 16]
```

For each rung, the executor selects the largest allowed world size that fits
the allocation and the number of candidates allowed to run concurrently. It
keeps each candidate's configured `batch_size`, adjusts only
`gradient_accumulation_steps`, and uses a fixed-global-batch sampler. Thus the
same shuffled examples form each optimizer update even when a promoted
candidate moves from one GPU to two, four, or more. Learning-rate and scheduler
configuration remain unchanged because the effective global batch and number
of optimizer updates per epoch remain unchanged. The selected plan is recorded
in `state.json` and reused on retries.

Choose `effective_global_batch_size` from the scientific baseline you want to
preserve: `batch_size * gradient_accumulation_steps * world_size`. For example,
a prior two-GPU run with local batch 1 and accumulation 16 has an effective
global batch of 32, even though its single-process configuration alone appears
to describe a batch of 16.

`effective_global_batch_size` must be divisible by
`batch_size * world_size`. If the requested world size is incompatible, the
runner warns, preserves the user's batch size, and selects the largest smaller
compatible world size. The warning recommends a compatible local batch size;
Samudra never silently overrides that scientific choice. Fixed anchors stay on
one GPU, using accumulation to preserve the configured effective global batch.
Because resource plans are persisted, retry a local run with at least as many
visible GPUs as its largest planned candidate. The executor fails loudly
instead of waiting indefinitely if a retry cannot satisfy that plan.

Adaptive data parallelism is supported by the `local` and
`slurm_allocation` executors. The separately submitted `slurm` executor rejects
it because its array jobs do not share one allocation. Adaptive candidate
training configs must use `backend: auto`, allowing the same saved candidate to
run either as a single process or through the existing distributed
initialization path.

On homogeneous multi-node Slurm allocations, ranks are spread uniformly over
the smallest number of nodes that can host them. For example, an eight-rank
trial on two six-GPU nodes runs four ranks per node. If a custom allowed world
size has no uniform placement, planning warns and selects the next smaller
placeable size rather than failing after workers are submitted.

### Use a whole Slurm allocation for independent trials

Include `search/slurm-allocation.yaml` to treat an existing Slurm job as a
resource pool. This distinct executor launches concurrent, exclusive `srun`
steps. A step uses one GPU by default or the planned GPU group when adaptive
data parallelism is enabled. On a homogeneous multi-node allocation it
derives total capacity from `SLURM_GPUS` or from
`SLURM_NNODES * SLURM_GPUS_ON_NODE`; CPU cores are divided evenly among GPUs
when Slurm exposes `SLURM_CPUS_ON_NODE`. Step memory is divided proportionally
from `SLURM_MEM_PER_NODE` when Slurm exposes it, so one worker does not reserve
the entire allocation's memory and serialize the GPU pool. It fails loudly
outside an allocation.

For example, Empire AI Alpha+ has eight H100 or H200 GPUs per HGX node. A batch
allocation shaped like the following lets sixteen candidates occupy two nodes:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=<all CPUs on one node>
#SBATCH --mem=0

module load <modules-needed-by-samudra>
source <shared-x86-environment>/bin/activate
python -m samudra.search path/to/search.yaml
```

Launch the controller directly from the batch script, as above. Do not wrap it
in `srun`: the executor needs to create its own job steps. The source tree,
Python executable, candidate configs, data, and output directory must be
visible at the same paths on every node. Alpha is x86_64; an environment built
for Alpha must not be reused on Empire's ARM systems. Use an architecture-
appropriate environment or multi-architecture container for those systems.

Before relying on allocation-wide pooling, verify that the target cluster
admits concurrent exclusive job steps. Some Slurm installations serialize
nested `srun` steps even when their GPU, CPU, and memory requests do not
overlap. From an interactive allocation, a useful smoke check is to start two
exclusive one-GPU `srun` commands concurrently and confirm with `squeue --steps`
that both enter `RUNNING`. This scheduler policy cannot be inferred from the
allocation environment. On a single node where the batch shell sees every
allocated GPU, use the `local` executor if concurrent job steps are unavailable.

The Slurm-allocation executor is synchronous: the allocation remains active
through every rung, and a worker failure stops promotion after the other
running workers have exited. For separately queued, resumable jobs and
controller dependencies, use the regular Slurm executor instead.

Clusters that expose the container runtime through environment modules should
set `executor.apptainer_module` to the exact loadable module name (for example,
`singularity-ce/4.3.3` on Torch). The resolved value is exported to every worker
and retained in the search configuration.

Every invocation gets a readable instance identifier such as
`perceiver-2deg--20260813T192612.123456Z`. The stable `name` describes the
experiment design; this generated `run_id` names its filesystem, object-store,
Slurm, and W&B resources so repeated trials cannot collide. Set `run_id`
explicitly only when an external system needs to allocate the identity. The
runner still refuses to overwrite an existing local or published instance.

The uniquely named search directory contains:

- `config.yaml`: the fully resolved, validated search configuration;
- `candidates/`: one fully resolved training configuration per candidate;
- `state.json`: internal resumable scheduler state;
- `results.csv` and `results.parquet`: one analysis-ready row per candidate and
  rung, including all requested metrics, timing, checkpoint lineage, W&B
  identity, and job ID;
- `epochs.parquet`: full epoch-level training, validation, inference, timing,
  throughput, and variable/depth/channel metrics from every completed run;
- `artifacts.parquet`: hashes, sizes, media types, and public/queryable locations
  for every published artifact;
- `analysis/report.md`: a continuously updated leaderboard, rung history,
  outcome (once available), and explicit candidate failures;
- `logs/`: scheduler output when using Slurm.

### Fail-fast worker probes

For costly Slurm searches, set `executor.rung0_probe: true`. Before releasing
the first candidate array, the executor runs the first candidate through the
real data loader, model forward/backward path, and one optimizer update with
W&B disabled. The bulk array is submitted only when the probe records a finite
training batch and at least one optimizer step. A missing or failed probe marks
the search terminal, publishes its diagnostics, and consumes no candidate or
fixed-anchor array allocation. Fixed anchors are submitted only after the same
probe succeeds.

Every search-managed training process atomically maintains
`search_worker_status.json` with its lifecycle history:
`launched`, `initialized`, `first_batch`, first `optimizer_step`, and
`completed` or `failed`. Events include batch/optimizer counts, loss and loader
timings when available, Slurm identity, and an error type/message on failure.
These files are copied into the public research record, and their latest stage
is included in failed result rows and reports. A scheduler process merely starting is
therefore not treated as evidence that training occurred.

`results.csv` is deliberately denormalized and readable directly with pandas:

```python
import pandas as pd

results = pd.read_csv(
    "/scratch/USER/searches/my-search--20260813T192612.123456Z/results.csv"
)
print(results.sort_values(["rung", "validation_loss"]))
```

## Publish an inspectable research record

Artifact publication is opt-in and independent of the executor. Local and
Slurm searches first create the same record on their working filesystem, then
an optional publisher mirrors it to another local directory or any
S3-compatible object store. This keeps compute scheduling separate from where
research results are retained.

For the public m2lines OSN pod, uncomment the following line in a copied search
config:

```yaml
artifacts: !include osn-artifacts.yaml
```

The packaged template publishes under
`s3://m2lines-pubs/Samudra/experiments/searches/<run-id>/`. It does not contain
credentials. On every machine that may run the search controller, provide
write credentials through the normal environment:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# Only when the credentials are temporary:
export AWS_SESSION_TOKEN=...
```

Slurm submissions inherit these variables; secrets are neither embedded in
the search config nor written to its artifacts. The S3 destination carries the
OSN endpoint explicitly. Publication is required once configured: an upload
failure stops promotion loudly instead of silently creating an incomplete
public record.

The published record includes resolved search and training configs, Git
provenance, scheduler state, per-rung outcomes (including failures), a Markdown
summary report, full epoch histories, structured worker status, W&B identities,
and an SHA-256 inventory.
`results.parquet` is created and published with a stable empty schema before
jobs are submitted, so its public URL can be queried immediately rather than
returning 404 while rung zero is running.
Search-level or per-run analysis hooks have a simple artifact contract: write
their tables, reports, or figures beneath an `analysis/` directory and the
publisher includes them automatically with their hashes and locations. This
keeps future diagnostics independent of Local versus Slurm execution.
Checkpoint publication is configurable:

- `none` publishes no model weights;
- `final` (the default) publishes the best-validation checkpoint from each
  successful final-rung run and fixed baseline (falling back to its latest
  checkpoint when necessary);
- `all` publishes every saved checkpoint from every completed rung.

Raw scheduler stdout/stderr and `experiment.log`/`error.log` are excluded from
artifact publication by default because jobs inherit credentials and arbitrary
process output is not safe to mirror into a public bucket. Set
`artifacts.logs: all` only for a destination with an appropriate access policy
and after auditing the workload's logging. Structured worker errors redact
credential-like environment values before publication.

The local filesystem always retains the checkpoints required for promotion.
`final` is normally enough to reproduce a promising model and run deferred
rollout diagnostics without paying object-storage costs for every eliminated
candidate.

The Parquet tables can be queried in place by agents or collaborators. For a
public HTTP endpoint, DuckDB needs no local download:

```sql
SELECT candidate, rung, epochs, validation_loss, error
FROM read_parquet(
  'https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/searches/my-search--20260813T192612.123456Z/results.parquet'
)
ORDER BY rung, validation_loss;
```

`epochs.parquet` supports deeper questions such as which model learned fastest,
which variable or depth stalled, whether throughput or optimizer-step counts
differed, and where divergence began. `artifacts.parquet` lets an agent locate
and verify the exact config, logs, or checkpoint behind any row.

These tables are intentionally single Parquet files, not shards. Their row
counts grow with candidates times rungs or candidates times epochs, rather than
with the ocean dataset, and are expected to remain small compared with one
checkpoint. A future search large enough to need sharding can publish a
partitioned Parquet dataset with the same columns and query interface; no
current workload benefits from that complexity.

### Deferred model diagnostics

Search currently records and publishes the inputs needed for later analysis
rather than coupling training to a particular scientific metric suite. Future
post-search jobs can load published checkpoints, generate matched validation
rollouts, and write tables or figures beneath `analysis/`. Keeping that work
separate preserves fast promotion decisions while leaving room to add the most
informative diagnostics after experience with real searches.

## Configure candidates and metrics

Search configuration uses the same Pydantic/YAML system as training and eval,
including `!include`, packaged presets, command-line overrides, generated JSON
schemas, and `@config.yaml` values.

```yaml
name: perceiver-2deg

algorithm:
  type: successive_halving
  rungs: [1, 3, 6, 12]  # cumulative total epochs
  promotion_fraction: 0.5
  minimum_promoted: 1

objective: {metric: validation_loss, mode: min}
metrics: [validation_loss, train_loss, best_validation_loss]

executor: !include torch.yaml

candidates:
  - name: control
    fixed: true
    config: experiments/control/train.yaml
  - name: sdpa-perceiver
    config: experiments/sdpa/train.yaml
    args: [--batch_size=2, --learning_rate=0.0006]
```

The objective is the single metric used for promotion. `metrics` controls the
additional columns collected in `results.csv`. Any finite scalar emitted by
training, validation, or inference can be named here, including namespaced
diagnostics such as `val/seam/window_jump_ratio/zos`. A missing or non-finite
metric, an incomplete epoch budget, or a missing checkpoint makes that result
ineligible.

`minimum_promoted` is a floor, so promotion can intentionally stop reducing the
candidate pool once it reaches that size. It cannot exceed the number of
non-fixed candidates configured for the search; if worker failures leave fewer
eligible candidates, every eligible candidate advances.

A search is `complete` only when every scheduled candidate result across all
rungs and fixed anchors is eligible. If the search can continue after one or
more worker failures, its terminal status is `partial`; the report identifies
the best completed finalist without presenting it as an uncontested winner. A
rung with no eligible candidates is `failed`.

Candidates marked `fixed` are reference baselines. They run independently at
the largest budget and do not consume promotion slots. Rung budgets are total
epochs, not extra epochs: a survivor resumes its complete checkpoint from epoch
1 to epoch 3, then from epoch 3 to epoch 6. When a candidate's scheduler omits
`target_epochs`, the search snapshots the largest rung as the common scheduler
horizon for every rung and fixed anchor. This prevents a resumed cosine
scheduler from restoring the first rung's short `T_max`. An explicitly
configured `target_epochs` is preserved.

Controller jobs inherit the verified search commit explicitly rather than
requiring an editable Git checkout in the controller environment. Their
partition, CPU count, memory, and walltime are configurable with
`controller_partition`, `controller_cpus_per_task`, `controller_memory`, and
`controller_time`. Published-object hashes are recorded locally after each
successful upload, so retries skip unchanged multi-gigabyte checkpoints.

The resumable `state.json` contract is versioned and validated on every read
and write. A malformed or stale state therefore fails at the controller
boundary with a schema error instead of producing a later key error.

## W&B

Each candidate uses the unique search run ID as its W&B group and receives the
tags `search`, the stable search name, and the candidate name. The timestamped
run ID is deliberately omitted from tags because W&B limits tags to 64
characters; it remains available as the group and in structured search
metadata. This makes repeated trials separately filterable while preserving a
stable tag for cross-run comparisons. Search identity, rung, objective, epoch
budget, executor, job ID, parent checkpoint, and the public artifact root are
stored under `config.experiment.search`.
Promoted rungs resume the same W&B run from the checkpoint, preserving one
continuous learning curve per candidate.

W&B is used for curves and interactive comparison. Promotion reads the local
training summary so temporary W&B or network failures do not control scheduling.

## Python API

`SearchConfig.from_yaml_and_cli()` loads and validates configuration,
`config.build()` selects the configured search algorithm, and `search.start()`
submits or runs it. Internal worker entry points are implementation details used
by executors.

Compute-specific logic lives in `samudra.search.executors`. Built-in executor
classes are selected by a small dictionary in `successive_halving.py`. Adding a
future Empire AI executor requires implementing the same `submit_anchors` and
`submit_rung` interface and adding one dictionary entry; no plugin registration
system is imposed today. The search algorithm has a separate typed-config and
factory boundary. A future Hyperband implementation can coordinate several
successive-halving brackets through the existing executor submissions and use
the algorithm-neutral artifact publisher. A scheduler that is not rung-based
may require broadening the executor task interface; the current API does not
pretend every possible explore/exploit strategy already fits it.

::: samudra.search

::: samudra.search.config

::: samudra.search.successive_halving
