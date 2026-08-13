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
rung; users do not manually advance the search. For a local laptop or Colab
notebook run, include `search/local.yaml` instead of `search/torch.yaml`;
candidates then run sequentially in the current environment.

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
- `logs/`: scheduler output when using Slurm.

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
`s3://m2lines-pubs/FOMO/experiments/searches/<run-id>/`. It does not contain
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
provenance, scheduler state, per-rung outcomes (including failures), full epoch
histories, experiment/error logs, W&B identities, and an SHA-256 inventory.
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

The local filesystem always retains the checkpoints required for promotion.
`final` is normally enough to reproduce a promising model and run deferred
rollout diagnostics without paying object-storage costs for every eliminated
candidate.

The Parquet tables can be queried in place by agents or collaborators. For a
public HTTP endpoint, DuckDB needs no local download:

```sql
SELECT candidate, rung, epochs, validation_loss, error
FROM read_parquet(
  'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/my-search--20260813T192612.123456Z/results.parquet'
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
additional columns collected in `results.csv`. A missing or non-finite metric,
an incomplete epoch budget, or a missing checkpoint makes that result
ineligible.

Candidates marked `fixed` are reference baselines. They run independently at
the largest budget and do not consume promotion slots. Rung budgets are total
epochs, not extra epochs: a survivor resumes its complete checkpoint from epoch
1 to epoch 3, then from epoch 3 to epoch 6. Use a scheduler whose meaning does
not change when the target epoch increases; `scheduler: null` is appropriate
for short architecture comparisons.

## W&B

Each candidate uses the unique search run ID as its W&B group and receives the
tags `search`, the stable search name, the run ID, and the candidate name. This
makes repeated trials separately filterable while preserving a stable tag for
cross-run comparisons. Search identity, rung, objective, epoch budget,
executor, job ID, parent checkpoint, and the public artifact root are stored
under `config.experiment.search`.
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
