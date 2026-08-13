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
rung; users do not manually advance the search. For a local laptop run, include
`search/local.yaml` instead of `search/torch.yaml`; candidates then run
sequentially in the current environment.

The search directory contains:

- `config.yaml`: the fully resolved, validated search configuration;
- `candidates/`: one fully resolved training configuration per candidate;
- `state.json`: internal resumable scheduler state;
- `results.csv`: one analysis-ready row per candidate and rung, including all
  requested metrics, timing, checkpoint lineage, W&B identity, and job ID;
- `logs/`: scheduler output when using Slurm.

`results.csv` is deliberately denormalized and readable directly with pandas:

```python
import pandas as pd

results = pd.read_csv("/scratch/USER/searches/my-search/results.csv")
print(results.sort_values(["rung", "validation_loss"]))
```

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

Each candidate uses the search name as its W&B group and receives the tags
`search`, the search name, and the candidate name. This makes runs filterable by
group or tag in W&B. Search identity, rung, objective, epoch budget, executor,
job ID, and parent checkpoint are stored under `config.experiment.search`.
Promoted rungs resume the same W&B run from the checkpoint, preserving one
continuous learning curve per candidate.

W&B is used for curves and interactive comparison. Promotion reads the local
training summary so temporary W&B or network failures do not control scheduling.

## Python API

`SearchConfig.from_yaml_and_cli()` loads and validates configuration.
`build_search()` selects the configured search algorithm, and `search.start()`
submits or runs it. Internal worker entry points are implementation details used
by executors.

Compute-specific logic lives in `samudra.search.executors`. Built-in executor
classes are selected by a small dictionary in `successive_halving.py`. Adding a
future Empire AI executor requires implementing the same `submit_anchors` and
`submit_rung` interface and adding one dictionary entry; no plugin registration
system is imposed today. The search algorithm similarly has a single factory
boundary where another strategy can be added later.

::: samudra.search

::: samudra.search.config

::: samudra.search.successive_halving
