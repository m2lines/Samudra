<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2 single-scale architecture search

## Status

This is a living lab notebook for the first full-training search over the
Perceiver v2 SamudraMulti architecture. The search is running on the public
2-degree OM4 dataset. This document records the questions, hypotheses, and
experimental design before inspecting the validation results. Results,
discussion, conclusions, and future work will be completed after the search.

The immutable run is
`perceiver-v2-2deg-architecture--20260814T171003.874785Z`. Its code revision is
[`6bac8ff4`](https://github.com/m2lines/Samudra/tree/6bac8ff4f2acb1edddcf184f1cbd9cfe0f00a762),
and its artifacts are published under the
[`m2lines-pubs` search directory](https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/).

## Research questions

The long-term objective is a Perceiver-based SamudraMulti that can share an
architecture across ocean datasets at multiple resolutions, eventually
including much larger LLC grids. Before testing weight sharing across scales,
this experiment asks whether the model can learn a strong predictor on one
small, real ocean dataset and which parts of its input/output bottlenecks matter
most.

The primary question is:

> Which Perceiver encoder/decoder design reaches the lowest autoregressive
> validation loss for a fixed amount of optimization on 2-degree OM4 data?

The supporting questions are:

1. Does querying processed spatial tokens directly outperform the original
   full Perceiver IO decoder, which introduces a second learned latent bank?
2. How much encoder compression is useful? Can 64 patch latents match or beat
   the 256-latent control?
3. Is the direct decoder's value-transport width a limiting bottleneck for 77
   prognostic channels?
4. How much neighboring patch context is needed during decoding?
5. Does a coarser physical patch decomposition improve learning by reducing
   the processor's latent-grid size, or does it discard important local
   structure?
6. Are apparent architecture improvements stable across two plausible learning
   rates, or are they optimizer-specific?
7. Can the search system reject non-training jobs early, rank candidates
   reproducibly, preserve enough evidence to diagnose failures, and usefully
   allocate more compute to promising models?

## Hypotheses

These hypotheses were recorded before the search results were available.

### H1: remove the second decoder bottleneck

The direct output-query decoder will outperform the full Perceiver IO decoder.
After the encoder and spatial processor have constructed useful tokens, another
learned latent bank is an unnecessary routing bottleneck. This is the principal
architectural hypothesis, motivated by the decoder root-cause investigation and
the earlier real-data patch probe in
[`perceiver_v2_direct_decoder.md`](perceiver_v2_direct_decoder.md).

### H2: preserve encoder capacity until the decoder is healthy

Reducing the encoder from 256 to 64 latents will make the model cheaper, but is
more likely to hurt validation loss than help it. A favorable result would show
that the existing encoder is over-parameterized for this patch size and provide
a promising path toward cheaper high-resolution training.

### H3: decoder transport width is under-sized

Increasing direct decoder cross-attention from two to four 64-dimensional heads
(128 to 256 transported values) will improve validation loss. Reducing it to
one head (64 values) should hurt. A monotonic 64 -> 128 -> 256 trend would be
especially strong evidence that information transport, rather than decoder
latent depth, is limiting the model.

### H4: some spatial context is necessary, but more is not always better

Removing neighboring context will hurt because ocean dynamics and patch-edge
predictions depend on nearby state. Expanding from one to two context rings may
help, but could dilute local attention or add enough computation that the gain
is not worthwhile. The expected ordering is two rings approximately one ring,
with both better than zero rings.

### H5: finer patches will retain more useful structure

The control's 6 x 10 degree physical patches will outperform 10 x 20 degree
patches, despite the latter's smaller latent grid and lower processor cost. If
the coarse-patch candidate wins, it would indicate that reducing spatial-token
count is more valuable at 2 degrees than retaining fine patch-local structure.

### H6: good architectural effects should survive the learning-rate pair

The ranking of the strongest architectural families should be broadly similar
at learning rates `4e-4` and `8e-4`. A candidate that wins at only one rate is a
useful optimization clue, but weaker architectural evidence than a family that
performs well at both.

## Experimental design

### Common model and training setup

All candidates use SamudraMulti's physical-grid -> patch-local Perceiver
encoder -> spatial ConvNeXt U-Net processor -> query decoder -> physical-grid
path. The common model is defined in
[`model.yaml`](../../src/samudra/configs/perceiver_search_2deg/model.yaml), and
the complete candidate matrix and promotion policy are defined in
[`search.yaml`](../../src/samudra/configs/perceiver_search_2deg/search.yaml).

Unless explicitly varied below, every candidate shares:

- the same public 2-degree OM4 source, train/validation split, and normalization;
- random seed 15;
- four-step autoregressive training and validation;
- batch size 1 with 32 batches accumulated per optimizer update;
- MSE loss, no learning-rate scheduler, and no residual prediction;
- a 128-dimensional embedding and the same ConvNeXt U-Net processor;
- 256 encoder latents of dimension 64;
- 6 x 10 degree physical patches;
- the direct decoder with 128-dimensional queries, two 64-dimensional
  cross-attention heads, six output-window patches, and one context ring; and
- Samudra-owned attention blocks using PyTorch scaled dot-product attention
  with automatic backend selection.

At 2-degree resolution, the control patch covers 3 x 5 grid cells and produces
a 30 x 36 processor grid (1,080 spatial tokens). The coarse 10 x 20 degree
candidate covers 5 x 10 cells and produces an 18 x 18 grid (324 tokens). With
six window patches, zero, one, and two context rings expose each output query to
6 x 6 = 36, 8 x 8 = 64, and 10 x 10 = 100 processor tokens respectively.

The relevant implementations are:

- [native scaled dot-product attention and Perceiver blocks](../../src/samudra/models/modules/perceiver.py);
- [patch-local Perceiver encoder](../../src/samudra/models/modules/encoder.py);
- [direct and full Perceiver IO decoders](../../src/samudra/models/modules/decoder.py); and
- [SamudraMulti encoder/processor/decoder composition](../../src/samudra/models/samudra_multi.py).

### Interventions

There are nine architectural hypotheses, each run at `4e-4` and `8e-4`, for 18
initial candidates.

| Family | Architectural intervention | Purpose |
| --- | --- | --- |
| `direct-control` | Direct decoder; 256 encoder latents; transport width 128; one context ring; 6 x 10 degree patches | Reference for all single-factor direct-decoder ablations |
| `direct-enc64` | Encoder latents 256 -> 64 | Test whether encoder compression is excessive or beneficial |
| `direct-transport64` | Decoder heads 2 -> 1; transport width 128 -> 64 | Test a narrower output information path |
| `direct-transport256` | Decoder heads 2 -> 4; transport width 128 -> 256 | Test a wider output information path |
| `direct-no-context` | Decoder context rings 1 -> 0 | Measure whether neighboring patches are necessary |
| `direct-context2` | Decoder context rings 1 -> 2 | Test a larger local receptive field |
| `direct-coarse-patch` | Patch extent 6 x 10 -> 10 x 20 degrees | Trade local detail for a 70% smaller processor-token grid |
| `pio-lean` | Full Perceiver IO decoder; 64 decoder latents; query dimension 128 | Compare direct decoding with a reduced second latent bottleneck |
| `pio-control` | Full Perceiver IO decoder; 256 decoder latents; query dimension 64 | Reproduce the original large decoder bottleneck as an architectural control |

Each family has `-lr4` and `-lr8` candidates. These paired rates reduce the risk
of selecting an architecture solely because the control rate was unfavorable.
They are not independent random seeds and must not be interpreted as replicates
for statistical significance.

### Compute allocation and promotion

Successive halving trains candidates to cumulative budgets of 1, 3, 6, and 12
epochs. At each boundary, candidates are ranked by validation loss and the best
half advance, with at least two candidates retained. All candidates therefore
receive the inexpensive first epoch, while later compute is concentrated on
the most promising configurations.

Each worker requests one RTX 6000 GPU, four CPUs, and 32 GiB of host memory. Up
to eight workers run concurrently. A disposable preflight candidate must load
real data, complete 32 microbatches, and perform an optimizer step before the
full first rung is released. A candidate is ineligible for promotion unless it
reports verified training progress.

The promotion objective is validation MSE after the candidate's current
cumulative epoch budget. We will also inspect train loss, best validation loss,
learning-curve shape, optimizer-step count, runtime, and failure state. The
first rung is primarily a pruning signal: small differences after one epoch
should not be treated as final model-quality estimates.

## Reproducing the observations with DuckDB

The search publishes JSON and Parquet artifacts to a public object-store
prefix. The following queries run in DuckDB and are deliberately tied to the
questions above. During an active rung, `results.parquet` and `epochs.parquet`
may not exist until the controller reaches a publication boundary. W&B remains
the live per-batch view; the public Parquet files are the durable comparison
record.

Start DuckDB and load its HTTP and JSON extensions:

```sql
INSTALL httpfs;
LOAD httpfs;
INSTALL json;
LOAD json;
```

### Did the search actually train?

This checks the controller state, optimizer-step gate, and first array job.

```sql
SELECT
    status,
    created_at,
    provenance.commit AS code_commit,
    rungs[1].epochs AS first_rung_epochs,
    len(rungs[1].candidates) AS first_rung_candidates,
    rungs[1].probe.status AS probe_status,
    rungs[1].probe.optimizer_steps AS probe_optimizer_steps,
    rungs[1].job_id AS first_rung_array_job
FROM read_json_auto(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/state.json'
);
```

The preflight lifecycle provides a second, lower-level correctness check. A
valid probe must reach `optimizer_step` and then `completed`.

```sql
SELECT
    stage,
    candidate,
    batches_seen,
    optimizer_steps,
    updated_at,
    list_transform(history, event -> event.stage) AS lifecycle
FROM read_json_auto(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/probe/direct-control-lr4/search_worker_status.json'
);
```

### Which candidates are best at the latest completed rung?

This is the primary model-selection query. Ineligible or failed workers remain
visible but sort below candidates with finite validation loss.

```sql
SELECT
    candidate,
    rung,
    epochs,
    eligible,
    validation_loss,
    train_loss,
    best_validation_loss,
    optimizer_steps,
    round(train_seconds / 60, 2) AS train_minutes,
    round(validation_seconds / 60, 2) AS validation_minutes,
    worker_stage,
    error
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/results.parquet'
)
ORDER BY rung DESC, eligible DESC, validation_loss ASC NULLS LAST;
```

### Which architectural families are robust to learning rate?

This removes the learning-rate suffix and compares the two members of each
family at their latest common completed rung. A low mean with a small range is
stronger evidence than one unusually good member and one poor member.

```sql
WITH results AS (
    SELECT
        regexp_replace(candidate, '-lr[48]$', '') AS family,
        candidate,
        rung,
        validation_loss
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/results.parquet'
    )
    WHERE eligible AND validation_loss IS NOT NULL
),
latest_shared_rung AS (
    SELECT max(rung) AS rung
    FROM (
        SELECT rung
        FROM results
        GROUP BY rung, family
        HAVING count(*) = 2
    )
)
SELECT
    family,
    rung,
    count(*) AS learning_rates_present,
    avg(validation_loss) AS mean_validation_loss,
    min(validation_loss) AS best_validation_loss,
    max(validation_loss) - min(validation_loss) AS learning_rate_range
FROM results
WHERE rung = (SELECT rung FROM latest_shared_rung)
GROUP BY family, rung
HAVING count(*) = 2
ORDER BY mean_validation_loss;
```

Successive halving may promote only one member of a pair after the first rung.
For the fairest complete-family comparison, explicitly use `WHERE rung = 0`
after rung zero has completed.

### How quickly does each candidate learn?

This supplies the learning curves needed to distinguish a poor architecture
from a promising candidate that is merely learning more slowly.

```sql
SELECT
    candidate,
    rung,
    epoch,
    train_loss,
    validation_loss,
    "progress/optimizer_steps" AS optimizer_steps,
    round(epoch_train_seconds / 60, 2) AS train_minutes,
    round(epoch_validation_seconds / 60, 2) AS validation_minutes
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/epochs.parquet'
)
ORDER BY candidate, epoch;
```

### Which interventions buy skill efficiently?

This compares the best observed loss with total GPU-worker time. It is useful
for identifying an intervention that is slightly better but disproportionately
expensive, especially the larger decoder context and transport-width variants.

```sql
SELECT
    candidate,
    max(epoch) AS cumulative_epochs,
    min(validation_loss) AS best_validation_loss,
    round(
        sum(epoch_train_seconds + epoch_validation_seconds) / 60,
        2
    ) AS observed_minutes,
    round(
        sum(epoch_train_seconds + epoch_validation_seconds)
            / nullif(max("progress/optimizer_steps"), 0),
        3
    ) AS seconds_per_optimizer_step
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/epochs.parquet'
)
GROUP BY candidate
ORDER BY best_validation_loss;
```

### What evidence is available for deeper diagnosis?

This inventories public metrics, reports, and logs before attempting a
postmortem or launching a follow-up evaluation.

```sql
SELECT
    kind,
    candidate,
    rung,
    artifact,
    bytes,
    public_url
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/artifacts.parquet'
)
WHERE kind IN ('log', 'metrics', 'report', 'checkpoint')
ORDER BY kind, candidate, rung, artifact;
```

## Interpretation guardrails

- Architecture families are the unit of inference; learning-rate members are a
  sensitivity check, not statistical replicates.
- Rung-zero rankings are noisy. Prefer conclusions that persist at larger
  cumulative budgets and across both learning rates.
- Successive halving censors learning curves for pruned candidates. A model
  that starts slowly cannot be shown to be bad at 12 epochs by this search.
- Validation MSE is appropriate for rapid architectural screening, but it does
  not establish long-rollout stability, regional skill, conservation, spectral
  fidelity, or performance at other resolutions.
- Parameter count and wall time should accompany loss comparisons. The goal is
  not merely the lowest 2-degree loss, but an architecture with a credible path
  to much larger grids.
- Any winning configuration should be rerun with multiple seeds before it is
  treated as a durable architecture improvement.

## Results

_Pending completion of the search._

## Discussion

_Pending analysis of completed-rung results, learning curves, and diagnostic
artifacts._

## Conclusions

_Pending._

## Future work

_Pending._
