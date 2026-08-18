<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2 single-scale architecture search

## Status

This is the lab notebook for the first full-training search over the Perceiver
v2 SamudraMulti architecture on the public 2-degree OM4 dataset. The questions,
hypotheses, and experimental design below were recorded before inspecting
validation results. Successive halving completed through its 12-epoch budget,
although a cluster-side cancellation initially interrupted two of the three
finalists. Both interrupted runs were subsequently resumed from their last
complete checkpoints and reached epoch 12. The controller and public artifacts
were reconciled on 2026-08-18, so all three finalists now have directly
comparable results at the full budget.

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
    worker_error AS error
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

### Preliminary rung zero

All 18 W&B workers completed epoch one and 89 optimizer updates. The values in
this section were read from their finished W&B summaries on 2026-08-14. At that
time, the public controller state still reported `running`, contained no validated
rung results, and had not published `results.parquet` or `epochs.parquet`.
Consequently, these are worker-reported preliminary results rather than the
durable controller-validated record.

| Architecture family | LR `4e-4` | LR `8e-4` | Two-rate mean | Change from direct-control mean | Mean train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct-coarse-patch` | [0.342571](https://wandb.ai/ocean_emulators/default/runs/oitxuuuu) | [0.324788](https://wandb.ai/ocean_emulators/default/runs/a04sz1f5) | **0.333679** | **-8.3%** | 20.3 min |
| `direct-transport256` | [0.354275](https://wandb.ai/ocean_emulators/default/runs/oj6rkd03) | [0.319645](https://wandb.ai/ocean_emulators/default/runs/rvv5lvh9) | **0.336960** | **-7.4%** | 27.6 min |
| `direct-enc64` | [0.363536](https://wandb.ai/ocean_emulators/default/runs/l8q3isnz) | [0.321942](https://wandb.ai/ocean_emulators/default/runs/ftjt32xh) | 0.342739 | -5.8% | 23.0 min |
| `direct-no-context` | [0.356488](https://wandb.ai/ocean_emulators/default/runs/9a1vcecr) | [0.332763](https://wandb.ai/ocean_emulators/default/runs/xn2e6a5j) | 0.344625 | -5.3% | 26.7 min |
| `direct-transport64` | [0.365438](https://wandb.ai/ocean_emulators/default/runs/otyae5f0) | [0.345617](https://wandb.ai/ocean_emulators/default/runs/02pi4nme) | 0.355527 | -2.3% | 27.5 min |
| `direct-context2` | [0.383464](https://wandb.ai/ocean_emulators/default/runs/65429i8w) | [0.344039](https://wandb.ai/ocean_emulators/default/runs/zzzzbej1) | 0.363752 | -0.1% | 27.2 min |
| `direct-control` | [0.366762](https://wandb.ai/ocean_emulators/default/runs/gd21xu96) | [0.361208](https://wandb.ai/ocean_emulators/default/runs/x3i0h8a6) | 0.363985 | control | 27.0 min |
| `pio-lean` | [0.403511](https://wandb.ai/ocean_emulators/default/runs/ywymchma) | [0.400348](https://wandb.ai/ocean_emulators/default/runs/g6nw1n4d) | 0.401929 | +10.4% | 40.8 min |
| `pio-control` | [0.421770](https://wandb.ai/ocean_emulators/default/runs/zfi0l44s) | [0.415711](https://wandb.ai/ocean_emulators/default/runs/gwim0j07) | 0.418740 | +15.0% | 38.4 min |

<details>

<summary>DuckDB query to reproduce this table</summary>

This query will work after the controller validates rung zero and publishes
`results.parquet`.

```sql
WITH rung_zero AS (
    SELECT
        regexp_replace(candidate, '-lr[48]$', '') AS family,
        CASE
            WHEN candidate LIKE '%-lr4' THEN '4e-4'
            WHEN candidate LIKE '%-lr8' THEN '8e-4'
        END AS learning_rate,
        validation_loss,
        train_seconds
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/results.parquet'
    )
    WHERE rung = 0 AND eligible
),
family_summary AS (
    SELECT
        family,
        max(validation_loss) FILTER (
            WHERE learning_rate = '4e-4'
        ) AS lr_4e_4,
        max(validation_loss) FILTER (
            WHERE learning_rate = '8e-4'
        ) AS lr_8e_4,
        avg(validation_loss) AS mean_validation_loss,
        avg(train_seconds) / 60 AS mean_train_minutes
    FROM rung_zero
    GROUP BY family
),
control AS (
    SELECT mean_validation_loss
    FROM family_summary
    WHERE family = 'direct-control'
)
SELECT
    family,
    round(lr_4e_4, 6) AS lr_4e_4,
    round(lr_8e_4, 6) AS lr_8e_4,
    round(family_summary.mean_validation_loss, 6) AS two_rate_mean,
    round(
        100 * (
            family_summary.mean_validation_loss
                / control.mean_validation_loss
                - 1
        ),
        1
    ) AS percent_from_direct_control,
    round(mean_train_minutes, 1) AS mean_train_minutes
FROM family_summary
CROSS JOIN control
ORDER BY family_summary.mean_validation_loss;
```

</details>

The two-rate mean is a sensitivity summary, not a replicate mean: each cell has
one seed, and the two members differ by learning rate rather than random seed.
The primary ranking remains each candidate's validation loss at a completed
cumulative budget.

### Successive-halving trajectory

Nine candidates advanced to three epochs, five advanced to six epochs, and
three advanced to the nominal 12-epoch rung. Validation loss decreased
monotonically for every promoted candidate. The table shows each candidate's
observed loss at promotion boundaries and the completed finalist ranking.

| Candidate | Epoch 1 | Epoch 3 | Epoch 6 | Latest later result | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `direct-no-context-lr4` | 0.356488 | 0.272234 | 0.234455 | **0.198794 (epoch 12)** | Recovered; selected winner |
| `direct-transport256-lr8` | 0.319645 | **0.263441** | **0.231851** | 0.201974 (epoch 12) | Completed; second |
| `direct-enc64-lr8` | 0.321942 | 0.265365 | 0.236658 | 0.203297 (epoch 12) | Recovered; third |
| `direct-coarse-patch-lr8` | 0.324788 | 0.267757 | 0.237423 | — | Pruned after epoch 6 |
| `direct-transport256-lr4` | 0.354275 | 0.272753 | 0.237599 | — | Pruned after epoch 6 |
| `direct-no-context-lr8` | 0.332763 | 0.274045 | — | — | Pruned after epoch 3 |
| `direct-context2-lr8` | 0.344039 | 0.276251 | — | — | Pruned after epoch 3 |
| `direct-coarse-patch-lr4` | 0.342571 | 0.282131 | — | — | Pruned after epoch 3 |
| `direct-transport64-lr8` | 0.345617 | 0.285886 | — | — | Pruned after epoch 3 |

The final comparison is now complete. `direct-no-context-lr4` reached 0.198794,
1.6% lower than `direct-transport256-lr8` and 2.2% lower than
`direct-enc64-lr8`. It also had the lowest epoch-12 train loss, 0.810475 versus
0.840199 and 0.844502. The advantage was not an isolated final validation:
no-context led at epoch 10, epoch 11, and epoch 12, and its validation margin
over transport widened from 0.001397 to 0.003180 across those boundaries.

The two interrupted jobs were making normal optimizer progress when Slurm
cancelled them simultaneously on `gr101` at 2026-08-15 03:00 UTC. Accounting
records report `CANCELLED by 0`; logs show neither an exception nor an
out-of-memory condition. `direct-no-context-lr4` retained an epoch-10
checkpoint, while `direct-enc64-lr8` retained an epoch-11 checkpoint. The
search controller correctly kept their incomplete results in the durable table
as ineligible, but incorrectly summarized the overall run as simply
`complete` once one finalist succeeded. On 2026-08-18, recovery jobs `15959944`
and `15959945` resumed the exact optimizer states. Encoder-64 completed its
remaining epoch in 23 minutes, while no-context completed two epochs in 53
minutes. The reconciled controller now marks all finalists eligible and the
public Parquet tables contain the complete comparison.

<details>

<summary>DuckDB query for the successive-halving trajectory</summary>

```sql
SELECT
    candidate,
    max(validation_loss) FILTER (WHERE epoch = 1) AS epoch_1,
    max(validation_loss) FILTER (WHERE epoch = 3) AS epoch_3,
    max(validation_loss) FILTER (WHERE epoch = 6) AS epoch_6,
    max(validation_loss) FILTER (WHERE epoch = 10) AS epoch_10,
    max(validation_loss) FILTER (WHERE epoch = 11) AS epoch_11,
    max(validation_loss) FILTER (WHERE epoch = 12) AS epoch_12,
    max(epoch) AS latest_completed_epoch
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/epochs.parquet'
)
WHERE candidate IN (
    SELECT DISTINCT candidate
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/results.parquet'
    )
    WHERE rung >= 1
)
GROUP BY candidate
ORDER BY coalesce(epoch_12, epoch_11, epoch_10, epoch_6, epoch_3) NULLS LAST;
```

</details>

## Analysis: why the no-context model won

The result supports a routing explanation more strongly than a raw-capacity
explanation. The winning decoder presents each output query with the 36 tokens
from its own six-by-six processor window. The one-context-ring alternatives
present 64 tokens, adding 28 neighboring tokens without an explicit relative
position that identifies how each neighbor should contribute to the queried
cell. Removing those tokens reduces the attention search space by 44% and
eliminates spatially plausible but incorrectly routed values.

This decoder is not globally context-free. Before decoding, the ConvNeXt U-Net
processor has already mixed information across the spatial latent grid. The
zero-context intervention only prevents the final cross-attention operation
from reintroducing anonymous neighboring patch tokens. A plausible mechanism is
therefore:

1. The encoder and processor construct spatially contextualized features.
2. The direct decoder queries the processed tokens assigned to its own output
   window.
3. Extra unanchored tokens make the final routing problem harder without adding
   reliably identifiable information.

The learning curve adds an optimization qualification. Wider transport at
`8e-4` learned fastest and led through epoch 6, while no-context at `4e-4`
started poorly but overtook it before epoch 10 and continued improving. Its
lower final train and validation losses suggest the win is not merely reduced
validation overfitting. It may combine a simpler routing problem with a learning
rate that is slower initially but better late in this short schedule.

This search does not isolate those two effects at epoch 12: the finalists differ
in both context and learning rate, and the no-context `8e-4` arm was pruned after
epoch 3. Nor does one seed establish statistical stability. The most informative
follow-up is a factorial comparison of context rings zero and one, transport
widths 128 and 256, and both learning rates, with enough budget to avoid pruning
the slower-starting no-context family. Explicit relative-position context should
then test the proposed routing mechanism directly.

## Discussion

### H1: direct output queries

H1 receives strong support as an early-learning result. Every one of the 14
direct-decoder candidates beat all four full Perceiver IO candidates at epoch
one, and no Perceiver IO candidate survived the first promotion. The
direct-control family mean was 13.1% lower than the PIO-control mean, while the
best direct families were approximately 20% lower and materially faster. The
eventual winner also used direct output queries. This agrees with the prior
diagnosis that the second learned decoder latent bank creates an unnecessary
routing bottleneck.

Successive halving only allocated one epoch to the full Perceiver IO arms, so
the experiment establishes inferior early optimization and compute efficiency,
not their hypothetical 12-epoch asymptote. That distinction matters, but early
efficiency is itself important for a model intended to scale to much larger
grids.

### H2: encoder latent count

The result contradicts H2's expected direction. Reducing the patch encoder from
256 to 64 latents improved the rung-zero two-rate mean by 5.8%, reduced training
time by about four minutes per epoch, survived every promotion boundary, and
reached a validation loss of 0.203297 at epoch 12. It finished 0.001324 behind
the wider-transport candidate and 0.004504 behind the no-context winner.

This does not show that aggressive spatial compression is harmless: both
variants mean-pool their internal latent bank to one processor vector per patch.
It does show that internal encoder latent count is not the limiting capacity
measure in this experiment and makes the smaller encoder a strong candidate for
follow-up work, especially given its lower cost.

### H3: decoder transport width

H3 receives partial support. The
256-transport, `8e-4` candidate ranked first at epochs 3 and 6 and finished
second at epoch 12 with 0.201974. Both learning-rate members of the
256-transport family survived to epoch 6, whereas the surviving 64-transport
candidate was pruned at epoch 3.

The evidence is not a clean monotonic width sweep: width 64 beat the 128-wide
control at `8e-4` in epoch one, only one seed was used, and the narrower
no-context model ultimately won. Wider transport remains a promising early-
learning intervention, not the best current default. A zero-context plus
256-transport combination and variable-wise diagnostics are needed to establish
whether width specifically relieves an output information bottleneck.

### H4: decoder context

H4 is contradicted. The zero-context `4e-4` candidate became a finalist and was
the best model at epochs 10, 11, and 12, finishing at 0.198794. The zero-context
`8e-4` member also survived epoch one, while the two-context-ring candidate was
pruned after epoch 3. This is consistent with prior evidence that unanchored
neighboring tokens can compete with the correct spatial route.

This result does not imply that ocean prediction needs no neighboring context.
The spatial processor already mixes information before decoding, and the
current decoder context lacks an explicit physical-position anchor. A better
next test is anchored or relative positional context, not simply more anonymous
tokens.

### H5: patch granularity

H5 is only partially supported. The coarse-patch family had the best epoch-one
two-rate mean and trained about 25% faster than the direct control. Its `8e-4`
member remained competitive through epoch 6, reaching 0.237423, only 0.005572
behind the leader, but it was then pruned. The 324-token processor grid is
therefore a useful efficiency point, though not the best observed validation
loss under the promotion policy.

Aggregate MSE still cannot tell whether the coarser representation learns an
efficient large-scale predictor or is rewarded for smoothing. Velocity/depth
errors, high-wavenumber power, amplitude ratios, and patch-edge diagnostics are
required before carrying this intervention to higher-resolution training.

### H6: learning-rate robustness

H6 receives mixed evidence. `8e-4` beat `4e-4` in all nine families at epoch
one, and three of the five epoch-6 candidates used `8e-4`, making it the strong
default for rapid screening. However, the `4e-4` no-context candidate became a
finalist and won the full epoch-12 comparison. Learning rate therefore
interacts with architecture rather than providing one universal ordering. A
follow-up should retain rate sensitivity for the best families and consider a
short range test before expanding another full matrix.

### Search-system observation

The preflight gate worked as intended: it proved that real data could be loaded
and that an optimizer update occurred before releasing the full array. Durable
Parquet publication also preserved learning curves, ineligible final results,
worker errors, timings, logs, and checkpoint inventory well enough to reconstruct
this report independently of W&B.

Three orchestration issues appeared. First, promotion controllers initially
failed because the immutable controller command did not load Apptainer on the
login environment; the search was manually resumed without retraining completed
rungs, and the launcher has since been corrected. Second, public controller
state lagged completed workers long enough to obscure whether promotion was
stuck. Third, Slurm simultaneously cancelled two healthy final workers, yet the
controller reported the overall search as simply `complete` because one finalist
succeeded. Manual checkpoint recovery obtained the missing results, but the
controller had to be reconciled and republished explicitly afterward. A robust
search should expose “workers complete, controller pending,” distinguish
complete from partial-final-rung completion, and automatically retry or requeue
workers whose scheduler termination is consistent with preemption.

## Conclusions

The search winner is `direct-no-context-lr4`: the patch-local Perceiver
encoder, direct query decoder, 128-value decoder transport, 256 encoder latents,
zero context rings, 6 x 10 degree patches, and learning rate `4e-4`. Its
validation loss fell from 0.356488 at epoch one to 0.198794 at epoch 12, a 44.2%
reduction, over 1,068 optimizer updates.

The most durable architectural conclusion is broader than that exact winner:
direct query decoding is substantially more effective and compute-efficient
than the tested full Perceiver IO decoder. Wider output transport is the best
supported early-learning capacity intervention, but removing decoder context is
the best final-budget intervention. A 64-latent encoder is also competitive,
while coarse patches offer a promising loss-versus-cost tradeoff. Additional
anonymous decoder context is not supported.

The winner remains provisional because this search used one seed and validation
MSE alone. The completed comparison removes the earlier right-censoring but does
not separate the winner's context intervention from its learning rate.

## Future work

The mechanistic implications and proposed next search are developed in
[`perceiver_v2_next_round_synthesis.md`](perceiver_v2_next_round_synthesis.md).
The immediate priorities are:

1. Run the durable ocean metrics and visualization suite on the finalists,
   including variable/depth error, spectra, amplitude, bias, patch seams, and a
   short rollout-stability probe.
2. Repeat the strongest configurations with multiple seeds before adopting a
   default architecture.
3. Run a matched factorial test of zero versus one context ring, transport
   widths 128 versus 256, and both learning rates; include the smaller encoder
   and physically anchored context as targeted follow-ups.
4. Improve the harness with explicit partial-completion state, automatic retry
   of preempted workers, bounded controller-lag alerts, and reliable monotonic
   W&B resume logging.
5. Confirm the selected design on a second resolution before treating
   single-scale gains as evidence of resolution-sharing ability.
