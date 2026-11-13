<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Large-scale error penalty on top of normalized-delta (2 in / 1 out, 1°)

**Experiment record, not intended to merge.** It is stacked on PR #984
(`experiments/normalized_delta_2in1out`) and changes exactly one thing: the
loss gains a penalty on large-scale error.

## The problem this targets

### A residual model adds up its per-step push

A residual emulator predicts each state as `x[t+1] = x[t] + Δ̂`. Write the
error in the predicted change as two parts:

- a part `b` that is the same every step (a *bias*, or systematic push);
- a part `ε` that varies from step to step.

After `n` steps of a rollout:

- the bias part grows linearly, as `n·b`;
- the random part grows only like `√n·|ε|`, and less if the dynamics damp it.

Over years, the bias wins. At 5 days per step, a bias of 0.001 °C per step
becomes about 0.6 °C over the 584 steps of an eight-year evaluation.

### The evidence that such a push exists

**One-step maps of the paper model** (PR #980, [one-step figures at
1°](https://github.com/m2lines/Samudra/blob/40ffb670/experiments/compression_baselines/reports/onestep_figures_onedeg.md),
finding 2). Starting from the true ocean, a single step of Samudra 2 pushes
the state in a structured way:

- the tropical surface about 0.08 °C colder per step;
- a colder Arctic interior;
- a colder deep North Atlantic;
- banded warm and cold patches at 250 m.

These are large-scale patterns: zonal bands and basin-sized patches.

**The rollout of PR #969.** It has the same residual setup as #984 and drifts
in the same direction: mean SST −0.25 °C and surface salinity −0.07 psu after
eight years, with a local maximum SST climbing to 38.7 °C.

### Why the per-cell loss barely sees it

The loss averages the squared error over cells. For any block of cells, that
average splits exactly into two parts:

```
mean over block of e²  =  (block mean of e)²  +  variance of e within the block
```

The systematic push lives in the first part. For Samudra 2 at the surface,
the time-mean bias is about 0.08 °C against a one-step RMSE of 0.32 °C, so the
bias is about **6% of the squared error**. The optimizer gets little reward
for removing it, even though the bias decides where the rollout ends up.

## The change

The loss adds a penalty, `large_scale_weight` × the squared area-weighted
mean error over blocks of `large_scale_block_cells` × `large_scale_block_cells`
cells, averaged over blocks by wet area. It is measured in the same change
units as the rest of the loss, `(pred − target) / q`. That keeps it balanced
across levels and variables in the same way.

Here, the weight is 10 and the blocks are 10 × 10 cells (10° at 1°). So the
block-mean part of the error counts 11 times instead of once, and the
within-block part is unchanged.

`tests/test_normalized_delta.py` pins down the contract. A uniform error of one
change unit costs `1 + weight` per channel. An error pattern that averages to
zero within every block costs exactly what it does without the penalty.

### Relation to published loss designs

- **GraphCast** (Lam et al. 2023) and **Keisler** (2022) weight squared error
  by cell area, by level and by variable, but they don't separate error by
  scale.
- **Spectral and multi-scale losses** (e.g. the amplitude-and-phase losses
  used against blurring in medium-range weather emulators) do separate it.
  They usually target small scales, to restore sharpness.
- This penalty is the opposite end of that idea. It upweights the *lowest*
  wavenumbers, because for multi-year ocean rollouts the large-scale mean
  error drives drift.
- **Conservation penalties** in climate emulators (global mass or energy
  budget terms) are the limit of one block covering the whole globe.

  Ten-degree blocks also catch the zonal-band biases that a single global
  mean would let cancel out.

## Predictions, made before training

Against #984, which is identical apart from the penalty:

1. **Per-step large-scale bias shrinks.** Measured with #980's one-step bias
   maps, the mean |bias| of temperature and salinity at the surface and
   250 m is at least 30% lower. ~65%.
2. **Rollout drift shrinks.** Eight-year drift in global-mean SST and surface
   salinity is at least 30% smaller in magnitude. ~55%.
3. **One-step RMSE barely changes,** within ±5%, because the penalty mostly
   reshapes error rather than removing it. ~70%.
4. **No effect on the local SST runaway.** That is a cell-scale problem. ~60%.
5. **Deep variance damping is unchanged.** ~75%.

## How to run

```bash
CODE_COMMIT=<sha> CFG=experiments/nd2in1_large_scale_penalty \
  NAME=2026-11-13-nd2in1-large-scale-penalty \
  bash experiments/normalized_delta_2in1out/submit_variant.sh
```
