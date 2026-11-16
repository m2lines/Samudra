<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Per-cell change scale for normalized-delta (2 in / 1 out, 1°)

**Experiment record, not intended to merge.** It is stacked on PR #984
(`experiments/normalized_delta_2in1out`) and changes exactly one thing: the
change scale `q` becomes a map over the grid instead of one number per
channel.

## From first principles

### What the change scale does

In #984, as in Keisler (2022) and GraphCast (Lam et al. 2023), the network
predicts a unit-scale increment `d`. The next state is

```
x[t+1] = x[t] + q · d
```

and the loss is the mean of `((x̂ − x[t+1]) / q)²`. Here `q = τ / σ`:

- `τ` is the typical five-day change;
- `σ` is the state's normalization spread.

That one choice of `q` does two jobs:

1. **Output conditioning.** The network's target `(x[t+1] − x[t]) / q` has a
   spread of about 1 everywhere it is applied. One output layer serves every
   channel.
2. **Loss weighting.** A forecast that only matches persistence scores the
   same everywhere `q` is uniform. So every part of the ocean the loss
   averages over counts in proportion to how much *skill over doing nothing*
   it offers.

### Why one number per channel doesn't fit the ocean

Keisler and GraphCast use one `τ` per variable and level. That suits the
atmosphere, whose day-to-day change varies across the globe by perhaps a
factor of a few.

The ocean's five-day change varies **by orders of magnitude within one
channel**:

- in western boundary currents, the Antarctic Circumpolar Current and the
  equatorial band, temperature at 250 m moves by about a degree in five days;
- in the subtropical gyre interiors at the same depth, it moves by
  hundredths of a degree.

With one `τ` per channel:

- **In the quiet interior,** the true target `(x[t+1] − x[t]) / q` is tiny,
  far below 1. Errors there are cheap in the loss, so the network can add
  unit-scale noise times `q`, which is large compared with the real change,
  and barely pay for it.
- **At fronts,** the target is much larger than 1, and those cells dominate
  the loss.

### The evidence

PR #980 ([one-step figures at
1°](https://github.com/m2lines/Samudra/blob/40ffb670/experiments/compression_baselines/reports/onestep_figures_onedeg.md),
finding 4) shows this pattern in the paper model.

- **Where it wins:** along boundary currents, fronts and the equator, it beats
  persistence.
- **Where it loses:** in the quiet interior, where the true change is tiny.
  89% of the area at 250 m is worse than persistence.
- The report's reading: Samudra 2's per-step error is roughly the same size
  everywhere, so it costs most where the ocean changes least.

The compression ladder in the same PR points the same way. Its largest
single parametric gain (`lpc_ctx`, on 44 of 77 channels) comes from a spread
that adapts to local context, with the same mean prediction.

## The change

Compute `τ(variable, y, x)`: the RMS five-day change over time at each wet
cell, on the training period.

```bash
python scripts/compute_tendency_stats.py --per-cell \
  --data $DATA_ROOT/OM4.zarr --output $DATA_ROOT/OM4_increment_rms_per_cell.zarr \
  --start 1975-01-03 --end 2013-09-30
```

Then use `q = max(τ_cell, 0.1 · τ_channel) / σ`, where `τ_channel` is the
channel's RMS over wet cells: the same `τ` as #984. That `q` sets both the
model's residual scale (`model.residual_scale_per_cell`) and the loss, which
is the mean of `((x̂ − x[t+1]) / q)²`.

- **With the floor at 1:** this reduces to #984 exactly.
- **With the floor at 0.1:** nearly stagnant cells can't receive more than
  100× a typical cell's weight, so the per-cell weights stay bounded.
- **Alignment check:** the loader fails if the map's land cells don't match
  each channel's wet mask. That also catches a map written on a flipped or
  shifted grid.

What this does to the balance:

- **Unchanged:** each channel's share of the loss at persistence is the same
  as in #984, proportional to its ocean area.
- **Changed:** within a channel, every wet cell now counts equally at
  persistence. Quiet regions gain weight and energetic regions lose it.

### Relation to other work

- **Keisler (2022) and GraphCast (Lam et al. 2023):** the per-variable,
  per-level residual normalization that this extends to per cell.
- **Learned per-cell spreads** (Nix & Weigend 1994; and #980's [losses from
  bits](https://github.com/m2lines/Samudra/blob/5565cafb/experiments/compression_baselines/reports/losses_from_bits.md),
  section 3): a network head predicts a spread per cell and situation, trained
  with an NLL loss. This experiment is the fixed, climatological version. It
  needs no new head and can't collapse its own spread. If it helps, the
  learned version is the next step.

## Predictions, made before training

Against #984, which is identical apart from `q`:

1. **One-step skill in the quiet interior improves.** In #980's bits and
   skill maps, the area of T and S at 250 m that is worse than persistence
   falls by at least a third. ~65%.
2. **Fronts and boundary currents get somewhat worse,** relative to #984, in
   one-step RMSE. ~60%.
3. **Global one-step RMSE is roughly unchanged,** within ±10%, because RMSE
   is dominated by the energetic regions this change de-emphasizes. The
   improvement shows up in skill and bits, not RMSE. ~60%.
4. **The local SST runaway is reduced or gone.** If it sits in a quiet,
   enclosed region, a smaller per-cell `q` there limits how far one step can
   move the state. ~45%; this depends on where the hot spot is.
5. **Deep variance damping is unchanged.** ~70%.

## How to run

```bash
CODE_COMMIT=<sha> CFG=experiments/nd2in1_per_cell_tau \
  NAME=2026-11-13-nd2in1-per-cell-tau \
  bash experiments/normalized_delta_2in1out/submit_variant.sh
```

`submit_variant.sh` checks that `OM4_increment_rms_wet.zarr` exists, which is
true on Torch. This run reads `OM4_increment_rms_per_cell.zarr`.
