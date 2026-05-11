<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Experiment: Multi-Scale Parallel-Dilation ConvNeXt Blocks

Tests whether E1's dilation win can be *compounded* with E2's per-branch-BN
training-stability win, by combining multiple receptive-field scales in
parallel inside a single block. Generalizes RepLKNet's "main + parallel
small kernel" recipe along the **dilation** axis instead of the kernel
axis.

## What's in this experiment

Three new model configs, all using one of two new block types:

| Config | Block type | Main branch | Parallel pattern |
|---|---|---|---|
| `model_multiscale_e14.yaml` | `multiscale_conv_next_block` | dense k=3 | parallels at d=16 stages only: `[4, 1]` |
| `model_multiscale_e15.yaml` | `multiscale_conv_next_block` | dense k=3 | parallels at every stage; bottleneck `[4, 1]` |
| `model_multiscale_e16.yaml` | `multiscale_true_conv_next_block` | depthwise k=3 | same per-stage pattern as E15 |

The block types live in `src/ocean_emulators/models/modules/blocks.py`:

- `MultiScaleConvNeXtBlock` — dense main spatial conv + 0+ depthwise
  parallel branches with per-branch `BatchNorm`. Canonical ConvNeXt recipe
  (single spatial conv + `pw_expand → act → pw_project`).
- `MultiScaleTrueConvNeXtBlock` — depthwise main spatial conv + 0+ depthwise
  parallel branches. Same structure as the dense version with depthwise
  groups on the main conv. Generalizes `RepConvNeXtBlock` to multi-dilation.

Both block types accept a `parallel_dilations: list[int]` per stage, plumbed
from the config's `parallel_dilations: dict[int, list[int]]` map keyed by
that stage's main dilation. Stages whose main dilation isn't in the map (or
maps to an empty list) run as standard single-conv blocks.

## Motivation: dilation is sparse, parallels fill the gaps

The mental model:

- **Kernel size** controls *how densely* a patch is sampled.
- **Dilation** controls *how widely* a patch is sampled, at the cost of
  sparsity (intermediate cells are skipped).

E1's win (`[1, 8, 16, 16]` dense + dilation at 1°: deep R² −16.14 → −1.88)
came from giving the bottleneck near-global reach via dilation-16. But
dilation-16 at the bottleneck literally samples 3 cells per row spaced 16
apart — it skips the cells in between. A parallel depthwise branch at
dilation-1 (or -4) fills those cells in. Per-branch `BatchNorm` lets each
scale learn its own normalization statistics, which is the only reason
training is stable when branches at very different dilations produce
gradients with very different magnitudes (per Ding et al. RepLKNet §5).

## Per-stage receptive-field design

At 1° the bottleneck is 22×45, and each latent cell encodes ~8° × 8° of
physical ocean (4 downsamples from 180×360). So the dilations translate
to physical reaches as:

| Stage | Feature map | Cells/latent | Dilation → physical reach |
|---|---|---|---|
| 0 (surface) | 180×360 | ~1° | d=1 → 1–2° (local eddies, fronts) |
| 1 (subbasin) | 90×180 | ~2° | d=2 → ~8° (coastal currents) |
| 2 (basin) | 45×90 | ~4° | d=4 → ~32° (gyres, synoptic systems) |
| 3 (bottleneck) | 22×45 | ~8° | d=16 → ~256° (pole-to-pole, cross-Pacific teleconnections) |

E15 and E16 use this exact schedule with `dilation: [1, 2, 4, 16]`. The
parallel-dilation sets fill in the intermediate scales:

- Stage 3 (d=16) gets parallels `[4, 1]` — d=4 (~32°, basin corrector)
  and d=1 (~1°, local-cell anti-aliasing).
- Stages 1–2 (d=2, 4) get a single parallel d=1 — local-detail corrector.
- Stage 0 (d=1) has no parallels — surface needs only local detail.

E14 reuses E1's existing `[1, 8, 16, 16]` schedule and applies multi-scale
only at d=16 stages. This keeps the spatial schedule identical to E1 so
the multi-scale effect can be isolated.

## What each experiment tests

| | Hypothesis | Primary metric to watch |
|---|---|---|
| E14 | Adding parallels at the deep stages recovers the upper-ocean R² and 2000m correlation E1 regressed, without sacrificing deep R². | 2000m correlation (E1: 0.17 → paper: 0.36); upper R² (E1: 0.59 → paper: 0.87) |
| E15 | Multi-scale at every stage compounds the bottleneck benefit. Higher risk of training instability. | Same as E14 + train stability (loss curves should match E14's, not diverge) |
| E16 | E2's training-stability benefit transfers to dilation-axis parallels. Depthwise variant approaches E1's deep R² (−1.88) at ~10× lower parameter cost. | Deep R²; parameter count vs E1 (E1: 84M, E16 expected ~10M) |

## How to run on Torch (NYU HPC)

```bash
# E14 — multi-scale at bottleneck
export CONFIG=configs/samudra_om4_v2/train.yaml
export NAME_SUFFIX=om4_samudra_v2_multiscale_e14_v1   # bump for each run
export ARGS="--model=@configs/samudra_om4_v2/model_multiscale_e14.yaml"

sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  scripts/slurm_apptainer_train.sbatch

# E15 — multi-scale at all stages. Identical command, swap model path.
export NAME_SUFFIX=om4_samudra_v2_multiscale_e15_v1
export ARGS="--model=@configs/samudra_om4_v2/model_multiscale_e15.yaml"

# E16 — depthwise multi-scale. Identical command, swap model path.
export NAME_SUFFIX=om4_samudra_v2_multiscale_e16_v1
export ARGS="--model=@configs/samudra_om4_v2/model_multiscale_e16.yaml"
```

For eval, swap `CONFIG` to `configs/samudra_om4_v2/eval.yaml`, keep the
same `ARGS`, and submit via `scripts/slurm_apptainer_eval.sbatch`. Same
pattern as the existing dense-dilated runs.

## Recommended run order

1. **E16 first** (depthwise, ~10M params, fastest to train).
   - If E16 lands deep R² in the [−5, −3] range, it's a major
     parameter-efficiency result and changes the headline story.
   - If E16 is closer to E2's −9, the dense vs depthwise gap is real
     and we move on.
2. **E14 second** (dense, comparable cost to E1).
   - If E14 lifts upper R² and 2000m correlation without regressing
     deep R², it's a clean compound win on top of E1.
3. **E15 last** (most ambitious).
   - Only worth running if E14 succeeds. Otherwise E15's per-stage
     multi-scale adds risk without buying anything new.

## Caveats

- **Recipe deviation from E1.** `MultiScaleConvNeXtBlock` uses the
  canonical ConvNeXt recipe (single spatial conv + `pw_expand → act →
  pw_project`). E1's existing `ConvNeXtBlock` has two stacked dense
  spatial convs. So E14 vs E1 mixes "recipe change" with "multi-branch
  addition." A clean ablation would also run a "single dense + no
  parallels" variant of this block — same recipe, no multi-branch — to
  isolate the recipe change from the multi-branch effect.
- **No inference-time fold in v1.** Parallel branches are run sequentially
  at inference. For deep stages (small feature maps) this is cheap. A
  proper fold would require a sparse kernel spanning
  `max(d_par) * (k-1) + 1` cells with nonzero weights only at the union
  of branch stencils — possible but materially more complex than
  `RepConvNeXtBlock`'s single-kernel fold. Not v1 scope.
- **`parallel_dilations` is keyed by main dilation, not stage index.**
  If two stages share a main dilation, they get the same parallel set
  (correct for E14; intentional for E1's `[1, 8, 16, 16]` where stages
  2 and 3 both at d=16). To give different parallel sets to two stages
  with the same main dilation, use a per-stage-unique dilation schedule
  like E15's `[1, 2, 4, 16]`.
- **Per-branch BN requires `norm: batch`.** Instance/layer norm is not
  supported (the fold identity is BN-specific, and the per-branch
  training-stability rationale leans on BN's per-branch running stats).

## References

- Ding et al. 2022, *Scaling Up Your Kernels to 31×31* — per-branch BN.
  [arXiv:2203.06717](https://arxiv.org/abs/2203.06717)
- Chen et al. 2017, *Rethinking Atrous Convolution for Semantic Image
  Segmentation* (ASPP) — parallel dilated branches without per-branch BN.
  [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)
- Liu et al. 2022, *A ConvNet for the 2020s* (ConvNeXt) — base recipe.
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- `configs/samudra_om4_v2/exp_kernel.md` — the predecessor experiment
  (E1 and friends) whose results motivate this one.
