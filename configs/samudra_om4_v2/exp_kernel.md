# Experiment: Large Depthwise Kernels

Tests the hypothesis that **large depthwise kernels improve ocean
emulation accuracy**, motivated by Ding et al. 2022 ("Scaling Up Your
Kernels to 31×31") §5.3 + Table 11, which shows that simply replacing
the 7×7 depthwise convs in ConvNeXt with kernels up to 31×31 yields
significant gains on dense-prediction tasks without requiring any other
changes (no re-parameterization, no architectural reshaping).

## What's in this experiment

A single new model file: `model_large_kernel.yaml`. The training,
evaluation, and visualization configs are unchanged from V2 — to run the
experiment, point `train.yaml`'s `model: !include` at the new file (or
override at the CLI), and bump `experiment.name` so outputs don't clash
with the V2 baseline run.

## How to run on Torch (NYU HPC)

```bash
export CONFIG=configs/samudra_om4_v2/train.yaml
export NAME_SUFFIX=om4_samudra_v2_large_kernel_v1   # bump for each run
export ARGS="--model=@configs/samudra_om4_v2/model_large_kernel.yaml"

# Container selection (pick one)
export CONTAINER_HASH=<git_sha>
# export CONTAINER_TAG=25.11-manual-<branch>
# export IMAGE_REF=ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-<git_sha>

sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  scripts/slurm_apptainer_train.sbatch
```

Notes:

- Paths inside `ARGS` are resolved relative to `/workspace/` inside the
  container, so the relative path above is correct.
- Bump `NAME_SUFFIX` (e.g. `_v2`, `_v3`, ...) every new run. The harness
  refuses to overwrite an existing `${OUTPUT_BASE}/$NAME` directory.
- For evaluation, swap `CONFIG` to `configs/samudra_om4_v2/eval.yaml`,
  set `TARGET_CHECKPOINT` to the trained run's checkpoint, keep the
  same `ARGS="--model=@..."` so the eval pipeline reconstructs the
  right model, and submit via `scripts/slurm_apptainer_eval.sbatch`.

## What changes from `model.yaml` (V2 baseline)

Only two things change — everything else is held constant to isolate the
kernel-size variable:

1. **Block type** flips from `conv_next_block` (the misnamed
   dense-spatial-conv block) to `true_conv_next_block` (the canonical
   ConvNeXt block: 1×1 pre-projection → depthwise k×k → norm → 1×1
   expand → activation → 1×1 project + residual).
2. **Per-stage kernel sizes** become `[31, 21, 13, 7]` (decreasing,
   shallow-to-deep), replacing V2's uniform `kernel_size: 3` with
   dilation `[1, 2, 4, 8]`.

Channel widths, expansion factor, norm, activation, padding, and
down/upsamplers are all identical to V2.

## Why decreasing kernels (`[31, 21, 13, 7]`)

Three considerations, all pointing the same direction:

- **Paper precedent.** RepLKNet uses `[31, 29, 27, 13]` and the ConvNeXt
  ablations show the same shallow-to-deep decreasing pattern works well.
  Deeper stages don't need big kernels because their effective receptive
  field is already large via composition through the network.
- **V2's effective RF.** V2 achieves an effective receptive field of
  roughly `[3, 5, 9, 17]` per stage via dilation. This experiment's
  kernels exceed that at every stage, so we're strictly testing "more
  spatial context" vs. V2.
- **Feature-map budget.** At 1° the deepest U-Net stage has spatial
  dims ~22×45. A 31×31 kernel there would require `N_pad=15`, more than
  half the feature height — wasted compute on padding. The decreasing
  schedule keeps padding ≤15% at every stage.

## Caveats

- `true_conv_next_block` is structurally distinct from `conv_next_block`
  (depthwise vs. dense, one spatial conv per block vs. two), so this
  experiment vs. V2 mixes "block design" and "kernel size." A clean
  ablation would compare against a `true_conv_next_block` control with
  `kernel_size: [3, 3, 3, 3]` — worth running if this result is
  interesting.
- Re-parameterization (the paper's parallel small-kernel branch, folded
  at inference) is intentionally **not** used. Per Ding et al. §5.3,
  ConvNeXt accommodates large kernels without it. If training shows
  instability at the largest kernels, that's the next thing to add.
- Layer scale γ is also not used, for the same reason — one block per
  stage doesn't need it.
