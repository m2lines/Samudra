<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# One-degree spatial-grid Perceiver full run

## Status

Prepared for review. No full training allocation has been submitted.

## Question

Can a coordinate-tied, locally compressed Perceiver encoder and a direct,
smoothly assembled decoder become a strong single-scale one-degree predictor
without the patch-periodic errors observed in earlier architectures?

This run is deliberately single scale. It tests whether the
encoder-processor-decoder is credible before parameter sharing across
resolutions is added.

## Evidence and hypothesis

The preceding 2-degree searches found that:

- direct output cross-attention was consistently better than adding a second
  Perceiver latent bottleneck in the decoder;
- physical residual prediction reduced short-budget validation loss by roughly
  75%;
- one-ring overlap assembly greatly reduced window discontinuities at nearly
  unchanged loss;
- smooth processor conditioning was the cleanest reviewed residual arm;
- pixel refinement did not provide matched positive evidence and its reviewed
  snapshots retained rectilinear structure; and
- fixed DCT transport lowered early loss but produced strong, visibly
  unphysical periodic errors.

The hypothesis is that coordinate-tied spatial encoder outputs preserve the
useful phase information indicated by the DCT result without imposing a fixed
patch basis. This is a compositional prediction: the exact conditioned
`spatial_grid` model has not yet been trained to convergence.

The full evidence synthesis remains on the
[`experiment/perceiver-v2-structured-transport-2deg`](https://github.com/m2lines/Samudra/blob/experiment/perceiver-v2-structured-transport-2deg/docs/experiments/samudra_multi_encoder_decoder_meta_analysis.md)
branch.

## Intervention

The complete preset is
[`perceiver_spatial_grid_1deg`](../../src/samudra/configs/perceiver_spatial_grid_1deg/README.md).

| Component | Selection |
| --- | --- |
| Data | OM4 one degree only, 1975--2013 train and 2013--2014 validation |
| Encoder | 6 x 10 degree groups, 2 x 2 coordinate-tied spatial queries |
| Processor grid | 60 x 72 at 3 x 5 degree spacing |
| Processor | Two-level ConvNeXt U-Net, widths 380 and 480 |
| Prediction | Same-grid physical residual |
| Decoder | Direct SDPA cross-attention, transported width 128 |
| Routing | Six-patch windows, zero anonymous context |
| Assembly | One-ring cosine overlap-add |
| Conditioning | Zero-initialized smooth processor path |
| Excluded | DCT transport and pixel refinement |
| Training | 70 epochs, LR `6e-4`, cosine schedule, normalized MSE |
| Intended allocation | Two GPUs, per-rank batch 2, accumulation 8, effective global batch 32 |

The model uses PyTorch SDPA through the native Perceiver implementation from
PR #842. `perceiver_implementation: auto` permits PyTorch to select Flash
Attention when the installed CUDA runtime and tensor shapes support it.

## Preflight gates

Before the full allocation:

- [x] train, eval, viz, data, and model presets validate against their schemas;
- [x] focused spatial-encoder and decoder forward/backward tests pass;
- [x] lint, formatting, typing, schema validation, secret scanning, and REUSE
      checks pass;
- [ ] run one real one-degree optimizer update on the intended Torch image;
- [ ] record peak memory, optimizer-step time, data-wait time, and selected SDPA
      backend;
- [ ] verify W&B receives loss, optimizer-step, snapshot, seam, and periodic
      metrics;
- [ ] confirm the requested GPU count and accumulation preserve effective batch
      32; and
- [ ] estimate completion within the cluster's requeue/time-limit contract.

No 70-epoch job should be released until the real-data optimizer probe passes.

## Evaluation and visualization plan

Evaluate the validation-selected checkpoint and the terminal checkpoint if
their losses differ materially. The configured evaluation produces a 25-step
rollout. Report:

- normalized validation loss and physical-unit variable/depth RMSE;
- skill relative to persistence at leads 1, 2, 4, 10, and 20;
- scalar and velocity gradient/high-wavenumber power;
- window and patch jump and periodic-power diagnostics; and
- common-color-limit error maps for `so_11`, `so_13`, `so_14`, `thetao_2`,
  `thetao_7`, and `thetao_8`.

The visualization preset consumes the evaluation Zarr output. Low loss cannot
promote a checkpoint with strong rectilinear or periodic error morphology.

## Results

Pending.

## Analysis

Pending.

## Conclusion and future work

Pending.
