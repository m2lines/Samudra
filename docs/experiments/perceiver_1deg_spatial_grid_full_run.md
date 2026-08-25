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
| Data | One-degree OM4 with raw `hfds` forcing; 1975--2013 train and 2013--2014 validation |
| Encoder | 6 x 10 degree groups, 2 x 2 coordinate-tied spatial queries |
| Processor grid | 60 x 72 at 3 x 5 degree spacing |
| Processor | Three-level ConvNeXt U-Net, widths 380, 480, and 520 |
| Prediction | Same-grid physical residual |
| Decoder | Direct SDPA cross-attention, transported width 128 |
| Routing | Six-patch windows, zero anonymous context |
| Assembly | One-ring cosine overlap-add |
| Conditioning | Zero-initialized smooth processor path |
| Excluded | DCT transport and pixel refinement |
| Training | 70 epochs, LR `6e-4`, cosine schedule, normalized MSE |
| Intended allocation | Two GPUs, per-rank batch 2, accumulation 8, effective global batch 32 |
| Trainable parameters | 75,194,225 |

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

## Deferred attention-depth ablation

The encoder is a complete local Perceiver IO: input cross-attention, two latent
self-attention stages, and output-query cross-attention. The decoder is
intentionally only the Perceiver IO output head--one direct cross-attention plus
a feed-forward residual--because the processor has already performed spatial
mixing and the root-cause experiments rejected a second decoder latent bank.

After this run establishes a convergence and rollout baseline, a focused screen
can compare encoder depths `{1, 2, 4}` and direct decoder cross-attention depths
`{1, 2}`. Adding output-query self-attention is not in that first ablation: its
cost is quadratic in output pixels within each window and it would mix physical
outputs in the decoder rather than leaving spatial dynamics to the processor.
It should require separate evidence before being introduced.

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

### Preliminary epoch-28 rollout diagnostic

While the full run was training, the rolling EMA checkpoint was snapshotted at
epoch 28 and evaluated over 2014-10-10--2015-10-05. The one-GPU job
`16350863` completed in 91 seconds and wrote 70 physical-space frames at
five-day spacing. The corresponding W&B run is
[`ezvjkkkk`](https://wandb.ai/ocean_emulators/default/runs/ezvjkkkk). This is an
interim checkpoint diagnostic, not the final 70-epoch result.

For each variable family, the saved rollout was compared with two baselines:

- persistence repeats the most recent input frame; and
- climatology repeats the scalar training mean for each channel.

The table reports area-weighted normalized RMSE across every wet cell and depth
channel in a variable family. Persistence skill is
`1 - MSE(model) / MSE(persistence)`; positive values favor the model.

| Lead | Variable | Model RMSE | Persistence RMSE | Persistence skill |
| ---: | :--- | ---: | ---: | ---: |
| 5 d | `uo` | 0.430 | 0.328 | -0.715 |
| 5 d | `vo` | 0.661 | 0.480 | -0.898 |
| 5 d | `thetao` | 0.027 | 0.018 | -1.313 |
| 5 d | `so` | 0.028 | 0.018 | -1.395 |
| 5 d | `zos` | 0.052 | 0.037 | -0.964 |
| 10 d | `uo` | 0.440 | 0.481 | 0.163 |
| 10 d | `vo` | 0.642 | 0.710 | 0.182 |
| 10 d | `thetao` | 0.028 | 0.031 | 0.175 |
| 10 d | `so` | 0.030 | 0.030 | 0.000 |
| 10 d | `zos` | 0.053 | 0.053 | 0.021 |
| 100 d | `uo` | 0.893 | 1.063 | 0.295 |
| 100 d | `vo` | 1.061 | 1.124 | 0.108 |
| 100 d | `thetao` | 0.090 | 0.168 | 0.711 |
| 100 d | `so` | 0.090 | 0.102 | 0.228 |
| 100 d | `zos` | 0.093 | 0.115 | 0.347 |
| 350 d | `uo` | 0.924 | 0.905 | -0.043 |
| 350 d | `vo` | 1.126 | 1.123 | -0.006 |
| 350 d | `thetao` | 0.166 | 0.085 | -2.795 |
| 350 d | `so` | 0.152 | 0.073 | -3.316 |
| 350 d | `zos` | 0.160 | 0.106 | -1.275 |

The full lead-time table, including physical-unit surface RMSE and climatology
skill, is stored alongside the rollout as
`persistence_climatology_skill.csv`.

## Analysis

The velocity gap is real in normalized space, but it is not evidence that every
other variable is solved. At the surface, the five-day physical RMSEs are
0.0613 m/s for `uo` and 0.0690 m/s for `vo`; their much larger normalized losses
also reflect the smaller training standard deviations and faster decorrelation
of velocity than of temperature or salinity.

More importantly, skill alternates between the two frames emitted by each
autoregressive call. At 5 days every variable is worse than persistence; at 10
days four families beat persistence and salinity ties it. The same weaker-odd,
stronger-even pattern remains visible at 15/20 and 25/30 days. With `hist: 1`,
channels are arranged as two time blocks. Residual assembly adds the first
decoded output block to the older input frame and the second block to the newer
input frame. Both are therefore ten-day residual targets, even though the saved
rollout exposes five-day-spaced frames. The model can attend to both inputs, but
the identity shortcut for the first output is older than the conventional
latest-state persistence baseline. This is a plausible structural cause of the
parity effect and should be tested directly before attributing the velocity
behavior primarily to patch extent.

The model nevertheless learns nontrivial dynamics. At 100 days it beats
persistence for every variable family, with particularly strong temperature
skill. Relative to climatology at 100 days, skill is 0.278 for `uo`, -0.087 for
`vo`, and approximately 0.99 for `thetao`, `so`, and `zos`. Thus `vo` has a
genuine medium-range deficiency: after about 50 days it is slightly worse than
the training-mean baseline even while retaining modest skill over persistence.
The very negative 350-day persistence skill for thermodynamic variables should
not be read as catastrophic drift in isolation. A nearly annual lead compares
similar seasons, making a fixed October initial state an unusually strong
baseline; climatology skill remains strongly positive for all variables except
`vo`.

These results motivate three separate checks on the final checkpoint: compare
current same-slot residual assembly with residuals anchored to the latest input
frame (and with direct prediction), inspect `uo`/`vo` vorticity and spectra, and
measure error conditioned on encoder-patch and decoder-window boundaries. The
lead-time parity test isolates residual/output assembly; the latter two tests
isolate missing dynamics and spatial patch artifacts.

## Conclusion and future work

Pending.
