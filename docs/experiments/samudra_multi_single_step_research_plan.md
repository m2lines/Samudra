<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# SamudraMulti single-step baseline research plan

## Scope

This plan starts from the completed 1-degree SamudraMulti baseline documented in
[`samudra_multi_single_step_mse.md`](samudra_multi_single_step_mse.md). The goal is a
stronger, faster, and more transparent baseline before another multi-resolution
training run.

All training and promotion targets in this plan are single-step. Use plain normalized
MSE as the optimization loss and report unweighted normalized MSE by variable group
and depth. Do not add autoregressive training or longer-rollout targets. Keep the
physical patch extent fixed at 3 degrees by 5 degrees unless that constraint is
explicitly revisited.

The all-channel full-data 1-degree gate remains:

- at most 0.05: promotion-ready;
- 0.05 to 0.075: diagnostic follow-up;
- above 0.075: do not add resolutions.

In addition, do not promote a candidate beyond 1 degree while it remains more than
roughly two times the matched v2 control.

## Bottom line and starting point

The historical multi-resolution run was genuinely much worse than v2, and its
headline W&B loss was distorted by extreme dynamic weights. Replacing that objective
with plain MSE made the comparison transparent but did not close the quality gap.
The completed 1-degree baseline reached 0.29469 all-channel MSE, 12.5 times the
supplied v2 reference of 0.0236. On the matched 512-timestamp screen, SamudraMulti
reached 0.38508 while v2 reached 0.04287, a 9.0-times gap.

The largest remaining suspects are the encoder bottleneck, normalization and
receptive-field choices, and incomplete optimization parity. The higher resolutions
are not needed to reproduce the failure. Keep the completed run as the baseline and
do not launch another full multi-resolution run until the cheap 1-degree funnel
closes most of the v2 gap.

## Where the errors are

The validation-selected epoch-12 errors from the completed full-data, 1-degree,
single-step, plain-MSE baseline are:

| Variable group | SamudraMulti | v2 1-degree reference | Ratio |
|---|---:|---:|---:|
| Temperature | 0.04279 | 0.00149 | 28.7x |
| Salinity | 0.08267 | 0.00138 | 59.9x |
| Zonal velocity | 0.50774 | 0.0361 | 14.1x |
| Meridional velocity | 0.55915 | 0.0565 | 9.9x |
| SSH | 0.03611 | 0.00239 | 15.1x |
| All channels | 0.29469 | 0.0236 | 12.5x |

These are directly logged unweighted normalized MSEs; no dynamic loss weights need
to be undone. The main observations are:

- Velocity dominates the absolute error.
- Salinity has regressed the most relative to v2.
- Plain MSE fixes dashboard interpretability but not model quality.
- Full-data validation improved from 0.42385 at epoch 1 to 0.29469 at epoch 12,
  about 30%, and was still improving at the end.
- The matched proxy preserves the decisive v2-over-SamudraMulti ranking, so it is
  useful for screening large changes even though it does not predict absolute
  full-data MSE.
- Single-step maps, depth-resolved summaries, spectra, persistence/increment
  baselines, and patch-seam diagnostics are still needed to localize the errors.

## Most likely causes, in priority order

### 1. The historical dynamic loss was effectively unconstrained

The historical multi-resolution configuration allowed channel weights to span a
factor of about 2,043; the deepest salinity weight reached roughly 3,175 while the
lowest weight was around 1.55. This distorted optimization and made W&B's scaled
loss difficult to interpret. The completed baseline demonstrates the correction:
use plain MSE, not dynamic inverse-error weighting or a capped variant, for this
baseline funnel.

This remains consistent with the failure mode discussed in the
[Samudra2 paper](https://arxiv.org/abs/2606.02610): inverse-error weighting can amplify
channels with little learnable signal and degrade higher-variance variables.

### 2. The encoder creates an aggressive resolution-dependent bottleneck

Each 3-degree by 5-degree patch becomes one 128-dimensional vector:

- 1 degree: 15 pixels times 160 channels to 128 values, about 19-times compression.
- 1/2 degree: 60 pixels times 160 channels to 128 values, about 75-times compression.
- 1/4 degree: 240 pixels times 160 channels to 128 values, about 300-times compression.

The Perceiver uses 256 latent tokens internally but averages them into one vector per
physical patch. The decoder then reconstructs 154 output channels at every pixel
without a full-resolution encoder-to-decoder path. See the
[model configuration](../../configs/samudra_multi_om4/model.yaml) and
[encoder implementation](../../src/samudra/models/modules/encoder.py).

Perceiver IO makes compute scale favorably through a latent bottleneck, but that
bottleneck still needs enough capacity for the desired output structure
([Perceiver IO](https://arxiv.org/abs/2107.14795)). V2's U-Net skip connections do not
force all local information through one patch vector.

### 3. Optimization comparisons need update/sample parity

The completed baseline restored effective global batch 32, but its 12-epoch run used
a cosine schedule configured for 70 target epochs. Nominal epochs also represent
different numbers of samples and optimizer updates across full, proxy, and
multi-resolution configurations. Define schedules and comparisons by optimizer
updates or processed samples, and record both alongside wall time.

### 4. Inputs, normalization, and receptive field differ from v2

- The Rust-loader baseline uses `tau_hfds` and omits `hfds_anomalies`; add the anomaly
  input only after Rust support exists and its effect can be isolated.
- SamudraMulti uses InstanceNorm, while v2 uses BatchNorm at 1 degree.
- SamudraMulti processor dilation is `[1, 1, 1]`, versus v2's broader multiscale
  receptive field.

InstanceNorm may remove sample-specific spatial levels important to thermohaline
fields and SSH. Test channel LayerNorm, GroupNorm, or a resolution-conditioned
normalization rather than assuming BatchNorm is the only alternative.

### 5. Multiple resolutions may produce conflicting gradients

Similar scalar errors do not prove that resolution tasks cooperate. Before using a
multi-task optimizer, measure gradient norms and cosine similarity between 1-degree,
1/2-degree, and 1/4-degree losses on the shared processor. If conflicts are frequent,
then GradNorm or PCGrad becomes justified
([GradNorm](https://proceedings.mlr.press/v80/chen18a.html),
[PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)).

### 6. Plain MSE can encourage smooth fine-resolution predictions

Networks commonly learn low frequencies first, and ordinary MSE may hide loss of
high-frequency energy and patch artifacts
([spectral bias](https://proceedings.mlr.press/v97/rahaman19a.html)). Retain MSE as the
transparent baseline objective. Consider small spectral-amplitude or spatial-gradient
terms only after representation and optimization controls are understood, and always
continue reporting plain single-step MSE.

## A. Make experimentation faster and more transparent

### A1. Install single-step diagnostics before tuning

Log unweighted MSE separately for each resolution, variable, and depth. Add
single-step prediction maps, power spectra, increment and persistence baselines, and
patch-seam summaries. Log optimizer updates, processed samples, throughput, CPU/GPU
memory, and gradient cosine by resolution. Do not add rollout metrics to this plan.

### A2. Run encoder/decoder identity tests

Overfit one batch, then 32 to 128 fixed samples, at each resolution on identity
reconstruction. Inspect reconstruction MSE, spectra, and patch seams. If the 1/4-degree
identity task cannot approach zero, optimization tuning cannot repair the current
bottleneck. These tests should finish in minutes to roughly an hour.

### A3. Improve the deterministic 1-degree proxy

The existing 512-timestamp contiguous proxy is reproducible and gives about a
five-times end-to-end saving, but the current `TimeConfig` interface cannot select an
arbitrary stratified index set. Extend the interface if it can be done without making
the loader brittle; otherwise retain the existing slice as the documented fallback.

The desired proxy uses:

- one resolution and one-step prediction;
- 256 to 512 timestamps stratified across seasons and decades;
- a fixed one-step validation set;
- two seeds for screening and three for finalists.

Calibrate it by checking whether it ranks representative controls and candidates in
the same order as a medium- or full-data 1-degree run.

### A4. Increase fidelity progressively, while remaining single-step

Promote configurations through:

1. 1-degree, 256-to-512-timestamp single-step proxy.
2. Full-data 1-degree single-step training.
3. 1 degree plus 1/2 degree, sampling one resolution per optimizer update.
4. Add 1/4 degree with the same balanced single-step sampling.
5. Full-data, single-step, multi-resolution training.

Do not leave 1 degree while the corrected model remains more than roughly two times
v2 or exceeds the full-data 0.075 no-promotion threshold.

### A5. Remove avoidable implementation overhead

The decoder loops sequentially over 30 windows per forward. Vectorize independent
windows across the batch dimension and benchmark identical outputs, memory, and
throughput. Then benchmark:

- selective activation checkpointing, because the current wrapping may be nested;
- disabling heavyweight `wandb.watch` logging during sweeps;
- repacking or prefetching 1/4-degree Zarr data only after compute profiling shows
  that I/O is material.

The completed 1-degree proxy already reduces SamudraMulti from 353 to 63 rank-local
microbatches per epoch and from 2:16:50 to 0:27:27 training time. Preserve that
funnel while optimizing per-batch overhead.

## B. Improve single-step quality

Apply changes in this order and isolate each change against the completed controls.

### B1. Finish parity controls with plain MSE

Keep plain MSE, effective global batch 32, and the Rust loader. Move scheduling to
optimizer-update or processed-sample units. Add `hfds_anomalies` only if Rust-loader
support is implemented and test it as its own ablation.

### B2. Reduce the bottleneck

If identity tests confirm a representation limit, test multiple spatial output tokens
per physical patch, a fine-scale encoder-to-decoder connection, or
resolution-specific encoder/decoder heads or adapters. Keep the physical patch extent
fixed during this plan so representation capacity is the isolated variable.

### B3. Use a single-step curriculum

Pretrain at 1 degree, introduce 1/2 degree only after the 1-degree gate passes, and
then introduce 1/4 degree. Sample one resolution per optimizer update and retain the
single-step objective throughout. Do not train or evaluate multi-step rollouts in this
plan.

### B4. Test normalization and receptive field

Compare InstanceNorm with channel LayerNorm, GroupNorm, or a resolution-conditioned
normalization. Restore larger processor dilation or an additional processor level as
separate ablations.

### B5. Address resolution conflicts only if measured

Use balanced resolution sampling first. Apply GradNorm, PCGrad, or separate heads only
if shared-gradient diagnostics show persistent conflict.

### B6. Add spectral and gradient terms last

After the representation and optimization baseline is competitive, test a small
spectral-amplitude or spatial-gradient term as an ablation alongside plain MSE. Keep
unweighted single-step MSE as the primary promotion metric and do not jump directly
to an adversarial objective.

## First comparison and decision rule

The first funnel is:

1. matched v2 proxy control;
2. completed current-architecture SamudraMulti proxy;
3. plain-MSE parity candidate with update-based scheduling and, only if supported,
   an isolated `hfds_anomalies` ablation;
4. normalization and receptive-field ablations;
5. bottleneck changes only if the identity tests support them.

Promote a proxy candidate to full-data 1-degree training only when its all-channel MSE
is at most twice the matched v2 proxy (currently approximately 0.08575). Apply the
full-data 0.05/0.075 gate after promotion. If parity, normalization, and receptive
field changes cannot approach v2, change the encoder/decoder representation rather
than continuing optimizer tuning.
