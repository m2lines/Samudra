<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Coarse-latent, high-resolution dynamics results

This is the living results ledger for
[`coarse_latent_highres_dynamics_plan.md`](coarse_latent_highres_dynamics_plan.md).
It records failures and plan revisions as well as promoted results. An empty
stage means that it has not yet run, not that it passed.

## Current status

| Stage | Status | Current conclusion |
|---|---|---|
| Plan and harness specification | Complete | Patch-compression reconstruction and counterfactual dynamics probes are implemented and tested |
| S0-R synthetic reconstruction | Complete | Two seeds promote the 16-moment encoder plus continuous anchored hybrid |
| S0-D synthetic subgrid closure | Complete | Two seeds show that the promoted pair retains and dynamically uses subpatch phase |
| S1 OM4 learned inverse | In progress | Production integration passes; Torch proxy bring-up is next |
| S2 frozen-inverse dynamics | Pending | — |
| S3 full validation | Pending | — |

## Evidence inherited from the decoder investigation

The earlier position-anchored and resampling comparisons used one source latent
per physical source cell:

- learned-encoder decoder inputs had shape `[B, 32, H, W]`;
- outputs had shape `[B, 16, H_out, W_out]`; and
- tested routes were `8->8`, `16->8`, shifted `8->8`, and `8->16`.

They did not compress \(3\times5\) or \(6\times10\) physical cells into one
processor token. Their result—that coordinate resampling beat attention for a
cell-aligned representation—must not be used as evidence that bilinear expansion
is sufficient for a coarse patch representation.

The historical `main` configuration instead produces a \(60\times72\) latent
grid from every physical resolution:

| Physical resolution | Physical grid | Cells represented by one token |
|---|---|---|
| one degree | \(180\times360\) | \(3\times5\) |
| half degree | \(360\times720\) | \(6\times10\) |
| quarter degree | \(720\times1440\) | \(12\times20\) |

This correction of scope is the reason for S0-R and S0-D.

## Run ledger

| Run ID | Stage | Commit | Configuration | Status | Result |
|---|---|---|---|---|---|
| `local-s0-r-smooth-s0` | S0-R | working tree | 12 encoder/decoder arms, width 160, 300 updates, spectral decay 1, seed 0 | Complete | Smooth-field screen; provisional only |
| `local-s0-r-highfreq-s0` | S0-R | working tree | E1/E2 x D0/D1/D2/D3, width 160, 500 updates, equal-energy Fourier modes, seed 0 | Complete | Corrected query-residual D2 wins error but over-amplifies some fine-grid spectral bands |
| `local-s0-r-moments-s0` | S0-R | working tree | 12-moment encoder x D0/D1/D2/D3, width 160, 500 updates, equal-energy Fourier modes, seed 0 | Complete | D2 retains subpatch power and wins error; fine-grid patch seams remain excessive |
| `local-s0-r-moment-sweep-s0` | S0-R | working tree | Moments `{8,12,16}` x position bias `{2,8}`, 1,500 updates, LR 0.001, seed 0 | Complete | 16 moments and bias 8 win error/power, but containing-patch query coordinates create seams |
| `local-s0-r-continuous-s0` | S0-R | working tree | Continuous per-neighbor D2/D3, 16 moments, 1,500 updates, LR 0.001, seed 0 | Complete | Seam defect largely removed; optimization trails the discontinuous prototype |
| `local-s0-r-continuous-lr-s0` | S0-R | working tree | Continuous D3, 16 moments, 1,500 updates, LR 0.003, seed 0 | Complete | Low error, faithful subpatch power, and no material seam defect; confirmed by seed 1 |
| `local-s0-d-s0` | S0-D | working tree | Mean/E1/E2 with query-residual D2, 1,500 updates, seed 0 | Complete | Learned encoders recover a modest 12.6--13.1% of the held-out counterfactual response; mean-only recovers none |
| `local-s0-d-long-s0` | S0-D | working tree | E1/E2 with query-residual D2, 5,000 updates, seed 0 | Complete | E2 remains stable but plateaus; E1 destabilizes after 2,500 updates at LR 0.003 |
| `local-s0-d-moments-s0` | S0-D | working tree | Resolved plus 12 learned coordinate moments with query-residual D2, 1,500 updates, seed 0 | Complete | Retains subpatch phase strongly, but over-predicts counterfactual response amplitude |
| `local-s0-confirm` | S0-R/S0-D | working tree | 16 moments plus continuous D3, 1,500 updates, LR 0.003, reconstruction seeds `{0,1}`, dynamics seeds `{0,1}` | Complete | Passes reconstruction, spectral, seam, and counterfactual gates |

## S0-R synthetic reconstruction

### Harness validation

The new probe uses a \(12\times12\) latent grid. Its physical grids are
\(36\times60\) and \(72\times120\), so each token compresses exactly
\(3\times5\) or \(6\times10\) cells. It trains all four input/output-resolution
routes in rotation. Thirteen reconstruction shape/gradient tests and two
dynamics tests pass.

### Smooth-field screen

The first one-seed, 300-update screen used Fourier amplitudes decaying inversely
with wavenumber. Mean normalized MSE across the four routes was:

| Encoder | Decoder | Mean normalized MSE | Runtime, seconds |
|---|---|---:|---:|
| resolved plus subgrid | bilinear | **0.01253** | 11.8 |
| attention pool | zero-init hybrid | 0.01280 | 21.3 |
| attention pool | bilinear | 0.01312 | 11.8 |
| resolved plus subgrid | zero-init hybrid | 0.01483 | 21.2 |
| resolved plus subgrid | anchored correction prototype | 0.02333 | 16.0 |
| attention pool | anchored correction prototype | 0.02436 | 16.1 |
| existing patch Perceiver | bilinear | 0.03886 | 95.9 |
| existing patch Perceiver | coordinate MLP | 0.04508 | 94.2 |
| existing patch Perceiver | anchored correction prototype | 0.05994 | 99.6 |
| existing patch Perceiver | zero-init hybrid | 0.06251 | 104.6 |

This establishes that a smooth-field task can favor a coarse bilinear renderer
even when its source tokens summarize multiple cells. It does not establish that
bilinear rendering retains dynamically relevant subpatch modes.

The screen also exposed a defect in the tested anchored correction:
`LocalCoordinateAttentionCorrection` sent the query only into attention weights.
With radius zero, every output within a coarse patch was therefore identical;
with radius one, variation could only arise by reweighting neighboring tokens.
This is not the intended direct cross-attention architecture. D2 was revised to
add an explicit query residual before the output feed-forward block. The result
above is retained as a diagnostic rather than silently overwritten.

### Equal-energy subpatch screen

The corrected screen added modes above the \(12\times12\) coarse Nyquist with no
wavenumber decay. After 500 updates:

| Encoder | Decoder | Mean normalized MSE | Mean legacy high-k ratio | Runtime, seconds |
|---|---|---:|---:|---:|
| resolved plus subgrid | query-residual anchored | **0.11679** | 2.252 | 26.0 |
| resolved plus subgrid | coordinate MLP | 0.17317 | 3.207 | 15.9 |
| resolved plus subgrid | query-residual hybrid | 0.18461 | 2.000 | 35.5 |
| attention pool | query-residual anchored | 0.30065 | 0.692 | 27.0 |
| attention pool | query-residual hybrid | 0.30338 | 0.484 | 35.4 |
| resolved plus subgrid | bilinear | 0.30757 | 0.461 | 20.1 |
| attention pool | bilinear | 0.31043 | 0.459 | 19.7 |
| attention pool | coordinate MLP | 0.33850 | 2.677 | 16.0 |

The resolved-plus-subgrid encoder and query-residual anchored decoder reduce
error by 62.0% relative to the matching bilinear arm. This is the first direct
evidence in this project that learned query-conditioned prolongation is useful
when a token actually compresses subpatch structure.

The spectral result is not yet acceptable. On \(3\times5\)-cell outputs the
anchored model's legacy high-k ratios are about 0.65, while on
\(6\times10\)-cell outputs they are about 3.86. That metric uses a grid-relative
FFT threshold and changes its physical meaning across resolutions. The harness
now additionally reports power above the fixed coarse-grid Nyquist. Promotion
requires longer runs and the physically comparable diagnostic; the current
error win alone is insufficient.

The 12-moment encoder improves the comparison further:

| Decoder | Mean normalized MSE | Mean power above coarse Nyquist | Mean seam/all gradient-error ratio |
|---|---:|---:|---:|
| query-residual anchored | **0.06756** | **0.782** | 1.487 |
| coordinate MLP | 0.08424 | 0.703 | 1.362 |
| query-residual hybrid | 0.09223 | 0.688 | 1.353 |
| bilinear | 0.29612 | 0.084 | **0.942** |

Relative to the same encoder with bilinear rendering, anchored attention reduces
normalized error by 77.2% and restores most power above the coarse Nyquist. This
is strong evidence for learned restriction and query-conditioned prolongation.
However, on \(6\times10\)-cell outputs its patch-seam gradient-error ratio is
1.89--1.92. The next screen therefore sweeps the position bias and moment count
rather than promoting the lowest-MSE arm unchanged.

At 1,500 updates and learning rate 0.001, position-bias strength 8 consistently
beats strength 2. Moment count 16 gives the best mean NMSE and subpatch power:

| Moments | Position bias | Mean NMSE | Mean subpatch-power ratio | Fine-grid seam ratio |
|---:|---:|---:|---:|---:|
| 16 | 8 | **0.04141** | **0.875** | 1.940 |
| 12 | 8 | 0.05584 | 0.814 | 1.867 |
| 8 | 8 | 0.05688 | 0.823 | 1.884 |
| 16 | 2 | 0.17135 | 0.555 | **1.668** |
| 12 | 2 | 0.17307 | 0.582 | 1.684 |
| 8 | 2 | 0.17523 | 0.568 | 1.689 |

Inspection localizes the seam defect to the query residual's relative coordinate:
“position within the containing patch” jumps from \(+1\) to \(-1\) at every
boundary. A revised D2 evaluates a value function for every neighboring token
using the query's offset from that token, blends those values with anchored
attention, and sends only continuous absolute coordinates and scale ratios
through the direct query residual.

That continuous decoder changes the tradeoff:

| Decoder | Mean NMSE | Mean subpatch-power ratio | Fine-grid seam ratio |
|---|---:|---:|---:|
| continuous per-neighbor D2 | 0.12098 | 0.566 | **1.061** |
| bilinear base + continuous D2 residual | **0.10728** | **0.658** | 1.161 |

Thus the structural seam issue is largely fixed, but the continuous value
function trains more slowly at learning rate 0.001. Raising the learning rate to
0.003 for the continuous hybrid produces:

| Mean NMSE | Subpatch-power ratio by route | Fine-grid seam ratio |
|---:|---|---|
| **0.02401** | `{0.955, 0.949, 0.928, 0.922}` | `{1.074, 1.060}` |

This beats the discontinuous prototype while removing its seam defect. Seed 1
confirms mean NMSE 0.02892, subpatch-power ratios 0.837--0.873, and fine-grid
seam ratios 0.990--1.014. Across seeds, mean NMSE is 0.02646. The discontinuous
0.041 result remains a diagnostic, not a promotable renderer.

## S0-D synthetic subgrid closure

The probe generates fine-grid fields as a patch mean plus zero-mean subpatch
modes, advances them by one fine longitude cell, and trains a coarse convolutional
residual processor. Its counterfactual evaluation constructs pairs with identical
initial patch means but different anomaly positions, for which the next patch
means differ after advection. A patch-mean encoder is included as the negative
control.

The first 1,500-update results are:

| Encoder | Reconstruction NMSE | Forecast NMSE | Coarse forecast NMSE | Latent pair RMS | Predicted/true counterfactual response | Counterfactual difference NMSE |
|---|---:|---:|---:|---:|---:|---:|
| patch mean only | 0.5049 | 0.5057 | 0.02316 | \(3.85\times10^{-8}\) | \(2.22\times10^{-6}\) | 1.000 |
| attention pooling | **0.4468** | **0.4440** | **0.00825** | 0.0272 | **0.1307** | 0.833 |
| resolved plus subgrid | 0.4483 | 0.4449 | 0.00946 | 0.0144 | 0.1261 | **0.811** |

The initial patch-mean difference of every counterfactual pair is below
\(4.8\times10^{-7}\), while the true post-advection coarse difference RMS is
0.11394. The patch-mean encoder is therefore correctly blind. Both learned
encoders preserve information that changes the predicted coarse tendency and
reduce coarse forecast error by 59--64% relative to the mean-only control.

The recovered counterfactual amplitude is only about 13%, and reconstruction
still loses roughly 45% of normalized variance. This is positive but not yet a
promotion result. At 5,000 updates the resolved-plus-subgrid arm remains stable
but essentially unchanged: reconstruction NMSE is 0.443, forecast NMSE is 0.442,
and the counterfactual response ratio is 0.132. The attention-pooling arm becomes
unstable after 2,500 updates at learning rate 0.003 as its processor residual
scale grows from about 0.006 to 0.072. Its final forecast NMSE regresses to 0.501.
The plateau therefore cannot be resolved merely by extending this learning-rate
schedule.

The pooled encoders emit only one aggregate per attention head. A new
resolved-plus-moments candidate instead projects patch anomalies onto 12 learned
continuous functions of relative within-patch coordinates and packs those
coefficients into latent channels. At the same 1,500-update budget:

| Encoder | Reconstruction NMSE | Forecast NMSE | Coarse forecast NMSE | Latent pair RMS | Predicted/true counterfactual response | Counterfactual difference NMSE |
|---|---:|---:|---:|---:|---:|---:|
| resolved plus 12 moments | **0.3850** | **0.3858** | **0.00782** | **0.2441** | **1.4846** | **0.778** |

The moment bank demonstrates the desired mechanism: a spatially coarse token can
retain which side of a patch contains an anomaly, and the coarse processor uses
that information to alter the next coarse tendency. Its response is 48% too
large, so this is evidence for representational sufficiency rather than a
calibrated dynamics result. Follow-up must sweep moment count, latent width, and
a lower learning rate, and must evaluate in-distribution held-out subpatch modes
separately from the deliberately shifted counterfactual.

The corrected continuous hybrid was then used in two 1,500-update dynamics
seeds:

| Seed | Reconstruction NMSE | Forecast NMSE | Coarse forecast NMSE | Response ratio | Counterfactual difference NMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3788 | 0.3811 | 0.00554 | 1.322 | 0.299 |
| 1 | 0.3785 | 0.3802 | 0.00234 | 1.060 | 0.073 |
| Mean | 0.3786 | 0.3806 | **0.00394** | **1.191** | **0.186** |

The mean-only control's response ratio was \(2.2\times10^{-6}\) and its
counterfactual difference NMSE was 1.0. The promoted representation therefore
does not merely reconstruct fine detail: its coarse processor uses subpatch
phase to predict a future patch-mean change. The response remains 19% high on
average and seed spread is material, so calibration and uncertainty remain S1/S2
evaluation requirements.

## S1 OM4 learned inverse

Promoted synthetic components:

- encoder: resolved patch mean plus 16 learned continuous relative-coordinate
  moments;
- decoder: physical-coordinate bilinear base plus a zero-initialized,
  continuous per-neighbor position-anchored attention residual;
- latent width: 160 for the first OM4 proxy;
- attention: four heads, head width 32, one coarse-token ring, position bias 8;
- key normalization with an unnormalized latent value path; and
- no additive absolute position or scale embeddings in the reconstructive
  latent.

Production implementation and configuration are in progress.

The production modules and proxy configuration are now implemented:

- `PatchMomentEncoder` uses latitude-area-weighted patch means and 16 learned
  continuous relative-coordinate moments;
- `ContinuousCoordinateAttentionCorrection` chunks output queries, keeps keys
  normalized and values raw, and conditions every local value on its continuous
  query offset;
- `ContinuousResampleAttentionResidualDecoder` preserves the coordinate
  resampling base exactly at initialization; and
- `train_cross_1_halfdeg_coarse_moment_attention_proxy.yaml` fixes the latent
  grid at \(60\times72\) for both physical resolutions.

Nineteen synthetic harness tests and 106 relevant encoder, decoder, config, and
SamudraMulti tests pass. The new training and model YAML files also pass schema
validation, and all pre-commit checks pass. The broader CPU suite reports 540
passes, two skips, and ten expected failures; six setup errors are confined to
the pre-existing `implementation: auto` Perceiver test configuration selecting
Flash Perceiver in a local environment where `flash_perceiver` is not installed.
None reaches the new modules. Cluster bring-up remains pending.

## S2 frozen-inverse latent dynamics

Pending S1 promotion.

## S3 full validation

Pending S2 promotion and review.

## Decision log

### 2026-07-24: reopen coarse-patch decoder selection

Position-anchored direct cross-attention is restored as a primary decoder
candidate for a spatially coarse latent grid. Coordinate resampling remains the
native-cell baseline and a possible coarse base path, but the earlier experiments
did not test within-patch decompression.

### 2026-07-24: require an explicit query residual in D2

The retained `LocalCoordinateAttentionCorrection` is suitable as a correction
to a spatial base, but its query affects only attention weights. The sole-renderer
D2 candidate now carries continuous output-query features directly into the
decoded hidden state. Results from the earlier correction-only prototype remain
listed but cannot select the direct-attention architecture.

### 2026-07-24: replace single pooled anomaly summary with a moment bank

Extending E1/E2 training from 1,500 to 5,000 updates does not improve the
subgrid counterfactual. A bank of learned continuous relative-coordinate moments
immediately exposes the missing phase information. This candidate is added to
S0-R and becomes the leading encoder for the next closure sweep; the original
resolved-plus-one-pooled-summary arm remains as an ablation.

### 2026-07-24: make query conditioning continuous across patches

Directly feeding within-containing-patch coordinates through the decoder query
residual produces excellent MSE but boundary discontinuities. The candidate
decoder now conditions each neighbor's value on its own continuous query offset
and blends overlapping predictions. The zero-initialized hybrid remains a
candidate because it currently optimizes better than the attention-only form.
