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
| S1 OM4 learned inverse | In progress | One-/half-degree bring-up and expanded audit pass; matched 60-epoch convergence, seed, and bilinear-control runs are running |
| S2 frozen-inverse dynamics | Implementation complete; execution pending | Physical-only, latent-teacher-only, and combined frozen-inverse objectives are implemented and tested; launch awaits the matched S1 selection |
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
| `2026-07-24-coarse-moment-attn-s1-proxy-s0` | S1 | `6332322c` | OM4 1°/½°, fixed 60×72 latent, width 160, 32 samples, 10 epochs, eight RTX6000 GPUs | Canceled (`14717201`) | Replaced before launch because the Lzanna allocation was at its CPU limit |
| `2026-07-24-coarse-moment-attn-s1-proxy-h200-s0` | S1 | `6332322c` | OM4 1°/½°, fixed 60×72 latent, width 160, 32 samples, 10 epochs, eight H200 GPUs | Canceled (`14717724`) | Still limited by Courant aggregate GPU quota after bring-up interval; replaced by two-GPU job |
| `2026-07-24-coarse-moment-attn-s1-proxy-2h200-s0` | S1 | `6332322c` | Same cross-resolution proxy on two H200 GPUs | Canceled (`14718063`) | H200 quota remained saturated; replaced by RTX6000 request |
| `2026-07-24-coarse-moment-attn-s1-proxy-2rtx-s0` | S1 | `6332322c` | Same cross-resolution proxy on two RTX6000 GPUs, 350 GB | Canceled (`14718370`) | Courant aggregate memory limit; request was much larger than measured need |
| `2026-07-24-coarse-moment-attn-s1-proxy-2rtx-64g-s0` | S1 | `6332322c` | Same cross-resolution proxy on two RTX6000 GPUs, 64 GB | Canceled (`14718372`) | Replaced while testing preemptible admission |
| `2026-07-24-coarse-moment-attn-s1-proxy-prem-s0` | S1 | `2e79511b` | Attempted generic H200/RTX6000 preemptible partitions by setting `--partition` directly | Rejected before submission | The rejection was caused by bypassing Torch's comment-driven preemption routing, not by lack of project authorization |
| `2026-07-24-coarse-moment-attn-s1-proxy-courant-requeue-s0` | S1 | `2e79511b` | Same proxy on two H200 GPUs, 64 GB, Courant partition | Canceled (`14722730`) | Replaced by the correctly routed preemption-only submission before launch |
| `2026-07-24-coarse-moment-attn-s1-proxy-preempt-only-s0` | S1 | `2e79511b` | Same proxy on two H200 GPUs, 64 GB; no explicit partition; `h200` constraint and `preemption=yes;preemption_partitions_only=yes;requeue=true` comment | Complete (`14723265`) | Ten epochs/640 updates completed in 9m12s with train/validation normalized MSE 0.161/0.161 and checkpoints for every epoch |
| `2026-07-24-coarse-moment-attn-s1-60ep-s0` | S1 | `2e79511b` | Resume the cross-resolution hybrid seed-15 checkpoint from epoch 10 through epoch 60 on two preemptible H200 GPUs | Canceled (`14724411`) | The two-GPU request remained under `gpu48` with `QOSMaxGRESPerUser`; replaced by a one-GPU, batch-two request with the same global batch and update count |
| `2026-07-24-coarse-moment-attn-s1-60ep-s1` | S1 | `2e79511b` | Independent cross-resolution hybrid run through epoch 60 with model seed 16 on two preemptible H200 GPUs | Canceled (`14724412`) | Replaced by the matched one-GPU request before launch |
| `2026-07-24-coarse-moment-bilinear-s1-60ep-s0` | S1 | `2e79511b` | Cross-resolution 16-moment encoder with bilinear-only D0 through epoch 60 on two preemptible H200 GPUs | Canceled (`14724413`) | Replaced by the matched one-GPU request before launch |
| `2026-07-24-coarse-moment-attn-s1-60ep-s0-1h200` | S1 | `2e79511b` | Resume the hybrid seed-15 checkpoint through epoch 60 on one preemptible H200 with batch size two | Running (`14724824`) | Preserves global batch two, route-homogeneous batches, and 64 optimizer updates per epoch |
| `2026-07-24-coarse-moment-attn-s1-60ep-s1-1h200` | S1 | `2e79511b` | Independent hybrid seed 16 through epoch 60 on one preemptible H200 with batch size two | Running (`14724820`) | Seed-stability run at the full 3,840-update S1 proxy budget |
| `2026-07-24-coarse-moment-bilinear-s1-60ep-s0-1h200` | S1 | `2e79511b` | Bilinear-only D0 control through epoch 60 on one preemptible H200 with batch size two | Running (`14724821`) | Matched control for the continuous anchored decoder branch |
| `2026-07-24-coarse-moment-attn-s1-proxy-preempt-only-s0-audit-v1` | S1 audit | `34b6bbba` | Wet-cell, gradient-power, seam, synchronized-latent, and cross-output patch-mean audit of the completed epoch-10 hybrid | Failed (`14724704`) | Route diagnostics completed, but the new cross-resolution pass assumed a Python loader attribute absent from the configured Rust loader |
| `2026-07-24-coarse-moment-attn-s1-proxy-preempt-only-s0-audit-v2` | S1 audit | `49c897d4` | Corrected native-loader-compatible audit of the same epoch-10 checkpoint | Complete (`14725135`) | Four routes audited over 73 batches each; synchronized latent cosine similarity 0.972 and output patch-mean symmetric normalized MSE 0.0158 |
| `local-s1-1deg-bringup-s0` | S1 | working tree after `6332322c` | Local OM4 1°, fixed 60×72 latent, one training sample and three validation samples | Complete | 1.62 s training step; 1.23 GB peak GPU memory; full model/data/checkpoint path succeeds |
| `local-s1-1deg-proxy-s0` | S1 | working tree after `6332322c` | Local OM4 1°, fixed 60×72 latent, width 160, 32 samples, seed 15; 10 epochs then resumed to 60 | Complete | Train/validation normalized MSE 0.058/0.058 after 1,920 updates; no seam spike; retains 62% of target gradient power |
| `local-s1-1deg-bilinear-s0` | S1 | working tree after `6332322c` | Same as local attention proxy, but D0 bilinear decoder only | Complete | Validation MSE 0.208; the continuous anchored hybrid is 29.3% lower at matched updates |
| `local-s1-1deg-radius0-s0` | S1 | working tree after `6332322c` | Same as local attention proxy, but decoder neighborhood radius zero | Complete | Validation MSE 0.153; one-ring blending improves error 3.9% at identical parameter count |
| `local-s2-latent-only-realdata-smoke` | S2 integration | working tree after `49c897d4` | One-degree OM4, one optimizer update at depth one, latent-only objective, frozen randomly initialized inverse | Complete | Native loader, 60×72 rollout, target re-encoding, wet-token loss, backward pass, physical validation, and checkpoint writing succeed; integration evidence only |
| `2026-07-24-coarse-latent-s2-checkpoint-smoke` | S2 integration | `1aa6672a` | Epoch-10 cross-resolution S1 checkpoint, half-degree OM4, one latent-only optimizer update at depth one | Complete (`14726481`) | Partial finetune loaded the frozen inverse exactly, initialized 67 allowlisted processor/boundary tensors, and completed training, physical validation, and checkpoint writes in 70 seconds |

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

The production modules and proxy configuration are implemented:

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
None reaches the new modules.

### One-/half-degree Torch bring-up

The first synchronized cross-resolution run trained all four `1->1`,
`1->1/2`, `1/2->1`, and `1/2->1/2` routes on two H200 GPUs with global batch
size two. Every input was encoded onto the same \(160\times60\times72\) latent.
The mixed schedule produced 64 optimizer updates per epoch, so ten epochs
correspond to 640 updates. Training completed without preemption or error in
9 minutes 12 seconds; peak per-rank GPU memory reported by the training loop
was approximately 5.5 GB.

Mean normalized MSE decreased monotonically:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.346 | 0.250 | 0.214 | 0.190 | 0.181 | 0.174 | 0.171 | 0.167 | 0.164 | **0.161** |
| Validation | 0.266 | 0.234 | 0.197 | 0.185 | 0.178 | 0.174 | 0.171 | 0.168 | 0.165 | **0.161** |

This establishes cross-resolution trainability but is not yet the selection
result: the matched seed, bilinear control, and structural audits require the
full 60-epoch/3,840-update budget. Those runs use one H200 with batch size two,
which preserves the original global batch and route-homogeneous optimizer
steps while avoiding a two-GPU queue limit.

The expanded audit of this epoch-10 checkpoint then completed on all 73 held-out
batches per route. Synchronized one- and half-degree inputs produce coarse
latents with mean token cosine similarity 0.972 and symmetric normalized MSE
0.091. Decoding the same latent onto both output grids gives patch-mean
symmetric normalized MSE 0.0158. Wet-cell channel-mean MSE is 0.236 for
`1->1`, 0.351 for `1->1/2`, 0.241 for `1/2->1`, and 0.353 for
`1/2->1/2`. The prediction/target gradient-power ratios are respectively
0.281, 0.139, 0.275, and 0.137. Seam/all gradient-error ratios remain below
one on all routes (0.831, 0.904, 0.836, and 0.896), so the early
cross-resolution weakness is attenuated fine-grid structure rather than a patch
boundary discontinuity. The epoch-60 audits are required to determine how much
of that attenuation is optimization-limited.

### Local one-degree bring-up

The first real-data run compresses each \(180\times360\), 154-channel state to
a \(160\times60\times72\) latent and reconstructs on the original grid. The
model has 536,436 trainable parameters. On a local NVIDIA GB10, the first
training step took 1.62 seconds and used 1.23 GB peak GPU memory; warmed-up
epochs averaged 0.15--0.19 seconds per update. The full 320-update proxy
completed in 3 minutes 22 seconds.

Validation normalized MSE decreased monotonically by epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.276 | 0.232 | 0.216 | 0.202 | 0.182 | 0.166 | 0.160 | 0.154 | 0.151 | **0.147** |

The candidate was then resumed to epoch 60, or 1,920 optimizer updates. Final
train and validation normalized MSE are both 0.058. The 50-epoch continuation
took 18 minutes 26 seconds. This demonstrates that the promoted inverse is
trainable on actual OM4 and that the local machine is fast enough for
one-degree ablations. It is not yet a full promotion result because one-degree
data alone cannot test resolution consistency.

The matched encoder with a bilinear-only decoder reaches validation MSE 0.208
after the same 320 updates, versus 0.147 for the continuous anchored hybrid:
a 29.3% relative error reduction. The control has 328,026 parameters versus
536,436 for the hybrid, so this result establishes the value of the learned
decoder branch but does not by itself distinguish coordinate routing from added
capacity. The parameter-matched radius-zero decoder reaches 0.153. The one-ring
hybrid is therefore 3.9% better at identical capacity, while most of the 29.3%
gain over bilinear remains attributable to learned coordinate-conditioned
within-patch unpacking. The synthetic seam metric provides the stronger evidence
for overlapping neighbor predictions: radius-zero query coordinates are
piecewise patch-local, whereas the promoted decoder blends continuous
per-neighbor predictions.

The checkpoint audit uses latitude-area-weighted MSE over wet cells only, so its
absolute MSE is larger than the training loss, which averages masked zeros over
the rectangular grid. Update-matched epoch-10 results are:

| Decoder | Wet-cell channel-mean MSE | Median channel MSE | Gradient-power ratio | Seam/all gradient-error ratio | Correction/prediction RMS |
|---|---:|---:|---:|---:|---:|
| Bilinear D0 | 0.372 | 0.108 | 0.049 | 0.942 | — |
| Continuous hybrid, radius 0 | 0.279 | 0.064 | 0.157 | **0.792** | 0.689 |
| Continuous hybrid, radius 1 | **0.270** | **0.054** | **0.204** | 0.830 | 0.733 |

At matched updates, learned coordinate-conditioned decoding reduces wet-cell
MSE 27.3% relative to bilinear and retains over four times as much physical-grid
gradient power. One-ring blending improves wet-cell MSE 3.4% over the
parameter-matched radius-zero decoder. None has a patch-seam error spike:
boundary gradient error is below the all-edge average.

At epoch 60, the radius-one candidate reaches wet-cell channel-mean MSE 0.115
and median channel MSE 0.030. It retains 61.8% of target gradient power, has a
seam/all gradient-error ratio of 0.798, and the learned correction RMS is 79.5%
of prediction RMS. Thus the correction is the dominant fine-structure route,
not a small perturbation of bilinear interpolation. The remaining error is
strongly channel dependent: the worst channels are deep `vo` and `uo`, with
wet-cell MSE roughly 0.33--0.44. The principal remaining reconstruction concern
is gradient attenuation, not patch seams.

## S2 frozen-inverse latent dynamics

Implementation is complete and execution awaits the matched S1 checkpoint
selection. The S2 model keeps the S1 `PatchMomentEncoder` and
`ContinuousResampleAttentionResidualDecoder`, changes encoder geometry mode from
`none` to parameter-free `sidecar`, enables one shared processor invocation per
physical step, and freezes all `encoder.*` and `decoder.*` parameters.

For selected depth \(n\), the model now exposes the latent rollout before
decoding and can optimize

\[
w_x L_x+
\lambda_z
\frac{\sum_{b,c,i,j}m_{ij}
\left(z^{\rm forecast}_{bcij}
-\operatorname{sg}E(x_{t+n})_{bcij}\right)^2}
{\sum_{b,c,i,j}m_{ij}},
\]

where \(m\) marks a coarse token wet if any target channel/cell mapped to that
token is wet. The target encoding is evaluated under `no_grad`. A latent-only
arm does not invoke the decoder during the training loss, while ordinary
validation and inference still decode physical forecasts.

The exact implementation/configuration map is:

- `physical_forecast_loss_weight` is \(w_x\);
- `latent_teacher_loss_weight` is \(\lambda_z\);
- [`SamudraMulti.latent_rollout`](../../src/samudra/models/samudra_multi.py)
  applies the processor \(n\) times with the \(m\)-th aligned boundary state;
- [`SamudraMulti.latent_teacher_loss`](../../src/samudra/models/samudra_multi.py)
  defines the stop-gradient wet-token objective;
- [`model_iterable_inverse_coarse_moment_attention.yaml`](../../configs/samudra_multi_om4/model_iterable_inverse_coarse_moment_attention.yaml)
  fixes the S2 architecture; and
- [`train_halfdeg_coarse_latent_dynamics_proxy.yaml`](../../configs/samudra_multi_om4/train_halfdeg_coarse_latent_dynamics_proxy.yaml)
  trains only from half-degree targets at depths `{1,2,4}` for the first causal
  screen; and
- [`validate_cross_1_halfdeg_coarse_latent_dynamics.yaml`](../../configs/samudra_multi_om4/validate_cross_1_halfdeg_coarse_latent_dynamics.yaml)
  evaluates each resulting checkpoint on all four one-/half-degree routes; and
- [`audit_coarse_dynamics.py`](../../scripts/audit_coarse_dynamics.py) checks
  frozen-inverse equality, synchronized latent agreement through depths
  `{0,1,2,4}`, latent-teacher error, forcing sensitivity, cross-output
  patch-mean consistency, and all 160 learned latent-channel residual scales;
  and
- [`submit_coarse_latent_s2.sh`](../../scripts/submit_coarse_latent_s2.sh)
  submits the four objective arms from one explicitly selected inverse with
  matched seed, data, update budget, resource request, and W&B group.

The four objective arms are `(w_x, lambda_z) = (1,0)`, `(0,1)`, `(1,0.01)`,
and `(1,0.1)`. The proxy contains 768 optimizer updates, 256 at each depth,
with global batch two. Per-epoch selection uses a fixed disjoint three-month
half-degree validation interval; the checkpoint-only audit uses the full
held-out year and all four physical input/output routes. Focused model/config
tests, full config-schema validation, Ruff, and MyPy pass.

A one-degree real-data latent-only smoke test also completed one optimizer
update and physical validation on the local GPU. It froze all 536,436 inverse
parameters and trained the 60,850,520-parameter boundary/processor path,
re-encoded the target on the same \(60\times72\) grid, and wrote a resumable
checkpoint. Because no S1 checkpoint was loaded for this integration-only
smoke, its numerical losses are not scientific results.

The corresponding Torch checkpoint-composition smoke then loaded the completed
epoch-10 S1 inverse, froze the same 30 tensors/536,436 parameters, initialized
exactly 67 allowlisted tensors belonging to the processor, geometry sidecar,
boundary encoder, and latent residual scale, and completed a half-degree
latent-only update plus physical validation. Job `14726481` exited zero after
70 seconds. This closes the checkpoint-compatibility risk; its losses are also
not used for model selection.

## S3 full validation

Pending S2 objective selection. The prepared
[`train_cross_1_halfdeg_coarse_latent_dynamics_full.yaml`](../../configs/samudra_multi_om4/train_cross_1_halfdeg_coarse_latent_dynamics_full.yaml)
uses all four one-/half-degree routes, physical depths `{1,2,4}`, global batch
32 on eight GPUs, and approximately the same 6,392-update budget as the
completed native-grid reference. Quarter-degree data are deliberately absent.

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
