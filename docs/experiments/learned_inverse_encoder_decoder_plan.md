<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Learned encoder/decoder inverse experiment plan

## Decision and scope

The encoder should remain a learned representation model. This plan does **not**
initialize it as an identity map, reserve physical-state channels, tie its weights
to the decoder, or require a global mathematical inverse. The evidence does not
justify those restrictions: in the completed one-degree, processor-bypassed
factorial, the learned Perceiver encoder with a direct decoder reached MSE
`0.025431`, compared with `0.012083` for the direct encoder and decoder. Changing
only the decoder increased MSE to `0.279303`. The decoder is the primary failure;
the existing encoder is a viable starting point.

The operational requirement is instead

```text
z_0       = E_r(x_t, b_t; g_r)
z_k       = P^k(z_0; b, g), k = 0, ..., N
x_hat_s,k = D_s(z_k; g_s)
```

where `r` and `s` denote input and output grids and `g` contains grid geometry. At
processor depth zero, same-grid reconstruction should satisfy
`D_r(E_r(x)) ~= x`. For different grids, the target is the same physical field on
the requested output grid, not an exact algebraic inverse. The decoder should also
remain usable after any supported processor depth. This is a learned left-inverse
condition on ocean states, not a demand that every latent vector or every encoder
input (history and forcing included) be recoverable.

The candidate decoder is:

```text
base       = pointwise_decode(coordinate_resample(z_k, source_grid, output_grid))
correction = zero_output(local_position_attention(z_k, source_grid, output_grid))
prediction = base + correction
```

Only the correction's final output projection is initialized to zero. The encoder
and base projection use their ordinary learned initialization and train jointly.
The implementation is:

- production resampling base: `src/samudra/models/modules/decoder.py`,
  `ResampleProjectionDecoder`;
- production hybrid: the same module's
  `ResampleAttentionResidualDecoder`;
- position-anchored correction: the same module's
  `LocalCoordinateAttentionCorrection`;
- zero-to-N model path and zero-depth decode: `src/samudra/models/samudra_multi.py`;
- processor geometry sidecar: `src/samudra/models/modules/augment_input.py`,
  `ProcessorGeometryConditioner`.

The former `use_fine_scale_queries` feature has been removed. That path could carry
the original prognostic input around the processor, allowing the decoder to
reconstruct without using `P^k(E(x))`. Static geometry and future forcing may
condition the processor and decoder, but the prognostic state must pass through the
latent route.

## Implementation status

The development prerequisites are now present on this branch:

- identity diagnostics use disjoint deterministic train and held-out samples;
- `identity_eval_only: true` loads an explicit finetune checkpoint and evaluates
  every fixed held-out route once, without a redundant training-window pass,
  backward passes, or optimizer updates;
- paired reconstruction can read a destination resolution at the exact input
  timestamps, balance fixed samples across shape-distinct routes, and report both
  learned and deterministic-resampler errors in normalized and physical units;
- the diagnostic separately reports interpolation in physical space and directly
  in the source-normalized coordinate system, exposing cross-resolution errors
  caused by resolution-specific means and standard deviations;
- encoder geometry can be additive, absent, or supplied to each processor call as
  a zero-initialized position/scale sidecar;
- physical-coordinate resampling and the bounded zero-initialized local-attention
  correction are production decoder options;
- `SamudraMulti` exposes encode/process/decode, supports one shared processor at
  depths `0`, `1`, `2`, `4`, or any other non-negative integer when widths match,
  and has gradient tests across repeated applications;
- forecast training can opt into a same-grid zero-depth reconstruction auxiliary
  loss through `zero_depth_reconstruction_weight`; and
- a zero-depth encoder/decoder checkpoint can initialize a processor-present
  finetune with `finetune_allowed_missing_prefixes: ["processor.",
  "processor_geometry."]`; every non-allowlisted missing key and every unexpected
  checkpoint key remains fatal, and EMA restarts for the composed model; and
- the synthetic probe now learns its encoder rather than copying target channels
  into the decoder input, uses fresh analytic coefficients and disjoint evaluation
  coefficients, and reports amplitude, bias, and high-wavenumber diagnostics.

The checked-in zero-depth controls are
`model_learned_inverse_resample.yaml` and
`model_learned_inverse_hybrid.yaml`; the processor-present candidate is
`model_iterable_inverse_hybrid.yaml`. No experiment result is implied by these
implementations. Training and evaluation remain to be run, and the candidate set
and defaults below remain provisional under the selection-logic revision rule.

The completed learned-encoder S0 screen supplies the first revision. At 2,000
updates and three seeds, physical-coordinate resampling plus projection beat both
the hybrid and position-only attention on same-grid, downsampled, longitude-shifted,
and upsampled routes. S1 therefore promotes the simple physical resampler. The
hybrid remains implemented as a fallback but does not receive a parallel ocean
sweep unless S1 spectra reveal an error the base cannot represent.

## Questions to resolve

The experiments are designed to answer six questions separately.

1. Can a jointly learned encoder and resampling decoder generalize the zero-depth
   reconstruction map to held-out ocean states rather than merely fit a fixed set?
2. Do the encoder's additive position and scale embeddings help the processor, or
   do they make a shared pointwise decoder spend capacity removing location-dependent
   signals from the content representation?
3. Is position/scale information best added to content, omitted from the encoder
   output, or carried separately as immutable geometry?
4. Does a local position-anchored residual improve reconstruction and
   cross-resolution rendering beyond the resampling base without damaging the
   stable base route?
5. Should attention keys be normalized while values remain unnormalized so physical
   amplitude does not have to pass through LayerNorm?
6. Does a candidate retain zero-depth reconstruction while the processor is trained
   and evaluated at multiple iteration counts?

## Required development

Implement these changes in order. Each step should land with focused unit tests and
should be usable without enabling later steps.

### D0. Make the reconstruction diagnostic measure generalization

Extend `src/samudra/identity.py` with independent deterministic training and held-out
sample sets. The current diagnostic repeatedly trains and evaluates on the same
fixed validation batches, which is useful for capacity tests but cannot distinguish
a learned inverse from memorization.

Proposed configuration fields:

```yaml
identity_train_samples: 32
identity_eval_samples: 32
identity_train_offset: 0
identity_eval_offset: 32
identity_input_source: 0
identity_output_source: 0
processor_iterations: 0
```

Log train and held-out metrics separately. Retain all existing channel, variable,
depth, spectrum, high-wavenumber, and seam diagnostics. Add:

- prediction standard-deviation ratios and mean bias in target-standard-deviation
  units by channel;
- base MSE, final MSE, and correction-only MSE;
- `RMS(correction) / RMS(base)`;
- encoder, base-decoder, and correction gradient norms;
- parameter count, peak memory, samples/second, and optimizer updates.

Add an evaluation mode that loads one checkpoint and evaluates every supported
same-resolution and cross-resolution source/target pair without further fitting.

### D1. Make encoder geometry injection configurable

The current `PerceiverEncoder` adds learned position and scale embeddings directly
to its content output. Add one encoder option with three modes:

```yaml
encoder:
  geometry_mode: additive  # additive | none | sidecar
```

- `additive` is the current behavior and remains the control.
- `none` removes the post-Perceiver position and scale additions. Intra-patch
  Fourier coordinates remain available to the Perceiver, so this does not make the
  encoder spatially unaware inside a patch.
- `sidecar` keeps content and geometry separate. The encoder returns learned content
  plus immutable source-grid coordinates/scale. The decoder uses the geometry for
  resampling and attention routing, and the processor receives it as conditioning
  at every iteration rather than as a one-time addition to content.

Do not silently concatenate geometry into the reconstructive tensor; that would
confound geometry mode with latent width. Source coordinates now travel separately
to the coordinate-aware decoder. At processor depth zero, `none` and `sidecar` are
therefore the same model and must not be reported as independent experimental arms.
Complete the distinction in D4 by conditioning every processor application on the
sidecar, then compare `none` with `sidecar` only in processor-present runs.

Tests must establish:

- `additive` is bitwise-equivalent to the current default;
- `none` changes only the post-Perceiver additions;
- geometry tensors are deterministic, batch-independent, correctly periodic in
  longitude, and excluded from decoder content values;
- a change in output resolution changes routing geometry but does not mutate the
  learned source content.

### D2. Replace shape-only resizing with physical-coordinate resampling

`ResampleProjectionDecoder` currently calls `F.interpolate` using only output
shape. Introduce a coordinate resampler that consumes source and target latitude and
longitude arrays. Its first implementation should be deterministic and have no
learned parameters:

- periodic linear interpolation in longitude;
- latitude interpolation using actual cell-center coordinates;
- explicit masks for invalid/padded source cells;
- exact no-op when source and target coordinates match;
- float32 interpolation accumulation under bfloat16 model execution.

Keep the current `F.interpolate` implementation as a named control. Unit tests
should cover identical grids, longitude wraparound, shifted longitude grids,
nonuniform latitude, 1-to-1/2-degree upsampling, 1/2-to-1-degree downsampling, masks,
and gradients.

### D3. Promote the hybrid decoder from probe to production code

Add a production decoder containing the coordinate-resampling base and a local
position-anchored attention correction. The attention implementation should use:

```text
Q = projection(output coordinates and optional query features)
K = projection(LayerNorm(source content))
V = projection(source content)                 # no LayerNorm on values
```

This isolates normalization to routing. The base path never passes through
LayerNorm. Initialize only the correction's final output projection to zero. Add a
configuration switch for the old normalized-value behavior as an ablation, not as
the default.

The correction must use a bounded physical neighborhood rather than attend to an
entire output window. Start with a 3-by-3 or 5-by-5 source-cell neighborhood,
periodic longitude, a validity mask, and relative physical-coordinate features.
Do not introduce a second learned latent bank.

Required tests:

- the hybrid is exactly equal to its base at initialization;
- both branches receive gradients after one optimizer update;
- raw-value and normalized-value modes have the expected data paths;
- same-grid, upsampled, downsampled, shifted, periodic, and masked cases work;
- changing output grid does not change parameter shapes;
- local attention memory scales linearly with output pixels times neighborhood size,
  not quadratically with global grid size.

### D4. Expose encode, process, and decode and make processor iteration legal

Refactor `SamudraMulti.forward_once` into independently callable `encode`, `process`,
and `decode` operations, while preserving the existing one-call behavior. The
current encoder emits 128 channels and the configured UNet emits 380, so the same
processor cannot simply be applied twice. Introduce an explicit canonical latent
width and require the repeated processor core to map that width to itself.

The least disruptive first control is:

- learned Perceiver encoder output width equals the processor's first/output width;
- the UNet processor maps `C -> C`;
- the decoder always accepts `C`, including at iteration zero;
- one shared processor instance is applied zero through `N` times.

Start with `C=380`, matching the existing processor output. A later width sweep may
change the whole processor consistently; do not place an iteration-dependent
adapter between `P` applications. Feed sidecar geometry to every processor
application in the same way.

Add equivalence tests for the existing one-processor path and tests for iterations
`0`, `1`, `2`, and `4`, including checkpointing and gradients. Log reconstruction
at iteration zero throughout forecast training so the encoder/decoder contract
cannot regress unnoticed.

The processor contract is now settled: one application is one physical time step.
For step `m`, separately encode exactly one aligned boundary state `b_m`, pass it
to the shared processor, and keep the prognostic state latent:

```text
z_0 = E_state(x_t)
z_m = P(z_{m-1}, E_boundary(b_m), geometry)
x_hat_{t+m} = D(z_m)
```

Depth `N` is supervised by `x_{t+N}`. The decoder is an output/supervision head;
it is never fed back through the state encoder. Inference chunking must carry
`z_m`, not the last decoded field. Backward checkpoint compatibility is explicitly
out of scope: retrain the corrected state-only inverse rather than migrating the
joint state-and-boundary encoder.

### D5. Add losses without constraining the encoder architecture

Use the same learned encoder in every arm. For reconstruction-only runs optimize:

```text
L_recon = normalized_MSE(D_r(E_r(x)), x)
```

For cross-resolution pairs optimize:

```text
L_cross = normalized_MSE(D_s(E_r(x_r)), x_s)
```

For forecast training retain the existing forecast loss and add a zero-depth
reconstruction batch or auxiliary term:

```text
L = L_forecast + lambda_0 * L_recon
```

Sweep `lambda_0` only after a candidate passes the reconstruction funnel. Do not
add a latent-cycle loss initially: `E(D(z)) ~= z` would constrain unreachable
latent directions and is stronger than the stated requirement.

## Small-scale experiment funnel

Use successive halving rather than a full Cartesian sweep. Every comparison must
retain the same fixed sample indices, output target, seed, optimizer-update count,
precision, and mask.

### S0. Synthetic coordinate and normalization screen

Extend `scripts/probe_perceiver_decoder.py` so its input first passes through a
small learned encoder; do not copy target channels into latent channels. Train on
fresh analytic fields and evaluate unseen coefficients.

Common settings:

| Setting | Value |
|---|---|
| Source grids | `8x8`, `16x16` |
| Output grids | same grid, `8->16`, `16->8`, half-cell longitude shift |
| Channels | 16 input/output, 32 learned latent |
| Train/eval batches | fresh coefficients; 256 unseen evaluation samples |
| Optimizer | Adam, no weight decay |
| Updates | 500 screen, 2,000 finalists |
| Batch size | 32 |
| Learning rate | `3e-4`, `1e-3`, `3e-3` on the base control; freeze the winner |
| Seeds | 0 and 1 for screens; add 2 for finalists |

Compare:

1. shape-only bilinear resampling plus projection;
2. physical-coordinate resampling plus projection;
3. position-anchored attention alone;
4. physical resampling plus zero-init attention correction.

For the hybrid, sweep in this order:

- attention values: `{LayerNorm(value), raw value}`;
- neighborhood: `{3x3, 5x5}`;
- position-bias strength: `{1, 2, 4, 8, 16}` using a 100-update screen, then train
  the best two for 500 updates;
- heads and head width: `{1x64, 2x64, 4x32}` only after choosing neighborhood and
  bias;
- correction output initialization: `{exact zero, std=1e-3}` as a confirmation
  that exact zero does not delay useful learning.

Promote raw values unless normalized values improve held-out error by at least 10%
without amplitude bias. A candidate must preserve same-grid performance while
improving at least one off-grid direction relative to physical interpolation.

### S1. One-degree learned-inverse screen

Use the existing learned Perceiver encoder, processor depth zero, the one-degree
dataset, 32 deterministic training samples, and 32 disjoint held-out samples.

Common settings:

| Setting | Value |
|---|---|
| Physical patch extent | `1.0 x 1.0` degrees |
| Encoder | current Perceiver encoder, depth 2 |
| Encoder latents | 1 per one-cell patch for the primary control |
| Encoder latent dimension | 64 |
| Learned representation width | 128 |
| Decoder base | physical-coordinate resampling plus learned 1x1 projection |
| Samples | 32 train + 32 held out |
| Batch size | 1 |
| Optimizer | existing Adam, no weight decay |
| Updates | 640 (`20 epochs x 32 samples`) |
| Scheduler | none |
| Precision | float32/naive on L40S; retest promoted architecture on a validated flash runtime |
| Seed | 15; add 16 only for finalists |

Torch bring-up on 2026-07-21 isolated an L40S runtime failure: both selective
checkpointing and checkpointing-disabled bfloat16/flash runs hit a CUDA illegal
memory access on the first backward, while the otherwise matched
naive-attention/float32 run completed. S1 therefore uses naive/float32 for every
cell so the architecture comparison remains controlled. This is an execution
runtime result, not evidence against the encoder architecture; flash/bfloat16 must
be revalidated separately before a full-scale promotion run.

Screen the full `{3e-4, 6e-4, 1e-3}` by `{additive, none}` matrix on the
resampling-only control. This parallelized revision avoids serial public-queue
latency and reveals any learning-rate/geometry interaction directly. Freeze the
best setting for subsequent structural comparisons.

Then run the encoder-geometry isolation on the promoted physical base:

| Encoder geometry | Physical resampling base |
|---|---:|
| Current additive position/scale | run |
| No post-encoder position/scale | run |
| Geometry sidecar | same as `none` at depth zero; do not duplicate |

This is the primary test of the position/scale concern. Do not conclude that
position embeddings are harmful from training-set MSE alone. Selection uses held-out
MSE, spectra, amplitude ratios, and later processor behavior. If `none` wins at
depth zero, `sidecar` must still be tested with a processor before removing geometry
from the model.

Run one hybrid ocean guard only if the promoted base passes the S1 reconstruction
gate but exhibits a repeatable spectral, seam, or variable-group defect. This is a
deliberate evidence-based reduction from the original six-cell decoder/geometry
factorial, not a permanent prohibition on the correction.

Run two capacity checks only on the best geometry/decoder pair:

- learned representation width `{128, 256, 380}`;
- encoder latent dimension `{64, 128}`.

Do not sweep encoder latent count on a one-token patch beyond `{1, 4}`; prior runs
show that manufacturing many latents for one input token adds cost without improving
the identity map.

Promotion from S1 requires:

- held-out all-channel MSE no worse than the resampling-only learned-encoder control;
- no variable group more than 10% worse;
- mean high-wavenumber power ratio in `[0.85, 1.15]`, with velocity reported
  separately;
- correction/base RMS below `0.5` unless the correction produces a clear held-out
  gain;
- no NaN, mask leakage, or longitude seam regression.

#### S1 primary-matrix result

The six seed-15 runs completed on Torch L40S using the controlled naive/float32
runtime described above. Metrics are from the disjoint 32-sample held-out set;
`best` and `final` are reported separately because the two higher additive learning
rates bounced after their minima.

| Encoder geometry | LR | Best epoch | Best MSE | Final MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---|---:|---:|---:|---:|---:|---:|---:|
| additive | `3e-4` | 20 | 0.021202 | 0.021202 | 0.8851 | 0.9561 | 0.0092 |
| additive | `6e-4` | 19 | 0.015452 | 0.020102 | 0.9264 | 0.9784 | 0.0125 |
| additive | `1e-3` | 19 | 0.013569 | 0.015551 | 0.9325 | 0.9814 | 0.0118 |
| none | `3e-4` | 20 | 0.020776 | 0.020776 | 0.8372 | 0.9310 | 0.0149 |
| none | `6e-4` | 20 | 0.013780 | 0.013780 | 0.9247 | 0.9745 | 0.0098 |
| none | `1e-3` | 20 | **0.011844** | **0.011844** | 0.9283 | 0.9758 | **0.0076** |

`none + 1e-3` wins every variable-group MSE against `additive + 1e-3`. Its
per-variable high-wavenumber ratios are `0.929` (temperature), `0.953` (salinity),
`0.894` (zonal velocity), `0.929` (meridional velocity), and `1.094` (SSH), all
inside the gate. Its amplitude ratios range from `0.960` to `0.994`. The selected
model therefore removes additive encoder geometry at depth zero; this does not
remove geometry from the eventual processor, where the sidecar remains mandatory
to test.

The one-cell patch seam ratio is undefined on the one-degree grid because every
edge is a patch edge and there are no within-patch edges for its denominator.
Seams are evaluated on the half-degree and cross-resolution routes in S2 instead.

The promoted geometry/LR pair then produced this capacity result at the best
held-out epoch:

| Embedding width | Latent dimension | Best epoch | Best MSE | Final MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 20 | 0.011844 | 0.011844 | 0.9283 | 0.9758 | **0.0076** |
| 256 | 64 | 19 | 0.010375 | 0.010903 | 0.9378 | 0.9798 | 0.0223 |
| 380 | 64 | 20 | 0.009505 | 0.009505 | **0.9703** | **0.9956** | 0.0221 |
| 128 | 128 | 20 | **0.007937** | **0.007937** | 0.9256 | 0.9707 | 0.0135 |

Increasing internal encoder latent dimension is decisively better than widening
the external representation. The `128/128` model wins aggregate and every
variable-group MSE; its per-variable high-k ratios range from `0.889` to `0.987`,
inside the gate. It also keeps the processor interface at width 128. The selected
S2 contract is therefore embedding width 128, latent dimension 128, geometry
`none`, and learning rate `1e-3`. The zero-depth model has about 0.92M parameters;
the width-380 control only rises to 0.97M there, but requiring every processor
block to carry 380 channels would be the material downstream cost.

### S2. Small cross-resolution ocean reconstruction

Paired same-timestamp loading is implemented in `src/samudra/datasets.py` through
the default-off `target_time_mode: current` option. The identity harness uses it in
`configs/samudra_multi_om4/identity_cross_1_halfdeg.yaml`, with 32 training and 32
held-out samples balanced over these four routes in every fixed sample cycle:

```text
1 degree   -> 1 degree
1/2 degree -> 1/2 degree
1 degree   -> 1/2 degree
1/2 degree -> 1 degree
```

Use a shared encoder and decoder. Normalize each physical source with its existing
dataset statistics, and compute metrics in both normalized and de-normalized units.
Treat the independently regridded OM4 source as the target; also report the error of
the deterministic physical resampler so the model is not credited for differences
already present between dataset products.

Common settings are batch size 1, Adam, no scheduler, 1,280 optimizer updates,
learning rate inherited from S1, seed 15, and balanced route sampling. Compare only:

1. winning encoder geometry plus resampling base;
2. winning encoder geometry plus hybrid;
3. current additive geometry plus hybrid as a guard against overfitting the geometry
   conclusion to one degree.

#### S2 coarse-patch control and plan revision

The first four-route control completed 40 epochs using the selected
Perceiver/resampling architecture. Its resolved config was naive attention with
bfloat16 (despite an inaccurate `naivefp32` run suffix); this is also evidence that
the earlier L40S fault requires the flash path rather than bfloat16 alone. The final
route results are:

| Route | Model MSE | Mean high-k ratio | Patch-seam ratio |
|---|---:|---:|---:|
| 1 degree -> 1 degree | 0.01954 | 1.158 | undefined on one-cell patches |
| 1 degree -> 1/2 degree | 0.06027 | 0.508 | 0.898 |
| 1/2 degree -> 1 degree | 0.01832 | 1.066 | undefined on one-cell patches |
| 1/2 degree -> 1/2 degree | 0.05737 | 0.496 | 0.875 |

The defect follows output bandwidth, not cross-grid direction. Both half-degree
outputs lose about half their high-wavenumber power, including the half-to-half
route where decoder coordinates already match the target. The one-to-half learned
MSE also reaches the interpolation reference while remaining spectrally coarse.
This localizes the new failure upstream of decoder rendering: a single vector per
one-degree physical patch cannot preserve four native half-degree cells.

This evidence revises the original three-way attention comparison. Do not run the
hybrid for this defect; decoder attention cannot recover information discarded by
the encoder. The existing `spatial_query_shape` path is not sufficient unchanged,
either: it packs ordered query results into channels at one coarse patch center,
while a shared pointwise resampling decoder has no intra-patch position with which
to select those channels. With one encoder latent its output queries are also
query-blind. A future learned-query candidate must spatially unpack queries onto a
finer canonical grid and use multiple or directly anchored encoder tokens.

The smaller immediate falsification test is the checked-in
`CanonicalResampleEncoder`: apply an ordinarily initialized learned pointwise
channel projection at every native cell, then physically resample those latent
features onto the finest configured grid. The encoder remains learned and receives
no identity initialization or prognostic bypass. Its latent grid is independent of
the requested output, so the same state can be processed zero to N times and
decoded at flexible resolutions.

The one-degree held-out screen strongly supports this route:

| Encoder | External width | Held-out MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---|---:|---:|---:|---:|---:|
| Perceiver, latent dim 128 | 128 | 0.007937 | 0.9256 | 0.9707 | 0.0135 |
| Canonical pointwise/resample | 128 | 0.002857 | 0.9677 | 0.9965 | 0.0020 |
| Canonical pointwise/resample | 160 | **0.002197** | **0.9728** | **0.9971** | **0.0018** |

The width-160 candidate wins every variable-group MSE and is now running the
matched naive/bfloat16 four-route S2 comparison. Width 128 remains the conservative
processor-width fallback because it is already 64% better than the selected
Perceiver control; the 160-width gain is 23% and must survive cross-resolution and
processor tests before becoming permanent.

That first fixed-finest-grid comparison was stopped after epoch 10 because it
exposed a second deterministic tradeoff:

| Route | Epoch-10 MSE | Mean high-k ratio |
|---|---:|---:|
| 1 degree -> 1 degree | 0.03178 | 0.446 |
| 1 degree -> 1/2 degree | 0.06298 | 0.430 |
| 1/2 degree -> 1 degree | 0.03208 | 0.629 |
| 1/2 degree -> 1/2 degree | **0.00821** | **0.968** |

It confirms the spatial diagnosis--half-to-half error falls 86% relative to the
fully trained coarse-patch control and fine-scale power is restored--but it is not
a shared inverse. Bilinearly upsampling one-degree latent features to the fixed
half-degree grid and then decoding back to one degree is a smoothing round trip.
The job was deliberately cancelled after this invariant was measured instead of
spending the remaining 960 updates optimizing around it.

The next revision keeps the successful learned pointwise channel projection but
leaves its spatial grid at the native input resolution. The processor is fully
convolutional and can apply the same weights zero to N times on either native grid;
the decoder alone performs physical-coordinate resampling to any requested output.
This makes both same-grid routes exact in spatial transport, preserves target-
independent encoding, and avoids both one-degree patch compression and an
upsample/downsample round trip. The checked-in candidates are
`model_learned_inverse_native_projection.yaml` and
`model_iterable_inverse_native_projection.yaml`.

The original deterministic half-to-one reference was invalid because destination
wet cells without any source-wet interpolation support were filled with physical
zero, producing normalized MSE near 39.4. The diagnostic now uses the destination
channel climatology only at unsupported points and retains wet-mask-renormalized
physical interpolation everywhere else. The completed model trajectory is
unaffected. The corrected normalized MSE is `0.005429` for half-to-one;
one-to-half remains `0.061161`, and both same-grid references are numerical zero.

The native-grid run completed all 1,280 updates and passed its structural gate.
Held-out same-grid MSE is `0.00358` at one degree and `0.00348` at half degree,
with mean channel high-wavenumber ratios `0.992` and `1.000`. One-to-half MSE is
`0.06120`, effectively equal to the physical interpolation reference `0.06116`.
Half-to-one remains `0.02934`, however, or 5.4 times the corrected physical
reference `0.00543`. The excess survived continued optimization while both
same-grid errors kept falling, so it is not evidence for more spatial encoder
capacity.

That asymmetric remainder first motivated a matched normalization control before
processor promotion. The encoder and decoder in this candidate are affine 1-by-1
maps and therefore commute with unmasked interpolation. With independently
normalized sources, one shared map cannot be the identity on both same-grid routes
and also apply two different source-to-target mean/std transforms. The checked-in
`identity_cross_1_halfdeg_common_stats.yaml` keeps both native products and masks
but normalizes every channel using the one-degree scalar statistics. A second
diagnostic, `source_normalized_resampler_mse`, measures interpolation performed
directly in each source's normalized coordinates.

The epoch-10 common-statistics result improved half-to-one MSE from `0.03459` to
`0.03013` at the matched point, but did not approach its `0.00543` reference.
Normalization is therefore a contributor, not a sufficient root cause. The more
important non-commuting operation is channel-wise masking: the baseline resamples
160 latent channels after the encoder has mixed prognostic variables with
different depth-dependent land masks, while the physical reference resamples each
of 154 prognostic channels using its own wet-neighbor denominator. No shared latent
mask can reproduce that operation.

The completed checkpoint-only diagnostic reaches the same conclusion without
training-trajectory ambiguity. On the original independently normalized data,
direct interpolation in the source-normalized basis gives MSE `0.00971` for
half-to-one, versus `0.00543` when physical values are transformed into the target
basis. That roughly `0.00428` normalization penalty explains only a small part of
the learned model's `0.02391` excess over the physical reference.

The matched `identity_cross_1_halfdeg_common_stats_masked.yaml` control therefore
projects latent features back to prognostic channels on the native source grid and
only then applies coordinate resampling with the immutable source wet masks. This
does not bypass the learned representation or leak targets: the encoder and
decoder remain learned pointwise maps, the processor still runs zero to N times,
and the masks carry geometry only. Promote this ordering if it closes the
half-to-one excess without regressing same-grid reconstruction; revisit scale
conditioning or learned local correction only if it does not.

The checkpoint-only decoder-order swap provides the causal result. Using the exact
epoch-25 common-statistics weights, same-grid MSEs are unchanged to below `4e-9`:
`0.00513708` at one degree and `0.00564119` at half degree. One-to-half improves
slightly from `0.0877352` to `0.0867701`. Half-to-one falls from `0.0260986` to
`0.00996908`, removing 78.0% of the learned excess above the `0.00542936` physical
floor. This isolates mask ordering as the primary remaining cross-grid cause. A
fresh matched run is still required to measure how much of the residual `0.00454`
excess is removable by optimizing with the corrected ordering from initialization.

Fresh-run bring-up agrees with the swap. At epoch 5, half-to-one MSE is `0.02121`
versus `0.03737` for the common-statistics latent-resampling control at the same
point; aggregate MSE is `0.03856` versus `0.04350`. Same-grid MSE remains comparable
(`0.01681`/`0.01883` versus `0.01774`/`0.01971`). At epoch 5, the masked-resampling
training portion takes about 2.2 times as long (`156` versus `71` seconds for 32
training samples). `identity/epoch_seconds` stops before route evaluation, so this
is a training-throughput measurement rather than an evaluator timing. Quarter-
degree promotion must still include independent memory and throughput gates.
Subsequent runs log `identity/training_seconds`, `identity/evaluation_seconds`, and
`identity/total_seconds` separately.

At epoch 10, half-to-one improves further to `0.01352`, compared with `0.03013`
for the matched control. Same-grid MSEs are `0.00877` and `0.00993`, and aggregate
MSE `0.03063` has already surpassed the control's final `0.03079` after one quarter
of the optimizer budget.

At epoch 15, aggregate MSE falls to `0.02830`. Half-to-one reaches `0.01121`
against the `0.00543` deterministic masked-resampling floor; the one- and
half-degree same-grid MSEs are `0.00634` and `0.00726`. One-to-half is `0.08839`
against its `0.07900` deterministic floor. The corrected ordering therefore keeps
closing both learned-projection and cross-grid excess without trading away the
same-grid inverse.

At epoch 20, half-to-one reaches `0.01003`, effectively reproducing the
checkpoint-only order swap's `0.00997` from a fresh initialization. Same-grid MSEs
are `0.00509` and `0.00588`, one-to-half is `0.08736`, and aggregate MSE is
`0.02709`. This independent optimization result makes the mask-order conclusion
robust to the original checkpoint's training path.

The 40-epoch run finishes at its validation best with aggregate MSE `0.02491`.
Final route MSEs are `0.00304` for one-to-one, `0.08498` for one-to-half,
`0.00805` for half-to-one, and `0.00357` for half-to-half. Half-to-one is only
`0.00262` above its `0.00543` deterministic floor, while the same-grid and
half-to-one high-wavenumber ratios are `0.936`, `0.938`, and `0.934`. Use this
validation-selected checkpoint for processor-depth experiments.

The epoch-10 checkpoint also completes a zero-shot nine-route evaluation after
adding the quarter-degree grid. Quarter-degree same-grid MSE is `0.01055`, close to
the one- and half-degree same-grid errors from the same checkpoint (`0.00865` and
`0.00981`). One-to-quarter MSE is `0.10897` versus the deterministic interpolation
floor `0.10903`; half-to-quarter is `0.03747` versus `0.02959`; and quarter-to-half
is `0.01158` versus `0.00268`. This is strong evidence that the learned native-grid
inverse generalizes to an unseen resolution and that most upsampling error is the
unrecoverable physical interpolation floor.

Quarter-to-one exposes a separate scale-aware transport issue. Its MSE `0.07040`
beats the bilinear deterministic reference `0.07933`, but aggregate high-wavenumber
power is `1.35`, including `1.73` for velocity. The underlying OM4 products were
conservatively regridded, whereas the current decoder point-samples with bilinear
interpolation even for a fourfold restriction. A fixed-checkpoint masked regular-
index area diagnostic confirms that mechanism while showing the operator must be
scale aware:

| Route | Bilinear floor MSE | Area floor MSE | Model high-k | Area high-k |
|---|---:|---:|---:|---:|
| 1/2 -> 1 | 0.00592 | 0.00741 | 0.888 | 0.997 |
| 1/4 -> 1/2 | 0.00268 | 0.00399 | 0.874 | 1.010 |
| 1/4 -> 1 | 0.07933 | 0.01123 | 1.351 | 1.001 |

For the fourfold restriction, area averaging removes 86% of the bilinear floor
and restores the spectrum. For both twofold restrictions it restores spectral
power but raises MSE by 25--49%. Revise the promoted transport to use channel-
masked conservative or antialiased restriction for substantially coarser outputs,
retaining coordinate bilinear interpolation for prolongation. Do not promote
regular-index area pooling unchanged at 2x; compare a physically weighted
conservative operator and a better low-pass kernel on the fixed samples first.

The first unchunked evaluator completed in 58 minutes of model work and held about
`52 GiB` of GPU memory in a spot check on a 96-GiB RTX PRO 6000. The exact chunked
eval-only rerun reproduces all nine route MSEs and aggregate `0.0399052` exactly.
Its held-out pass takes 1,162.5 seconds (19:22.5), with peak CUDA allocation
`10.30 GiB` and reservation `24.43 GiB`; training time is zero. The cold Slurm job
takes 56:20 because its one-time SIF construction consumes about 36 minutes; the
warm area-control job completes in 15:54. The bounded-memory evaluator therefore
passes the engineering gate. A first cold pull also exposed 574 GiB of stale
shared Apptainer cache; that recoverable cache was removed and the Slurm harness
now forces OCI layers and build temporaries onto `SLURM_TMPDIR`, while retaining
the keyed final SIF in shared scratch. The clean rerun is Slurm job `14574139`
(W&B `pod127bl`); the area diagnostic is job `14579543` (W&B `rqlh7bke`).

After selecting among them, add quarter degree first with `identity_eval_only:
true`, `finetune: true`, `epochs: 1`, and the selected checkpoint. The completed
zero-shot result is finite and geometrically sensible except for the expected
large-ratio bilinear restriction aliasing above. Validate the bounded-memory and
scale-aware restriction changes, then repeat the balanced screen with all nine
input/output pairs among 1, 1/2, and 1/4 degree for 2,880 updates. Diagnose any
remaining coordinate or capacity failure before training on quarter-degree targets.

The earlier physical-patch and subcell-query ranges are no longer a mandatory
sweep. Retain them as fallback hypotheses only if the native-grid candidate fails
or its measured quarter-degree processor cost is unacceptable. In that event,
start from physical patch extent `{1x1, 2x2 degrees}`, encoder latent count
`{16, 32}`, and an explicitly spatially unpacked coordinate-query design; revise
those values using the failure measurement rather than treating them as fixed.

### S3. Processor-depth contract

After D4, pretrain the selected encoder/decoder at `k=0`, then introduce the
processor without freezing either head.

Use `finetune: true`, the zero-depth checkpoint path, and the explicit missing-key
allowlist above. Keep the encoder, decoder, embedding width, and geometry mode
identical across the checkpoint boundary; only the processor and its optional
geometry conditioner may be newly initialized.

The first diagnostic uses depths `{0,1,2,4}` and logs a decode at every depth. Keep
the latent width fixed at the selected canonical width. Use two loss-weight checks:

```text
lambda_0 in {0.05, 0.2}
```

plus `lambda_0=0` as the forecast-only control. Do not sweep processor architecture
simultaneously. The zero-depth held-out reconstruction MSE may worsen by at most 20%
from its pretrained value. If it worsens more, first test alternating reconstruction
and forecast batches before increasing `lambda_0`.

The zero-depth term is now explicitly a source-grid MSE. It builds a reconstruction
context from `input_resolution_cpu` and the immutable `input_mask`, decodes before
any processor call, and never uses the batch's possibly different target grid. This
makes the inverse constraint valid on mixed-resolution forecast batches without a
prognostic bypass or target leakage. The exact staged config is
`identity_cross_1_halfdeg_common_stats_masked_processor.yaml`; it initializes only
`processor.*` and `processor_geometry.*` outside the selected zero-depth checkpoint.

An initial one-GPU bring-up completed successfully as Slurm job `14563629` (W&B
run `joo4bptt`). It trained the `lambda_0=0.05` candidate for five epochs and 160
optimizer updates on 32 samples per epoch, then decoded the same held-out latent
state at all requested depths:

| processor depth | held-out MSE |
| ---: | ---: |
| 0 | 0.021896 |
| 1 | 0.022411 |
| 2 | 0.030988 |
| 4 | 0.107452 |

Depth one is only 2.4% worse than depth zero, while depths two and four degrade
substantially on every resolution route. This is evidence that the interface is
legal and trainable, but that repeated-processor stability is not obtained merely
by training a single call. It does not yet distinguish inadequate exposure to
positive depths from intrinsically unstable processor dynamics. Exact-window
baseline job `14567329` evaluates the untouched S2 checkpoint at `0.024567` MSE.
The smoke's depth-zero MSE is therefore 10.9% better, and its depth-one MSE remains
8.8% better, so the 20% reconstruction-preservation gate passes. Depth two is
41.5% worse than the smoke's depth zero and depth four is 391% worse. The matched
longer `lambda_0=0` and `lambda_0=0.05` jobs remain the decision runs; this smoke
result is not a substitute for them.

The matched decision runs are Slurm jobs `14581788` (`lambda_0=0`) and `14581789`
(`lambda_0=0.05`). Each uses two RTX PRO 6000 GPUs and accumulation four, preserving
effective global batch eight, 256 samples and 32 optimizer updates per epoch, and
1,280 total training samples. They load the same validation-selected S2 checkpoint
and initialize only the 65 allowlisted processor/geometry tensors. Earlier four-
L40S duplicates `14562662` and `14562663` were canceled while still pending after
both RTX jobs passed startup validation; they consumed zero runtime.

Both matched RTX runs complete successfully (W&B `s8rig9hd` and `u1ki1obr`):

| `lambda_0` | depth 0 | depth 1 | depth 2 | depth 4 |
|---:|---:|---:|---:|---:|
| 0 | 0.024406 | 0.022151 | 0.033715 | 0.135192 |
| 0.05 | 0.021861 | 0.022284 | 0.030805 | 0.113853 |

The untouched exact-window checkpoint is `0.024567`. Thus `lambda_0=0` merely
preserves it (0.7% better), while `lambda_0=0.05` improves depth zero by 11.0%.
The regularized arm is only 0.6% worse at depth one and is 8.6% and 15.8% better
at depths two and four. Promote `lambda_0=0.05`. However, its depth-four MSE is
still 5.2 times depth zero and is slightly worse than the short smoke's `0.10745`.
More single-depth exposure therefore does not make the shared map stable for
arbitrary iteration. Before proxy forecasting, train positive depths sampled from
`{1, 2, 4}` toward the same refinement target, retain the source-grid depth-zero
term, and require improvement at depths two/four without regressing depth zero/one.

That causal test is complete as Slurm job `14590678` / W&B `9yart3ax`. It uses
the same seed, checkpoint, 1,280 samples, 160 optimizer updates, effective batch
eight, and `lambda_0=0.05`; the only training change is a deterministic balanced
cycle over positive depths `{1,2,4}`:

| positive-depth training | depth 0 | depth 1 | depth 2 | depth 4 |
|---|---:|---:|---:|---:|
| fixed depth 1 | 0.021861 | 0.022284 | 0.030805 | 0.113853 |
| cycle `{1,2,4}` | 0.022202 | 0.021475 | 0.021805 | 0.024028 |

Balanced depth exposure improves depths one, two, and four by 3.6%, 29.2%, and
78.9%, respectively, while depth zero changes by only +1.6% and remains 9.6%
better than the untouched exact-window checkpoint. Depth four is now only 8.2%
worse than depth zero rather than 5.2 times worse. Every depth-four route improves:
the four route MSEs contract from `0.08589--0.14571` to `0.00594--0.07354`, with
the remaining maximum on the known one-degree-to-half-degree interpolation route.
This supports inadequate positive-depth exposure as the repeated-iteration root
cause. Do not add a global processor residual solely to fix this failure; advance
the multi-depth checkpoint to the forecast proxy and retain residualization only
as a fallback if forecasting reintroduces iteration drift.

These completed runs establish only a same-target refinement control. They do not
establish the physical-time contract above and must not be used to promote a
latent-autoregressive model.

## Forecast and full-scale validation

Only one structural candidate and one conservative fallback should enter this
stage. The fallback is the learned Perceiver encoder plus physical resampling
projection without the attention correction.

### V0. Calibrated one-degree proxy

The results recorded below are historical. Their depth cycling used the same
`t+1` target at depths one, two, and four. One-step values remain representation
evidence; the multi-depth values are not physical rollout evidence.

Use the existing stratified 512-window, one-step proxy and matched v2 control:

- 12 epochs and 192 optimizer updates;
- plain normalized MSE;
- effective global batch 32;
- Adam with learning rate `6e-4` unless S1 shows a decisive alternative;
- optimizer-update schedule targeting 6,160 updates;
- seeds 15 and 16, adding 17 only for the winner;
- no residual prediction, prognostic bypass, spectral loss, or dynamic loss.

Run:

1. resampling-only fallback;
2. hybrid candidate;
3. hybrid candidate with `lambda_0=0.05` zero-depth reconstruction regularization;
4. matched v2 reference if its existing result is not directly comparable after
   data or training-loop changes.

The selected inverse candidate is instantiated by
`train_1deg_iterable_inverse_masked_mse_stratified_updates_proxy.yaml`. Its defaults
use batch 2, eight ranks, and two accumulation steps for effective global batch 32
and 192 optimizer updates. Set `model.zero_depth_reconstruction_weight=0` for the
forecast-only control rather than changing any other architecture or schedule.

The current references are proxy MSE `0.051655` for the Perceiver-encoder/resampling
decoder, `0.041278` for direct one-cell, and `0.042390` for matched v2. Promote a
candidate only if its two-seed mean is at most `0.05`, beats persistence, retains
the S1 zero-depth contract, and has no material spectral or variable-group
regression. Prefer the simpler fallback if the hybrid improves aggregate MSE by less
than 5%.

The first two-seed forecast proxy is complete from the multi-depth S3 checkpoint.
It used 512 stratified samples per epoch, 12 epochs, 192 optimizer updates, and
effective global batch 32. Validation selected the terminal epoch in all four runs:

| source-grid zero-depth weight | seed 15 | seed 16 | mean |
|---:|---:|---:|---:|
| 0 | 0.0315724 | 0.0315876 | 0.0315800 |
| 0.05 | 0.0316064 | 0.0315465 | 0.0315765 |
| 0.2 | 0.0315684 | 0.0316211 | 0.0315947 |

All three weights are forecast-equivalent within 0.06%; all beat the matched v2
proxy by about 25.5%, the old Perceiver/resampling result by 38.8--38.9%, and the
direct one-cell proxy by about 23.5%. Regularization improves velocity
high-wavenumber power without a material variable-group regression; weight 0.2
raises the two-seed zonal ratio to about `0.733`, versus about `0.718` at 0.05.
The W&B runs are `bznwj3is`, `h73bbty2`, `4zaqxkwg`, `gr0pvlvr`, `fbmv4ccl`,
and `4mhuysks`.

Post-forecast reconstruction and depth audits expose the remaining failure. On the
same held-out cross-resolution identity window, the two-seed depth-zero mean is
`0.0374764` for weight zero and `0.0281127` for weight 0.05. Relative to the
pre-forecast multi-depth checkpoint's `0.0222023`, those regress by 68.8% and
26.6%. Thus 0.05 is useful but narrowly misses the provisional 25% preservation
gate. On a separate true `t+1` target, decoding the same input after successive
processor calls gives:

| zero-depth weight | depth 0 | depth 1 | depth 2 | depth 4 |
|---:|---:|---:|---:|---:|
| 0 | 0.0137334 | 0.0478494 | 0.0968079 | 0.291519 |
| 0.05 | 0.00631293 | 0.0476432 | 0.0880912 | 0.219623 |

Depth zero is a persistence-like inverse control on this future-target audit; depth
one is the trained one-step forecast. Applying the processor two or four times
toward that same refinement target deteriorates sharply. This reproduces the S3
exposure failure under forecast training and rules out treating successful
multi-depth autoencoding as a permanent property of the checkpoint. Regular
training now has a default-off `train_processor_depths` option that cycles positive
depths continuously and identically on every rank, restores the configured depth
for validation, and logs the selected depth. The iterable-inverse proxy and full
configs select `[1, 2, 4]`. The next matched proxy must use this exposure before a
full-scale launch.

The weight-0.2 fixed-depth inverse audit passes: its two-seed depth-zero mean is
`0.0241583`, an 8.8% regression from the pre-forecast `0.0222023`, versus 26.6%
for weight 0.05. The matched multi-depth rerun therefore used weight 0.2 and the
same two seeds:

| forecast training depths | seed 15 | seed 16 | mean |
|---|---:|---:|---:|
| fixed `{1}` | 0.0315684 | 0.0316211 | 0.0315947 |
| cycled `{1,2,4}` | 0.0360968 | 0.0361911 | 0.0361440 |

At the deliberately fixed 192-update budget, multi-depth exposure costs 14.4%
aggregate MSE but still beats matched v2 by 14.7%, direct one-cell by 12.4%, and
the old Perceiver/resampling proxy by 30.0%. Its two-seed `t+1` depth audit is:

| depth | 0 | 1 | 2 | 4 |
|---:|---:|---:|---:|---:|
| mean MSE | 0.00371387 | 0.0413635 | 0.0455987 | 0.0485074 |

Depth four is only 17.3% worse than depth one, compared with roughly 4.6 times
worse under fixed-depth training. The current-state inverse mean is `0.0246888`,
only 11.2% worse than the pre-forecast checkpoint. Those facts validate the
same-target refinement control only. They do not pass the corrected physical V0
gate and do not promote inverse weight `0.2`. The historical forecast W&B runs are
`9i47kib4` and `t7855wfw`; their depth audits are `dp1y2chf` and `3yluahc9`; their
inverse audits are `142he1n3` and `hufdi48y`.

The corrected V0 sequence is:

1. Retrain the selected state-only encoder and decoder from scratch on zero-depth
   autoencoding, first one degree through the standard trainer with
   `train_1deg_state_only_autoencoder_proxy.yaml` (the equivalent identity-only
   diagnostic is `identity_1deg_state_only_native_masked_projection.yaml`), then matched
   one/half-degree routes with
   `train_cross_1_halfdeg_state_only_autoencoder_proxy.yaml` (with
   `identity_cross_1_halfdeg_common_stats_masked.yaml` retained for detailed route
   diagnostics).
2. Confirm same-grid, cross-grid, and unseen quarter-degree reconstruction before
   introducing a processor. Compare against the prior joint-encoder numbers, but
   do not load that checkpoint.
3. Initialize a fresh boundary encoder and processor from the new inverse. Train
   physical leads `{1,2,4}` with four-step data windows, aligned forcing at every
   call, and true targets `{t+1,t+2,t+4}`.
4. Run seeds 15 and 16 at the existing 512-window/12-epoch proxy scale. Sweep
   zero-depth weight `{0,0.05,0.2}`; use learning rate `6e-4`, effective batch 32,
   and 192 updates as starting values subject to the selection-logic revision rule.
5. Audit decoded MSE at each true lead, zero-depth inverse retention, forcing
   ablations/shuffles, spectra, and latent chunk continuity. Promote only after the
   predicted lead beats persistence and the processor is demonstrably sensitive
   to the correctly aligned boundary state.

Corrected V0 execution is now underway on exact, checksum-verified code overlays.
The first clean-break result is decisive: the state-only one-degree inverse
(`14616192`, W&B `4xbqyev0`, code `b3f7d823`) reaches held-out MSE `0.000648411`
after 40 epochs. The prior joint prognostic-plus-boundary inverse reached about
`0.00304`. The exact fixed-window audit `14619746` confirms the result at
`0.000654260`, a 78.5% reduction with high-wavenumber power ratio `0.988` and
standard-deviation ratio `0.999`.

The fresh one/half-degree mix run (`14618325`, W&B `27t8occ3`, code `f4a57cb8`) completes at
aggregate MSE `0.0210660`, split as `0.00365078` on one-degree outputs and
`0.0384813` on half-degree outputs. The corresponding deterministic persistence /
coordinate-resampling baselines are `0.00259272` and `0.0372132`. Its exact
route audit `14619373` is more informative: relative to the earlier joint encoder,
aggregate MSE improves from `0.02491` to `0.0218603`; one-to-one from `0.00304`
to `0.00108228`; one-to-half from `0.08498` to `0.0789610`; half-to-one from
`0.00805` to `0.00617820`; and half-to-half from `0.00357` to `0.00121959`.
Every route improves even though the corrected checkpoint trained for 10 rather
than 40 epochs. One-to-half is fractionally better than its `0.0789996`
deterministic floor; half-to-one has `0.0007488` excess over its `0.00542936`
floor, and the same/fine-to-coarse high-wavenumber ratios are `0.979--0.985`.
This is direct evidence that removing boundary channels from the learned state is
beneficial, not merely harmless.

The nine-route audit `14619374` then applies that one/half-trained checkpoint to
the unseen quarter-degree product. Quarter same-grid MSE is `0.00136714`, versus
`0.01055` for the earlier joint encoder. One-to-quarter is `0.0974551` versus its
`0.109031` deterministic floor; half-to-quarter is `0.0260758` versus `0.0295858`;
and quarter-to-half is `0.00322509` versus `0.00268161`. The same-grid quarter
high-wavenumber ratio is `0.958`, and quarter-to-half is `0.986`. Thus flexible
output resolution survives the clean state separation and improves substantially.
Quarter-to-one remains the known scale-aware restriction exception: MSE `0.0698314`
beats bilinear resampling's `0.0793258`, but its aggregate high-wavenumber ratio is
`1.502`. This supports an antialiased/conservative deterministic restriction path
for large downsampling factors; it does not revive an unstructured attention
decoder or justify adding the zero-initialized attention residual everywhere.

A two-epoch physical-time bring-up (`14619138`, W&B `o6gi7wvi`, code
`455be788`) loaded the state-only inverse, initialized only the boundary encoder
and processor, and completed the encode-once rollout without decode/re-encode.
True-lead validation improved from `{0.101246, 0.165935, 0.277784}` after epoch 1
to `{0.0811572, 0.132833, 0.198276}` after epoch 2 for leads `{1,2,4}`. This is a
functionality and optimization smoke, not a promotion result. The definitive
two-seed, three-weight proxy jobs use code `ed7e7cb9` and additionally log a
lead-matched persistence baseline, masked zero-depth inverse retention, and zeroed,
batch-shuffled, and time-reversed boundary controls.

Three failed setup jobs contribute no model evidence: `14616135` invoked an
identity config through the training entry point, `14616181` duplicated the data
subdirectory in `DATA_ROOT`, and `14616295` exposed the now-fixed cross-grid
persistence shape bug after training but before completing epoch-one validation.
The route-audit harness failure `14619195` likewise performed no evaluation; the
harness now explicitly supports `samudra.identity` and its replacements are pinned
to the completed cross-resolution checkpoint.

### V1. Full-data one-degree run at v2-like scale

After corrected V0 promotion, train its winner on the 1975--2013 one-degree
training interval and 2013--2014 validation interval:

- 70 epochs;
- effective global batch 32;
- 6,230 realized optimizer updates, preserving the known final partial
  accumulation behavior;
- Adam, initial learning rate `6e-4`, cosine schedule in optimizer-update units;
- plain one-step normalized MSE and absolute-field prediction;
- seed 15 first; run seed 16 only if seed 15 passes the gate;
- evaluate the validation-selected checkpoint and terminal checkpoint;
- retain zero-depth reconstruction evaluation at every image-validation epoch.

The corresponding checked-in full-data config is
`train_1deg_iterable_inverse_masked_mse_updates.yaml`; it uses the same learned
heads, processor sidecar geometry, optimizer-update schedule, and effective global
batch as the proxy.

Seed 15 runs from exact code/image commit
`f5366fdd89dfb82c1e6f42a9d00b17939c440a41`. Cluster fragmentation left no node
with eight available RTX6000s, so it uses two GPUs with gradient accumulation eight
to preserve effective global batch 32 and the exact 6,230-update schedule. Initial
job `14605300` made no optimizer step: both ranks timed out in the first NCCL
barrier on `gr102`. The first scratch-backed replacement, job `14605887`, also
made no optimizer step because a silently corrupted SIF copy failed to mount; its
digest differed despite matching size and a successful copy exit. The SIF was
rewritten with checksummed `rsync`, verified against SHA-256
`0667e69d7079823cb2ffda2d15784c282ff176c2494cc18fea3d3a24b4123729`, and passed
`apptainer inspect`. Job `14607299` was the exact replacement with Torch's
documented `NCCL_P2P_DISABLE=1` workaround and scratch-backed logs, caches, and
verified SIF. It completed about 88 updates (one epoch) before being cancelled
immediately after the physical-time semantics were clarified. It is a
mis-specified run and supplies no model-selection evidence. Jobs `14605300` and
`14605887` made zero updates.

The fresh corrected run will be the requested v2-scale validation: full one-degree
data, roughly the existing v2/Samudra parameter and update scale, and the same
primary metrics. Compare
against the quoted v2 full-data MSE `0.023600` and the direct one-cell result
`0.015976`. Promotion requires all-channel MSE at most `0.025`, no variable group
more than 25% worse than v2, and zero-depth reconstruction no more than 25% worse
than its pre-forecast checkpoint.

Profile before launch. Existing evidence suggests one RTX6000 is sufficient for
proxy screening; the completed full direct one-cell run used two GPUs, peaked near
17.3 GiB per rank, and took about 8 hours 20 minutes. The hybrid must demonstrate
bounded-neighborhood memory scaling in a one-epoch smoke before receiving the full
budget.

### V2. Full multi-resolution validation

Do not begin with a three-resolution full-budget run. Use two gates:

1. Train 1-degree plus 1/2-degree on the full time interval with balanced resolution
   sampling and a total compute-matched budget of 6,230 optimizer updates.
2. If both resolutions pass their same-grid and cross-grid gates, add quarter degree
   and repeat at 6,230 total updates.

Use one resolution pair per optimizer update and report sample counts by route. Set
microbatch size independently by resolution, using gradient accumulation to preserve
effective global batch 32; begin with batch 2 at one degree and batch 1 at half and
quarter degree. Do not interpret the compute-matched run as converged if each
resolution receives fewer updates. For the final winner only, run an exposure-matched
budget of 6,230 updates per resolution (12,460 for two resolutions or 18,690 for
three).

Evaluate every same-resolution route and every cross-resolution route at fixed held-
out timestamps. Promotion requires:

- no same-resolution all-channel MSE more than 25% worse than the corresponding
  single-resolution model;
- no cross-resolution route worse than the deterministic coordinate-resampling
  reference by more than 10% unless it materially improves high-wavenumber fidelity;
- high-wavenumber ratios between `0.8` and `1.2` for temperature, salinity, and SSH,
  with velocity ratios reported and required to improve over the resampling-only
  fallback;
- no longitude seam, latitude-edge, mask, or grid-shift artifact;
- stable zero-depth reconstruction throughout processor training.

## Selection logic

The plan is intentionally falsifiable and open to revision. New measurements,
implementation constraints, or better hypotheses may change the experiment order,
candidate set, sweep ranges, budgets, and promotion gates. The numerical thresholds
and exact requirements below are initial decision aids, not immutable specifications.
Any departure should be recorded with the evidence and reasoning that motivated it
so the plan evolves deliberately rather than being followed mechanically.

- If removing additive encoder geometry improves `k=0` but hurts processor runs,
  retain geometry as a sidecar rather than restoring content addition automatically.
- If additive geometry is best on held-out reconstruction and forecasting, keep it;
  the concern is then resolved empirically.
- If the hybrid does not beat the physical resampling base by at least 5% on the
  proxy or by a clear spectral margin, ship the simpler base.
- If raw attention values are no better than normalized values, LayerNorm is not the
  limiting residual-path concern; it still remains absent from the stable base.
- If cross-resolution error is dominated by encoder patch compression, increase
  learned subcell representation capacity before changing decoder attention.
- If zero-depth reconstruction collapses during processor training, adjust the
  training mixture or processor interface before making the encoder more rigid.
- If the full one-degree candidate cannot reach MSE `0.025`, do not spend the
  three-resolution exposure-matched budget.

## Expected artifacts

Each stage should produce:

- checked-in configs or a generated sweep manifest with one immutable row per run;
- resolved configuration, commit/container identity, random seed, sample indices,
  and optimizer-update count;
- train and held-out reconstruction tables by variable and depth;
- forecast, persistence, and deterministic-resampling comparisons;
- spectra, high-wavenumber ratios, seam diagnostics, amplitude ratios, and maps;
- peak GPU/host memory and throughput;
- a short decision record saying which arm advances and which hypothesis was
  falsified.

The final report should update
`docs/experiments/perceiver_decoder_root_cause.md` rather than replacing its decoder
diagnosis. The completed evidence selects the learned native-grid projection plus
projection-before-channel-masked coordinate resampling, with processor geometry in
a sidecar and no attention correction. The remaining corrected state-only inverse,
physical-time proxy, and full-scale runs test whether that decoder recommendation
survives true latent autoregression; contrary evidence should revise it.
