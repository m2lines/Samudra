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
- encoder geometry can be additive, absent, or supplied to each processor call as
  a zero-initialized position/scale sidecar;
- physical-coordinate resampling and the bounded zero-initialized local-attention
  correction are production decoder options;
- `SamudraMulti` exposes encode/process/decode, supports one shared processor at
  depths `0`, `1`, `2`, `4`, or any other non-negative integer when widths match,
  and has gradient tests across repeated applications;
- forecast training can opt into a same-grid zero-depth reconstruction auxiliary
  loss through `zero_depth_reconstruction_weight`; and
- the synthetic probe now learns its encoder rather than copying target channels
  into the decoder input, uses fresh analytic coefficients and disjoint evaluation
  coefficients, and reports amplitude, bias, and high-wavenumber diagnostics.

The checked-in zero-depth controls are
`model_learned_inverse_resample.yaml` and
`model_learned_inverse_hybrid.yaml`; the processor-present candidate is
`model_iterable_inverse_hybrid.yaml`. No experiment result is implied by these
implementations. Training and evaluation remain to be run, and the candidate set
and defaults below remain provisional under the selection-logic revision rule.

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

Before implementing training targets for `k>1`, decide whether a processor
application denotes one physical time step or one refinement step. If it is a
physical step, compare `D(P^k(E(x_t)))` with `x_{t+k}` and supply the matching
forcing sequence. If it is refinement depth, compare all trained positive depths
with the same forecast target. Do not mix these semantics in one experiment.

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
| Precision | production bfloat16/flash path; repeat finalist in float32/naive |
| Seed | 15; add 16 only for finalists |

First sweep learning rate `{3e-4, 6e-4, 1e-3}` on the current additive-geometry,
resampling-only control. Freeze the best setting for all structural comparisons.

Then run the six-cell isolation:

| Encoder geometry | Base only | Base + zero-init correction |
|---|---:|---:|
| Current additive position/scale | run | run |
| No post-encoder position/scale | run | run |
| Geometry sidecar | same as `none`; do not duplicate | same as `none`; do not duplicate |

This is the primary test of the position/scale concern. Do not conclude that
position embeddings are harmful from training-set MSE alone. Selection uses held-out
MSE, spectra, amplitude ratios, and later processor behavior. If `none` wins at
depth zero, `sidecar` must still be tested with a processor before removing geometry
from the model.

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

### S2. Small cross-resolution ocean reconstruction

Develop paired same-timestamp loading for different sources. Start with 32 training
and 32 held-out timestamps and balance these four routes per optimizer update:

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

After selecting among them, add quarter degree in evaluation first, without
training. If zero-shot `1/2 <-> 1/4` behavior is finite and geometrically sensible,
repeat the balanced screen with all nine input/output pairs among 1, 1/2, and 1/4
degree for 2,880 updates. Otherwise diagnose coordinate or capacity failure before
training on quarter-degree targets.

For the all-resolution screen, sweep physical patch extent `{1x1, 2x2 degrees}` and
encoder latent count `{16, 32}` for patches containing multiple native pixels. Keep
the representation learned. If high-wavenumber loss indicates that one vector per
physical patch is insufficient, compare the existing coordinate-query encoder with
`spatial_query_shape=[4,4]`, `spatial_query_channels={8,16}`, and
`queries_dim={64,128}`. Those queries produce an ordered learned subcell
representation; they are not an identity initialization.

### S3. Processor-depth contract

After D4, pretrain the selected encoder/decoder at `k=0`, then introduce the
processor without freezing either head.

The first diagnostic uses depths `{0,1,2,4}` and logs a decode at every depth. Keep
the latent width fixed at the selected canonical width. Use two loss-weight checks:

```text
lambda_0 in {0.05, 0.2}
```

plus `lambda_0=0` as the forecast-only control. Do not sweep processor architecture
simultaneously. The zero-depth held-out reconstruction MSE may worsen by at most 20%
from its pretrained value. If it worsens more, first test alternating reconstruction
and forecast batches before increasing `lambda_0`.

For refinement-depth semantics, train positive depths sampled uniformly from
`{1,2,4}` toward the same one-step target. For physical-time semantics, use the
corresponding `t+k` targets and forcing sequence. In either case, compare shared
processor weights with the existing single-call processor at matched parameter
count and optimizer updates.

## Forecast and full-scale validation

Only one structural candidate and one conservative fallback should enter this
stage. The fallback is the learned Perceiver encoder plus physical resampling
projection without the attention correction.

### V0. Calibrated one-degree proxy

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

The current references are proxy MSE `0.051655` for the Perceiver-encoder/resampling
decoder, `0.041278` for direct one-cell, and `0.042390` for matched v2. Promote a
candidate only if its two-seed mean is at most `0.05`, beats persistence, retains
the S1 zero-depth contract, and has no material spectral or variable-group
regression. Prefer the simpler fallback if the hybrid improves aggregate MSE by less
than 5%.

### V1. Full-data one-degree run at v2-like scale

Train the promoted candidate on the 1975--2013 one-degree training interval and
2013--2014 validation interval:

- 70 epochs;
- effective global batch 32;
- 6,230 realized optimizer updates, preserving the known final partial
  accumulation behavior;
- Adam, initial learning rate `6e-4`, cosine schedule in optimizer-update units;
- plain one-step normalized MSE and absolute-field prediction;
- seed 15 first; run seed 16 only if seed 15 passes the gate;
- evaluate the validation-selected checkpoint and terminal checkpoint;
- retain zero-depth reconstruction evaluation at every image-validation epoch.

This is the requested v2-scale validation: full one-degree data, roughly the
existing v2/Samudra parameter and update scale, and the same primary metrics. Compare
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
diagnosis. The new evidence will determine whether the recommended production model
is the simple physical resampling decoder or the resampling decoder plus a learned
local correction, and how encoder geometry should be represented.
