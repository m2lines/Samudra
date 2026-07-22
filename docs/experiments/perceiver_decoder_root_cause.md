<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Final report: Perceiver IO decoder root causes

## Executive conclusion

The observed failure is not one undifferentiated "Perceiver problem." Three
separable mechanisms are supported by controlled evidence:

1. the naive one-latent decoder is exactly query-blind for multi-query windows;
2. the configured 64-dimensional cross-attention value path is narrower than the
   77-channel target and independently loses channel information;
3. with multiple latents, spatial correspondence is unanchored, expensive to learn,
   and unnecessary for fixed-set memorization.

The most defensible near-term architecture is the existing resolution-flexible
resampling projection, upgraded to periodic/coordinate-aware interpolation. The
best research candidate is that stable path plus a zero-initialized local attention
residual with relative physical position. Retaining the current Perceiver IO
decoder would require at minimum a genuinely wider cross-attention path and an
explicit spatial route; increasing latent count or latent width alone does not
address the measured causes.

## Objective

Isolate why the SamudraMulti Perceiver IO decoder performs poorly on identity
reconstruction, then identify architectural changes that preserve flexible output
resolutions. This report combines the completed real-ocean experiments with new
decoder-only controls; follow-up experiments are explicitly separated from the
evidence supporting the conclusions.

## Prior evidence

The completed 32-sample pure-autoencoder factorial documented in the
[single-step research results](samudra_multi_single_step_research_results.md)
localizes nearly all of the
surprising error to the decoder. With the processor bypassed, direct/direct reaches
MSE `0.012083`, Perceiver/direct reaches `0.025431`, direct/Perceiver reaches
`0.279303`, and Perceiver/Perceiver reaches `0.285353`. Replacing the Perceiver
decoder with bilinear resampling plus a pointwise projection subsequently reduces
the paired one-degree forecast proxy from the original SamudraMulti result of
`0.381735` to `0.051655`.

Those experiments demonstrate localization and provide a successful fallback, but
they do not distinguish mathematical degeneracy, information capacity, spatial
routing, optimization, and memorization. The controls below remove the encoder,
processor, data loader, ocean mask, boundary fields, and bfloat16 arithmetic. They
feed 128-channel processor-like features directly to the production
`PerceiverDecoder`; the first 77 feature channels are independent standard-normal
targets and the remaining channels are zero.

The reproducible harness is `scripts/probe_perceiver_decoder.py`.

## Root cause 1: a one-latent, multi-query decoder is query-blind

`perceiver-pytorch==0.8.8` performs the final decoder cross-attention without a
query residual. With one internal latent, the attention softmax has length one and
is identically one for every query. The output therefore has no dependence on the
query. This is an architectural invariant, not an optimization failure.

A four-by-four, 77-channel synthetic run confirms all three predictions after 20
updates:

| Measurement | Result |
|---|---:|
| Maximum output difference across 16 spatial queries | `0.0` |
| Query-embedding gradient norm | `0.0` |
| Output/target spatial-variance ratio | `0.0` |

One latent remains a useful negative control, but it is not a viable configuration
for a window containing multiple output locations.

## Root cause 2: the configured attention path is narrower than the target state

The production task emits 77 prognostic channels per spatial query. The model
config sets `queries_dim=64`. At the time of the recorded experiments,
`PerceiverConfig.build_io` did not expose the dependency's `cross_heads` or
`cross_dim_head`; their defaults were one head of width 64. Increasing
`latent_dim` therefore did not widen either cross-attention value path. This branch
now exposes and tests both controls for the matched follow-up configuration.

To remove spatial routing, the next experiment uses a two-by-two field decoded as
four independent one-token/one-query windows. It trains on a fresh batch of 64
random channel vectors on every update and evaluates 128 unseen vectors after 500
updates. Consequently, static coordinate memorization cannot solve the task.

| Latent / query width | Cross-attention inner width | Held-out MSE | Output/target spatial variance |
|---:|---:|---:|---:|
| 64 / 64 | 64 | `0.437818` | `0.545` |
| 128 / 128 | 64 | `0.239298` | `0.745` |
| 128 / 128 | 128 | **`0.021848`** | `0.986` |
| 128 / 128 | 2 heads x 64 | **`0.021848`** | `0.986` |

The one-head width-128 and two-head width-64 runs are numerically identical for
this one-token case. Widening the true attention value path reduces held-out error
by 90.9% relative to widening only the latent and query representations, and by
95.0% relative to the current 64-wide path. The widened curve is still decreasing
at update 500, so `0.021848` is not interpreted as an asymptotic floor.

This establishes a channel-information bottleneck independent of spatial routing.
It also explains why prior "width 128" controls could improve without resolving
the failure: the underlying cross-attention head width remained 64.

## Root cause 3: unanchored routing admits memorization and learns copying slowly

Multi-latent Perceiver IO is not universally incapable of fitting a copy task. On
four fixed four-by-four samples, both 16 and 64 width-64 latents reach roughly
`1e-3` training MSE after 500 updates, while a width-128 model transiently reaches
below `1e-6`. The fit is memorization rather than a learned identity operator: the
64-latent model has held-out MSE `1.660140`, worse than predicting zero for
standard-normal targets.

Attention instrumentation makes the mechanism visible. For each cross-attention
stage, the harness records normalized entropy, stable rank, and maximum mass. It
also multiplies the head-averaged query-to-latent and latent-to-data attention
matrices to obtain an approximate query-to-input transport matrix. Self-attention
and value transformations mean this product is not a complete model Jacobian, but
its same-cell mass is an interpretable routing diagnostic.

| Training task and width | Held-out MSE | Approx. diagonal mass | Approx. top-1 alignment | Output/target spatial variance |
|---|---:|---:|---:|---:|
| Four fixed fields, width 64 | `1.660140` | `0.064` | `0.070` | `0.998` on training fields |
| Fresh fields, width 64 | `0.682370` | `0.905` | `0.938` | `0.287` |
| Fresh fields, width 128 | `0.300376` | `0.921` | `0.938` | `0.651` |

The fixed-sample model reconstructs its training fields with diffuse transport. It
can encode a sample signature into global latents and use coordinate queries to
emit a memorized field, without learning input-cell-to-output-cell correspondence.
Fresh random fields prevent this shortcut. They force both widths to discover a
mostly diagonal route, but the route emerges only late in optimization and the
model still under-reconstructs spatial variance after 300 updates.

This is consistent with the prior ocean controls: increasing unanchored latent
count from 1 through 256 did not solve the 12-by-12-window task, whereas reducing
the task to one input and one output cell closed most of the gap. The evidence now
supports an optimization and inductive-bias diagnosis: a valid multi-latent route
exists, but the unordered bottleneck makes learning spatial correspondence
unnecessarily difficult and permits misleading fixed-set memorization.

## Architectural evidence: direct and position-anchored cross-attention

Two structure-changing controls retain coordinate queries and therefore admit
arbitrary output grids:

1. **Direct cross-attention** removes the internal learned latent bank. Output
   queries attend directly to processor tokens, followed by a query residual,
   feed-forward block, and 77-channel projection.
2. **Position-anchored direct cross-attention** adds a strong relative-position
   logit bias to the same learned attention. On a matching grid it starts at the
   corresponding input cell; on a different output grid the same construction can
   be evaluated from physical input/output coordinates. Content-dependent logits
   remain free to learn corrections to the geometric route.

All following models use a 128-wide, one-head attention path and train on fresh
random fields. The four-by-four Perceiver IO and plain direct controls use eight
training samples per update; the 12-by-12 controls use four because of the larger
attention matrix. Held-out evaluation uses respectively 16 and 8 unseen samples.

| Architecture | Grid / context | Parameters | Held-out MSE at update 300 | Diagonal or max attention mass | Spatial variance ratio |
|---|---|---:|---:|---:|---:|
| Perceiver IO | 4x4 / none | `1,040,589` | `0.300376` | `0.921` approximate diagonal | `0.651` |
| Direct cross-attention | 4x4 / none | `307,661` | **`0.010044`** | `0.995` diagonal | `0.973` |
| Direct cross-attention | 12x12 / none | `307,661` | `0.500880` | `0.549` diagonal | `0.476` |
| Direct cross-attention | 12x12 / one ring | `307,661` | `0.857778` | `0.399` maximum | `0.072` |
| Position-anchored direct | 12x12 / none | `307,661` | **`0.005325`** | `>0.999999` diagonal | `0.986` |
| Position-anchored direct | 12x12 / one ring | `307,661` | **`0.005244`** | `>0.999999` maximum | `0.986` |

Removing the latent bank improves four-by-four held-out MSE by 96.7%, uses 70.4%
fewer parameters, and runs more than twenty times faster in the local CPU probe.
It does not by itself solve scale: at 12-by-12 the route is still being discovered
at update 300, and irrelevant context makes that discovery harder.

The explicit spatial prior removes the scale and context sensitivity. At
12-by-12 it reduces held-out MSE by 98.9% without context and 99.4% with context,
relative to plain direct attention. The context/no-context learning curves are
nearly identical because context tokens no longer compete equally with the
corresponding source cell. This supports a flexible-resolution decoder built around
geometric routing plus learned residual correction, rather than an unordered
second latent bottleneck.

## Cross-resolution control: anchoring alone is not enough

The hard same-grid anchor above must not be mistaken for proof of useful
cross-resolution behavior. A second synthetic task evaluates five smooth analytic
latitude/longitude basis fields with independent coefficients for every sample and
channel. It observes them on an 8-by-8 source grid and reconstructs the same
continuous fields on a 16-by-16 output grid. Training again uses fresh random
coefficients and evaluation uses unseen coefficients.

As geometry-only references, direct bilinear interpolation has MSE `0.015970` and
nearest-neighbor interpolation has MSE `0.029651`; the target variance is
`0.488414`. Circularly padding longitude before bilinear interpolation lowers the
reference to `0.015492`, a further 3.0% improvement. The learned pointwise
projection reaches `0.015562` at update 300 when trained with a suitable learning
rate, consistent with the nonperiodic bilinear floor.

| Decoder geometry | Same-grid random MSE | 8-to-16 analytic MSE | Interpretation |
|---|---:|---:|---|
| Sharp position attention, strength 16 | `0.005244` | `0.044621` | Excellent identity, approximately nearest-neighbor off-grid |
| Broad position attention, strength 2 | `0.040001` | `0.012272` | Learns useful interpolation, weakens exact identity |
| Bilinear resampling + projection | `0.0000148` | `0.015562` | Stable identity and predictable interpolation |
| Bilinear base + zero-initialized attention residual | **`0.000000196`** | **`0.012608`** | Preserves identity and improves the interpolation base |

The attention-only results demonstrate a real temperature tradeoff: a kernel sharp
enough to enforce identity cannot interpolate, while a kernel broad enough to
interpolate allows neighboring tokens to contaminate same-grid copying. The hybrid
removes that conflict. Its synthetic base projection is initialized to copy the 77
aligned input channels, so the absolute same-grid number is an oracle diagnostic,
not a claim that processor features already equal output channels. The relevant
architectural result is that a zero-initialized residual cannot damage the base at
initialization and subsequently reduces cross-resolution error by about 21%.

The production analogue should therefore be:

1. resample the canonical processor grid deterministically to the requested output
   coordinates, using periodic longitude and coordinate-aware latitude rather than
   edge-clamped index interpolation;
2. project channels with the learned pointwise head already validated by the ocean
   proxy;
3. add a zero-initialized local attention residual whose logits include relative
   physical position, periodic longitude, and an explicit validity mask.

This preserves the successful resampling decoder exactly at initialization, keeps
output-grid flexibility, and gives the model a route to learn corrections where
bilinear interpolation is inadequate.

## Learned-encoder S0 confirmation

The earlier oracle-copy probe placed target channels directly in the decoder input.
The follow-up S0 screen removes that shortcut: a two-layer pointwise encoder learns
the 16-to-32-channel representation jointly with each decoder from fresh analytic
coefficients, and evaluation uses 256 unseen coefficient draws. The table reports
three-seed held-out means after 2,000 Adam updates at learning rate `3e-3`.

| Decoder | Same 8-to-8 | Down 16-to-8 | Shifted 8-to-8 | Up 8-to-16 |
|---|---:|---:|---:|---:|
| Physical-coordinate resampling + projection | **0.000182** | **0.000208** | **0.004716** | **0.003459** |
| Physical base + zero-init local attention | 0.000301 | 0.000326 | 0.004890 | 0.003487 |
| Position-anchored attention only | 0.001035 | 0.000652 | 0.005568 | 0.006257 |

The preceding 500-update screen also isolated coordinate semantics. Shape-only
bilinear interpolation tied physical interpolation on the identical grid, but was
10.0 times worse when downsampling, 8.7 times worse on the half-cell longitude
shift, and 3.6 times worse when upscaling. This is direct evidence that the dominant
flexible-resolution architectural fix is physical-coordinate resampling, not more
attention capacity.

The longer confirmation revises the earlier hybrid recommendation. Its correction
helped same-grid and down-grid error at 500 updates, but the simpler learned base
continued improving and won every route by 2,000 updates. The hybrid was also about
4--7 times slower on CPU. Under the plan's successive-halving rule, it does not
advance as a co-equal S1 candidate; its checked-in implementation remains a fallback
if ocean spectra expose a residual that a pointwise projection cannot learn.

Position-only anchored attention was not rejected because it cannot express the
desired mapping. It was rejected as the primary renderer because, under the same
learned-encoder budget, it lost every route, required about 8--10 times the runtime
of the base, suppressed shifted-grid high-wavenumber power to `0.815`, and amplified
upscaled high-wavenumber power to `1.264`. It remains an informative control, while
the zero-initialized hybrid is the safer way to add it if later evidence justifies
the cost.

## Real-ocean learned-inverse confirmation

S1 tests the promoted physical-coordinate resampling decoder with the production
two-layer Perceiver encoder on 154-channel, masked one-degree OM4 states. The
processor is bypassed, but neither the encoder nor the decoder projection is given
an identity initialization. Every result is measured on 32 held-out states that
are disjoint from the 32 fixed training states.

| Encoder geometry | LR | Best held-out MSE | Final MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---|---:|---:|---:|---:|---:|---:|
| additive | `3e-4` | 0.021202 | 0.021202 | 0.8851 | 0.9561 | 0.0092 |
| additive | `6e-4` | 0.015452 | 0.020102 | 0.9264 | 0.9784 | 0.0125 |
| additive | `1e-3` | 0.013569 | 0.015551 | 0.9325 | 0.9814 | 0.0118 |
| none | `3e-4` | 0.020776 | 0.020776 | 0.8372 | 0.9310 | 0.0149 |
| none | `6e-4` | 0.013780 | 0.013780 | 0.9247 | 0.9745 | 0.0098 |
| none | `1e-3` | **0.011844** | **0.011844** | 0.9283 | 0.9758 | **0.0076** |

At the useful `1e-3` learning rate, omitting post-encoder additive position/scale
embeddings wins every variable-group MSE and lowers mean absolute normalized bias.
The selected model's high-wavenumber ratios are `0.929` for temperature, `0.953`
for salinity, `0.894` for zonal velocity, `0.929` for meridional velocity, and
`1.094` for SSH; all satisfy the preregistered `[0.85, 1.15]` gate. The low-rate
result prevents a blanket claim that geometry removal always helps: this is an
optimization-by-geometry interaction, with a clear advantage at the learning rate
that best trains both variants.

This result supports a learned inverse, not a dumb encoder. The encoder still must
compress each masked ocean state into the canonical latent representation. Only
the decoder's spatial correspondence is anchored by physical-coordinate
resampling; its channel projection remains learned. Geometry is also retained as
a sidecar for the processor experiments, where a nonzero-depth latent evolution
may need it even though adding it directly to depth-zero encoder features hurts
reconstruction.

The selected `0.011844` is consistent with closing the earlier decoder gap, but it
is not treated as a strict head-to-head improvement over the historical
direct/direct `0.012083`: the harnesses and held-out sample contracts differ. The
controlled conclusion comes from the six S1 runs above. Width `{256, 380}` and
latent-dimension `128` confirmations then isolated whether more internal encoder
capacity or a wider processor interface was preferable.

| Embedding width | Latent dimension | Best held-out MSE | Final MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 0.011844 | 0.011844 | 0.9283 | 0.9758 | **0.0076** |
| 256 | 64 | 0.010375 | 0.010903 | 0.9378 | 0.9798 | 0.0223 |
| 380 | 64 | 0.009505 | 0.009505 | **0.9703** | **0.9956** | 0.0221 |
| 128 | 128 | **0.007937** | **0.007937** | 0.9256 | 0.9707 | 0.0135 |

The `128/128` model wins aggregate MSE and every variable group. Its high-k ratios
are `0.950` for temperature, `0.941` for salinity, `0.889` for zonal velocity,
`0.919` for meridional velocity, and `0.987` for SSH. This selects extra internal
encoder capacity rather than a wide external state: the canonical processor
interface remains 128 channels, while the Perceiver encoder latent dimension rises
to 128. The zero-depth parameter count changes only from about 0.92M to 0.97M over
the width sweep, but a 380-channel interface would make every repeated processor
application substantially wider.

## Multi-resolution root cause: encoder spatial bandwidth

The first balanced one/half-degree ocean control uses one shared Perceiver encoder
and physical-resampling decoder for all four input/output routes. It completed 40
epochs (1,280 batch-one updates) with one learned vector per one-degree patch. The
resolved runtime was naive attention with bfloat16; the run suffix incorrectly says
`fp32`. Stability through completion shows that bfloat16 alone does not reproduce
the earlier flash/bfloat16 L40S fault.

| Route | Held-out MSE | Mean high-k ratio | Patch-seam ratio |
|---|---:|---:|---:|
| 1 degree -> 1 degree | 0.01954 | 1.158 | undefined |
| 1 degree -> 1/2 degree | 0.06027 | 0.508 | 0.898 |
| 1/2 degree -> 1 degree | 0.01832 | 1.066 | undefined |
| 1/2 degree -> 1/2 degree | 0.05737 | 0.496 | 0.875 |

The failure follows output bandwidth rather than cross-resolution direction. Both
half-degree outputs retain only about half the target high-wavenumber power,
including the half-to-half route where decoder source and output coordinates are
already identical. The one-to-half MSE reaches the deterministic interpolation
scale while retaining the same coarse spectrum. Thus the physical decoder is doing
the expected rendering; the encoder has already compressed four native
half-degree cells into one one-degree token.

This is evidence against spending the next run on decoder attention. It also
reveals a limitation in the checked-in packed spatial-query encoder: ordered query
outputs are stored as channels at a coarse patch center, but the shared pointwise
decoder has no intra-patch coordinate with which to select them. With one internal
latent, those PerceiverIO output queries are query-blind as well. A viable query
version must spatially unpack outputs onto a finer canonical grid and use multiple
or directly anchored encoder tokens.

The simpler control now projects every native input cell into learned latent
channels and physically resamples those features onto the finest configured grid.
It uses ordinary learned initialization, carries no prognostic bypass, and fixes
the latent grid independently of the requested output. It therefore still forms a
learned representation that can be processed zero to N times.

| One-degree encoder | External width | Held-out MSE | Mean high-k ratio | Mean amplitude ratio | Mean absolute bias / target std |
|---|---:|---:|---:|---:|---:|
| Perceiver, latent dim 128 | 128 | 0.007937 | 0.9256 | 0.9707 | 0.0135 |
| Canonical pointwise/resample | 128 | 0.002857 | 0.9677 | 0.9965 | 0.0020 |
| Canonical pointwise/resample | 160 | **0.002197** | **0.9728** | **0.9971** | **0.0018** |

Both canonical-grid controls decisively beat the Perceiver encoder at the same
zero-depth task. Width 160 wins every variable group and is the active matched
four-route candidate; width 128 remains the lower-cost processor fallback.

The fixed-finest-grid four-route run then provided a useful falsification by epoch
10. Half-to-half MSE fell from the coarse control's final `0.05737` to `0.00821`,
and its high-k ratio rose from `0.496` to `0.968`. But one-to-one MSE was `0.03178`
with high-k ratio `0.446`. Upsampling native one-degree features to a half-degree
canonical grid and linearly sampling them back is a smoothing operation, not an
inverse. Because this is a spatial invariant of the fixed-grid transport, the run
was cancelled after 320 updates rather than completing the remaining 960.

The revised candidate retains the learned pointwise channel representation on the
native input grid. The shared processor remains fully convolutional and can run the
same weights zero to N times at any supported input resolution; only the decoder
resamples to the requested output coordinates. This is target-independent, keeps
both same-grid spatial paths exact, and avoids both measured bottlenecks. It does
trade constant processor resolution for native-resolution compute, which quarter-
degree memory and throughput tests must quantify.

The first deterministic half-to-one baseline was itself invalid: physical zero was
used where a destination wet cell had no wet source neighbor, yielding normalized
MSE near 39.4. The corrected diagnostic uses destination climatology at unsupported
locations while retaining mask-renormalized physical interpolation elsewhere. This
does not change any learned model result above. The corrected normalized references
are `0.005429` for half-to-one, `0.061161` for one-to-half, and numerical zero for
same-grid routes.

The native-grid candidate then completed all 1,280 matched updates:

| Route | Held-out MSE | Mean high-k ratio | Physical resampler MSE |
|---|---:|---:|---:|
| 1 degree -> 1 degree | **0.00358** | 0.992 | 0 |
| 1 degree -> 1/2 degree | 0.06120 | 0.438 | 0.06116 |
| 1/2 degree -> 1 degree | 0.02934 | 0.621 | 0.00543 |
| 1/2 degree -> 1/2 degree | **0.00348** | 1.000 | 0 |

This closes the encoder-bandwidth diagnosis: both same-grid routes reconstruct
accurately with essentially exact spectral amplitude, and one-to-half reaches its
irreducible interpolation error. The remaining asymmetric half-to-one error is a
different contract failure. One contributor is that each product is standardized
with different scalar channel means and standard deviations, while the native
encoder and resampling decoder are shared affine 1-by-1 maps. Those maps commute
with unmasked interpolation but cannot be identity maps in two different
normalized coordinate systems and also apply a route-dependent source-to-target
affine transform.

The first matched control therefore uses one common channel-normalization basis
across native products and also reports a source-normalized interpolation baseline.
At epoch 10 it reduces half-to-one MSE from `0.03459` to `0.03013`, but remains far
above the `0.00543` reference. This does not support normalization as the sole root
cause.

A checkpoint-only decomposition makes the scale of that effect explicit. With
the original independent statistics, half-to-one interpolation performed directly
in the source-normalized basis has MSE `0.00971`; the physical-to-target-normalized
reference is `0.00543`. Thus normalization accounts for about `0.00428`, while the
trained model is `0.02391` above the physical reference. One-to-half similarly
changes from `0.06116` in the physical reference to `0.05620` in the
source-normalized control. The much larger remaining learned error requires a
second mechanism.

The stronger structural mismatch is mask ordering. The baseline encoder mixes
prognostic and boundary channels before the decoder interpolates 160 latent
channels. Ocean validity is instead channel dependent: depth levels and variables
have different wet masks, and the physical reference independently renormalizes
the four interpolation weights for each of 154 output channels. Interpolation is
linear for a fixed mask, but these different per-channel denominators do not
commute with learned channel mixing. Same-grid routes conceal the problem because
no interpolation occurs.

The next matched control decodes latent features to prognostic channels on the
native source grid, then performs physical-coordinate interpolation with the
immutable channel-wise input masks. It retains a learned representation, does not
copy prognostic values around the encoder, supports arbitrary output coordinates,
and leaves the shared processor callable zero to N times. Its implementation is
the `project_before_resample` path in `ResampleProjectionDecoder`; the data path
carries source geometry through `GridContext.input_mask`.

That ordering has now been isolated with an exact checkpoint swap:

| Route | Latent-resample baseline | Project then channel-mask resample | Change |
|---|---:|---:|---:|
| 1 degree -> 1 degree | 0.00513708 | 0.00513707 | < `4e-9` |
| 1 degree -> 1/2 degree | 0.0877352 | 0.0867701 | -0.000965 |
| 1/2 degree -> 1 degree | 0.0260986 | **0.00996908** | **-0.016130** |
| 1/2 degree -> 1/2 degree | 0.00564119 | 0.00564118 | < `2e-9` |

Both evaluations use the exact epoch-25 common-statistics weights; no optimizer
step occurs. Relative to the half-to-one physical floor `0.00542936`, the ordering
change removes 78.0% of excess error while leaving both same-grid functions
unchanged. This is direct evidence that per-channel wet-mask interpolation cannot
be reconstructed after latent channel mixing. The remaining `0.00454` excess is
similar in scale to the learned same-grid projection error and is being tested with
a fresh matched optimization.

The fresh run has passed its epoch-5 bring-up gate. Half-to-one is already
`0.02121`, compared with `0.03737` for the common-statistics latent-resampling
control at epoch 5, while same-grid errors remain comparable. Aggregate MSE improves
from `0.04350` to `0.03856`. At epoch 5, the channel-wise masked training path
costs `156` seconds for 32 samples versus `71` seconds for the control, an observed
2.2x training-throughput penalty. `identity/epoch_seconds` excludes the subsequent
route evaluation, whose cost must be measured separately before quarter-degree
training. Explicit training, evaluation, and total-duration metrics are now emitted
for subsequent runs.

By epoch 10, half-to-one reaches `0.01352` versus the control's matched `0.03013`;
same-grid errors are `0.00877` and `0.00993`. Aggregate MSE `0.03063` is already
below the latent-resampling control's epoch-40 result `0.03079`, establishing that
the corrected ordering improves optimization as well as the checkpoint swap.

The epoch-15 result strengthens that conclusion: aggregate MSE is `0.02830`,
half-to-one is `0.01121` versus its `0.00543` deterministic floor, and same-grid
MSEs are `0.00634` and `0.00726`. One-to-half is `0.08839` versus its `0.07900`
floor. Thus the remaining error is increasingly comparable to the learned
same-grid projection error, rather than the much larger latent mask-order failure.

At epoch 20, fresh half-to-one MSE is `0.01003`, effectively reproducing the
checkpoint-only order swap's `0.00997`. Same-grid errors are `0.00509` and
`0.00588`, one-to-half is `0.08736`, and aggregate MSE is `0.02709`. Agreement
between the intervention on fixed weights and independent optimization is strong
evidence that projection-before-channel-masked-resampling addresses the causal
failure rather than merely favoring one training trajectory.

The fresh run finishes at its epoch-40 validation best. Aggregate MSE is `0.02491`;
route MSEs are `0.00304` for one-to-one, `0.08498` for one-to-half, `0.00805` for
half-to-one, and `0.00357` for half-to-half. Half-to-one is only `0.00262` above
its `0.00543` deterministic floor. Same-grid and half-to-one high-k ratios are
`0.936`, `0.938`, and `0.934`, so the accuracy gain does not come from spectral
smoothing.

## Quarter-degree zero-shot evidence

The epoch-10 one/half-degree checkpoint was evaluated without optimization on all
nine routes after adding the unseen quarter-degree grid:

| Route | Model MSE | Deterministic MSE | Excess | High-k ratio |
|---|---:|---:|---:|---:|
| 1 -> 1 | 0.00865 | 0.00000 | 0.00865 | 0.899 |
| 1 -> 1/2 | 0.08824 | 0.08265 | 0.00559 | 0.551 |
| 1 -> 1/4 | 0.10897 | 0.10903 | -0.00006 | 0.651 |
| 1/2 -> 1 | 0.01348 | 0.00592 | 0.00756 | 0.888 |
| 1/2 -> 1/2 | 0.00981 | 0.00000 | 0.00981 | 0.886 |
| 1/2 -> 1/4 | 0.03747 | 0.02959 | 0.00788 | 0.635 |
| 1/4 -> 1 | 0.07040 | 0.07933 | -0.00892 | 1.351 |
| 1/4 -> 1/2 | 0.01158 | 0.00268 | 0.00890 | 0.874 |
| 1/4 -> 1/4 | 0.01055 | 0.00000 | 0.01055 | 0.848 |

Quarter same-grid error is comparable to the trained resolutions, and the two
coarse-to-quarter routes remain close to their deterministic floors. The learned
inverse therefore generalizes to a new native grid without a finest-grid query
bypass or scale embedding in the encoder.

Quarter-to-one reveals a new transport-specific failure rather than an encoder
failure. Its aggregate high-k ratio `1.35` includes velocity at `1.73`; bilinear
point sampling aliases a fourfold restriction, while the target OM4 products were
conservatively regridded. The promoted architecture should keep native-grid learned
projection followed by channel-masked transport, but use conservative or
antialiased restriction for substantially coarser outputs and bilinear coordinate
prolongation for finer outputs.

The fixed-checkpoint area diagnostic isolates the restriction effect:

| Route | Bilinear floor MSE | Area floor MSE | Model high-k | Area high-k |
|---|---:|---:|---:|---:|
| 1/2 -> 1 | 0.00592 | 0.00741 | 0.888 | 0.997 |
| 1/4 -> 1/2 | 0.00268 | 0.00399 | 0.874 | 1.010 |
| 1/4 -> 1 | 0.07933 | 0.01123 | 1.351 | 1.001 |

Area averaging is decisive for 4x restriction: MSE is 86% lower than bilinear and
high-k power is essentially exact. At 2x it restores high-k power but raises MSE,
so the production operator should be scale aware. Use a physically weighted
conservative or antialiased restriction for large ratios; retain bilinear
prolongation; and test a better low-pass kernel before replacing bilinear at 2x.

The unchunked evaluator took 58 minutes after container setup and held about
`52 GiB` of GPU memory in a spot check. The exact chunked eval-only rerun matches
all nine route MSEs and aggregate `0.0399052`, takes 1,162.5 seconds for its sole
held-out pass, and peaks at `10.30 GiB` allocated / `24.43 GiB` reserved CUDA
memory. Its cold Slurm wall time is 56:20 because SIF construction takes about
36 minutes; the warm area control completes in 15:54. The bounded-memory flexible-
resolution path therefore passes its engineering gate. The clean and area jobs
are Slurm `14574139` / W&B `pod127bl` and Slurm `14579543` / W&B `rqlh7bke`.

## Architecture decision matrix

| Candidate | Same-grid identity | Flexible output grid | Learned nonlocal correction | Evidence-backed decision |
|---|---|---|---|---|
| Current Perceiver IO | Poor at production scale | Yes | Yes | Do not promote unchanged |
| Wider Perceiver IO | Channel bottleneck reduced | Yes | Yes | Run the matched ocean control; insufficient without spatial anchoring |
| Direct query-to-token attention | Better at 4x4, slow at 12x12 | Yes | Yes | Useful ablation, not sufficient alone |
| Position-only anchored attention | Learnable but slower and less accurate | Yes | Yes | Retain as a control; do not promote as sole renderer |
| Physical-coordinate resampling projection | Best learned-encoder S0 result | Yes | No | Promote as the primary architecture |
| Resampling + zero-init attention residual | Preserves base at initialization | Yes | Yes | Retain as fallback; S0 does not justify its added cost |
| Native-grid latent resampling + learned projection | Excellent same-grid inverse and spectra | Yes | No | Baseline: cross-grid error exposes normalization and mask-order contracts |
| Native-grid learned projection + channel-masked resampling | Exact same-grid transport by construction | Yes | No | Primary matched candidate; validate against physical interpolation floor |

The production and control implementations behind the matrix are:

- current and widened Perceiver IO: `PerceiverDecoder` in
  `src/samudra/models/modules/decoder.py`;
- direct and position-anchored attention controls: `DirectCrossAttentionIO` and
  `AnchoredCrossAttentionIO` in `scripts/probe_perceiver_decoder.py`;
- physical-coordinate base: `coordinate_bilinear_resample` and
  `ResampleProjectionDecoder` in `src/samudra/models/modules/decoder.py`;
- source-mask transport: `GridContext` in `src/samudra/utils/ctx.py`, populated by
  `TrainingShard` in `src/samudra/datasets.py` and consumed by `SamudraMulti.decode`;
- production hybrid: `LocalCoordinateAttentionCorrection` and
  `ResampleAttentionResidualDecoder` in that same module;
- learned encoder geometry modes: `PerceiverEncoder` in
  `src/samudra/models/modules/encoder.py`;
- non-destructive processor geometry sidecar: `ProcessorGeometryConditioner` in
  `src/samudra/models/modules/augment_input.py`; and
- shared zero-to-N encode/process/decode path: `SamudraMulti` in
  `src/samudra/models/samudra_multi.py`.

The same `SamudraMulti.reconstruct_once` path now constructs a source-grid context
for the zero-depth MSE. Consequently the inverse regularizer is independent of the
chosen forecast output resolution, while the processor geometry sidecar remains
outside the encoder and is applied once per processor iteration.

Channel-conditioned queries are not the first remedy. They would multiply query
count by up to 77 while the current spatial routing failure remains. Producing all
channels from one spatial query is reasonable once the query/value path is at least
as wide as the state; variable-group heads may be worth testing later if velocity
continues to lag after routing is fixed.

## Limitations and remaining evidence

- The first L40S ocean bring-up isolated a CUDA illegal-memory access in the
  bfloat16/flash backward both with and without activation checkpointing. The
  completed fresh ocean run validates naive attention with bfloat16, but the flash
  backend remains a production-runtime limitation to diagnose separately.
- The multiplied attention matrices are a routing diagnostic, not an exact model
  Jacobian because self-attention and value transformations intervene.
- The production checkpoint swap and fresh run confirm the mask-order mechanism on
  actual OM4 coordinates, periodic longitude, nonuniform latitude, and channel-
  specific wet masks. Quarter-degree zero-shot scaling is numerically successful;
  the bounded-memory rerun and scale-aware restriction comparison remain.
- The hybrid synthetic control initializes its base projection to copy aligned
  target channels. Real processor features require a learned channel projection.
  The completed ocean resampling proxy is the evidence that such a projection is
  trainable in the real model.
- S2 confirms the learned inverse across independently regridded one/half-degree
  products, and the quarter zero-shot run confirms unseen-resolution transfer.
  Processor-depth, forecast proxy, and full-scale forecast validation remain.

## Reproduction

The core local controls are reproducible from the repository root:

```bash
# Exact one-latent query blindness.
uv run python scripts/probe_perceiver_decoder.py \
  --grid-size 4 --samples 4 --steps 20 --num-latents 1

# One-cell channel bottleneck: current true width versus 128-wide attention.
uv run python scripts/probe_perceiver_decoder.py \
  --grid-size 2 --window-patches 1 --samples 64 --eval-samples 128 \
  --steps 500 --fresh-batches --num-latents 1 --latent-dim 64 \
  --queries-dim 64 --cross-dim-head 64
uv run python scripts/probe_perceiver_decoder.py \
  --grid-size 2 --window-patches 1 --samples 64 --eval-samples 128 \
  --steps 500 --fresh-batches --num-latents 1 --latent-dim 128 \
  --queries-dim 128 --cross-dim-head 128

# Production-window spatial anchor.
uv run python scripts/probe_perceiver_decoder.py \
  --architecture anchored-cross-attention --grid-size 12 \
  --context-patches 1 --samples 4 --eval-samples 8 --steps 300 \
  --fresh-batches --queries-dim 128 --cross-dim-head 128

# Flexible 8-to-16 interpolation base plus learned residual.
uv run python scripts/probe_perceiver_decoder.py \
  --architecture resample-attention-residual --data-mode analytic \
  --grid-size 8 --output-grid-size 16 --samples 4 --eval-samples 8 \
  --steps 300 --fresh-batches --queries-dim 128 --cross-dim-head 128 \
  --position-bias-strength 2 --learning-rate 0.003
```

The follow-up real-data control is fully specified by
`configs/samudra_multi_om4/identity_1deg_decoder_wide.yaml` and runs through
`python -m samudra.identity`.

## Recommended follow-up validation

1. Retain the completed width-160 native-grid encoder/decoder, no additive encoder
   geometry, and projection-before-channel-masked-resampling selection.
2. Do not add the zero-initialized attention residual: neither S2 nor quarter
   zero-shot transfer exposes a residual defect that justifies its cost.
3. Compose the validation-selected zero-depth checkpoint with the shared processor
   and verify decoded depths 0, 1, 2, and 4. The first `lambda_0=0.05` smoke run
   produced MSEs `0.02190`, `0.02241`, `0.03099`, and `0.10745`, respectively.
   The untouched checkpoint is `0.02457` on the exact same held-out window, so the
   inverse is preserved through one call. The depth-two/four growth requires the
   matched longer processor runs before the repeated-dynamics design is promoted.
4. Retain the completed bounded-memory quarter evaluator. Prototype physical
   conservative restriction for 4x downsampling, where area lowers the floor 86%,
   and compare a better low-pass kernel at 2x, where naive area improves spectra
   but worsens MSE.
5. Run the one-degree proxy and full v2-scale forecast validations only after the
   inverse and resource gates pass.
