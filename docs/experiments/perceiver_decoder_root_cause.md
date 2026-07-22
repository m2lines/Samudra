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

## Architecture decision matrix

| Candidate | Same-grid identity | Flexible output grid | Learned nonlocal correction | Evidence-backed decision |
|---|---|---|---|---|
| Current Perceiver IO | Poor at production scale | Yes | Yes | Do not promote unchanged |
| Wider Perceiver IO | Channel bottleneck reduced | Yes | Yes | Run the matched ocean control; insufficient without spatial anchoring |
| Direct query-to-token attention | Better at 4x4, slow at 12x12 | Yes | Yes | Useful ablation, not sufficient alone |
| Position-only anchored attention | Learnable but slower and less accurate | Yes | Yes | Retain as a control; do not promote as sole renderer |
| Physical-coordinate resampling projection | Best learned-encoder S0 result | Yes | No | Promote as the primary architecture |
| Resampling + zero-init attention residual | Preserves base at initialization | Yes | Yes | Retain as fallback; S0 does not justify its added cost |

Channel-conditioned queries are not the first remedy. They would multiply query
count by up to 77 while the current spatial routing failure remains. Producing all
channels from one spatial query is reasonable once the query/value path is at least
as wide as the state; variable-group heads may be worth testing later if velocity
continues to lag after routing is fixed.

## Limitations and remaining evidence

- The new synthetic probes use the naive Perceiver implementation in float32. The
  production ocean decoder uses flash attention and bfloat16; the checked-in wide
  ocean autoencoder control is required to quantify implementation-specific gains.
- The multiplied attention matrices are a routing diagnostic, not an exact model
  Jacobian because self-attention and value transformations intervene.
- The cross-resolution analytic control uses a regular normalized grid. A production
  residual must use actual OM4 coordinates, periodic longitude, nonuniform latitude,
  and masks for padded or invalid context tokens.
- The hybrid synthetic control initializes its base projection to copy aligned
  target channels. Real processor features require a learned channel projection.
  The completed ocean resampling proxy is the evidence that such a projection is
  trainable in the real model.
- The prior ocean factorial and forecast proxy are the real-data evidence used in
  this report. The prepared wide-decoder ocean run is follow-up validation of the
  measured channel bottleneck, not a prerequisite for the mathematical or
  synthetic conclusions.

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

1. Run the checked-in 128-wide cross-attention configuration on the fixed 32-sample
   ocean autoencoder with the processor bypassed.
2. Implement the position bias from physical source/query coordinates rather than
   the square-window index control, including periodic longitude and nonuniform
   latitude. The current cross-resolution control uses normalized regular-grid
   coordinates.
3. Compare the production hybrid with the existing resampling projection on the
   fixed 32-sample ocean autoencoder.
4. Promote only candidates that learn fresh synthetic copying, preserve
   high-wavenumber variance, and materially close the direct-decoder ocean
   autoencoding gap.
