<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Coarse-latent, high-resolution dynamics experiment plan

## Status and interpretation

This plan is intentionally revisable. Its numerical budgets, gates, candidate
widths, and exact losses are starting points rather than immutable requirements.
New evidence should change the plan rather than be forced into its present
selection logic. Results and changes of interpretation are recorded in
[`coarse_latent_highres_dynamics_results.md`](coarse_latent_highres_dynamics_results.md).

The decoder experiments completed before this plan compared renderers whose
source representation contained one latent vector per physical source cell.
They did **not** compare decoders that must expand one coarse token into a
multi-cell physical patch. Consequently, those experiments support coordinate
resampling for a native-cell latent grid but do not reject position-anchored
attention for the coarse patch grid used by `main`.

## Scientific objective

Learn a spatially coarse but feature-rich latent state whose dynamics can be
trained from high-resolution ocean fields:

\[
x_t^{(r)}
\xrightarrow{E_r}
z_t\in\mathbb{R}^{C_z\times H_c\times W_c}
\xrightarrow{P^N}
z_{t+N\Delta t}
\xrightarrow{D_r}
\hat{x}_{t+N\Delta t}^{(r)}.
\]

For the first production proxy, physical patches are \(3^\circ\times5^\circ\),
so \(H_c\times W_c=60\times72\) regardless of whether the physical input is
one degree, half degree, or quarter degree. At those resolutions a coarse token
summarizes respectively \(3\times5\), \(6\times10\), or \(12\times20\) cells.
The processor is shared and operates only on the coarse grid. One processor
application advances one \(\Delta t\); \(N\) applications advance \(N\Delta t\).
One aligned boundary state is separately encoded for every processor call.

This objective has two distinct requirements:

1. the encoder must retain within-patch information relevant to future
   coarse fluxes and high-resolution reconstruction; and
2. the decoder must unpack a coarse feature vector at arbitrary physical
   output coordinates without making the physical evolution depend spuriously
   on the requested rendering grid.

An exact inverse is information-theoretically impossible for arbitrary fields
when a patch is compressed. The experiments therefore measure how the required
latent width grows with subpatch complexity and whether the retained information
is dynamically useful, rather than assuming perfect reconstruction is possible.

## Architectural contract

### Geometry and normalization

- Relative within-patch coordinates, cell area, resolution, and masks may
  determine attention routing.
- Absolute coarse-grid geometry is supplied to the processor as an immutable
  sidecar rather than added to reconstructive latent values.
- Attention keys may be normalized. Reconstructive values retain a raw-amplitude
  path and do not pass through LayerNorm by default.
- Longitude routing is periodic. Latitude uses physical coordinates and does
  not wrap.
- Wet masks are channel-aware when physical channels are transported.

### Encoder candidates

**E0: existing patch Perceiver.** Retain `main`'s shared Perceiver patch encoder
as the production baseline.

**E1: position-aware attention pooling.** Produce one token per physical patch:

\[
z_p=\sum_{i\in p}
\operatorname{softmax}_i\left(
q^\top k(x_i,\Delta r_i,A_i,M_i)
\right)v(x_i).
\]

The value path is unnormalized, while keys may be normalized.

**E2: resolved-plus-subgrid encoder.** Reserve an explicit amplitude-preserving
route for area-weighted patch means and use a learned branch for anomalies:

\[
\bar{x}_p =
\frac{\sum_i A_iM_ix_i}{\sum_i A_iM_i},\qquad
z_p =
\left[
W_{\rm mean}\bar{x}_p,\;
E_{\rm sgs}(x_i-\bar{x}_p,\Delta r_i,A_i,M_i)
\right].
\]

This is not an identity encoder: only a subspace is anchored to resolved
moments, while the remaining channels learn a compact subgrid state.

### Decoder candidates

**D0: coarse bilinear projection.** Project or bilinearly resample the coarse
token grid with a pointwise output head. This is a deliberately weak control
when a token represents multiple output cells.

**D1: coordinate-conditioned local decoder.**

\[
\hat{x}(q)=
M_\theta\left(z_{p(q)},\Delta r_q,A_q,r_{\rm target}\right).
\]

This tests whether within-patch coordinates and a local feature vector suffice
without attention.

**D2: position-anchored local direct cross-attention.**

\[
\hat{x}(q)=H_\theta\left[
\sum_{p\in\mathcal N(q)}
\operatorname{softmax}_p\left(
q(r_q)^\top k(z_p)/\sqrt d-\beta d_{\rm phys}(q,p)^2
\right)v(z_p)
\right].
\]

The initial neighborhood is the containing patch plus either zero or one ring
of coarse neighbors. Queries use continuous physical coordinates, so output
resolution is not fixed.

**D3: coarse base plus zero-initialized anchored correction.**

\[
D_r(z)=R_r(H_{\rm coarse}(z))+C_{\rm anchored}(z,r).
\]

The correction is initialized to zero. Where practical, test constraining the
area-weighted correction mean within each patch to zero, so the coarse head owns
resolved patch means while attention owns subpatch organization.

## Experiment funnel

### S0-R: synthetic reconstruction and transport

Use a \(12\times12\) coarse latent grid and two physical grids:

| Physical grid | Cells per coarse patch |
|---|---|
| \(36\times60\) | \(3\times5\) |
| \(72\times120\) | \(6\times10\) |

Generate fresh held-out examples from smooth multiscale Fourier fields, sharp
fronts, translated vortices, and masked/coastal fields. Train and evaluate:

- \(3\times5\) input to \(3\times5\) output;
- \(6\times10\) input to \(6\times10\) output;
- \(6\times10\) input to \(3\times5\) output;
- \(3\times5\) input to \(6\times10\) output;
- half-cell shifted output coordinates; and
- unseen intermediate output coordinates.

The first screen crosses E0/E1/E2 with D0/D1/D2/D3 at latent widths
`{160, 320}`. Start with four attention heads, head widths `{32, 64}`,
neighborhood radii `{0, 1}`, and position-bias strengths `{2, 8, 16}`. Use
successive halving rather than completing the Cartesian product.

Primary metrics are normalized MSE, spectral power by wavelength relative to
the patch, front-location and gradient error, patch-seam error, area-mean
consistency between output grids, parameter count, runtime, and peak memory.

### S0-D: synthetic subgrid closure

Construct pairs of high-resolution states with identical patch means but
different within-patch front positions. Evolve them with known advection so
their next-step coarse tendencies differ when material crosses patch boundaries.

Measure:

1. latent distinguishability between paired initial states;
2. next-step coarse-tendency error;
3. high-resolution front displacement after decoding;
4. improvement over a processor that receives only patch means; and
5. stability for autoregressive depths `{1, 2, 4, 8}`.

This is the most direct falsification test for the hypothesis that a feature-rich
coarse latent can carry dynamically useful subgrid information.

### S1: OM4 learned-inverse proxy

Use the fixed \(60\times72\) latent grid with synchronized one- and half-degree
OM4 states. Train on routes `1->1`, `1/2->1/2`, `1/2->1`, and `1->1/2`.
Use disjoint timestamps, at least two seeds for small screens, and approximately
2,000--4,000 optimizer updates before promotion.

Successive-halving order:

1. cross all surviving encoders with D1 and D2;
2. retain the best two encoder/decoder pairs;
3. compare D2 against the zero-initialized D3 hybrid;
4. sweep latent width only after routing and normalization are selected.

Report latent agreement between synchronized resolutions,
\(\|E_{1^\circ}(x_{1^\circ})-E_{1/2^\circ}(x_{1/2^\circ})\|\), but do not
initially optimize it. Then compare no alignment loss with a weak shared-subspace
alignment loss. Do not force the entire latent to agree, because that could erase
subgrid information available only at higher resolution.

### S2: frozen-inverse latent dynamics

Freeze the selected encoder and decoder. Train only the boundary encoder,
processor, processor geometry, and per-channel residual scale:

\[
z_{m+1}=z_m+\alpha\odot P(z_m,E_b(b_m),g_c).
\]

Train and validate physical depths `{1, 2, 4}`. Compare:

1. decoded physical loss only;
2. latent teacher loss only; and
3. their combination,

\[
L =
\sum_r w_rL_{\rm physical}
\left(D_r(z_{t+n}),x_{t+n}^{(r)}\right)
+\lambda_z\left\|
z_{t+n}-\operatorname{sg}E_{\rm high}(x_{t+n}^{\rm high})
\right\|^2,
\quad \lambda_z\in\{0.01,0.1\}.
\]

The main causal test trains primarily from half-degree targets and evaluates the
same coarse latent forecast at both half and one degree.

### S3: full validation

First run the promoted architecture on full one- plus half-degree training and
validate every input/output route at depths `{1, 2, 4}`. Compare against
persistence, `main`, the completed native-grid model, and a patch-mean-only
encoder. Proceed to quarter-degree training only after review of this checkpoint.

Promotion requires:

- useful same-grid reconstruction despite compression;
- materially better front and high-wavenumber fidelity than D0;
- consistent patch means across requested output resolutions;
- no material patch-seam artifact;
- a positive S0-D counterfactual result;
- high-resolution supervision improving coarse forecast tendencies;
- stable repeated processor calls; and
- acceptable output-query runtime and memory.

No single numerical threshold in this section overrides scientific review.

## Implementation map

- Existing physical patch encoder:
  [`PerceiverEncoder`](../../src/samudra/models/modules/encoder.py)
- Existing position-anchored local attention:
  [`LocalCoordinateAttentionCorrection`](../../src/samudra/models/modules/decoder.py)
- Existing zero-initialized hybrid:
  [`ResampleAttentionResidualDecoder`](../../src/samudra/models/modules/decoder.py)
- Promoted coarse restriction:
  [`PatchMomentEncoder`](../../src/samudra/models/modules/encoder.py)
- Promoted continuous prolongation correction:
  [`ContinuousCoordinateAttentionCorrection`](../../src/samudra/models/modules/decoder.py)
- Promoted coarse inverse proxy:
  [`model_coarse_moment_attention_inverse.yaml`](../../configs/samudra_multi_om4/model_coarse_moment_attention_inverse.yaml)
- Historical coarse-grid configuration:
  [`model.yaml`](../../configs/samudra_multi_om4/model.yaml)
- Latent autoregressive model path:
  [`SamudraMulti`](../../src/samudra/models/samudra_multi.py)
