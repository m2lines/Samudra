<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# A native-grid learned inverse for multi-resolution ocean emulation

## Abstract

The SamudraMulti model implemented on `main` used Perceiver modules to map
heterogeneous physical grids into and out of a shared latent processor. Although
this design allowed variable regular output query grids, it performed poorly even
on one-step, one-degree prediction: full-data normalized mean-squared error (MSE)
was `0.29469`, compared with a quoted external Samudra v2 reference of `0.0236`.
Controlled auto-encoding experiments localized most of the unexpected error to
the representation heads rather than the convolutional processor. In particular,
fixed physical patches caused resolution-dependent information loss, while the
Perceiver IO decoder had a narrow and weakly anchored route from latent tokens to
spatial output queries.

The revised multi-resolution model retains an encoder--processor--decoder
factorization but changes its semantics. A jointly trained pointwise
encoder--decoder pair forms a learned approximate inverse on the native input
grid and is then frozen; deterministic, channel-masked coordinate resampling
provides output-grid flexibility; and a shared processor advances the state
autoregressively in latent space. Boundary forcing and grid geometry enter each
physical transition through separate paths rather than contaminating the
reconstructive state. A completed one/half-degree run achieves aggregate
normalized MSEs of `0.03982`, `0.05408`, and `0.06595` at physical leads 1, 2,
and 4, respectively, while preserving same-grid reconstruction MSE near
`1e-3`. These results support the new inverse and temporal contract, although
remaining velocity-spectrum and route-exposure limitations should be resolved
before quarter-degree training.

For reproducibility, “main” in this discussion means `origin/main` commit
`d689f92c`, inspected on 2026-07-24. The comparison is specifically against its
[`SamudraMulti` implementation](https://github.com/m2lines/Samudra/blob/d689f92c/src/samudra/models/samudra_multi.py#L49-L113)
and
[default model configuration](https://github.com/m2lines/Samudra/blob/d689f92c/configs/samudra_multi_om4/model.yaml);
Samudra v2 appears only as an external accuracy reference.

## Architectural comparison

The `main` implementation compressed each fixed \(3^\circ\times5^\circ\)
physical patch to a single learned vector before applying the spatial processor.
Consequently, increasing input resolution increased compression without increasing
the latent spatial grid: a patch contained 15 one-degree cells, 60 half-degree
cells, or 240 quarter-degree cells. The decoder then reconstructed every output
pixel through windowed Perceiver IO attention. Output latitude and longitude
formed spatial queries, and each query produced all prognostic channels at once.
This made output shape formally flexible, but required attention to learn both the
physical correspondence between source and destination cells and the channel
transformation. Boundary variables were encoded together with prognostic state,
and conventional autoregression decoded a physical prediction and encoded it
again at the next step.

The new model separates information representation, dynamics, and rendering:

\[
z_0=E_s(x_t),\qquad
z_m=z_{m-1}+\alpha\odot
P\!\left(z_{m-1}+E_b(b_m)+G(\ell)\right),\qquad
\hat{x}_{t+N}=R_{\ell_z\rightarrow\ell_o}\!\left(D(z_N),M\right).
\]

Here \(E_s\) and \(D\) are learned pointwise channel maps, \(P\) is the nonlinear
U-Net processor, \(E_b\) encodes one time-aligned boundary state per transition,
and \(G\) supplies position and cell scale to the processor. The residual scale
\(\alpha\) is initialized to zero. The renderer \(R\) performs periodic-longitude,
physical-coordinate bilinear interpolation and renormalizes it independently with
each prognostic channel's wet mask \(M\). The selected implementation is defined
by the
[native-grid model configuration](../../configs/samudra_multi_om4/model_iterable_inverse_native_masked_projection.yaml)
and the
[one/half-degree training configuration](../../configs/samudra_multi_om4/train_cross_1_halfdeg_iterable_inverse_masked_mse_updates.yaml);
the previous representation is specified by the linked `main` snapshot above.

| Property | SamudraMulti on `main` | Revised multi-resolution model |
|---|---|---|
| State encoder | Perceiver compression per fixed physical patch | Learned \(1\times1\) map at every native cell |
| Spatial latent grid | Fixed by patch extent | Equal to the input grid |
| Decoder transport | Learned Perceiver IO query routing | Deterministic physical-coordinate resampling |
| Output channels | Joint vector from each spatial query | Joint pointwise projection before channel-specific transport |
| Geometry | Added to encoder and decoder tokens; coordinates form decoder queries | Zero-initialized processor sidecar |
| Boundary forcing | Mixed into encoder input | Separate encoder called once per physical step |
| Autoregression | Decode and re-encode physical state | Encode once and carry latent state |
| Zero-step behavior | Production forward always applies \(P\); no reconstruction API | Frozen learned inverse \(D(E_s(x))\) |
| Output resolution | Flexible through query count | Flexible through requested coordinate arrays |

The revised encoder is not an identity or “dumb” front end: its \(160\)-channel
representation is learned jointly with the decoder. What is removed is premature
spatial compression. Spatial mixing is delegated to the dynamical processor, while
the decoder learns channel transformations and grid geometry specifies transport.

## Experimental interpretation

In a 32-sample autoencoder factorial derived from the `main` architecture, the
one-latent direct-encoder/Perceiver-decoder arm reached MSE `0.27930`; increasing
it to main's 256 decoder latents remained poor at `0.27802`. Direct/direct reached
`0.01208`, localizing the unexpected failure to the decoder. In a matched
one-degree forecast proxy, replacing only that decoder with resampling plus a
pointwise projection reduced MSE from `0.38174` to `0.05166` (86.5%). Widening
attention values corrected a channel bottleneck but not unanchored routing;
position-anchored attention remained slower and less accurate than coordinate
resampling with a learned encoder.

Multi-resolution controls exposed two further failures. Coarse patch tokens
discarded half-degree bandwidth before decoding; retaining one learned vector per
native cell restored it. Resampling latent mixtures before projecting prognostic
channels also failed to reproduce channel-specific coastal masks. Reversing those
operations removed 78% of the excess half-to-one error on identical weights.
Both resolutions now use common one-degree normalization statistics.

Fixed-one-step forecast training exposed two temporal problems: a soft
reconstruction term did not prevent encoder/decoder drift, and the processor was
not trained to associate repeated calls with successive physical leads. The new
procedure freezes \(E_s,D\), then trains the transition, boundary encoder,
geometry sidecar, and residual scale at true depths \(\{1,2,4\}\). Depth \(N\)
consumes \(N\) ordered boundary states and targets \(x_{t+N\,dt}\).

## Results and implications

The completed full-interval one/half-degree experiment trained for 6,392 optimizer
updates. Aggregate validation MSE was `0.03982` at lead 1, `0.05408` at lead 2,
and `0.06595` at lead 4, corresponding to reductions of 62.9%, 69.5%, and 73.9%
relative to lead-matched persistence. Every one-to-one, one-to-half,
half-to-one, and half-to-half route beat its persistence baseline at every
evaluated lead. The one-to-one lead-one MSE was `0.02642`, 91.0% below the
completed `main`-architecture full-data baseline of `0.29469`; because the
training contracts differ, this magnitude summarizes integrated progress rather
than a single-factor effect. Same-grid reconstruction remained exactly `0.001110`
at one degree and `0.001243` at half degree throughout processor training,
directly confirming that improved forecasts were not obtained by sacrificing the
learned inverse.

Boundary ablations support the intended physical interpretation. Setting the
per-step forcing to zero increased aggregate error by 23.4%, 61.1%, and 124.7%
at leads 1, 2, and 4; reversing its temporal order increased error by 9.8%,
19.2%, and 93.7%. The transition therefore uses both boundary values and
ordering. Spatial diagnostics are also substantially healthier than in the old
patch-compressed model: temperature, salinity, and sea-surface-height
high-wavenumber power ratios remain between `0.938` and `1.025` on all routes.

These comparisons should nevertheless be interpreted mechanistically rather than
as a single controlled leaderboard result. The historical full one-degree run and
the new multi-resolution run differ in training schedule, temporal supervision,
normalization, and representation, so their headline MSEs do not isolate one
factor. The decoder-only factorial and matched proxy provide the stronger causal
evidence for the rendering change; the full run validates the integrated model.
The full promotion run used one training seed; its independent endpoint audit
confirms checkpoint-evaluation reproducibility, not run-to-run uncertainty. Two
limitations remain. Half-degree same-grid velocity power is underrepresented
(`0.790/0.751` for zonal/meridional velocity), suggesting insufficient processor
exposure or capacity. Coarse-to-fine velocity power is lower still
(`0.288/0.149`), an expected coarse-source and deterministic-prolongation limit
of this architecture; a learned super-resolution head was not tested. In
addition, one-to-one lead-one MSE is 28.7% above the completed single-resolution
reference, slightly outside the provisional 25% degradation gate.

The principal conclusion is therefore architectural but bounded. The new
native-grid inverse resolves the dominant representation and decoder failures
while preserving flexible rectilinear output resolutions and a zero-to-\(N\)
processor contract. The next step should keep this inverse fixed, test processor
exposure or modest capacity increases, and validate conservative restriction for
large downsampling ratios. Quarter-degree training is not yet warranted.
The detailed experiments, definitions, and reproduction commands are collected
in the
[decoder root-cause report](perceiver_decoder_root_cause.md).
