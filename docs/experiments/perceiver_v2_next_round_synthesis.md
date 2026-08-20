<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2: prior evidence and next-round research synthesis

## Status

This report synthesizes Jesse's decoder-root-cause, learned-inverse,
single-step, and coarse-latent experiments with the completed 2-degree
Perceiver v2 architecture search. It records what the first search established,
what remains ambiguous, and the recommended experiment funnel for selecting a
second-generation Perceiver architecture.

The completed primary search is
`perceiver-v2-2deg-architecture--20260814T171003.874785Z`, running immutable
code revision
[`6bac8ff4`](https://github.com/m2lines/Samudra/tree/6bac8ff4f2acb1edddcf184f1cbd9cfe0f00a762).
Its pre-registered design is documented in
[`perceiver_v2_2deg_architecture_search.md`](perceiver_v2_2deg_architecture_search.md).
All three finalists reached the full 12-epoch budget, including two candidates
recovered from cluster-side cancellation, and the controller's public Parquet
record was reconciled on 2026-08-18.

## Evidence reviewed

This synthesis reviewed every Markdown source in Jesse's
[`docs/experiments`](https://github.com/m2lines/Samudra/tree/codex/decoder-root-cause-report/docs/experiments)
directory. The PDFs in that directory are rendered copies of several Markdown
reports and are not independent experiments.

- [single-step MSE baseline](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_mse.md)
- [single-step research plan](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_research_plan.md)
- [single-step research results](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_research_results.md)
- [decoder root-cause report](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/perceiver_decoder_root_cause.md)
- [learned inverse plan and results](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/learned_inverse_encoder_decoder_plan.md)
- [native-grid multi-resolution discussion](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_multires_model_discussion.md)
- [coarse-latent plan](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_plan.md)
- [coarse-latent results ledger](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_results.md)
- [coarse-latent final report](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_report.md)

## Executive synthesis

Jesse's work does not support the broad conclusion that Perceiver-like models
cannot work for SamudraMulti. It supports a narrower and more actionable
conclusion: the original combination of fixed-patch compression, unordered
learned decoder latents, and unanchored output routing is a poor inductive bias
for reconstructing spatial ocean fields.

The evidence separates three architectural regimes:

1. **Original Perceiver IO heads.** These fail badly. A one-latent decoder is
   mathematically query-blind, wider unanchored latent banks can memorize fixed
   fields without learning spatial correspondence, and the configured attention
   value path was narrower than the 77-channel target state.
2. **Native-grid learned inverse.** Pointwise learned state features plus
   deterministic, channel-masked physical resampling produce the strongest
   demonstrated one-/half-degree model. This is the best production baseline
   from Jesse's experiments, but its processor cost scales with the native grid.
3. **Structured coarse latent.** A fixed coarse grid can retain useful subpatch
   dynamics when it has an explicit resolved-mean path, learned phase-sensitive
   subpatch moments, and a continuous position-anchored decoder. This model beats
   persistence on every tested route and lead, but remains roughly twice as
   inaccurate as the native-grid model and loses too much velocity spectrum.

The resulting strategic position is:

> Continue Perceiver research, but treat set-based patch encoding and continuous
> query decoding as replaceable primitives. Do not preserve the full Perceiver
> IO decoder or an unordered bottleneck merely because they are named
> “Perceiver.”

For eventual LLC-scale data, a fixed or hierarchical coarse representation is
still attractive. Jesse's coarse-moment result is positive evidence that this
goal is possible. It also shows that generic latent count is the wrong capacity
measure: the latent must explicitly retain spatial phase, resolved quantities,
and physically anchored routes.

## Summary of Jesse's findings

### 1. The original SamudraMulti failure is reproducible at one scale

The full one-degree, one-step baseline reached normalized MSE `0.29469`, versus
the quoted Samudra v2 reference `0.0236`. On the calibrated 512-timestamp proxy,
SamudraMulti reached `0.38508` and v2 reached `0.04287`. The proxy preserved the
large model ranking at about one fifth of the end-to-end compute, making it a
useful architecture screen even though its absolute loss did not predict the
full-data value.

The failure was not repaired by normalization, processor dilation, or scalar
embedding-width changes. Those effects were smaller than seed variation. Plain
MSE improved interpretability but not architecture quality. Fixed-set identity
tests exposed severe smoothing, especially for velocity, and growing patch-seam
structure at finer resolutions.

### 2. The decoder caused most of the surprising autoencoder error

A processor-bypassed 32-sample factorial isolated the representation heads:

| Encoder | Decoder | Reconstruction MSE |
| --- | --- | ---: |
| Direct | Direct | `0.012083` |
| Perceiver | Direct | `0.025431` |
| Direct | Perceiver IO | `0.279303` |
| Perceiver | Perceiver IO | `0.285353` |

Changing only the encoder added about `0.0133` MSE; changing only the decoder
added about `0.2672`. Replacing the Perceiver IO decoder with physical resampling
plus a learned pointwise projection then reduced the matched forecast proxy from
`0.381735` to `0.051655`.

This supports retaining a learned encoder as a research object. It does not
support retaining the original decoder.

### 3. The decoder has several distinct causal failures

Jesse isolated mechanisms that should not be collapsed into “insufficient
capacity”:

- **One-latent query blindness.** With one decoder latent, output cross-attention
  has a length-one softmax. Without a query residual, every spatial query receives
  the same value; spatial output difference and query gradient are exactly zero.
- **Narrow value transport.** On a one-cell held-out copy task, widening the true
  attention value path from 64 to 128 reduced MSE from `0.239298` to `0.021848`.
  Widening latent or query embeddings without widening the transported values did
  not have the same effect.
- **Unanchored spatial assignment.** A many-latent model fit fixed fields near
  `1e-3` while reaching held-out MSE `1.660140`. It memorized sample/coordinate
  associations without learning an input-cell-to-output-cell identity route.
- **Context competition.** Plain direct attention learned small copy tasks but
  became much worse at larger windows and with irrelevant neighbor context.
  A physical position bias removed most of that scale/context sensitivity.
- **Interpolation-temperature conflict.** A sharp attention anchor copied
  matching grids but acted like nearest-neighbor off grid; a broader anchor
  interpolated better but contaminated identity. A deterministic physical base
  plus a learned correction removed that initialization conflict.

In learned-encoder synthetic tests, however, the simple physical-coordinate
resampling base eventually beat the attention hybrid on every route and was
four to seven times faster. This rejected attention as an automatic decoder
default for a native-cell latent grid. It did not reject anchored attention for
unpacking a genuinely coarse token.

### 4. Multi-resolution transport exposed encoder and mask-order failures

One vector per fixed physical patch discarded high-resolution structure before
the decoder saw it. Both half-degree output routes retained only about half the
target high-wavenumber power, including half-to-half reconstruction. Retaining a
learned feature vector at every native cell restored same-grid reconstruction
and spectral power.

Cross-resolution coastal transport then exposed a separate error. Learned latent
channel mixing does not commute with interpolation under variable- and
depth-specific wet masks. Projecting back to physical prognostic channels before
channel-wise masked resampling removed 78% of the half-to-one excess error on
identical weights, while changing same-grid MSE by less than `4e-9`.

Large downsampling also needs scale-aware restriction. At fourfold
quarter-to-one restriction, area averaging reduced the deterministic bilinear
floor by 86% and restored high-wavenumber power. At twofold restriction it
improved spectra but worsened MSE, so a conservative/antialiased operator must be
selected by scale rather than applied universally.

### 5. State, forcing, geometry, and time needed separate contracts

Encoding boundary forcing together with persistent state contaminated the inverse.
A state-only inverse reduced one-degree reconstruction by 78.5% and improved all
one-/half-degree routes. The selected contract therefore encodes state once,
supplies one separately encoded boundary state per physical transition, and adds
geometry to the processor as a sidecar rather than to reconstructive content.

A soft reconstruction penalty did not preserve the inverse during forecast
training: encoder and decoder weights co-adapted while forecast loss remained
good. Freezing the learned inverse preserved it exactly. A zero-initialized
per-channel latent residual transition then improved leads 1, 2, and 4 over a
replacement-form processor.

Repeated processor calls also needed supervision at their actual physical leads.
Training only depth one produced severe degradation at depths two and four;
cycling true depths `{1,2,4}` and using ordered boundary states repaired the
iteration contract. These are processor/exposure findings, not decoder findings.

### 6. The native-grid model is the strongest validated baseline

The frozen native-grid learned inverse plus latent ReZero transition completed
the full one-/half-degree run at aggregate lead MSE `{0.03982, 0.05408, 0.06595}`
for leads `{1,2,4}`. Every route beat lead-matched persistence, and same-grid
reconstruction stayed fixed near `1e-3`. Boundary-zeroing and time reversal
increasingly damaged longer leads, confirming that forcing magnitude and order
were used.

Its remaining problems were not renewed decoder query blindness. Half-degree
same-grid velocity power remained `0.790/0.751`, coarse-to-fine velocity power
was only `0.288/0.149`, and multi-resolution one-to-one lead-one loss was 28.7%
worse than the single-resolution model. These point toward processor exposure,
processor capacity, and source-information limitations.

### 7. A structured coarse latent works, but is not yet competitive

Jesse's coarse-latent experiment preserved a fixed `60 x 72` processor grid and
replaced generic patch compression with:

- cosine-latitude-weighted resolved patch means;
- 16 learned continuous within-patch moments;
- a coordinate-resampling base; and
- a zero-initialized, continuous position-anchored attention correction.

Synthetic counterfactual advection showed that this representation retained
subpatch phase and used it to predict different future coarse tendencies for
states with identical initial patch means. On OM4, the learned decoder reduced
inverse error by 62--72% relative to bilinear-only decoding and retained 13--22
times more gradient power.

The full one-/half-degree coarse model beat persistence on all 12 route/lead
comparisons at `{0.08275, 0.10085, 0.12806}` aggregate MSE. It causally depended
on the learned moment channels and aligned boundary forcing. Nevertheless, its
errors were 1.9--2.1 times the native-grid model, its half-degree velocity spectra
were strongly attenuated, and teacher-latent error grew with lead. Jesse therefore
retained it as the preferred coarse-latent research architecture, not as the
production replacement.

## Completed Perceiver v2 search

The search trained 18 initial candidates for one epoch, promoted nine to epoch
3, five to epoch 6, and three to epoch 12. Every promoted learning curve
improved monotonically. The complete finalist comparison is:

| Candidate | Epoch 1 | Epoch 3 | Epoch 6 | Epoch 12 | Final rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct-no-context-lr4` | 0.356488 | 0.272234 | 0.234455 | **0.198794** | **1** |
| `direct-transport256-lr8` | **0.319645** | **0.263441** | **0.231851** | 0.201974 | 2 |
| `direct-enc64-lr8` | 0.321942 | 0.265365 | 0.236658 | 0.203297 | 3 |

The winner is `direct-no-context-lr4`: a patch-local Perceiver encoder with
256 internal latents, direct query decoding, transport width 128, no decoder
context rings, 6 x 10 degree patches, and learning rate `4e-4`. Its validation
loss fell 44.2% from epoch 1 to epoch 12. It finished 1.6% below the wide-
transport finalist and 2.2% below the 64-latent finalist. It also had the lowest
epoch-12 train loss, and led at epochs 10, 11, and 12; the final result is not a
single noisy validation point.

The three finalists are close enough that their exact order should not be
treated as a universal architecture ranking. The search used one seed per
candidate, and successive halving pruned the `8e-4` no-context arm at epoch 3,
so context and learning rate remain partly confounded. The durable conclusion
is instead that all three viable directions use direct output queries and avoid
the original second Perceiver IO latent bank.

### Rung-zero screen

All 18 initial workers completed epoch one and 89 optimizer updates. The early
family comparison explains the promotion decisions and remains useful for
separating fast-learning interventions from the final-budget ranking.

| Architecture family | LR `4e-4` | LR `8e-4` | Two-rate mean | Change from direct-control mean | Mean train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct-coarse-patch` | [0.342571](https://wandb.ai/ocean_emulators/default/runs/oitxuuuu) | [0.324788](https://wandb.ai/ocean_emulators/default/runs/a04sz1f5) | **0.333679** | **-8.3%** | 20.3 min |
| `direct-transport256` | [0.354275](https://wandb.ai/ocean_emulators/default/runs/oj6rkd03) | [0.319645](https://wandb.ai/ocean_emulators/default/runs/rvv5lvh9) | **0.336960** | **-7.4%** | 27.6 min |
| `direct-enc64` | [0.363536](https://wandb.ai/ocean_emulators/default/runs/l8q3isnz) | [0.321942](https://wandb.ai/ocean_emulators/default/runs/ftjt32xh) | 0.342739 | -5.8% | 23.0 min |
| `direct-no-context` | [0.356488](https://wandb.ai/ocean_emulators/default/runs/9a1vcecr) | [0.332763](https://wandb.ai/ocean_emulators/default/runs/xn2e6a5j) | 0.344625 | -5.3% | 26.7 min |
| `direct-transport64` | [0.365438](https://wandb.ai/ocean_emulators/default/runs/otyae5f0) | [0.345617](https://wandb.ai/ocean_emulators/default/runs/02pi4nme) | 0.355527 | -2.3% | 27.5 min |
| `direct-context2` | [0.383464](https://wandb.ai/ocean_emulators/default/runs/65429i8w) | [0.344039](https://wandb.ai/ocean_emulators/default/runs/zzzzbej1) | 0.363752 | -0.1% | 27.2 min |
| `direct-control` | [0.366762](https://wandb.ai/ocean_emulators/default/runs/gd21xu96) | [0.361208](https://wandb.ai/ocean_emulators/default/runs/x3i0h8a6) | 0.363985 | control | 27.0 min |
| `pio-lean` | [0.403511](https://wandb.ai/ocean_emulators/default/runs/ywymchma) | [0.400348](https://wandb.ai/ocean_emulators/default/runs/g6nw1n4d) | 0.401929 | +10.4% | 40.8 min |
| `pio-control` | [0.421770](https://wandb.ai/ocean_emulators/default/runs/zfi0l44s) | [0.415711](https://wandb.ai/ocean_emulators/default/runs/gwim0j07) | 0.418740 | +15.0% | 38.4 min |

These losses come from four-step autoregressive validation, not Jesse's
processor-bypassed reconstruction task. They are more relevant to forecast
selection, but less able to localize whether an effect belongs to the inverse,
processor, or temporal feedback path.

### What the completed search says

1. **Direct decoding is strongly favored.** Every direct candidate beat every
   full Perceiver IO candidate at epoch one, and no full-PIO candidate survived
   promotion. The direct control mean was 13.1% below the PIO control mean, and
   the eventual winner also used direct output queries. This is the strongest
   topological decision from the search.
2. **Removing unanchored decoder context is the best final-budget
   intervention.** `direct-no-context-lr4` won at epoch 12. The processor already
   mixes neighboring spatial information; anonymous neighboring tokens in the
   final query operation appear to make routing harder rather than add useful
   context.
3. **Wider value transport accelerates early learning but is not the final
   winner.** Transport width 256 led at epochs 3 and 6 and finished second at
   epoch 12. The width sweep was not monotonic, and width remains confounded with
   head count, so the result does not establish a simple capacity law.
4. **Fewer encoder latents are competitive and cheaper.** The 64-latent encoder
   finished only 2.2% behind the winner and trained faster. Generic internal
   latent count is therefore not the principal encoder-capacity knob at 2
   degrees. Both models still mean-pool the latent bank to one patch vector, so
   neither tests spatial phase preservation.
5. **Coarser patches are efficient but not selected.** The 10 x 20 degree patch
   was the best epoch-one family and reduced the processor grid by 70%, but it
   was fourth at epoch 6 and was pruned. Aggregate loss still cannot determine
   whether its early advantage came from smoothing fine structure.
6. **Learning rate interacts with architecture and budget.** `8e-4` won every
   epoch-one pair, but the `4e-4` no-context candidate won at epoch 12. The next
   search should not multiply a wide rate sweep across every structural arm;
   it should retain `4e-4` and `8e-4` only for the small set needed to resolve
   this interaction, then fix the selected schedule.

### Subsequent decoder evidence

The later
[`perceiver_2deg_seam_removal_search.md`](perceiver_2deg_seam_removal_search.md)
does not change the primary search's direct-decoder conclusion, but it changes
what should be fixed before the next encoder search. One-ring overlap-add cut
the major absolute-model `zos` window seam by more than half, and the residual
overlap winner reached one-step validation loss `0.07291` versus `0.23530` for
the best absolute overlap model. Decoder context did not improve the matched
absolute overlap model. The scalable default should therefore move from hard
window assembly to overlap-add, while residual prediction remains the leading
parameterization pending short-rollout stability checks.

Two controlled follow-ups are currently running: a residual/absolute by
hard/overlap assembly experiment, and a residual/absolute by processor-
conditioning by pixel-refinement experiment. Their results should select the
decoder reconstruction details used by the next encoder search; they should not
delay defining or implementing the encoder candidates.

## Which open questions the current search answers

| Jesse's open mechanism | Current intervention | Evidence possible from this search | Remaining ambiguity |
| --- | --- | --- | --- |
| Redundant decoder latent bank | Direct decoder versus `pio-lean` and `pio-control` | Direct forecast loss, learning speed, runtime | No controlled zero-depth identity comparison in the same run |
| Narrow attention value path | Transport widths 64, 128, and 256 | Whether wider output transport improves forecast optimization | Head count and total width change together; no variable-wise transport diagnostic |
| Context competition | Context rings 0, 1, and 2 | Whether more local tokens improve forecast loss and runtime | No explicit physical position bias, so it cannot test whether anchoring makes context useful |
| Encoder latent count | 64 versus 256 Perceiver latents | Cost and optimization sensitivity to internal latent count | Both collapse to one mean-pooled patch vector; no test of multiple spatial outputs or subpatch moments |
| Fixed patch compression | 6 x 10 versus 10 x 20 degree patches | Loss/throughput tradeoff from a smaller processor grid | No native-grid/direct anchor, spectra, seam, or subpatch counterfactual test |
| Full forecast behavior | Four-step training and validation | Early autoregressive model selection | Does not isolate frozen inverse, true latent autoregression, boundary alignment, or decode/re-encode drift |
| Optimizer robustness | Learning rates `4e-4` and `8e-4` | Whether a family effect survives a rate pair | One seed per cell and the upper rate still wins universally |

## Important questions this search does not answer

The current search should not be used to conclude that Jesse's remaining
mechanisms are resolved. It does not test:

- position-anchored attention against plain direct attention;
- a deterministic coordinate route or a zero-initialized attention correction;
- normalized versus raw-amplitude attention values;
- additive encoder geometry versus no geometry versus a processor sidecar;
- state-only encoding and separately time-aligned boundary forcing;
- projection-before-channel-masked resampling;
- frozen-inverse latent autoregression at true physical leads;
- resolved means plus phase-sensitive subpatch moments;
- native-grid or direct one-cell diagnostic anchors;
- variable/depth MSE, velocity power, seams, bias, or amplitude; or
- cross-resolution generalization and scale-aware restriction.

These omissions are expected: the current search is an economical first screen,
not a reproduction of the full root-cause program.

## Remaining research questions

These questions should not be combined into one Cartesian search. The active
decoder-reconstruction experiments address parts of Q1--Q3; the recommended
next primary search concentrates on Q4--Q5.

### Q1: can explicit physical anchoring make context useful?

**Hypothesis.** A direct decoder with a continuous physical position bias will
outperform plain direct attention when neighboring context is present, while
remaining no worse with zero context. The effect should appear most clearly in
patch-edge metrics and later at mismatched input/output resolutions.

This directly combines the current no-context result with Jesse's synthetic
finding that anchoring removed context sensitivity.

### Q2: is the 256-wide decoder gain information transport or optimization?

**Hypothesis.** Increasing transported value width improves velocity and deep
channel losses more than scalar surface fields. If all variable groups improve
uniformly or the effect vanishes after several epochs, the rung-zero result is
more likely general optimization capacity than a 77-channel transport bottleneck.

Use parameter-matched head/width variants where practical so total transported
width and head count are not permanently confounded.

### Q3: does the coarse-patch winner preserve useful fine structure?

**Hypothesis.** The coarse-patch model's epoch-one MSE advantage is partly due to
smoother, easier predictions. It should lose more velocity high-wavenumber power
than the control unless the Perceiver patch encoder has learned meaningful
subpatch summaries.

The decision requires variable/depth losses, amplitude ratios, spectra, and
patch-seam diagnostics from identical checkpoints. Aggregate validation MSE is
not sufficient.

### Q4: what spatial information does the patch Perceiver retain?

**Hypothesis.** Generic latent count will matter less than the structure of the
patch output. A learned resolved-mean route plus multiple phase-sensitive
subpatch coefficients will outperform mean-pooling 64 or 256 unordered latents
at the same processor grid.

This should be tested against both the best current Perceiver encoder and a
native-grid/direct diagnostic anchor. The anchor is not necessarily the final
architecture; it quantifies the cost of compression.

### Q5: can a structured Perceiver subsume the coarse-moment encoder?

**Hypothesis.** A Perceiver can remain the learned set encoder while exposing
structured outputs: one resolved token and several coordinate-conditioned
subpatch tokens or coefficient blocks. The key requirement is that the outputs
remain spatially addressable through the processor and decoder rather than being
mean-pooled or packed into channels with no unpacking coordinate.

A strong result would reproduce the counterfactual property from Jesse's
coarse-moment experiment: two fields with equal patch means but different front
positions produce distinguishable latents and different future coarse tendencies.

### Q6: which learning-rate schedule compares structures fairly?

**Hypothesis.** `8e-4` is useful for fast screening, while `4e-4` has a better
late-budget asymptote for at least the no-context decoder. A small matched bridge
on the selected baseline should decide whether to use one fixed rate, a schedule,
or a longer first rung. Rates should be selected before the structural matrix
rather than multiplied across every expensive candidate indefinitely.

## Proposed experiment funnel

### Stage 0: completed primary architecture screen

The original 18-candidate search is complete and durably published. It selected
direct output queries, rejected the full Perceiver IO decoder as a primary path,
showed that unanchored decoder context is unnecessary, and established that a
64-latent patch encoder is a competitive compute control. The remaining metric
audit is useful characterization, but is no longer a prerequisite for choosing
the topology of the next search.

### Stage 1: finish decoder reconstruction selection

The seam-removal search selected overlap-add and residual prediction on one-step
evidence. The two active controlled searches should now determine whether the
residual advantage survives a matched assembly comparison and whether
processor conditioning or pixel refinement removes the remaining channel-tail
artifacts. Select one decoder recipe using loss, per-channel seam tails,
periodic-mode power, and matched short rollouts. Do not open another broad
decoder sweep unless these experiments falsify their mechanisms.

### Stage 2: recommended next primary search -- structured latent transport

The next primary search should permit paired encoder/decoder changes rather than
holding the direct decoder completely fixed. Its central question is whether a
scalable Perceiver can transport spatially addressable coarse and subpatch
state without forcing all information through one mean-pooled patch vector.

Three external architecture ideas sharpen this direction. The
[Hierarchical Perceiver](https://arxiv.org/abs/2202.10890) restores locality and
hierarchical grouping so Perceiver-like models can scale to much larger raw
signals. [GINO](https://arxiv.org/abs/2309.00583) maps irregular physical points
to and from a regular latent grid using local geometric operators. The
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895) motivates learning
between function spaces rather than binding every parameter to one grid,
although Samudra should not assume that Fourier truncation alone preserves the
fine ocean spectrum. These suggest testing locality, explicit latent geometry,
and multiresolution transport as mechanisms, not importing any entire model.

Use the following architecture-level candidate set:

| Candidate family | Paired encoder/decoder contract | Role |
| --- | --- | --- |
| `mean64-direct` | Current 64-latent mean-pooled patch encoder plus selected direct overlap decoder | Efficient Perceiver baseline |
| `resolved-anomaly` | Exact masked, area-weighted patch mean plus one learned Perceiver anomaly summary; deterministic mean prolongation plus learned anomaly correction | Minimal physically anchored intervention |
| `spatial-latent-grid` | Cross-attend each patch into a small coordinate-tied latent grid, such as 2 x 2, preserve those token coordinates through processing, and query them directly on decode | Tests whether spatially addressable latents remove the phase/routing failure |
| `hierarchical-perceiver` | Local coordinate-tied latents at two levels, with grouped restriction to coarse global tokens and symmetric local prolongation | HiP-inspired scalable candidate for LLC-sized inputs |
| `mean-detail-pyramid` | Exact conservative coarse state plus masked Haar/lifting-style detail coefficients; use Perceiver blocks to encode and evolve learned detail features, then reconstruct through the paired inverse transform | Preserves resolved means and subpatch phase by construction |
| `local-operator-bridge` | Geometry- and mask-aware local cross-attention from physical cells to a fixed latent mesh, and local coordinate queries back to the output grid | GINO-like test of patch-free, resolution-flexible transport |
| `native-grid-anchor` | Native-cell learned inverse and direct output projection | Information-preserving diagnostic ceiling, not a scalable finalist |

Jesse's coarse-moment encoder remains a valuable positive control in the
representation gate, but it need not consume a full forecast arm if
`resolved-anomaly` reproduces its reconstruction and counterfactual behavior.
Likewise, the already-tested 256-latent mean-pooled encoder should not be rerun:
the completed search showed that generic latent count is low-information.

The highest-priority new candidate is `mean-detail-pyramid`. It combines an
exact low-frequency route, explicit phase-carrying detail state, linear local
transforms, and a natural hierarchy across resolutions. The
`spatial-latent-grid` candidate is the cleanest test of a more recognizably
Perceiver-native solution. The `local-operator-bridge` is the strongest
resolution-flexibility control and tests whether fixed patch boundaries are
the wrong abstraction entirely.

This should be a gated search rather than seven equally expensive full runs:

1. **Representation gate.** Test held-out reconstruction, velocity-spectrum
   retention, and an equal-patch-mean/different-front-position counterfactual.
   Reject an encoder if its latents cannot distinguish the counterfactual or if
   its decoder cannot reconstruct the difference.
2. **Forecast screen.** Train all passing scalable candidates for three epochs
   on the same 2-degree four-step task. Use the selected learning-rate schedule,
   not another full rate Cartesian product. Keep the native-grid anchor for
   diagnosis rather than promotion.
3. **Promotion.** Advance the best three scalable representations to 12 epochs.
   Rank by autoregressive validation loss subject to spectral, amplitude, and
   seam guardrails, rather than aggregate MSE alone.
4. **Confirmation.** Repeat the best two representations with two new seeds and
   short 5-, 10-, and 20-step rollouts. A difference smaller than seed spread is
   not an architecture decision.

Use transport width 128 and zero anonymous context for candidates that retain
the existing direct decoder. New paired decoders should use coordinate-local
routing, the selected overlap rule, and the same output projection capacity so
that the search compares representation contracts rather than arbitrary decoder
size. Retain one `mean64-direct` width-256 arm only as a bridge to the fast-
learning finalist.

### Stage 3: repair the temporal and multi-resolution contract

Once a single-scale representation is selected:

1. train a state-only inverse;
2. separate boundary encoding and geometry sidecar conditioning;
3. freeze the inverse and train a zero-initialized latent residual transition;
4. supervise true physical leads `{1,2,4}`;
5. add one-/half-degree routes and channel-masked transport; and
6. validate large-ratio conservative restriction before quarter-degree training.

This stage should reuse Jesse's demonstrated contract rather than asking the
architecture search to rediscover it indirectly.

## Search-system finding from this search

The public controller record and Parquet tables are now reconciled, but the run
exposed two important states that were not represented clearly while it was
active. First, all 18 initial W&B workers were terminal while the public
controller still showed a running empty rung. Second, two healthy finalists
were cancelled by Slurm, while the search temporarily reported `complete`
because one finalist succeeded. Both were manually resumed from checkpoints to
obtain the final comparison.

Before the next search, the durable system should distinguish:

- workers complete, controller pending;
- worker artifacts validated, promotion pending;
- publication pending or failed; and
- scientifically ineligible workers.

A periodic controller/reconciler heartbeat or a worker-completion manifest in
the public bucket would let an agent and a human detect this state without W&B
API access. The search should also alert when every array worker is terminal but
the rung has not advanced within a bounded grace period.

## Recommendation

Retire the full Perceiver IO latent bank from the primary decoder path. Use
direct output queries, zero anonymous context rings, and overlap-add assembly.
Select residual versus absolute prediction and optional full-resolution
refinement from the two active controlled searches before freezing the decoder
recipe.

The best next primary search is the structured latent-transport experiment in
Stage 2, not another broad sweep over decoder heads, context, latent count, and
learning rate. Encoder and decoder should be co-designed where the latent
contract requires it. The top three primary-search finalists are separated by
only 2.2%, while Jesse's evidence identifies loss of subpatch phase as a much
larger and more fundamental scaling limitation. A generic mean-pooled latent
bank is unlikely to close that gap simply by adding more latents, heads, or
epochs.

## Results and discussion

The completed result changes the next-round priorities in six concrete ways:

| Current finding | Planning consequence |
| --- | --- |
| Every direct decoder beats both full Perceiver IO families at the screen, and every finalist is direct | Remove full Perceiver IO from the primary path; retain it only as a historical diagnostic control |
| Transport width 256 leads through epoch 6 and finishes second | Keep one width-256 bridge arm, but do not make width the main next-round axis |
| Zero context wins at epoch 12 | Use zero anonymous decoder context by default; test context again only with explicit physical anchoring |
| Encoder 64 finishes within 2.2% of the winner at lower cost | Use it as the efficient baseline and shift encoder capacity experiments toward structured spatial outputs |
| Coarse patches win early but rank fourth at epoch 6 | Treat them as an efficiency point and require spectral/information gates before promotion |
| `8e-4` wins early, while `4e-4` wins the final budget | Resolve the schedule on a small bridge comparison instead of multiplying rates across every structural candidate |

The exact finalist order remains provisional because there is one seed per cell
and the no-context learning-rate pair did not both reach epoch 12. The strongest
conclusion is topological rather than numerical: the full decoder latent bank is
consistently worse, while where and how spatial information is compressed
remains unresolved.

The controller/publication lag and partial-finalist completion are also part of
the experimental result. The next search-system iteration should make
terminal-worker reconciliation, retry state, and publication health directly
queryable.

## Next-round decision record

Proceed with the structured latent-transport search after the active decoder
reconstruction searches select the common output assembly and prediction
parameterization. Implement paired encoders and decoders plus representation
and counterfactual gates first; then compare the scalable contracts on matched
three-epoch forecasts, promote three to epoch 12, and confirm the best two with
additional seeds and short rollouts. Do not combine this with another broad
decoder or optimizer Cartesian sweep.
