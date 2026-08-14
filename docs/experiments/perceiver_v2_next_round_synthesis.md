<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2: prior evidence and next-round research synthesis

## Status

This is a living planning report. It synthesizes Jesse's decoder-root-cause,
learned-inverse, single-step, and coarse-latent experiments with the preliminary
first-rung results from the current 2-degree Perceiver v2 search. It is intended
to define the questions for the next experiment round, not to select a final
architecture before the current successive-halving search finishes.

The current search is
`perceiver-v2-2deg-architecture--20260814T171003.874785Z`, running immutable
code revision
[`6bac8ff4`](https://github.com/m2lines/Samudra/tree/6bac8ff4f2acb1edddcf184f1cbd9cfe0f00a762).
Its pre-registered design is documented in
[`perceiver_v2_2deg_architecture_search.md`](perceiver_v2_2deg_architecture_search.md).

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

## Current Perceiver v2 search: preliminary rung-zero evidence

All 18 W&B workers completed epoch one and 89 optimizer updates. At the time of
this note, the public controller state still reports `running`, contains no rung
results, and has not published `results.parquet` or `epochs.parquet`. The values
below are therefore read directly from finished W&B summaries and are provisional
until the controller validates and publishes the rung.

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

### What rung zero tentatively says

1. **Direct decoding is strongly favored.** Every direct candidate beats every
   full Perceiver IO candidate. The direct control mean is 13.1% below the PIO
   control mean, and the direct path is substantially faster. This agrees with
   Jesse's diagnosis of the redundant unordered decoder latent bank.
2. **A wider decoder path remains promising.** Transport width 256 is the best
   individual run and second-best family mean. The result is consistent with
   Jesse's value-path bottleneck, although widths 64/128/256 are not monotonic,
   so the evidence does not yet establish a simple capacity law.
3. **Extra unanchored context does not help.** Zero context beats the control;
   two context rings are tied with or worse than it. This agrees with Jesse's
   finding that irrelevant context competes with the correct spatial route when
   attention lacks a physical anchor.
4. **Fewer encoder latents are not harmful at this budget.** Reducing 256 to 64
   improves the two-rate mean and training time. This falsifies the preliminary
   expectation that encoder latent count alone would be a limiting capacity
   measure at 2 degrees. It does not test spatial phase preservation: both
   configurations still mean-pool the latent bank to one vector per patch.
5. **Coarser patches optimize quickly, but may be smoothing.** The 10 x 20 degree
   patch is the best family by epoch-one mean and trains about 25% faster than the
   direct control because its processor grid is 70% smaller. Validation MSE alone
   cannot tell whether it retained velocity and high-wavenumber structure. Jesse's
   coarse-latent evidence makes this the most important result to audit before
   promotion, not an automatic architecture win.
6. **The higher learning rate wins every pair.** `8e-4` is better in all nine
   families. Architecture rankings may still change as training continues, but
   the next search should include at least one rate above `8e-4` or use a short
   learning-rate range test before spending the full architecture budget.

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

## Questions and hypotheses for the next round

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

### Q6: is `8e-4` still below the useful learning-rate range?

**Hypothesis.** At least part of the current ranking is undertraining. A short
range test or paired `8e-4`/`1.2e-3` screen on the promoted families will improve
sample efficiency without destabilizing validation. Rates should be selected
before a larger structural matrix rather than multiplied across every expensive
candidate indefinitely.

## Proposed experiment funnel

### Stage 0: finish and diagnose the current search

1. Allow successive halving to complete at cumulative epochs 3, 6, and 12.
2. Publish and compare `results.parquet` and `epochs.parquet` rather than selecting
   from W&B summaries alone.
3. Run the durable metric suite on at least the direct control, the final winner,
   the best full-PIO control, and any pruned candidate whose rung-zero result
   poses a distinct mechanistic question.
4. Report variable/depth MSE, velocity spectra, amplitude, bias, seams, parameter
   count, and runtime before deciding whether coarse patches advance.

### Stage 1: decoder-mechanism search on the fixed encoder

Keep the best current encoder/patch setting fixed and compare a small causal set:

1. plain direct decoder at the promoted transport width;
2. direct decoder with continuous physical position bias;
3. anchored direct decoder with zero versus one context ring;
4. raw-amplitude value path versus normalized value path; and
5. deterministic physical-coordinate base plus a zero-initialized anchored
   correction, if the latent/output coordinate contract permits it.

Use the promoted learning-rate range and at least two seeds for finalists. The
goal is not a Cartesian sweep; it is to determine whether anchoring, context, and
value transport explain distinct residual errors.

### Stage 2: encoder-information search with the selected decoder

Compare:

1. current mean-pooled Perceiver patch encoder;
2. fewer-latent current encoder as the compute control;
3. explicit resolved mean plus learned Perceiver anomaly summary;
4. resolved mean plus multiple coordinate-conditioned subpatch tokens or moment
   blocks;
5. Jesse's moment encoder as a positive coarse-latent control; and
6. native-grid/direct projection as an information-preserving diagnostic anchor.

First run reconstruction and synthetic counterfactual gates. Promote only
representations that retain fine structure and change future coarse tendencies,
then spend four-step OM4 forecast compute.

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

## Search-system finding from this rung

All 18 W&B runs report `finished`, finite validation loss, and 89 optimizer
updates, while the public state still reports a running first rung with zero
results. This is a correctness-observability gap even if the delayed Slurm
controller eventually reconciles it.

Before the next search, the durable system should distinguish:

- workers complete, controller pending;
- worker artifacts validated, promotion pending;
- publication pending or failed; and
- scientifically ineligible workers.

A periodic controller/reconciler heartbeat or a worker-completion manifest in
the public bucket would let an agent and a human detect this state without W&B
API access. The search should also alert when every array worker is terminal but
the rung has not advanced within a bounded grace period.

## Provisional recommendation

Continue the current search. If the direct-decoder advantage survives later
rungs, retire full Perceiver IO from the primary decoder path. Carry forward
the 256-wide direct decoder and the best learning-rate region, but do not select
coarse patches until spectral and variable-wise diagnostics are available.

For the next architecture round, prioritize physically anchored direct decoding
and structured Perceiver patch outputs. That direction is consistent with the
long-term need for scalable, resolution-flexible latent states and with Jesse's
strongest causal evidence. A generic mean-pooled latent bank is unlikely to close
the remaining gap simply by adding more latents, heads, or epochs.

## Results and discussion

The preliminary result changes the next-round priorities in six concrete ways:

| Current finding | Planning consequence |
| --- | --- |
| Every direct decoder beats both full Perceiver IO families | Remove full Perceiver IO from the primary path if the gap persists beyond rung zero; retain one arm as a diagnostic control |
| Transport width 256 is the best individual candidate | Carry a wide value path forward, then separate total width from head count and inspect variable-wise gains |
| Zero context beats one or two unanchored rings | Test physically anchored context rather than sweeping still larger unanchored neighborhoods |
| Encoder 64 beats encoder 256 | Stop treating generic latent count as the principal encoder-capacity knob; test spatially structured outputs |
| Coarse patches win early loss and throughput | Require spectra, velocity/depth losses, amplitude, and seam diagnostics before promotion |
| `8e-4` wins every learning-rate pair | Calibrate the upper learning-rate range before the next expensive architecture matrix |

These decisions are provisional because epoch-one successive-halving comparisons
favor fast starters, have one seed per cell, and do not yet include spatial
diagnostics. The strongest conclusion is topological rather than numerical: the
full decoder latent bank is consistently worse, while where and how spatial
information is compressed remains unresolved.

The controller/publication lag is also part of the experimental result. W&B
proves that workers trained, but durable search consumers cannot yet rank them.
The next search-system iteration should make terminal-worker reconciliation and
publication health directly queryable.

## Next-round decision record

_Pending review of this synthesis and the completed current search._
