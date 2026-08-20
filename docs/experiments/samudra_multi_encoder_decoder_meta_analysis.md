<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# SamudraMulti encoder/decoder architecture meta-analysis

## Status and scope

This review was written on 2026-08-20 after rereading Jesse Derer's complete
decoder-root-cause experiment series and the subsequent 2-degree Perceiver
search notebooks. It asks two related questions:

> What encoder/decoder architecture is most likely to make SamudraMulti a
> strong, scalable multi-resolution ocean predictor?

> Before sharing parameters across resolutions, what is the strongest
> encoder-processor-decoder we can train and evaluate at one degree?

The answer is intentionally not the candidate with the smallest number in one
table. The experiments differ in grid, target, training duration, temporal
contract, and degree of compression. This review first identifies conclusions
supported by controlled comparisons, then makes compositional predictions where
separate experiments support compatible mechanisms. Predictions without a
direct experiment are labeled as such and converted into proposed tests.

The current evidence supports three different answers:

1. **Best demonstrated single-scale architecture today:** Jesse's native-grid,
   state-only
   learned inverse with projection-before-channel-masked coordinate resampling,
   frozen during forecast training and followed by a latent ReZero transition.
   Its completed full-data one-degree run selected epoch 58 at lead-1/2/4 MSE
   `0.020519/0.034380/0.046903`, while its frozen inverse remained at
   `0.000648`. It is the accuracy and correctness reference through one and half
   degree.
2. **Best current single-scale Perceiver candidate:** a coordinate-tied
   `spatial-grid2` encoder, the direct output-query decoder, physical residual
   prediction, one-ring overlap assembly, and smooth processor conditioning,
   with no anonymous decoder context and no pixel-refinement block. This exact
   composition has not been trained to convergence; it is the most informative
   scalable candidate, not the measured champion.
3. **Best predicted multiscale architecture:** a hierarchical, conservative
   mean/detail transport with coordinate-tied local Perceiver latents and a
   paired synthesis decoder. It should preserve an exact resolved route and
   explicit spatial phase, use Perceiver attention for learned local detail
   compression and coupling, and retain Jesse's temporal, geometry, boundary,
   masking, and frozen-inverse contracts.

The second and third architectures have not yet been tested as a whole. That is
the main research opportunity identified by this review. The immediate
one-degree decision and the longer-term LLC design should therefore be reported
separately rather than forcing one architecture to answer both questions.

## Source record

### Jesse's investigation

The following documents were reviewed from the
[`codex/decoder-root-cause-report`](https://github.com/m2lines/Samudra/tree/codex/decoder-root-cause-report/docs/experiments)
branch:

- [SamudraMulti single-step MSE baseline](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_mse.md)
- [single-step research plan](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_research_plan.md)
- [single-step research results](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_single_step_research_results.md)
- [decoder root-cause report](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/perceiver_decoder_root_cause.md)
- [learned inverse plan](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/learned_inverse_encoder_decoder_plan.md)
- [native-grid multiresolution model discussion](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/samudra_multi_multires_model_discussion.md)
- [coarse-latent dynamics plan](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_plan.md)
- [coarse-latent results ledger](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_results.md)
- [coarse-latent final report](https://github.com/m2lines/Samudra/blob/codex/decoder-root-cause-report/docs/experiments/coarse_latent_highres_dynamics_report.md)

### Subsequent Perceiver experiments

- [direct output-query decoder](perceiver_v2_direct_decoder.md)
- [first 2-degree architecture search](perceiver_v2_2deg_architecture_search.md)
- [next-round synthesis](perceiver_v2_next_round_synthesis.md)
- [decoder seam-removal search](perceiver_2deg_seam_removal_search.md)
- [residual assembly and pixel de-aliasing searches](perceiver_2deg_residual_and_dealiasing_searches.md)
- [structured latent-transport search](perceiver_2deg_structured_transport_search.md)

The public OSN Parquet histories were queried directly for all four 2-degree
search records. Key W&B summaries were also checked for the primary finalists,
the seam winner, DCT-14, spatial-grid-2, processor conditioning, Jesse's native
multiresolution run, and Jesse's coarse-latent run. After the first draft, every
W&B image linked from Alex's visual-review sections in the residual/de-aliasing
and structured-transport notebooks was inspected at its original resolution.
That audit changes the DCT recommendation and qualifies the pixel-refinement
recommendation below. The durable OSN histories, rather than
dashboard-selected points, remain the numerical source for the 2-degree tables.

## Comparability and inference rules

The evidence has three levels.

### Level A: controlled evidence

Candidates share data, objective, budget, and almost all configuration. These
comparisons can support causal architectural claims. Examples are hard versus
overlap assembly at a matched epoch, or decoder transport 128 versus 256 inside
the first search.

### Level B: mechanistically consistent evidence

Experiments differ, but independently expose the same failure or intervention.
For example, Jesse's equal-mean front tests, his coarse moment-channel ablation,
and DCT-14's early forecast advantage all support the value of phase-preserving
subpatch state. Their numerical losses must not be combined, but their
mechanistic agreement is informative.

### Level C: compositional prediction

Two individually supported interventions have not been tested together. The
prediction that low late-stage learning rates and a wider information path will
work well together is one example. Such predictions are useful architecture
specifications, but they require a factorial confirmation before promotion.

Several caveats apply throughout:

- The first 2-degree search used one seed per cell and successive-halving
  censoring. Its 12-epoch finalists are comparable; pruned candidates do not
  have known 12-epoch asymptotes.
- The structured search has only two comparable epochs for four promoted
  candidates. Its result is a strong lead, not a final model selection.
- Jesse's native-grid and coarse-latent numbers use different grids, objectives,
  and training procedures from the 2-degree searches. They establish ceilings
  and mechanisms, not a numerical leaderboard with DCT-14.
- One-step validation cannot establish rollout stability. Artifact ratios near
  one do not establish physical skill, and low MSE does not establish spectral
  fidelity.
- W&B snapshot panels use independent color limits. They support comparisons of
  error morphology, not visual comparisons of amplitude, unless their limits
  are matched.
- “Width” refers to several different controls: internal Perceiver latent
  count, latent feature dimension, processor width, and decoder transported
  value width. Evidence for one is not automatically evidence for another.

## Evidence synthesis

### 1. The original architecture fails before multiscale complexity is added

The historical one-degree, single-step SamudraMulti reached all-channel MSE
`0.29469`, 12.5 times the quoted v2 reference `0.0236`. On the matched proxy it
was 9.0 times worse (`0.38508` versus `0.04287`). Tiny fixed-set identity fits
remained near `0.38--0.40` across one, half, and quarter degree and retained only
`0.16--0.19` mean high-wavenumber power.

This rules out an explanation based primarily on conflicting resolutions or
long rollout. The encoder/decoder representation was already inadequate on a
single-scale reconstruction and one-step task.

### 2. The original Perceiver IO decoder is the clearest rejected component

Jesse's pure-autoencoder localization is unusually decisive:

| Encoder / decoder | All-channel MSE |
| --- | ---: |
| Direct / direct | **0.012083** |
| Perceiver / direct | 0.025431 |
| Direct / Perceiver | 0.279303 |
| Perceiver / Perceiver | 0.285353 |

Changing the decoder accounts for nearly the entire failure. Increasing the
decoder latent count from 1 to 256 did not repair a production-size window,
while widening the actual one-cell value path and removing context
normalization did help. The later 2-degree architecture search independently
found every direct decoder ahead of every full Perceiver IO decoder at epoch 1;
the direct-control family mean was 13.1% below the PIO-control mean and no PIO
candidate was promoted.

**Judgment:** the decoder should not contain a second anonymous learned latent
bank. Perceiver attention remains useful as a direct, local, coordinate-aware
transport or correction, not as another compression stage after the processor.

### 3. Output routing and output assembly are separate problems

The first search selected zero anonymous decoder context at 12 epochs. The
processor already mixes neighboring information; adding unanchored tokens to
the final query makes spatial routing harder. The winner reached `0.198794`,
versus `0.201974` for the wider-transport finalist and `0.203297` for the
64-latent encoder finalist.

The seam searches then isolated assembly. At matched budgets, one-ring
overlap-add changed validation loss by less than 0.3%, yet reduced mean
decoder-window jumps by 42.8--52.5%, p90 jumps by 59.0--63.5%, and worst jumps
by 77.0--78.5%. Input context without blending did not fix the defect. Global
context was smooth but slower and less accurate.

**Judgment:** use direct output queries or paired synthesis, zero anonymous
context, and an explicit cross-boundary assembly mechanism. Context should be
reintroduced only with physical offsets or distance anchoring.

### 4. Residual prediction is powerful, but its production meaning is unresolved

At common epochs, physical residual prediction reduced 2-degree validation loss
by approximately 75% under both hard and blended assembly. The six-epoch seam
winner reached `0.07291`, compared with `0.23530` for the best absolute overlap
model. The independent de-aliasing search reproduced the residual advantage.

This is strong optimization evidence, but it does not directly settle the final
multiresolution contract. Jesse's strongest one-/half-degree model decodes an
absolute state from a frozen inverse and residualizes the *latent transition*:

\[
z_{m+1}=z_m+\alpha\odot P(z_m,E_b(b_m),g),\qquad \alpha_0=0.
\]

That contract reached aggregate lead-1/2/4 MSE
`0.03982/0.05408/0.06595`. A physical residual is also ambiguous when source
and destination grids differ unless the source state is first transported to
the destination grid.

**Judgment:** retain residual prediction in single-scale screens because it
greatly improves information efficiency. For the final multiresolution model,
prefer Jesse's validated latent residual transition and test a route-consistent
physical residual head defined as
`masked_resample(source_state) + predicted_increment`. Do not assume the
single-grid residual result transfers unchanged.

The image audit adds an important implementation qualification. In `so_13`,
`residual-condition` produced coherent basin-scale error without the sharp
rectilinear divisions visible in `residual-none`; `residual-condition-refine`
still showed a strong horizontal division and block structure. In `so_12`, the
unconditioned residual arm showed an especially clear zonal stripe. These are
early, unmatched checkpoints--the refinement arms reached only epochs 1--2--so
they do not estimate converged loss. They nevertheless support smooth processor
conditioning as the safer anti-aliasing path and provide no reason to add pixel
refinement to the full run.

### 5. Generic latent count is not the important encoder capacity axis

Reducing the original patch Perceiver from 256 to 64 internal latents improved
early loss, reduced runtime, and finished only 2.2% behind the first-search
winner. In the structured search, `mean64-wide` and `mean64-direct` were tied
through epoch 2. Packed spatial queries and learned four- or sixteen-moment
summaries were also tied with the mean-pooled baseline after one epoch.

These results do not imply that representation capacity is unimportant. They
show that adding unordered internal latents or widening the final bridge does
not restore spatial information that is discarded when the latent bank is
collapsed into one undifferentiated patch vector.

### 6. Spatial phase is the strongest encoder signal across investigations

Four lines of evidence agree:

1. Jesse's fine decoder queries reduced identity MSE from `0.324229` to
   `0.028478`; retaining 3 x 5 spatial encoder tokens improved it further to
   `0.026923`.
2. A native-cell direct representation achieved proxy MSE `0.041278` and
   full-data one-degree MSE `0.015976`, outperforming the quoted v2 reference.
3. Jesse's coarse mean-plus-moment model retained dynamically useful subpatch
   information. Zeroing its learned moment channels increased lead-one raw MSE
   by factors of roughly 2.4--2.9 in S2, and its full one-/half-degree model beat
   persistence on all 12 route/lead comparisons.
4. In the structured search, the complete 3 x 5 DCT basis was the only
   representation to separate strongly from the baseline. At epoch 2,
   `dct-detail14` reached `0.057458`, 23.8% below `mean64-direct`. DCT-4 was
   also better in loss but produced severe patch jumps, showing that an
   incomplete local basis can improve MSE while breaking reconstruction.

The result is not “DCT is proven best.” DCT-14's mean patch-periodic error power
was 4.15 times the baseline, despite good jump metrics. The post-search image
audit makes that metric concrete: DCT-14 errors contain dense high-frequency,
grid-aligned texture, while DCT-4 produces an unmistakable rectangular lattice.
The same morphology appears across the reviewed salinity and temperature
channels. By contrast, `spatial-grid2` and `mean64-direct` errors are dominated
by smoother, more physically plausible basin-scale structures, even where their
scalar MSE is larger.

DCT also uses a complete basis only because a 2-degree patch contains 3 x 5
cells; a complete basis grows to 60 or 240 coefficients per channel at half or
quarter degree. A flat complete DCT is therefore a diagnostic showing that
phase-addressed information improves short-budget MSE, not a candidate to
promote. Its low MSE is vetoed by a pre-registered spectral guardrail and the
corresponding images. This is an example of why the selection rule cannot be a
scalar-loss minimum.

**Judgment:** preserve explicit phase-addressed state, but learn or
coordinate-tie that state rather than using the fixed DCT basis in the immediate
model. `spatial-grid2` is the strongest existing Perceiver mechanism for doing
so. A future hierarchy may still include a conservative resolved stream, but it
must pass the same spatial-error and spectral gates as every learned route.

### 7. Native-grid transport is the demonstrated ceiling; coarse transport is the target

Jesse's native-grid learned inverse uses a learned pointwise state encoder,
pointwise physical-channel projection, and channel-masked physical-coordinate
resampling. Its same-grid reconstruction remained `0.001110` at one degree and
`0.001243` at half degree throughout forecast training. Its full
multiresolution aggregate lead losses were:

| Architecture | Lead 1 | Lead 2 | Lead 4 |
| --- | ---: | ---: | ---: |
| Native-grid frozen inverse | **0.03982** | **0.05408** | **0.06595** |
| Coarse moment/attention frozen inverse | 0.08275 | 0.10085 | 0.12806 |

The coarse model is approximately 1.9--2.1 times worse, but it preserves the
fixed 60 x 72 processor grid and beats persistence everywhere. Its inverse
already loses fine-scale power at depth zero, while teacher-latent error grows
with lead. Both representation and transition remain limiting.

**Judgment:** native-grid is the non-scalable reference that every compressed
architecture should approach, not the long-term processor topology. The next
model should report its fraction of the native-grid gap closed at equal routes
and leads.

### 8. Masking, geometry, forcing, and temporal ownership are architecture

Jesse's investigation provides high-confidence contracts that should not be
reopened in the first representation search:

- encode prognostic state separately from transient boundary forcing;
- supply exactly one aligned boundary state per physical processor step;
- keep geometry in a zero-initialized processor sidecar rather than adding it
  to reconstructive values;
- retain an unnormalized amplitude path;
- project latent features to physical channels before applying each channel's
  wet mask during spatial transport;
- encode once, remain latent across physical leads, and decode only requested
  outputs;
- freeze the learned inverse during transition training; and
- train the shared transition at true physical leads `{1, 2, 4}`.

The boundary path is causally active: zeroing it increased full native-grid
lead-1/2/4 error by 23.4%/61.1%/124.7%, and reversing it increased error by
9.8%/19.2%/93.7%. Soft reconstruction penalties did not prevent encoder and
decoder drift; freezing did.

## Optimization and width: what can reasonably be composed?

The first architecture search gives the pattern motivating a combined
prediction:

| Candidate | Epoch 1 | Epoch 3 | Epoch 6 | Epoch 12 |
| --- | ---: | ---: | ---: | ---: |
| No context, LR `4e-4`, transport 128 | 0.35649 | 0.27223 | 0.23446 | **0.19879** |
| Transport 256, LR `8e-4`, context 1 | **0.31965** | **0.26344** | **0.23185** | 0.20197 |
| Encoder latents 64, LR `8e-4` | 0.32194 | 0.26537 | 0.23666 | 0.20330 |

The wider model learned fastest, while the lower-rate, simpler-routing model
had the better late asymptote. It is reasonable to predict that no context,
transport width 256, and a lower late-stage rate will combine well. It is not a
measured result because the three-factor cell was never trained to epoch 12.

Newer evidence weakens the claim that width 256 is universally useful. Under
residual prediction at `4e-4`, `mean64-wide` and `mean64-direct` were identical
through epoch 2. Jesse's width screens also distinguish width types: increasing
external width from 128 to 380 improved inverse MSE from `0.011844` to
`0.009505`, while increasing latent feature dimension from 64 to 128 reached
`0.007937`. In a canonical native encoder, width 160 improved MSE from
`0.002857` to `0.002197` over width 128.

**Prediction:** use at least 160 channels for a structured coarse state and test
256 for the detail/transport path. Use a schedule that permits the fast early
behavior of `8e-4` but decays to or below `4e-4`, rather than assuming either
fixed rate is optimal. This schedule and width combination is a Level C
prediction and needs a small bridge experiment before multiplying it across the
architecture matrix.

## Single-scale decision for the one-degree run

### What “most powerful” means here

There are two legitimate one-degree choices, depending on the question the run
is intended to answer.

**For the lowest-risk, strongest one-degree predictor, use Jesse's native-grid
frozen-inverse architecture.** This is not merely an autoencoder result. Its
full-data forecast trained for 65 fragmentation-adjusted epochs (6,110 updates),
selected epoch 58, and reached normalized lead-1/2/4 MSE
`0.020519/0.034380/0.046903`. It beat matched persistence by large margins,
retained a frozen zero-depth reconstruction MSE of `0.000648`, and retained
high-wavenumber power ratios of `0.968/0.983/0.811/0.839/0.991` for
temperature/salinity/zonal velocity/meridional velocity/SSH. No Perceiver
candidate in this study has comparable full-data or rollout evidence. If the
only objective were one-degree skill, repeating that architecture would be the
evidence-based decision.

That full run has already been performed, however. Repeating it would validate
reproducibility but would add little architecture information. It also processes
the native input grid, so its cost scales with resolution and it does not test
the compression needed for LLC.

**For the strongest one-degree Perceiver experiment, use a spatially structured
Perceiver encoder and the already selected simple decoder.** The recommended
composition is:

| Component | One-degree choice | Reason |
| --- | --- | --- |
| Input contract | Prognostic state separated from aligned boundary forcing when the implementation permits | Jesse's causal and reconstruction evidence; concatenation remains a known current-code compromise |
| Encoder | `SpatialLatentGridEncoder`, query grid 2 x 2, group extent 6 x 10 degrees | Produces a 3 x 5 degree, 60 x 72 processor grid while retaining four coordinate-tied outputs per local group |
| Processor | Existing ConvNeXt U-Net processor | Keeps the experiment focused on transport and supplies cross-patch mixing |
| Temporal output | Same-grid physical residual for this screen | Approximately 75% short-budget loss reduction in two searches |
| Decoder routing | Direct output cross-attention, transport width 128, context 0 | Direct routing decisively beats Perceiver IO; wider transport did not help under the residual regime |
| Assembly | One-ring overlap-add | Large, controlled reduction in window discontinuities with negligible loss cost |
| Conditioning | Smooth processor conditioning enabled | Best matched residual comparison and best reviewed conditioned morphology |
| Refinement | Pixel refinement disabled | Unmatched training and visible divisions in both refined residual snapshots; no positive evidence yet |
| Optimizer | Initial LR `6e-4`, cosine decay over optimizer updates; retain the best checkpoint | Interpolates the separately observed fast/high-rate and better-late/lower-rate regimes and matches the validated full-run schedule class |
| Fixed basis | No DCT | Short-budget loss win is outweighed by 4.15x periodic power and visibly unphysical basis-aligned errors |

At one degree, `patch_extent: [6.0, 10.0]` with a 2 x 2 spatial query grid is
the resolution-scaled counterpart of the 2-degree `spatial-grid2` experiment:
each Perceiver group contains 6 x 10 input cells and emits four ordered tokens,
so the processor receives the historical 60 x 72 grid at 3 x 5 degree spacing.
It tests meaningful compression while avoiding the mean-pooled encoder's
one-vector-per-patch phase loss.

This is a Level C composition. `spatial-grid2` tied the mean-pooled baseline in
epoch-one loss, had better jump means, and showed more coherent error morphology,
but was not trained beyond that budget. Processor conditioning was established
with the mean-pooled encoder, not with `spatial-grid2`. The recommendation joins
compatible mechanisms; it is not presented as an already measured winner.

### Optional 2-degree decision screen before the EOD launch

The most informative small screen is not another broad successive-halving
search. Run these four arms for the same fixed three-epoch optimizer budget:

1. `mean64-direct`, conditioned;
2. `spatial-grid2`, unconditioned;
3. `spatial-grid2`, conditioned; and
4. `spatial-grid2`, conditioned, decoder transport width 256.

All four should use residual prediction, one-ring overlap, zero anonymous
context, no pixel refinement, fixed `4e-4`, the same seed, and the same batch and
update count. Arms 1--3 isolate representation and conditioning; arm 4 cheaply
tests whether width becomes useful once phase is retained. Do not prune before
three epochs.

Choose the one-degree architecture using validation loss *subject to* an
artifact veto. Inspect the same preregistered channels (`so_11`, `so_13`,
`so_14`, `thetao_2`, `thetao_7`, `thetao_8`) with common color limits, and reject
an arm with strong grid-periodic power, sharp window bands, or checkerboard
errors even if its scalar loss is smallest. If the screen does not complete in
time, launch arm 3. If arm 4 improves loss without worsening the image and
spectral diagnostics, promote its wider transport; otherwise retain width 128.

### Full-run evaluation contract

Train 70 epochs but select by validation checkpoint rather than the terminal
epoch. Evaluate both checkpoints if they differ materially. At minimum report:

- normalized validation loss and physical-unit variable/depth RMSE;
- 1-, 2-, 4-, 10-, and 20-step rollout skill against persistence;
- gradient/high-wavenumber power for scalar and velocity groups;
- window/patch jump and periodic-power diagnostics; and
- the preregistered snapshot panels above with common color limits.

The subsequent visualization run should use the selected checkpoint and include
error maps, spectra, and rollout curves. A one-step loss improvement without
stable rollout or with a newly periodic error mode is not a successful
single-scale architecture.

## Predicted best architecture

```text
prognostic state
  -> masked conservative mean/detail analysis
       -> fixed coarse tokens --------------------+
       -> coordinate-tied local detail tokens     |
              -> local SDPA Perceiver coupling ---+-> latent ReZero dynamics
                                                     + aligned boundary/geometry
  <- paired mean/detail synthesis <---------------+
  <- physically anchored continuous correction
  <- projection-before-channel-masked transport
  <- overlap/partition-of-unity assembly
  -> requested physical output grid
```

### Encoder: conservative resolved stream plus hierarchical detail stream

For each fixed physical region and prognostic channel:

1. Compute an exact, wet-mask- and area-aware resolved coefficient. This is the
   amplitude-preserving DC route and should never pass through LayerNorm.
2. Preserve a paired local mean/detail route for the anomaly, but do not assume
   that a fixed DCT is its implementation. The DCT experiment diagnoses the
   value of explicit phase and the danger of fixed patch bases simultaneously.
   Coordinate-tied learned tokens, a masked lifting scheme, or a learned
   multilevel basis are more plausible implementations because they can retain
   locality without imposing the observed periodic texture.
3. Keep detail coefficients as coordinate-tied tokens or explicit local spatial
   axes. Do not average them and do not pack them into channels unless the
   decoder has the exact inverse coordinate contract.
4. Use native SDPA Perceiver cross-attention inside local groups to compress and
   couple detail tokens. A small hierarchy should exchange information between
   local detail tokens and fixed coarse tokens. The Perceiver's role is learned
   set-to-structured transport, not unstructured mean pooling.
5. Produce two coupled states:
   - a fixed coarse grid for shared global dynamics; and
   - bounded local detail tokens at one or more levels for reconstruction and
     local dynamics.

This design is a hybrid of the strongest mechanisms rather than a vote between
“Perceiver” and “not Perceiver.” The fixed transform guarantees a phase route;
the Perceiver learns which detail combinations matter dynamically and provides
a scalable local-attention implementation for large inputs.

### Processor: shared coarse dynamics with local detail coupling

The coarse stream should use the existing spatial processor initially so that
the experiment isolates transport. The detail stream should receive local
updates conditioned on its coarse token and physically adjacent groups. Coarse
tokens should receive aggregated detail or flux summaries, since Jesse's
counterfactual experiment shows that equal coarse means can have different next
coarse tendencies.

Forecast training should use Jesse's validated contract:

- a separate, aligned boundary encoder;
- geometry as a sidecar;
- `z + alpha * P(...)` with zero-initialized per-channel `alpha`;
- frozen encoder/decoder after inverse pretraining; and
- true physical lead supervision at `{1, 2, 4}` with a small latent-teacher
  term as a controlled option.

### Decoder: paired synthesis base plus continuous correction

The decoder should:

1. Decode the resolved and detail streams through the paired conservative
   synthesis transform.
2. Project to physical channels before channel-specific masked spatial
   transport.
3. Add an optional zero-initialized continuous correction conditioned on
   physically anchored local tokens and a smoothly upsampled processor field.
4. Assemble any independently evaluated local supports with one-ring
   overlap-add or an equivalent partition of unity.
5. Avoid a second Perceiver latent bank and avoid anonymous context. If
   neighboring tokens are used, supply query-to-token physical offsets or a
   distance bias.

Smooth processor conditioning is the reasonable single-scale default: in the only
matched six-epoch residual comparison it improved loss by 1.83%, window-jump
p90 by 5.24%, and periodic-power means, and its reviewed error map was the cleanest
conditioned residual example. Full-resolution pixel refinement is not promoted:
its workers did not reach a matched budget and both reviewed refined residual
snapshots retained visible rectilinear structure.

### Near-term 2-degree instantiation

Before the hierarchy exists, the best risk-adjusted 2-degree spike is:

- coordinate-tied `spatial-grid2` encoding;
- residual prediction for the single-grid screen;
- transport width 128, with 256 tested only as a matched arm;
- smooth processor conditioning;
- one-ring overlap assembly;
- no anonymous decoder context; and
- no pixel refinement or fixed DCT basis.

Residual prediction, no anonymous context, overlap assembly, and conditioning
have direct support in adjacent controlled comparisons. The exact
`spatial-grid2` composition remains a prediction, but it is preferable to the
lower-loss DCT arm because it did not trigger the same spectral and visual
artifact vetoes.

### Implementation map

The current experiment branch already contains useful pieces of this design:

- [`DCTDetailEncoder`](../../src/samudra/models/modules/encoder.py) and
  [`DCTDetailDecoder`](../../src/samudra/models/modules/decoder.py) implement the
  paired 2-degree mean/detail spike;
- [`SpatialLatentGridEncoder`](../../src/samudra/models/modules/encoder.py)
  preserves coordinate-conditioned Perceiver outputs as spatial axes;
- [`PatchMomentEncoder`](../../src/samudra/models/modules/encoder.py) implements
  the learned resolved-plus-moment control;
- [`PerceiverEncoder`](../../src/samudra/models/modules/encoder.py) is the
  mean-pooled baseline;
- [`PerceiverDecoder`](../../src/samudra/models/modules/decoder.py) contains the
  direct-query, overlap-add, smooth-conditioning, and pixel-refinement paths;
  and
- [Samudra-owned Perceiver blocks](../../src/samudra/models/modules/perceiver.py)
  use PyTorch scaled dot-product attention.

The state-only native encoder, geometry sidecar, separate boundary encoder,
channel-masked coordinate resampler, and latent ReZero path are implemented on
Jesse's investigation branch. They should be integrated as an explicit temporal
and physical-contract layer after the next representation gate rather than
reimplemented indirectly inside another decoder.

## Hypothesis ledger

| Hypothesis | Evidence | Confidence | Architectural consequence |
| --- | --- | --- | --- |
| Full Perceiver IO decoder is harmful | Pure inverse localization and all 18 first-search arms | High | Remove second latent bank |
| Hard window assembly causes visible seams | Two controlled seam searches | High | Use overlap/partition-of-unity assembly |
| Anonymous decoder context is harmful | First search and seam controls | High at 2 degrees | Default to zero; reopen only with physical anchoring |
| Physical residual improves a same-grid screen | Two independent 2-degree searches | High for short same-grid training | Keep in screens; test route-consistent form for multiresolution |
| Latent ReZero is the better final temporal contract | Jesse's frozen-inverse proxy and full runs | High through one/half degree | Use residual latent transition and freeze inverse |
| Internal latent count is not the phase bottleneck | 64 vs 256 search and decoder latent sweeps | High | Stop sweeping unordered latent count |
| Explicit phase state is necessary | Native anchor, moment ablation, fronts, DCT result | High | Preserve resolved mean plus coordinate-tied details |
| DCT-14 should be promoted on loss | Two observed epochs, periodic power 4.15x baseline, visibly basis-aligned errors | Rejected | Retain only as mechanistic evidence that phase matters |
| Spatial-grid Perceiver is the strongest current compressed candidate | Better jump means, tied epoch-1 loss, more physical error morphology | Medium | Run the matched conditioned screen and a full single-scale test |
| Width 256 plus low late LR improves the structured model | Separate early-width and late-LR effects | Medium-low | Run a small factorial; do not assume additivity |
| Smooth processor conditioning helps | One matched residual pair | Medium | Carry one conditioned control forward |
| Full-resolution pixel refinement helps | No matched trained comparison; both early refined residual maps retain visible divisions | Low/unknown | Exclude from the one-degree run; revisit only with a controlled anti-aliasing design |
| Native-grid processing scales to LLC | Accuracy is strong; state and processor scale with all cells | Low | Use only as a diagnostic ceiling |

## Recommended next experiment

The next experiment should answer one central question:

> Can a structured, phase-preserving Perceiver transport close a meaningful
> fraction of the native-grid gap without reproducing patch-periodic artifacts?

### Gate 0: tactical composition check

Run the four-arm, fixed three-epoch 2-degree screen specified in the
single-scale section. This is optional for the EOD launch: it can select between
mean pooling and `spatial-grid2`, test whether conditioning composes with the
spatial encoder, and test width 256 once without reopening the whole search.
Use fixed optimizer steps and do not halve the candidates.

### Gate 1: full one-degree Perceiver forecast

Train the selected Perceiver composition for 70 epochs with checkpoint
selection, then run the evaluation and visualization contract above. Compare it
against the completed native-grid architecture and the historical Samudra v2
values, but describe those as external references because temporal and
representation contracts differ. This gate answers whether the compressed
Perceiver is a credible *single-scale* predictor; it does not establish shared
multiscale skill.

### Gate 2: representation tests for the hierarchy

For future candidates, measure held-out reconstruction, per-variable spectra,
coastal and wet-mask behavior, patch/window jumps, and equal-mean
different-front counterfactual distinguishability. Include:

1. native-grid learned inverse as the information ceiling;
2. `mean64-direct` as the phase-poor compressed control;
3. `spatial-grid2` as the current Perceiver lead;
4. a learned hierarchical mean/detail representation; and
5. a hybrid resolved stream plus coordinate-tied Perceiver detail grid.

DCT-14 may remain as a diagnostic control but is not promotion-eligible. Reject
any learned model that cannot retain the counterfactual or that gains MSE by
creating a large periodic mode.

Promotion should use a Pareto gate over:

- validation loss and physical-unit variable/depth RMSE;
- velocity and scalar gradient/spectral retention;
- window and patch jump mean, p90, and maximum;
- patch- and window-periodic power;
- runtime and peak memory; and
- identical 5-, 10-, and 20-step rollouts.

### Gate 3: temporal and multiresolution confirmation

For the best two representations:

1. pretrain a state-only inverse;
2. freeze it;
3. attach the boundary encoder, geometry sidecar, and ReZero transition;
4. train physical leads `{1, 2, 4}`;
5. compare absolute decoding with route-consistent transported-persistence plus
   increment decoding; and
6. add one-/half-degree routes before any quarter-degree allocation.

Use at least two new seeds. Report every route separately and compare against
the native-grid frozen-inverse reference, persistence, and the coarse
moment/attention model.

## Decision

The present mean-pooled patch Perceiver should not be scaled to finer data by
adding more unordered latents. The full Perceiver IO decoder should remain a
historical control. Jesse's native-grid inverse should remain the accuracy and
contract reference, but its processor grid is not the intended LLC-scale
solution.

For the immediate one-degree objective, the evidence-backed champion is Jesse's
native-grid frozen inverse. The recommended new full run is instead the
`spatial-grid2` Perceiver composition because it is the strongest experiment
that can improve the scalable architecture thesis. These statements are not in
conflict: one names what has won; the other names the most valuable candidate to
test next.

The most likely successful SamudraMulti architecture is a **hierarchical
resolved-plus-detail latent state**: exact conservative means, explicit
phase-addressed local details, coordinate-tied local Perceiver attention, a
fixed shared coarse processor, and a paired continuous decoder with smooth
cross-boundary assembly. It should inherit Jesse's state/boundary separation,
geometry sidecar, mask ordering, frozen inverse, and latent-autoregressive
training contract.

The strongest immediate experiment is not another generic hyperparameter
search. It is the small conditioned `spatial-grid2` composition check followed
by the 70-epoch one-degree train/eval/viz run. DCT-14 is no longer a finalist:
the agreement between its periodic-power metric and its visibly unphysical
errors is sufficient to reject it despite its lower two-epoch MSE.

### Evidence that would change this recommendation

This recommendation should be revised if any of the following occur:

- a trained `spatial-grid2` model fails to separate from mean pooling in loss,
  spectra, or front counterfactuals after a matched budget;
- the equal-mean front gate shows that a hierarchical detail bottleneck cannot
  retain dynamically relevant phase at half or quarter degree;
- a native-grid or sparse-native processor becomes computationally practical at
  LLC scale and preserves the native-grid accuracy advantage;
- channel-grouped moments close the velocity-spectrum deficit more efficiently
  than fixed details, supporting Jesse's proposed variable-grouped basis; or
- route-consistent physical residual prediction fails cross-resolution tests,
  leaving the absolute frozen decoder plus latent residual as the sole supported
  temporal contract.

## Reproduction queries

<details>

<summary>First-search learning-rate and architecture trajectory</summary>

```sql
WITH epochs AS (
    SELECT
        candidate,
        epoch,
        validation_loss,
        regexp_replace(candidate, '-lr[48]$', '') AS architecture_family,
        CASE
            WHEN candidate LIKE '%-lr4' THEN 0.0004
            ELSE 0.0008
        END AS learning_rate
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-v2-2deg-architecture--20260814T171003.874785Z/epochs.parquet'
    )
)
SELECT
    architecture_family,
    epoch,
    avg(validation_loss) FILTER (
        WHERE learning_rate = 0.0004
    ) AS loss_lr4,
    avg(validation_loss) FILTER (
        WHERE learning_rate = 0.0008
    ) AS loss_lr8
FROM epochs
WHERE epoch IN (1, 3, 6, 12)
GROUP BY architecture_family, epoch
ORDER BY epoch, least(loss_lr4, loss_lr8) NULLS LAST;
```

</details>

<details>

<summary>Original seam-search final-budget comparison</summary>

```sql
WITH latest AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY candidate ORDER BY epoch DESC
        ) AS recency
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-seam-removal-2deg--20260818T210753.565365Z/epochs.parquet'
    )
)
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/zos" AS window_jump_zos,
    "val/seam/window_jump_ratio/channel_mean" AS window_jump_mean,
    "val/seam/patch_jump_ratio/channel_mean" AS patch_jump_mean,
    "progress/optimizer_steps" AS optimizer_steps
FROM latest
WHERE recency = 1
ORDER BY validation_loss;
```

</details>

<details>

<summary>Matched residual/absolute hard-versus-overlap comparison</summary>

```sql
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/channel_mean" AS window_jump_mean,
    "val/seam/window_jump_ratio/channel_p90" AS window_jump_p90,
    "val/seam/window_jump_ratio/channel_max" AS window_jump_max,
    "val/seam/patch_jump_ratio/channel_mean" AS patch_jump_mean,
    "val/seam/patch_periodic_power_ratio/channel_mean" AS patch_power_mean
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-residual-assembly-2deg--20260820T005445.253301Z/epochs.parquet'
)
WHERE
    (candidate IN ('hard-absolute', 'blend1-absolute') AND epoch = 2)
    OR
    (candidate IN ('hard-residual', 'blend1-residual') AND epoch = 3)
ORDER BY candidate;
```

</details>

<details>

<summary>Pixel de-aliasing latest-observed comparison</summary>

```sql
WITH latest AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY candidate ORDER BY epoch DESC
        ) AS recency
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-pixel-dealiasing-2deg--20260820T005523.824824Z/epochs.parquet'
    )
)
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/channel_mean" AS window_jump_mean,
    "val/seam/window_jump_ratio/channel_p90" AS window_jump_p90,
    "val/seam/window_jump_ratio/channel_max" AS window_jump_max,
    "val/seam/patch_jump_ratio/channel_mean" AS patch_jump_mean,
    "val/seam/window_periodic_power_ratio/channel_mean" AS window_power_mean,
    "val/seam/patch_periodic_power_ratio/channel_mean" AS patch_power_mean,
    "progress/optimizer_steps" AS optimizer_steps
FROM latest
WHERE recency = 1
ORDER BY validation_loss;
```

</details>

<details>

<summary>Latest structured-transport loss and artifact metrics</summary>

```sql
WITH latest AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY candidate ORDER BY epoch DESC
        ) AS recency
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-structured-transport-2deg--20260820T035545.388436Z/epochs.parquet'
    )
)
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/channel_mean" AS window_jump_mean,
    "val/seam/patch_jump_ratio/channel_mean" AS patch_jump_mean,
    "val/seam/window_periodic_power_ratio/channel_mean" AS window_power_mean,
    "val/seam/patch_periodic_power_ratio/channel_mean" AS patch_power_mean,
    "progress/optimizer_steps" AS optimizer_steps
FROM latest
WHERE recency = 1
ORDER BY validation_loss;
```

</details>

<details>

<summary>Structured-transport matched epoch-2 comparison</summary>

```sql
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/channel_mean" AS window_jump_mean,
    "val/seam/window_jump_ratio/channel_p90" AS window_jump_p90,
    "val/seam/window_jump_ratio/channel_max" AS window_jump_max,
    "val/seam/patch_jump_ratio/channel_mean" AS patch_jump_mean,
    "val/seam/patch_jump_ratio/channel_p90" AS patch_jump_p90,
    "val/seam/patch_periodic_power_ratio/channel_mean" AS patch_power_mean,
    epoch_train_seconds / 60 AS train_minutes
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-structured-transport-2deg--20260820T035545.388436Z/epochs.parquet'
)
WHERE epoch = 2
ORDER BY validation_loss;
```

</details>

<details>

<summary>Worker eligibility and failure-state audit</summary>

Replace `<SEARCH_RUN>` with any of the four 2-degree search run identifiers
listed above. This query deliberately keeps incomplete workers visible.

```sql
SELECT
    candidate,
    rung,
    epochs,
    eligible,
    worker_stage,
    worker_optimizer_steps,
    coalesce(worker_error, error) AS error,
    scheduler_stderr_tail
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/<SEARCH_RUN>/results.parquet'
)
ORDER BY rung, eligible DESC, candidate;
```

</details>
