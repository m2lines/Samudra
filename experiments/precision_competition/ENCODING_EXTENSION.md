<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Continuous refinement and scale-weighted categorical losses

This extension adds six variants to the scalar-copy experiment in `ENCODING.md`.
Its 36 bundles use the same two learning rates (1e-4, 1e-5), three seeds
(11,22,33), batch size 64, fixed-point input/target, shared transformer,
auxiliary target/noise, 12,000-update solo training, qualification gates,
3,000-update paired continuations, and 4/6-bit milestone probes. The original
66 bundles remain unchanged and use their original immutable source archive.
These are experiment results, not intended to merge.

The question is whether repeated continuous correction and loss scaling explain
benefits attributed to categorical prediction. No benefit is assumed.

| Added variant | Correction and stage objective |
| --- | --- |
| cont_shrink_mse | Continuous correction; ordinary scalar-output MSE; radius shrinks eightfold |
| cont_shrink_normalized | Same predictions; normalized-residual MSE; radius shrinks eightfold |
| cont_fixed_mse | Continuous correction; ordinary scalar-output MSE; radius stays eight |
| ar_overlap_scaled | Existing hard overlapping CE, weighted by squared relative radius |
| ar_twohot_hard_scaled | Existing two-hot overlapping CE with hard decoding, weighted likewise |
| ar_twohot_soft_scaled | Existing two-hot overlapping CE with soft decoding, weighted likewise |

## Continuous decoder

Run the transformer once. The shared two-layer decoder takes its main query
representation, center/8, log2(radius/8)/16, and the refinement-level indicator,
exactly as for categorical refinement. Replace its 16 logits with one scalar
passed through tanh. Four successive predictions follow:

```
c0 = 0; r0 = 8
qk = tanh(decoder(h, stop_gradient(ck), rk, k))
pk = ck + rk * qk
ck+1 = stop_gradient(pk)
rk+1 = rk / 8        # shrinking arms
rk+1 = rk            # fixed-range control
```

The scalar correction qk lies in [-1,1]. The returned value is p3. The ordinary
stage loss is `(pk - target)^2`, measured in normalized scalar coordinates.
The normalized stage loss is `(qk - (target - ck)/rk)^2`. The current center and
radius are detached; therefore the ordinary loss equals rk squared times the
normalized loss at identical model parameters and states. Average the four
stage losses. Do not clip the residual target in either loss, even when the
current bounded correction cannot reach it; count those range misses.

For the first 1,000 solo updates only, replace the feedback center by
`ck + rk*clip((target-ck)/rk,-1,1)` after computing the stage loss from the
model's actual prediction. Subsequently use model-generated centers. Evaluation,
continuations and probes always use model-generated centers. Later losses do
not backpropagate through earlier center choices, but do train the common
representation and shared decoder. The fixed-range control retains the level
indicator and otherwise has the same architecture, training schedule, bounded
correction and feedback.

With fixed radius eight, normalized-residual MSE is ordinary output MSE divided
by 64 at every stage. The existing fixed initial gradient calibration cancels
that global factor, so a second fixed-range normalized variant would duplicate
the mathematical objective after calibration. It is omitted. This equivalence
is for the objectives and calibration, not a claim of bitwise floating-point
identity between independently implemented losses.

## Scale-weighted categorical loss

The existing four categorical losses have equal weights. In the added variants,
use `mean_k(wk * CEk)`, where `wk=(rk/8)^2`. The overlapping decoder has radii
8,1,1/8,1/64, so its weights are:

```
1, 1/64, 1/4096, 1/262144
```

The geometry, targets, logits, hard/soft decoding, teacher-forcing schedule,
parameter initialization and detach boundaries remain identical to the
corresponding unweighted categorical arm. Only the stage-loss weights change.
Weights are fixed, not learned. Radius-squared weighting reflects the physical
MSE penalty associated with a fixed relative displacement; CE itself is not
numerically equivalent to MSE. These weights deliberately make fine-stage
errors much cheaper. This tests sensitivity to fine-stage loss emphasis. Aggregate main/auxiliary
gradient diagnostics do not alone establish which stage caused an effect.

As in the original wave, calibrate the overall main-loss coefficient once at
initialization using target-derived prefixes and the same fixed calibration
examples, seed-matched reference head and initial backbone as the original
wave, then freeze it. This removes an initial global gradient-scale change,
while preserving relative stage weights. Log the coefficient and subsequent
main/auxiliary gradient diagnostics. No learned task weighting is introduced.

## Verification, execution and interpretation

Tests check target-independent inference and nonzero shared gradients for all
six variants; the exact radius schedule; ordinary/normalized loss identities;
and unchanged predictions with the categorical weighting enabled. A separate
source comparison confirms that all eleven original variants preserve exactly
their predictions and losses under both model-generated and target-derived
prefixes. Existing jobs retain the original source snapshot regardless.

`encoding_extension_smoke.json` runs six GPU smoke tasks with the original
20-solo/5-per-continuation engineering protocol; it bypasses scientific gates.
After all pass, `encoding_extension_manifest.json` launches 36 scientific
bundles with concurrency at most 12, at most one hour per slot and at most one
infrastructure retry per failed bundle. Reuse the original checkpoint/requeue
mechanism and require L40S GPUs. Preserve a separate source archive, output root,
manifest hashes, job IDs and W&B group. A dedicated quiet monitor owns this
extension through verified curves, figures, tables and published results in
experiment PR #967. Account for actual GPU time, including smoke and failures.

Compare continuous shrinking versus fixed range; ordinary versus normalized
continuous MSE; and each weighted categorical arm against its original
unweighted counterpart. Report solo acquisition, qualification failures,
per-stage RMSE, range misses, absolute accuracy, paired bits lost relative to
main-only continuation, auxiliary accuracy and raw/EMA results. Inspect source
and initialization matching across the two snapshots before combining tables.
Continuous heads have fewer output parameters than categorical heads; report
counts. Continuous arms report four stages of numerical errors and range
misses; categorical accuracy, margins, entropy and incorrect-first-choice
statistics do not apply to them. Continuous versus categorical comparisons also
change loss shape, treatment of unreachable targets (unclipped residual losses
versus endpoint categorical labels), and hard-arm warm-up centers. They compare
complete decoder/objective packages, not an isolated effect of representation.
Continuous bounded corrections impose no discrete quantization floor,
but finite precision and model errors still limit achieved accuracy.

A scale-weighted decoder can look stable because it never learned fine
precision. Compare absolute accuracy and acquisition with its interference
penalty. A fixed-range decoder may make large corrections at late stages; its
per-stage errors and range misses distinguish correction from degradation.
Report both observations rather than inferring a benefit from the architecture.
