<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# The Samudra 2 revision: what these experiments can answer

This note goes through the reviewer and editor questions in the revision deck
("Meeting 11.3 — Samudra 2 revision", 40 slides). For each one it asks three
things: what our experiments already say, what a next step could answer, and
what we recommend. It ends with a prioritized list of actions.

**Read this first.** The compression-baseline results so far are on the **2°**
grid. Samudra 2 in the paper is **1°**. No Samudra checkpoint has been scored.
So most items below are *proposed* contributions. The status column says which
are already backed by a result in this folder. Definitions are in the
[overview](../README.md) and [background](background.md).

## Summary

| # | Question (deck slides) | What we can offer | Status | Cost |
|---|---|---|---|---|
| 1 | Compare against simpler and recent baselines (1–5) | A one-step statistical baseline per channel; at 1° it could sit next to Samudra 2's one-step forecasts | 2° result; 1° proposed | Low to medium |
| 2 | What is new about dynamic weighting? (16–19) | Dynamic loss ≈ maximum likelihood with a per-channel spread, regularized | Algebra checked against code | None |
| 3 | Losses in normalized time-delta space (10–15) | Our score uses exactly that normalization; proper-score comparison of the target variants | Normalization: result. Comparison: proposed | Medium |
| 4 | How much skill comes from the prescribed forcing? (20–23) | Bits of information the forcing carries about the next state, per variable, depth and region | Proposed; next experiment | Low |
| 5 | Deep equatorial variance bands: physical or artificial? (30–37) | Input-group information ablation from OM4 alone; excess-bits maps for Samudra | Proposed | Low (ablation), medium (maps) |
| 6 | Trend/anomaly decomposition (10) | Deep changes are persistent (AR(1) skill), consistent with a predictable drift | Result; interpretation tentative | None |
| 7 | Changed-forcing generalization (6–9) | Possibly a "sticking" diagnostic for the subsurface cold bias | Proposed, speculative | Medium |
| 8 | Per-channel loss curves, overfitting (24–29) | Per-channel bits on validation as a comparable monitor | Proposed, minor | Low |
| 9 | Multiple starts, drift (38–40) | Baseline spread from refits on several windows | Proposed, minor | Low |

---

## 1. Baselines: "important baselines, including a physics-based one and recent ML ones"

**Slides 1–5.** The deck answers with Neptune: Samudra 2 has lower
depth-averaged RMSE over 2015–2022 rollouts.

**What we have.**
- On 2° OM4, the `mix` baseline beats persistence on every channel. It cuts
  five-day RMSE by 5–18% depending on variable
  ([v2](v2_om4_twodeg.md)).
- It is a transparent, CPU-cheap, purely statistical one-step forecaster.

**What it isn't.** It is not physics-based, so it doesn't meet that part of the
request. Its numbers are one-step, while the deck's table is rollouts.

**What a next step could add.** Slide 31 shows the group already has Samudra 2
forecasts with the true history refreshed every step (598 outputs, 2014–2022).
Those are one-step forecasts. Fitting the ladder at 1° on the same window would
give a per-channel table:

| | Persistence | Ladder (`mix`) | Samudra 2 |
|---|---|---|---|
| One-step RMSE | | | |
| One-step bits | | | via EMOS |

That shows *how much* Samudra 2 adds over cheap statistics, and where.

**Recommendation.**
- **Optional for the revision.** It strengthens the baseline story but doesn't
  replace a physics baseline.
- **Worth doing for us either way.** It's the "beat `mix`" bar for one-step
  quality.

## 2. "Clarify what is new about the proposed dynamic weighting"

**Reviewer 2 major 1 and reviewer 3 detailed 19 (slides 16–19).** The deck
shows that fixed final weights nearly match Samudra 2, and that normalized-delta
and Hu weighting behave differently.

**What we have.** A derivation, checked against `DynamicLoss`, in
[losses_from_bits.md](losses_from_bits.md), section 2, and
[background](background.md), section 17:
- A Gaussian likelihood with a learned spread per channel has, at its optimum,
  gradients equal to MSE weighted by 1 / (each channel's current MSE).
- `DynamicLoss` uses exactly those weights, plus a cap of 20 on the ratio between
  weights and a 25-step moving average.

**What it answers.**
- The dynamic loss is maximum likelihood with per-channel uncertainty
  (Kendall et al. 2018), regularized by the cap and smoothing.
- The novelty is not the objective. It is the regularization that stops
  near-perfect channels from taking over, and its application to ocean
  emulation.
- It also *predicts* slide 17's result: at convergence the weights settle where
  the likelihood puts them, so freezing the final weights should do about as
  well.

**Recommendation.** **Do this for the revision.** It costs no compute, gives
reviewer 2 a principled answer, and frames reviewer 3's control as a confirmed
prediction. Have one person check the reading against the training code and
the weight trajectories before it goes in. Related doc fix: #982.

## 3. "Computing loss in normalized space, using normalized time-delta targets"

**Reviewer 3 major 3 (slides 10–15).** The group trained normalized-delta and
background-anomaly variants.

**What we have.**
- v2 bins every variable at ε × σ_Δ, the spread of its five-day changes pooled
  over depth. That's the Keisler (2022) and GraphCast-style normalization the
  reviewer refers to. We have the 2° spreads (`results.json`, `sigma_delta`).
- The change from v1 to v2 is itself a small demonstration of why it matters
  for a *score*. Scaling by each channel's value range made deep bins mismatched
  with the dynamics. Scaling by time-difference spread fixed that
  ([v2](v2_om4_twodeg.md), finding 2).

**What a next step could add.**
- Score the three target variants (Samudra 2, normalized delta, background
  anomaly) on one-step forecasts in bits and RMSE, through EMOS.
- Bits are a proper score that rewards calibrated confidence and adds up across
  variables, so they compare the variants in the same normalized space the
  reviewer names, with one number per channel.

**Recommendation.** Useful if the one-step outputs of the variants are easy to
get. Otherwise it's a second-tier item. The existing rollout comparisons already
answer the reviewer.

## 4. "Quantify the contribution of the prescribed forcing"

**Reviewer 2 minor 1 and reviewer 3 major 4 (slides 6, 20–23).** The deck
answers with rollout ablations: wind-only, heat-only, seasonal and zero forcing.

**What a next step would add.**
- Fit the ladder with and without `tauuo`, `tauvo` and `hfds`. The bits saved
  by adding forcing estimate **how much information the forcing carries about
  the next ocean state**, independently of any emulator.
- It breaks down per variable, depth and region, and by forcing component
  (wind vs heat).
- The rollout ablations measure how much *Samudra* relies on the forcing over
  years. This measures how much the forcing *could* tell any forecaster at one
  step. Together they separate "the information is there" from "the model uses
  it".

It also tests our leading explanation for the large surface joint-coding gap.
If forcing removes most of it, the spatially correlated errors were unseen
weather.

**Recommendation.** **Do this next.** It's about an hour on the DTN at 2°, it's
already the next planned experiment, and it gives the revision a model-free
complement to slides 21–23. Repeat at 1° if it's cited in the paper.

## 5. Deep equatorial variance bands: physical or artificial?

**Reviewer 2 major 6 (slides 30–37).** The deck finds:
- the deep-salinity excess persists at one step even with the true history
  refreshed (slide 31);
- restricting inputs to deep T/S reduces the equatorial excess (slides 32–37);
- other ocean-state inputs are implicated.

**What a next step could add:**

- **An input-group information ablation from OM4 alone (cheap).** Refit the
  ladder for deep T/S channels while dropping one input group at a time: other
  variables at the same level, levels above and below, velocities. The bits each
  group carries say whether OM4's deep T/S changes actually depend on those
  inputs at one step.
  - If the other variables carry about 0 bits for deep T/S, the model-free
    evidence agrees with slide 32: those inputs give a network room to fit
    spurious structure, not signal.
  - If they carry real information, restricting them has a cost the deck
    should mention.
- **Maps of excess bits for Samudra (needs its one-step outputs).** Train a
  coder on OM4 and score Samudra's refreshed one-step deep forecasts.
  - If the equatorial band costs more bits than it does for OM4 truth, the
    structure is unlike anything in the true dynamics. That is direct evidence
    it's an artifact.
  - The same maps would show whether the excess is noise (too expensive) or
    something more coherent.

**Recommendation.** Run the **input-group ablation** at 2° now: it's cheap and
speaks directly to the deck's strongest new analysis. The Samudra maps follow
once one-step outputs are in hand.

## 6. "Decompose the prediction into trend and anomaly"

**Reviewer 3 detailed 16 (slide 10).**

**What we have.**
- The true deep tracer changes are positively persistent. AR(1) alone has RMSE
  skill against persistence of 0.176 for `thetao_18` and 0.214 for `so_18`
  (v2; v1 gave 0.205 and 0.220).
- Part of the deep five-day change is a slow, predictable drift. That is
  consistent with the background-anomaly target improving subsurface means
  (slides 12–13).

**Recommendation.** At most a supporting sentence. The link is suggestive, not
a test.

## 7. Generalization under changed forcing

**Reviewer 3 major 4 (slides 6–9).** The CM4 runs show a rapid subsurface cold
drift in the first year, then a flat bias (slides 7–8).

**What a next step could add (speculative).** That shape matches the
"attractor" failure mode: the state moves fast to a wrong mean, then stops
changing. The proposed sticking detector would score each rollout step with a
coder trained on CM4 truth. Steps that are *too cheap* to encode (changes
collapsed) while the mean error grows would characterize the bias as a
dynamical failure, not just an offset.

**Recommendation.** Not for this revision. It needs the detector to be
validated first (planned on degraded truth). Note it as future diagnostic work.

## 8. Per-channel loss curves and overfitting

**Reviewer 1 substantial 4–5 (slides 24–29).** The deck shows per-channel
training and validation MSE, and a mid-training rise in deep validation loss.

**What a next step could add.** Bits per value on validation, per channel, is
comparable across channels without a normalization choice. It would make the
heatmap on slide 25 interpretable on one scale. That helps presentation, not the
reviewer's question.

**Recommendation.** Skip for the revision.

## 9. Multiple starts and drift

**Reviewer 1 substantial 3 and 6 (slides 38–40).**

**What a next step could add.** The ladder can be refit on several training
windows in minutes, which gives an error bar on the baseline itself. Drift is
better served by the existing OHC and SSH diagnostics.

**Recommendation.** Skip for the revision.

---

## What to do next

### For the revision, in priority order

1. **Dynamic loss as likelihood (item 2).** No compute. Draft a paragraph for
   the rebuttal; one person checks it against the training code and logged
   weights.
2. **Forcing information in bits (item 4).** About an hour at 2°, then 1° if
   cited. Also the planned next experiment.
3. **Input-group ablation for deep T/S (item 5).** Cheap at 2°; supports
   slides 32–37 with model-free evidence.
4. **One-step baseline table at 1° (item 1).** Needs the 1° ladder and the
   refreshed one-step outputs from slide 31. Optional.

### For the research agenda beyond the revision

- **Validate the failure-mode detectors on degraded truth** (noise, sawtooth,
  sticking), then run them on Samudra rollouts. See
  [research_agenda.md](research_agenda.md).
- **Build the causal spatial coder**, to measure the joint-coding gap directly.
- **Losses** ([losses_from_bits.md](losses_from_bits.md), #981): compare the
  current loss, a per-cell spread head and a tolerance-discretized NLL on the new
  2° model. Measure Samudra's own joint-coding gap before investing in
  field-level losses.
- **Scale everything to 1°** once the 2° pieces work.
