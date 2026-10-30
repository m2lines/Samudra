<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Why velocity learning differs between MSE and CRPS

The investigation found a loss-dependent failure to learn strong currents at
these training budgets, and demonstrated a partial remedy. A velocity-only,
variance-corrected ensemble-mean MSE auxiliary improves mean velocity error,
pointwise CRPS and coverage together in a short controlled continuation. It
preserves the ordinary mean-MSE auxiliary's reconstruction gain without its
coverage loss. It does not recover realistic strong-current amplitudes or
improve every field.

All four 512-update continuations and their evaluations are complete. The
investigation used 6.2053 GPU-hours, including diagnostics, intermediate
evaluations and failed startup attempts. CPU toy experiments are additional.

## What explains the original gap?

**Strong currents are suppressed in the members themselves.** Across three
exactly aligned held-out OM4 dates, both deterministic MAE and stochastic fair
CRPS produce weak currents. For reference speeds of 0.3–1 m/s, the stochastic
CRPS ensemble mean retains only 1.2% of reference squared velocity, versus
35.4% for the stochastic per-member-MSE model. Individual-member maps and
spectra also lack current structure, so ensemble averaging alone cannot explain
the weakness. These are initialization reconstructions, not future forecasts.
The [multi-date report](velocity-multidate-diagnostics.md) includes maps, depth
breakdowns, spectra, reference-speed bins and exact metric definitions.

**MSE gives large errors much greater training pressure.** At identical model
outputs on fixed training examples, MSE produces about ten times larger output
gradients with respect to normalized outputs, at the configured coefficients,
than CRPS in the strongest-current 1%, while giving slightly smaller
gradients in the weakest half. OM4 forecast gradients have positive cosines
of 0.28–0.51 with velocity reconstruction gradients. These local probes provide
little evidence for consistent forecast-gradient opposition as the main cause.
They do not reconstruct historical Adam updates. See the
[gradient audit](velocity-gradient-audit.md).

**Better CRPS need not mean better current reconstruction.** In the original
stochastic CRPS run, going from 2k to 4k barely improves mean velocity RMSE.
Member absolute error worsens, but the subtracted spread term grows enough to
improve CRPS. Spread is a legitimate part of the score; this particular change
does not demonstrate better current patterns.

**The mean-MSE auxiliary introduces its own spread penalty.** For M independent, identically distributed
conditional members, the expected squared error of their sample mean contains
predicted variance divided by M. This motivated a corrected auxiliary:

\[
L_{\mathrm{aux}}=(\bar X-y)^2-\frac{s^2}{M},
\]

where X̄ is the member mean, y is the reference, and s² is sample variance with denominator M−1. The correction removes that
predicted-variance penalty in expectation. It remains differentiable and
unclamped; individual sampled losses can be negative and noisy. The
[analytic and synthetic checks](velocity-diagnostics/toy-ustat.md) verify the
identity and show improved coverage in a three-seed Gaussian toy. The earlier
[toy learning curves](velocity-diagnostics/toy-mechanisms.md) also demonstrate
that absolute-error objectives can learn narrow currents more slowly, then
recover with longer training. Neither toy alone establishes the ocean cause.

## The controlled ocean test

Four arms start from the same stochastic-initializer/deterministic-processor
model's raw 2k checkpoint and Adam state. Each adds 512 mixed updates: 256 OM4
and 256 observation updates, effective batch eight, learning rate 1e-4 and no
cooldown. EMA resets identically. Data order and stochastic draws match.
Training uses four members. Auxiliary coefficients apply to the existing
per-channel normalized velocities, retaining the 0.1 initialization coefficient
and 77-channel averaging denominator.
Only the OM4 initialization velocity loss differs; every observation,
forecast and non-velocity loss is unchanged.

The arms are unchanged CRPS, velocity CRPS weighted by 2.2, a unit ordinary
ensemble-mean velocity MSE auxiliary, and the same auxiliary with its variance
correction. The 2.2 multiplier approximately matches the ordinary auxiliary's
combined velocity-gradient norm on three training probes. It does not match
the full objective gradient or the Adam update. It tests a general increase in
velocity supervision against a change in loss shape and member coupling.

The following table uses final EMA weights and eight evaluation members.
RMSE and CRPS are in m/s. MSE averages equally across the three dates and 19
depths before its square root is taken; CRPS is averaged directly. Both use
cosine-latitude weights on wet cells. Coverage is the fraction of
reference values inside the eight-member min/max range; ideal continuous
exchangeability would give 7/9, about 78%.

| Arm | u mean RMSE | v mean RMSE | u CRPS | v CRPS | u/v coverage |
|---|---:|---:|---:|---:|---:|
| Continued CRPS control | 0.07206 | 0.05280 | 0.02633 | 0.01979 | 53.7% / 58.1% |
| Velocity CRPS weighted ×2.2 | 0.07139 | 0.05257 | 0.02534 | 0.01930 | 57.4% / 60.9% |
| Ordinary mean-MSE auxiliary | 0.06879 | 0.05193 | 0.02629 | 0.02028 | 46.5% / 50.6% |
| Variance-corrected auxiliary | 0.06883 | 0.05190 | 0.02516 | 0.01937 | 55.0% / 59.5% |

The corrected auxiliary improves u/v mean RMSE by 4.47%/1.71% and CRPS by
4.41%/2.12% versus control. It gives essentially the same mean accuracy as the
ordinary auxiliary, but restores 8.5/8.9 percentage points of coverage relative
to that arm. All three dates show the same signs for corrected-versus-control
mean RMSE, CRPS and coverage; raw-weight results agree in direction. This is
exploratory evidence from one training seed, not a significance test.

The weight-only arm mainly improves CRPS and coverage, with less than 1%
mean-RMSE improvement. Thus increasing velocity's overall loss weight does not
reproduce the auxiliary's reconstruction effect in this pilot. The
[full continuation report](velocity-continuation-results.md) retains all four
arms, raw/EMA curves, per-date contrasts, maps and spectra.

## What remains unresolved?

Strong-current squared-velocity retention rises from 1.46% in the continued
control to 3.06% with the corrected auxiliary. That is an improvement from a
very low starting point. The earlier MSE model retains about 35%, although it
is a different from-scratch training comparison, not a matched continuation.
The new models still fail to reconstruct much of the energetic current
structure. Coverage also remains well below the ideal reference.

The corrected auxiliary has collateral initialization costs: temperature RMSE
rises from 0.6065 to 0.6202 °C, salinity from 0.4746 to 0.4807 in the dataset's
units, and SSH from 0.02354 to 0.02649 m. Its unchanged nine-origin observation
forecast blend is 0.89870 versus 0.89759 for control, about 0.12% worse. All four
EMA forecast blends are within 0.13%; this does not erase the initialization
tradeoffs. The [forecast metric specification](loss-ablation-partial.md#observation-forecast-validation)
and full component tables are retained.

My interpretation is that limited strong-current learning under this objective
and training budget is a better-supported explanation than ensemble averaging
or consistent forecast-gradient interference. The controlled loss changes
support that interpretation, but change both error emphasis and member coupling.
The variance correction specifically improves the ordinary auxiliary's
probabilistic tradeoff. This is a useful candidate for further testing, not an
overall winning configuration or a complete velocity solution.

The next comparison worth considering is a longer matched control/corrected-
auxiliary run with the same diagnostics and explicit checks of SSH/T/S costs.
No such longer run was launched. Three validation dates and one training seed
cannot establish broad seasonal performance or calibration; dates used to
choose these interventions are not an independent test set. Six other candidate
histories were excluded for exact timestamp mismatch without interpolation.

## Reproducibility

The linked reports retain checkpoint/input fingerprints, job receipts, raw
scores and figure scripts. The [original matched loss pilots](loss-ablation-partial.md)
provide the starting evidence. [Continuation results and provenance](results/velocity-continuations/)
retain the final four-arm comparison; [gradient evidence](results/velocity-diagnostics/gradients/)
and [multi-date diagnostics](results/velocity-investigation/) retain the
mechanism checks. All allocated GPU time, including corrected startup errors,
is included in the cost above. No training jobs remain for this investigation.
