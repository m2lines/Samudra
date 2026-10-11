<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Early-data reproduction at the original 4k budget

**The 4k advantage repeats in these same-seed executions.** Both fresh early-data
runs completed 4,000 updates and all 96 held-out origins. Relative to the matched
U-global-repro4k control, early-fine improves validation/test RMSE blend by
**2.7%/2.6%**, and early-fine+latent by **3.0%/3.7%**. Both selected their final
checkpoint, so this result also holds at the fixed 4k endpoint.

This supports the earlier-data recipes at this budget. It does not establish
that latent channels or fine resolution alone cause the gain, or quantify
variation across independent seeds. The [6k comparison](early-rmse-6k-2026-10-10.md)
uses a stretched mixture as well as a larger budget; its intermediate update
4,000 is not this matched control.

Contents: [Results](#results) · [Curves](#curves) · [Experiment](#experiment) ·
[Verification](#verification).

## Results

All values use the same five-component, full-field **RMSE-only** validation
blend and frozen denominators; lower is better. Historical runs were rescored
from saved physical components, removing spectra from this comparison. Their
data/grid/statistics hashes and observation-support definitions match.

| Model | Execution | Best validation | Best update | Final validation | Final-five mean |
|---|---|---:|---:|---:|---:|
| U-global | Original 4k, Beta | 0.8815 | 4,000 | 0.8815 | 0.9088 |
| U-global-repro4k | Matched 4k control, Torch | 0.8861 | 4,000 | 0.8861 | 0.9306 |
| U-multitask-early-fine | Original 4k | 0.8704 | 4,000 | 0.8704 | 0.8889 |
| U-multitask-early-fine | New 4k reproduction | **0.8624** | 4,000 | **0.8624** | 0.9037 |
| U-multitask-early-fine-latent | Original 4k | 0.8559 | 4,000 | 0.8559 | 0.9206 |
| U-multitask-early-fine-latent | New 4k reproduction | **0.8595** | 4,000 | **0.8595** | 0.9053 |

The two reproduced early-data models also beat the original U-global's RMSE
minimum. The original fine+latent advantage was 2.9% on RMSE alone, so it was
not solely a spectral-score effect. The new latent advantage over the new
non-latent fine model is only **0.3% on validation**; that small difference does
not establish a reliable benefit from the extra channels.

The final-five mean uses scheduled validations at budget−400, −300, −200,
−100 and the endpoint. This descriptive diagnostic was declared before the
new 4k results were available; it does not select checkpoints. These correlated
points are not five independent trials. Their variation across historical and
reproduced runs remains substantial, including for U-global itself.

For context, the 6k early-fine/fine+latent final-five means are **0.8524/0.8502**,
versus **0.8830/0.8808** for the two U-global seeds. Thus the close 6k minima hide
a late-training difference. The combined evidence supports a budget- and
checkpoint-dependent comparison, rather than dismissing the original 4k gain
as noise. It does not show that 4k is optimal: the 6k models reach lower errors.

### Held-out comparison

These are the validation-selected checkpoints, evaluated on the same 96 monthly
test origins. Test data did not select models or weights. The test blend uses
common held-out climatology denominators, separately from validation.

| Model, matched 4k execution | Test RMSE blend | Own initialized persistence | Change vs U-global-repro4k |
|---|---:|---:|---:|
| U-global-repro4k | 0.9082 | 0.9829 | — |
| U-multitask-early-fine | 0.8849 | 0.9824 | −2.6% |
| U-multitask-early-fine-latent | 0.8746 | 0.9718 | −3.7% |

| Model | SST RMSE (°C) | Geostrophic velocity RMSE (m/s) | EKE RMSE (m²/s²) | OHC 0–700 m RMSE (10⁸ J/m²) | OHC 700–2000 m RMSE (10⁸ J/m²) |
|---|---:|---:|---:|---:|---:|
| U-global-repro4k | 0.6478 | 0.11491 | 0.025254 | 8.073 | 4.358 |
| U-multitask-early-fine | 0.6276 | 0.11503 | 0.025242 | 7.453 | 4.291 |
| U-multitask-early-fine-latent | 0.6396 | 0.11479 | 0.025218 | 7.244 | 4.153 |

The largest gains are in upper-ocean heat content: **7.7%/10.3%** for fine and
fine+latent. SST improves **3.1%/1.3%**; fine+latent also improves deep OHC by
**4.7%**. Velocity/EKE differences are tiny, and both still have worse EKE RMSE
than their own initialized persistence (0.022554 m²/s²). The result is mainly
thermodynamic improvement, not evidence of recovered eddy variability.

[All held-out components and persistence controls](artifacts/early-rmse-4k-2026-10-11/heldout-scores.csv).
Historical test scores are not retroactively assigned to the RMSE minima: the
historical selector could choose different weights.

## Curves

The complete comparison contains **722 verified validation events across 14
series**. All lines use the same RMSE blend. Solid lines are current 4k
reproductions, dotted lines original 4k runs, and dashed lines stretched 6k
runs. Colors identify model families. Markers are actual validation events;
no smoothing chooses minima. The late views explicitly restrict the displayed
range; the full view and CSV retain every point.

![Matched schedules and late validation](artifacts/early-rmse-4k-2026-10-11/validation-late.png)

[Full training curves](artifacts/early-rmse-4k-2026-10-11/validation-full.png) ·
[Observation-exposure curves](artifacts/early-rmse-4k-2026-10-11/validation-observation-exposure.png) ·
[Vector figure](artifacts/early-rmse-4k-2026-10-11/validation-late.pdf) ·
[All values and five physical components](artifacts/early-rmse-4k-2026-10-11/validation.csv) ·
[Minima, endpoints and final-five means](artifacts/early-rmse-4k-2026-10-11/minima.csv).

## Experiment

| Literal model name | Description | Recent OM4 1° | Earlier OM4 ¼° | Observation updates | Extra recurrent channels |
|---|---|---:|---:|---:|---:|
| U-global-repro4k | Recent-only matched control from the preceding wave | 2,000 | 0 | 2,000 | 0 |
| U-multitask-early-fine | Earlier fine-resolution examples through the existing encoder/decoder; best validation in the 6k RMSE wave | 1,000 | 1,000 | 2,000 | 0 |
| U-multitask-early-fine-latent | Same fine task plus recurrent channels on all tasks; best original 4k validation, including RMSE rescoring | 1,000 | 1,000 | 2,000 | 10 |

All three have a global 1° processor. The early models wrap it with the existing
encoder/decoder for the ¼° task; they do not evolve the globe at ¼°. New runs
start fresh at seed 1729, fixed LR 1e-4, eight-example accumulation, and unchanged
losses, normalization and splits. The 4k progressive mixture ends at exactly
2,000 observation updates; earlier/recent OM4 alternate in the remaining slots.
No cooldown is used. These are repeat executions of the same seed, not new
independent seeds. No qualification weights enter training.

**U-global** in the original wave is the prior 4k Beta control. **U-global** and
**U-global-seed1730** in the RMSE wave are the completed 6k controls with
3,000 updates per task and seeds 1729/1730. Historical early models have their
recorded Torch recovery history. Curve labels distinguish wave and budget;
the current reproductions and 6k runs share the same Torch training producer.

Selection uses the equal mean of SST, geostrophic velocity, EKE, OHC 0–700 m and
OHC 700–2000 m RMSE ratios to the frozen validation climatology. Surface errors
average day 5/15/30; OHC uses calendar-month means. There is no detrending,
deseasonalization or spectral term. Validation is saved every 100 total updates
and observation milestones. Test comparisons use 30-day, separately initialized
forecasts, not continuous multi-year rollouts.

This is a comparison at fixed mixed-task updates, not matched FLOPs: fine-task
updates cost more. The new reproductions test the combined earlier-data/fine-task
recipes; they do not independently isolate temporal variety from resolution.

## Verification

All jobs completed successfully on October 11 by **15:52:51 ET**: CPU data audit
**19580149**, qualification array **19580150**, and training/evaluation array
**19580151**. Each arm used one Torch RTX6000, 16 CPUs and 192 GB RAM, at most
two concurrent GPUs. Actual allocated time was **8.7656 GPU-hours**, including
qualification, startup, training and evaluation; **no retries**.

The unchanged successful training producer is
`eae2b5c56da44eb3686a6eaf72a80aa419a4b5a4`. Fresh qualification checked fitting,
gradient paths, exact I/O and bitwise deterministic checkpoint replay. The
completion auditor independently hashed both selected checkpoint files and
verified all 4,000 consecutive updates, exposures, finite losses/gradients,
fixed LR, seed, producer, frozen validation scores, and both complete 96-origin
evaluations. Plotting verified the shared data/support contracts and all saved
RMSE scores. Reporting changes did not alter training code.

[Completion/checkpoint audit and accounting](artifacts/early-rmse-4k-2026-10-11/verification.json) ·
[Curve provenance](artifacts/early-rmse-4k-2026-10-11/provenance.json).
Cluster root: `/scratch/jr7309/runs/2026-10-11-early-rmse-4k`.
Its launch contracts and immutable result/validation bundles are retained;
no source data or checkpoints were deleted. Reporting script:
`scripts/plot_early_rmse_reproductions.py`.
