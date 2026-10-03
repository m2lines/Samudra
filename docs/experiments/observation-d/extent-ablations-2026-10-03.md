<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native-patch transfer: cause-isolation follow-up

Authorized October 3, with approximately 40 hours before review (roughly
Monday October 5 at 9:53 a.m. ET). One seed, one four-GPU beta production
allocation, four alternatives to the original U-Net multitask run. This follows
the [completed screen](extent-wave-2026-10-02.md), whose U-multitask composite
was 19.9% worse than U-global on the held-out cohort. The local processor's gap
was smaller; this wave concentrates on the larger U-Net gap.

## Fixed setup and precise interventions

All runs start from the same random seed 1729 and U-Net weights as the original
U arms. Data, normalization, geometry inputs, architecture, batch accumulation,
observation losses, checkpoint-selection metric and held-out cohort stay fixed.
The original mixed schedule has 4,000 slots: 1,000 global OM4, 1,000 auxiliary
native-patch OM4 and 2,000 global observation slots. Global data are 180×360;
native quarter-degree patches are 128×128 with a central 64×64 scored core.
Global tasks always retain six five-day forecast steps. All task samples use
the original count-derived seeds, including the retained original global slots.

| Literal name | Auxiliary patch-slot behavior | Actual optimizer updates | Question |
|---|---|---:|---|
| U-omit-patch | Skip the slot completely: no forward/backward, Adam moments, weight decay, LR change or RNG draw | 3,000 | Does the original native task hurt compared with omitting it, or merely fail to replace the value of additional global OM4? |
| U-patch-lr01 | Original shared initializer and six-step native task, with patch LR 1e-5 instead of 1e-4 | 4,000 | Do weaker patch parameter updates reduce interference? |
| U-patch-truth | Supply the two true native interior states; six-step evolution loss only, no patch initializer or reconstruction/completion losses | 4,000 | Does removing the patch reconstruction problem improve transfer through the shared processor? |
| U-patch-1step | Original shared initializer, but score/predict only the first five-day native step; retain original reconstruction/completion losses | 4,000 | Is short-horizon supervision more useful than the six-step regional task? |

Global OM4 and observation learning rates remain 1e-4 in all four arms. The
smaller-patch-LR arm retains shared Adam moments; it is not a separate optimizer
or a simple loss-weight experiment. The true-state arm changes both the quality
of regional initial conditions and which parameters receive regional training;
it is a bundled diagnostic of the initializer burden, not a clean proof of
initializer gradient interference alone. No true future ocean states or boundary
conditions are supplied to any rollout. Observation inference always uses the
learned initializer, including for U-patch-truth.

The omission arm preserves 4,000 **schedule slots**, not 4,000 optimizer updates.
Its 1,000 skipped patch slots are explicitly recorded as `joint_skip` events and
in `EXPOSURE.json`; the inherited OM4 counters in training markers count scheduled
slots. It has only 1,000 actual global OM4 updates plus 2,000 observation updates.
This is intentionally a removal control, not a compute-matched challenger.

Interpret against both original controls: U-global uses 2,000 global OM4 updates
and 2,000 observation updates; U-multitask uses 1,000 global, 1,000 native patch,
and 2,000 observation updates. If U-multitask loses to U-omit-patch, that supports
harm from the auxiliary task in this setup. If it beats omission but loses to
U-global, the evidence favors a less useful allocation of updates. These are
single-seed comparisons, not population-level causal estimates.

## Diagnostics and evaluation

A separate small GPU diagnostic uses four deterministic training examples per
task at the original U-global and U-multitask validation-selected checkpoints.
It reports observation/global-OM4/native-patch losses, shared initializer and
processor gradient norms, and pairwise gradient cosines before clipping. It
excludes task-specific input adapters from the shared-parameter comparison.
There are no optimizer steps and no validation/test examples in this diagnostic.
Four examples are a mechanism probe, not a representative global survey.

The same frozen nine-month observation validation selects checkpoints by
integrated-plus-spectral composite, never RMSE alone. Every finished arm gets
96 independently initialized monthly held-out forecasts (2015–2022), the same
27 spectral keys, and initialized-state persistence/anomaly/climatology controls.
Report held-out composites using the same common held-out climatology denominators
as the first screen; distinguish those from validation-selection denominators.
The held-out cohort has been seen in earlier experiments, so these follow-ups
are exploratory comparisons rather than a fresh untouched-test claim.

Keep all completed original runs unchanged. No new seeds, LLC experiments,
additional arms or adaptive extensions are included in this wave. Queue waits
are expected; do not cancel/resubmit to chase capacity. Target less than one day
per run, with a 20-hour per-run training cap and 24-hour four-GPU allocation cap.
A progress snapshot is useful if queue delays prevent completion before review.

## Qualification and execution

A fresh fitting qualification is required for the pinned producer, then a
separate ten-slot joint/resume qualification for each intervention. Production
depends on all four probes succeeding. CPU tests verify true omission of optimizer
and RNG effects, task-local learning-rate changes, actual one/six-step forecast
calls, and absence of initializer gradients under true-state patch inputs.
The runtime, native cache and observation reference remain the verified ones from
the original screen. Job IDs, producer hashes and live outcomes follow here.
