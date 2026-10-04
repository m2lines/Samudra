<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# What information makes the fine-scale accessory target useful?

Predeclared October 4 around 8:50 a.m. ET, within the authorized iteration
window ending Monday around 9 a.m. ET. Completed validation, before collecting
the accessory arms' held-out scores, motivates this follow-up: U-aux01 reached
0.642149 versus U-global's 0.664336, a 3.3% improvement. U-aux10 reached 0.669779
and did not improve. All use the fixed integrated-plus-spectral observation
selection metric; the single-seed difference needs careful interpretation.

A training-only target decomposition found that static spatial means explain
67.8% of standardized target variance and per-cell calendar-month means explain
72.0%. These controls ask whether aligned fine-scale variability matters, or
whether the accessory head mainly supplies a geographic/seasonal regularizer.

## Fixed setup and literal model names

Every arm uses the exact U-aux01 architecture (62,967,809 parameters), coefficient
0.01, seed 1729, fresh initialization, optimizer and progressive schedule:
2,000 global 1° OM4 updates plus 2,000 global observation updates. There is no
regional evolution task. Native quarter-degree five-day-mean velocities supply
only the existing surface subcell-variance target. Data splits, physical input
normalization, forcing, masks, geometry, six-step OM4 horizon, observation
calendar alignment and validation/test metrics stay unchanged.

Let z(t, y, x) be the existing globally standardized log1p variance target.
All climatologies use only the same 2,829 training frames. They change the
accessory target, not physical input normalization or observation labels.

| Literal name | Target replacing z at every predicted OM4 lead | Question |
|---|---|---|
| U-aux01-static | Training time mean of z at each cell | Is a fixed spatial target sufficient? |
| U-aux01-seasonal | Training calendar-month mean of z at each cell | Is geography plus the annual cycle sufficient? |
| U-aux01-shuffled | z from a fixed permutation of training dates, seed 271828 | Does correctly aligned timing matter when the target's marginal distribution is preserved? |
| U-aux01-anomaly | z minus its training calendar-month mean at each cell | Can supervision focused on departures from the seasonal spatial pattern help? |

U-aux01, defined in the [representation report](extent-representation-2026-10-03.md),
uses the actual aligned z. U-global is the same original U-Net without an
accessory head or loss, defined in the [first screen](extent-wave-2026-10-02.md).
Every result table must use these literal names.

The shuffled target preserves full spatial maps and their marginal distribution,
but destroys the relation between model date and target date and disrupts target
sequence correlations. It tests alignment against potentially conflicting
supervision; failure alone would not prove a physical mechanism. Static,
seasonal and anomaly targets have different residual variance, so a common
coefficient does not imply matched accessory gradient magnitudes. Record target
moments and training losses to make that limitation visible. No target is
rescaled per cell or per mode.

## Qualification and execution plan

Pin a new producer without changing active runs. Test the temporal transforms,
qualify fitting and serialization/resume for every target mode on beta_test, and
require each probe to extrapolate below 12 training hours. Use one four-GPU beta
allocation with a 16-hour cap and 14-hour training safety cap, after the running
initializer follow-up releases its node. Allow overlap with the attention wave,
for at most eight production GPUs. Wait through normal queue delays.

Collect all 96 held-out monthly forecasts only for the validation-selected
checkpoint of each completed arm. Report test scores as exploratory, with the
same test-climatology denominators and persistence controls. These forecasts
cover 2015–2022 as independent monthly 30-day cases, not continuous eight-year
rollouts. Any unfinished arm at Monday review will be labeled unfinished.

## Qualification submitted at 8:49 a.m. ET

Producer `558d45cac2294f691d1d73a47db1783df1850c80`, archive SHA-256
`e79ee680969a14ff21bdda5de08e5d285cca1bf2b7ec775a2b41dd7a91b952e6`.
Local validation passed 22 targeted tests, Ruff, mypy for the four touched
Python implementation files, shell syntax and diff checks. The tests check
seasonal/anomaly decomposition, invalid-cell support, preserved shuffled maps,
unchanged global RNG, future-label alignment and gradient routing. No active
producer was replaced and no checkpoint crosses the new qualification contract.

| Stage | Arm | Job |
|---|---|---:|
| Fit | Shared accessory architecture | 209715 |
| Joint/resume probe | U-aux01-static | 209716 |
| Joint/resume probe | U-aux01-seasonal | 209717 |
| Joint/resume probe | U-aux01-shuffled | 209718 |
| Joint/resume probe | U-aux01-anomaly | 209719 |

All four probes depend on the fitting job succeeding. Each qualification uses
one GPU, 16 CPUs, 128 GiB and at most two hours on beta_test/test. Production
has not been submitted. Root:
`/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-04-extent-accessory-controls`.
[Exact receipts](artifacts/extent-accessory-controls-2026-10-04/qualification-jobs.json).
