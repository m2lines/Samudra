<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Final 96-month test exports

`summary.json` and `score-breakdown.json` preserve the evaluator's **frozen
validation-normalized test score, 0.7691**. The [matched reporting comparison](../comparison/heldout-comparison.json)
uses training-climatology errors evaluated on the held-out cohort, giving
**0.7161 versus 0.5927**. Neither normalization was used for test-based selection.

`calibration.json.gz` contains additive wet-area statistics for all 96 origins.
`spectra.json.gz` retains per-origin Pacific-patch powers of individual members,
their mean, and references. Maps show the same model's fixed January 2015, 2018
and 2021 cases at two image pixels per grid cell. Interiors are calendar-month
averages of each member; surface maps are five-day bins ending at day 30.

`COMPLETE.json.gz` records the checkpoint, data/training contract and all 96 raw
member-array hashes. Those arrays remain in Torch's
`/scratch/jr7309/diffusion-global-v1/evaluation/final-test-steps32/`.
