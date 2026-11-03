<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Temperature skyline pre-training audit

The local Torch quarter-degree fields are authorized for this experiment. All
coordinates, timestamps, time metadata, and 19 masks match the September averaged
release. Maximum absolute differences in means and standard deviations are both
zero over the 80 input channels. The reader uses the reference normalization and
cell areas. This is not a full payload comparison between releases.

The complete training-range scan used 2,830 dates. The precision audit used all
wet cells at each depth over all 72 valid five-day validation windows, with native
cell-area weighting and equal weight per date. Both compare against OM4 model
context, not observations. No validation data fitted the bin support.

| Depth (m) | Absolute-bin rounding RMSE (C) | Persistence RMSE (C) | Ratio |
| ---: | ---: | ---: | ---: |
| 2.5 | 0.006312 | 0.302938 | 2.08% |
| 550 | 0.003557 | 0.096466 | 3.69% |
| 1,400 | 0.002016 | 0.022980 | 8.77% |
| 5,000 | 0.000808 | 0.002946 | 27.42% |
| 6,000 | 0.000418 | 0.000943 | 44.33% |

No validation targets fell outside either tested support. Absolute-bin rounding
remains below persistence at every depth but is material in the deepest layers.
These are oracle nearest-bin errors, not learned model results. They do not yet
establish the gap to the selected Samudra checkpoint. The residual audit uses a
conservative support derived from the full training state range, not a separately
optimized increment vocabulary.

All 19 rows are in [precision.csv](precision.csv). [precision.json](precision.json)
and [support.json](support.json) retain the source metadata hashes, normalization,
bounds, split origins, and sampling/weighting definitions.

Execution receipts:

- Local pilot 21707293: completed 0:0 in 17 seconds.
- Full range audit 21707355: completed 0:0 in 1 minute 42 seconds;
  source commit `ddb594fe18c0087f25dfe8856daaf92d66c6c2db`.
- Whole-validation precision audit 21708346: completed 0:0 in 29 seconds;
  source commit `f799a0b1805be6a615546d4dd5aa4ec3df8e19e5`.
- GPU qualification 21707350: canceled before start on November 3 after the
  shared RTX6000 pool's eight-GPU quota blocked it.
- Replacement qualification 21759055: queued on November 3 for preemptible
  `rtx6000,h100,h200,b200` capacity, source
  `a5cfd7670486ae0373e45ba2d2cd2281214c1f39`. Its eight-update, 1.658B-parameter
  run is an engineering qualification only. It uses the full training support,
  saves every update, and resumes automatically on scheduler restart. Callback
  job 21759089 covers final termination. GPU execution is not yet established.

Torch root: `/scratch/jr7309/temperature-skyline-20261102`.
