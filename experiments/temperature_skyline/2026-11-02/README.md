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
- Replacement qualification 21759055: completed `0:0` on November 3,
  12:53:41–12:57:04 EST, on B200 node `gb013`, source
  `a5cfd7670486ae0373e45ba2d2cd2281214c1f39`. All eight optimizer updates had
  finite losses and gradients, with no targets outside the audited support.
  Peak allocated GPU memory was 28.53 GiB. Total job time was 3m23s; logged
  optimizer-step timings exclude checkpoint and other overhead.
  `COMPLETE.json` confirms step 8 and qualification status; latest optimizer
  checkpoint and both best-weight checkpoints exist. Validation ran, with
  mean-depth bin NLL 8.131 and normalized MSE 1.400. This short run establishes
  engineering qualification only. There were no preemptions or restarts, so
  eviction recovery remains untested. The pending completion callback 21759089
  was canceled after direct verification, and its reminder was revoked.

[Qualification run in W&B](https://wandb.ai/ocean_emulators/temperature-skyline/runs/ce73e0a348ff).

Initial smaller training pair submitted November 3, source
`e176caa58027a3a1e1f5bffb8e868b40fa86f348`:

- Job 21760465: absolute-temperature classifier, 210,067,456 parameters.
- Job 21760475: scalar regression control, 207,969,281 parameters.

Each run uses one preemptible GPU, 20,000 updates, batch 32, and matched data and
validation sampling. Both were queued at submission. Checkpoints occur every
500 updates; validation every 2,000. The assigned GPU is sampled every five
seconds, including I/O and validation. After startup and a five-minute window,
a mean utilization below 60% triggers checkpoint-and-stop for diagnosis.
Actual utilization and model skill are not yet established. Configurations are
`src/samudra/configs/temperature_skyline/small_{absolute,regression}.json`.

Both initial jobs stopped after one update because CUDA supplied a bare GPU
UUID and the monitor passed it to `nvidia-smi` without its required `GPU-`
prefix. They saved checkpoints before failing: 21760465 used 42 seconds on
H200 `gh125`; 21760475 used 39 seconds on RTX6000 `gr102` (0.0225 GPU-hours
combined). No utilization measurement was obtained. Source
`73e13c8bd52b2d89bc6a39f60021bb1ac0d8d177` fixes the identifier, with tests for
both bare and prefixed UUIDs (15 focused tests pass). Replacement jobs 21762617
(classifier) and 21762622 (regression) were queued to resume their step-1
checkpoints with unchanged training budgets and utilization thresholds. Prior
attempt receipts are retained under each run's `attempts/<old-job-id>/`.

Torch root: `/scratch/jr7309/temperature-skyline-20261102`.
