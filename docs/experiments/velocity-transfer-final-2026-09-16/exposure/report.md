<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# DUACS exposure diagnostic

These are best-so-far scores on the 12 checkpoint-selection validation dates, not the full held-out evaluation. RMSE is in m/s. Each pair is restricted to the same maximum number of DUACS window draws. D4 also sees OM4 examples. The limits do not match optimizer steps, total compute, learning-rate phase, geometry channels, or the number of validation opportunities, so this is a learning-curve diagnostic rather than a causal transfer estimate.

| DUACS draw cap | Seed | D0 10-day RMSE | D4 10-day RMSE | D4 improvement |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 15 | 0.102124 | 0.115864 | -13.45% |
| 4096 | 16 | 0.094830 | 0.099098 | -4.50% |
| 4096 | 17 | 0.101376 | 0.110817 | -9.31% |
| 16384 | 15 | 0.102124 | 0.103625 | -1.47% |
| 16384 | 16 | 0.094830 | 0.094455 | +0.40% |
| 16384 | 17 | 0.101376 | 0.110817 | -9.31% |
| 65536 | 15 | 0.102124 | 0.103625 | -1.47% |
| 65536 | 16 | 0.094830 | 0.094455 | +0.40% |
| 65536 | 17 | 0.101376 | 0.108225 | -6.76% |
| 262144 | 15 | 0.102124 | 0.101601 | +0.51% |
| 262144 | 16 | 0.094830 | 0.094455 | +0.40% |
| 262144 | 17 | 0.101376 | 0.101461 | -0.08% |
| 524288 | 15 | 0.102124 | 0.100559 | +1.53% |
| 524288 | 16 | 0.094830 | 0.094455 | +0.40% |
| 524288 | 17 | 0.101376 | 0.100888 | +0.48% |
| 671272 | 15 | 0.102124 | 0.100265 | +1.82% |
| 671272 | 16 | 0.094830 | 0.094455 | +0.40% |
| 671272 | 17 | 0.101376 | 0.100888 | +0.48% |

Training exposure is logged every ten updates; validation every 256. The CSV brackets D4 exposure between the preceding count and that count plus all intervening examples (at most 36 draws). A checkpoint is admitted only when its upper bound is within the cap. D0 exposure is exact because every update uses DUACS. The common final cap is the minimum, across all six runs, of the largest logged validation exposure lower bound. Later training without validation cannot extend these curves. Repeated log records after recovery are retained as observed validation opportunities.

The two longer-lead columns in matched-exposure.csv refer to the same 10-day-selected checkpoint, rather than independent lead-specific minima. Input history checksums are recorded in provenance.json so this snapshot can be distinguished from the final completed histories.
