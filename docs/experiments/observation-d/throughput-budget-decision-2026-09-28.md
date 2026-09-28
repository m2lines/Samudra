<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Production budget decision after throughput qualification

**Resolved: retain the original 16k-total comparisons.** The user clarified that
feasible GPU spend and update counts should be chosen within the three-day
wall-time budget. The 100-hour ceiling was an assistant planning assumption, not
a user constraint. Use 20-hour scratch and 16-hour other-arm allocations, with
internal limits 30 minutes shorter, and a 120 GPU-hour planning ceiling.

H200 job 18724971 completed 60 OM4 and 60 observation updates with the qualified
training producer. Over the last 20 updates of each task, mean wall time was
2.320 seconds/OM4 update and 4.140 seconds/observation update, including sample
access and optimization. GPU utilization near the end was 82–90%.

| Choice | Scratch, per arm | OM4 + observations, per arm | Observation-only tail | Estimated training GPU-hours across six arms |
| --- | ---: | ---: | ---: | ---: |
| Proposed bounded wave | 12,000 obs | 6,000 + 6,000 | 1,500 obs | 70.7 |
| Original update counts | 16,000 obs | 8,000 + 8,000 | 2,000 obs | 94.2 |

Estimates exclude validation, checkpoint writing, startup, evaluation and retries.
The original 12-hour per-job allocations are too short: estimated update time
alone is 18.4 hours for scratch and 14.4 hours for the other arms. The original
counts leave insufficient margin within the 100 allocated GPU-hour campaign cap.

The proposed bounded wave retains all six comparisons, the same one seed and
batch eight, exact per-task sample streams, optimizer, models, masking, losses,
and integrated-plus-spectral selection. It preserves the 50/50 task allocation
and reserves the final quarter of observation updates for the observation-only
finish. Scratch retains immutable 8k and 12k endpoints. Finish checkpoints include
4,500 observations at the boundary and early points after it.

Suggested job limits: 15 hours for each scratch arm and 13 hours for each other
arm (82 requested production GPU-hours total), with internal limits 30 minutes
shorter. Monitor actual accumulated use, retaining recovery/evaluation room under
100 GPU-hours. If keeping the original 16k counts is preferred, propose a
120 GPU-hour campaign cap and longer per-job limits instead.

The earlier request to reduce counts was rejected by automatic approval review.
After the user clarification, the original counts are retained and only runtime
limits/planning ceiling are increased. No further budget approval is pending.
