<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Three-day investigation: interim findings

Started 26 September 2026. **Final report due 29 September at 09:30 ET.**
This is an active investigation; the larger-backbone/mixed-task comparison has
not run yet. [Plan and scope](three-day-followup-2026-09-26.md).

## First result: much of the long-range advantage is reduced bias

These diagnostics reuse the completed InstanceNorm scratch and InstanceNorm
OM4 → observations models defined in the [preceding report](instance-norm-results-2026-09-26.md).
They use the same selected checkpoints and three continuous annual test origins;
there is no training, reselection or fitted test-time correction.

For each location we decompose error over the 73 forecast steps into its temporal
mean and residual temporal variation. The area/time-weighted MSE is exactly the
sum of those two components, including variable observation support. This
within-year mean is a diagnostic decomposition, not a deployable bias correction.
Values below average the three per-origin decompositions.

| Model | Field | Annual MSE | Temporal-mean bias contribution | Temporal-anomaly error contribution |
| --- | --- | ---: | ---: | ---: |
| InstanceNorm scratch | SST (°C²) | 3.087849 | 2.115475 | 0.972374 |
| InstanceNorm scratch | ADT (m²) | 0.032543 | 0.023446 | 0.009098 |
| InstanceNorm OM4 → observations | SST (°C²) | 1.016766 | 0.520636 | 0.496130 |
| InstanceNorm OM4 → observations | ADT (m²) | 0.015001 | 0.008632 | 0.006370 |

Approximately **77% of the SST MSE reduction and 84% of the ADT MSE reduction**
are in the temporal-mean bias component. The temporal-anomaly error also improves;
the advantage is not exclusively a mean offset. This does not yet identify
whether weights, initial state or forcing response causes the difference.

## Training climatology is a stronger year-end baseline than either model

The baseline is the existing training-only monthly surface climatology evaluated
at each forecast midpoint's calendar month, with the same observed support.
The correlation is the area-weighted centered spatial correlation of predicted
and reference anomalies relative to that climatology. Simply subtracting the
same climatology from prediction and truth cannot change RMSE; the separate
correlation/amplitude diagnostics answer a different question.

| Model | Field | Day-365 RMSE | Climatology RMSE | Day-365 mean bias | Centered anomaly correlation |
| --- | --- | ---: | ---: | ---: | ---: |
| InstanceNorm scratch | SST (°C) | 2.1102 | 1.0373 | -0.9670 | 0.2792 |
| InstanceNorm scratch | ADT (m) | 0.2192 | 0.0992 | -0.1282 | 0.0510 |
| InstanceNorm OM4 → observations | SST (°C) | 1.2906 | 1.0373 | -0.4106 | 0.3847 |
| InstanceNorm OM4 → observations | ADT (m) | 0.1536 | 0.0992 | -0.0564 | 0.0318 |

OM4 initialization improves year-end SST anomaly correlation, but not ADT anomaly
correlation. Both models have worse year-end SST and ADT RMSE than climatology.
The previous annual improvement therefore establishes reduced degradation
relative to scratch, not useful year-ahead predictive skill against this baseline.
One seed and three previously examined annual origins remain exploratory evidence.

## Execution and remaining work

The Samudra-2-matched processor has 83,872,357 parameters, versus approximately
31.6M in the previous pilot; the initializer stays approximately 121.7M. The
reference backbone/output padding and kernel are reused, retaining InstanceNorm
and fixed loss. Eighteen backbone/observation/evaluator tests passed. Immutable
producer `b9d6ebad9f240e5fd92f6e0bb86c4d774ccfe6ea` is built; qualification jobs
18580735 (OM4) and 18580736 (observations) are queued behind `QOSGrpCpuLimit`.
No new production training has been submitted.

Bias/anomaly diagnostic producer `6d8a8d26b` passed two decomposition tests and
repository checks. CPU job 18580782 failed before analysis because bare container
Python lacked an installed project dependency; corrected job **18580810 completed
in 23 seconds** using the container project environment. No GPU-hours were used.
Raw [scratch diagnostics](artifacts/2026-09-26-three-day/scratch-annual-decomposition.json)
and [transfer diagnostics](artifacts/2026-09-26-three-day/transfer-annual-decomposition.json)
retain evaluation and training-statistics hashes.

Next: qualify throughput/memory, freeze matched task-exposure budgets, implement
and qualify the single-loop sequential/mixed controls, and evaluate initial-state
and forcing interventions. Known preprocessing limits and quarter-degree runs
are out of scope for this three-day campaign.
