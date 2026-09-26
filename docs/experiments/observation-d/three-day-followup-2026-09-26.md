<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Three-day joint-training and mechanism investigation

Authorized 26 September 2026 at 13:30:45 UTC. **Report deadline: 29 September
2026 at 13:30:45 UTC (09:30:45 America/New_York).** The deliverable is conclusions
about performance and/or mechanism, not a requirement to exhaust compute or
complete an unbounded search. Finalize the report within that wall-time cap.

## Scope and priorities

1. Qualify a Samudra-2-backbone-matched evolution model: widths
   [280, 380, 480, 520], capped GELU, expansion two, zonally periodic upsampling,
   and the reference output kernel. Retain affine InstanceNorm, no running stats,
   and fixed loss initially. This is architecture matched, not a reproduction of
   the BatchNorm/dynamic-loss reference recipe. Keep the initializer fixed in
   architecture so the processor change is explicit.
2. Use one training loop with scratch-observation, sequential OM4/observation,
   and scheduled mixed OM4/observation arms. The sequential arm is a control for
   ordering; the mixed arm retains OM4 in its observation-heavy tail. Retain
   reconstruction as an auxiliary objective, rather than a separate observation
   reconstruction/adaptation phase. Start with the existing shared output/state
   interface and forcing adapter; extra decoders/task embeddings are conditional
   follow-ups, not automatic additions.
3. Diagnose existing completed checkpoints while the new comparison runs:
   case-specific initial-state information versus seasonal structure, perturbation
   sensitivity at short/long leads, and bias/seasonal versus anomaly errors.
   Use training-derived replacements; never derive climatology or tune
   interventions from held-out outcomes. Avoid raw state swaps between
   independently trained latent representations.
4. If time permits, test whether richer OM4 supervision is responsible for the
   benefit using observation-pattern versus full-state supervision on matched
   OM4 examples. Do not sacrifice completed main comparisons/reporting for this
   optional extension.

**Excluded:** the known velocity/preprocessing ceiling investigation and
quarter-degree training. Stay on the existing cheap one-degree grid; the user
intends quarter-degree work after the mechanism is better understood.

## Qualification and resource bounds

Use Torch/RTX by preference, seed1729 initially, existing verified local datasets,
and no repeat bulk transfer. Check actual scheduler capacity, quota and throughput
before production. Initial operational ceiling: **120 allocated GPU-hours**, with
all qualification/calibration/retry allocations charged. This is a new bounded
campaign, not leftover allocation from a prior wave. Do not infer authorization
for new seeds or unrelated searches. Production update counts are set from
qualification throughput and remaining wall time, then pinned before submission.
Reserve the final six hours for completing evaluations, analysis and reporting;
do not launch a job whose requested duration exceeds the remaining deadline.

Maintain approximately matched OM4 and observation exposure between sequential
and mixed arms. Record both task update counts, effective batch, loss weights,
optimizer policy and GPU-hours; sampling proportions and loss scales must not
silently double-count a task's intended weight. Preserve early checkpoints,
OM4 retention curves and observation learning curves. Separate final-budget raw
weights from observation-validation-selected weights.

Selection remains the fixed v3 integrated-plus-spatial-spectral observation
validation score. Report components, persistence, 30-day and annual held-out
metrics. Annual and intervention diagnostics do not alter the selected checkpoint.
No held-out tuning; one-seed limitations and confounded comparisons are explicit.

## Execution record

At start, Torch authenticated multiplexing works and the user queue is empty.
Scratch usage is 4.04 TB of 5 TB (approximately 0.96 TB free). Prior InstanceNorm
results and all source checkpoints remain intact. The old monitor timer stays
disabled; monitoring uses interruptible waits and meaningful updates, with
previously authorized Slack blocker alerts if user action is required.
