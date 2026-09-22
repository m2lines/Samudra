<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Diagnosing 550 m salinity reconstruction

The wave-3 D initializer improves aggregate error yet reconstructs 550 m salinity anomalies poorly. This follow-up distinguishes finite precision, inability to fit, competing training objectives, background-field learning, temporal distribution shift, and missing information. Diffusion is a future option; these diagnostics do not assume deterministic MSE averaging is the dominant cause.

## First stage: approved September 22, 2026

All runs begin at the original wave-3 D best checkpoint, use its existing normalization and train/validation split, the Rust loader and GPU frame cache. Evolution remains frozen and unused. One RTX GPU per run, BF16 except the explicit precision comparison, batch size 2 with accumulation 4 (effective batch 8), fresh AdamW at 1e-4, weight decay 0.01, clipping 1.0, seed 1729. TF32 is disabled. No new optimizer search.

| Run | Question | Protocol |
| --- | --- | --- |
| Precision | Is BF16 causing the structure errors? | Identical weights/batches, BF16 versus FP32 inference on 16 separated training origins and all 11 validation origins. Save paired maps and metrics, plus unchanged-D BF16 maps at all 99 held-out origins after the first-stage choices are locked. |
| Fit one | Can the model fit one field? | Only so_9 loss, one training origin, 300 updates. |
| Fit sixteen | Does fitting break as examples vary? | Only so_9 loss, 16 evenly spaced training origins, 300 updates. |
| Continued control | Does more ordinary optimization help? | Original all-interior-variable loss, 1,000 updates on all eligible training windows. |
| Field specialization | Is the field lost among competing objectives? | Only so_9 loss, the exact same starting checkpoint, examples and 1,000 updates as continued control. |

Both reconstructed times enter the training objective. Report current-time metrics separately for correspondence with the maps. For the two generalization arms, checkpoints are selected by mean validation so_9 MSE over both times; for memorization arms selection uses the fitting set. Save the last checkpoint too. Selection happens every 100 updates, and checkpointing at least every three minutes. The original checkpoint is an eligible baseline. No early stopping, so both generalization arms receive equal updates.

Every run writes its full protocol, checkpoint checksum, producer revision, Slurm request and W&B logs. First qualify the exact producer with three fitting updates. Initial scheduling cap is five single-GPU jobs with three-hour limits; this is a diagnostic cap, not a plan to consume all allocated walltime. Inspect useful throughput and recover failures before extending the wave.

No fitting uses the displayed 2018 test date. After all first-stage choices are locked, evaluate selected generalization checkpoints on the original 99 origins and save so_9 maps across seasons. Compare physical RMSE, climatology-relative MSE skill, anomaly correlation/amplitude, biases, and paired BF16/FP32 differences. Memorization is a sanity check, not held-out skill.

## Follow-ups driven by the first stage

1. If precision differences are material or tiny-set fitting stalls, repeat the relevant fitting test in FP32 and diagnose train/eval behavior before scaling.
2. Test a climatology-plus-anomaly output parameterization with a matched direct-output control. Simply subtracting climatology inside an MSE expression is algebraically identical and is not a valid ablation. Lock initialization and train-only climatology handling before this stage.
3. Use saved training/validation/test maps to compare error and anomaly distributions across years and seasons, spatial scales, and persistent bias. A similar test-period error does not rule out train/test shift. Any fitted bias/blur/displacement correction is diagnostic unless fitted on training data and evaluated out of sample.
4. If fitting succeeds but generalization remains poor, test additional information (surface salinity, longer history, then explicitly privileged interior inputs) with matched controls before attributing the remaining error to irreducible ambiguity.

Publish the diagnostic report, plots and code to the existing draft PR. Do not interpret prettier or sharper maps alone as increased conditional prediction skill.

## Launch record

Runtime producer: `349f6c9ec`. Smoke job `18259726` completed three updates and exact native/cache equivalence checks before production submission. The first-stage jobs are precision `18260077`, one-example fitting `18260078`, sixteen-example fitting `18260079`, continued control `18260080`, and field specialization `18260081`. All use RTX PRO 6000 Blackwell GPUs.

The inherited wave-3 `study-manifest.json` retains its original T/S selection description; for these diagnostics the selection rule above and `diagnostic-protocol.json` plus `selection` events are authoritative. Both generalization arms select on so_9 validation MSE.
