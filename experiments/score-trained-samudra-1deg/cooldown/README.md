<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Matched forecast-only cooldown after F/M training

User-authorized October 14, 2026: a small fine-tune of both models on 100% forecast data, with a
cooldown. This tests whether forecast adaptation recovers M's deficit while giving F the same additional
training. It does not erase the unequal forecast exposure during the original 70 epochs.

- Initialize F and M from their own final epoch-70 **EMA** checkpoints. Strict architecture loading;
  both configurations now set `model.mode: forecast`. All layers are trainable.
- Ten epochs each, fresh Adam optimizer and fresh cosine scheduler starting at 0.00006, reaching zero
  after epoch ten. This is a low-LR restart, not resuming the exhausted original 70-epoch cosine schedule.
  The last training epoch uses a small positive LR; the scheduler reaches zero after its update.
- Keep the inherited EMA tracker/age (decay 0.999), data splits, normalization, Rust loader, native
  pointwise backend, four-member CRPS objective, two-input/two-output and four-call autoregression.
- Two RTX PRO 6000 Blackwell GPUs per arm, batch four/rank, accumulation four, effective batch 32.
  Seed 15 in both cooldowns; task fractions must be 100% forecast. Online W&B and eight-member annual
  validation every epoch. Save every two epochs plus latest/final/EMA/validation-best checkpoints.
- Six-hour allocation ceiling per job, normal RTX partition, no automatic requeue. The trainer disallows
  `finetune` together with `preemptible`; any interruption recovery must resume the cooldown's own
  checkpoint with `finetune: false`, retaining optimizer/scheduler progress rather than restarting it.
- Expect roughly three hours per arm based on the original forecast throughput, approximately 13 RTX
  GPU-hours combined; actual allocation costs will be recorded, including any failed attempts.

Training reuses immutable source `8150d606e3ca5d640ee28d43c17be22525893e5c` and the verified
`sd-seeds-base-0b5320248.sif`. No training implementation changes. Resolved [F](F.yaml) and [M](M.yaml)
configs and the [submission harness](train_torch.sbatch) are included here. Remote campaign root remains
`/scratch/jr7309/runs/score-multitask-1deg-20261012`. New outputs are `outputs/cooldown-F-forecast`
and `outputs/cooldown-M-forecast`; existing results/checkpoints are retained.

Jobs **F19804204 / M19804205** were observed running on gr105, two GPUs each. Startup success still
requires checkpoint-load, online W&B and first-epoch per-step verification. CPU callback **19804212**
uses `afterany:19804204:19804205`. Quiet monitor `/root/forecast_cooldown_monitor` owns webhook
`4d8fab84-a337-4023-baed-321b45002d29`, with an hourly reminder and a first bring-up check after ten
minutes. The registration's initial placeholder job field is `pending-cooldown-F-M`; the actual mapping
is recorded here. Scratch preflight: 4.64 TB / 5 TB used (92.89%), with required dataset/SIF and online
W&B credentials present. No credential values are recorded.

After training, repeat the fixed held-out forecast comparison (30 origins, sixteen members), completion
retention diagnostic, and continuous rollout/within-versus-across-call salinity diagnostics. Final cooldown
EMA is primary, with final **raw** weights as a sensitivity check so inherited EMA lag cannot hide the
response to this short fine-tune. Report each arm before/after and the between-arm difference. Keep
validation selection separate from final-checkpoint results; do not select on held-out performance.

## Initial startup failure and recovery

F19804204 and M19804205 both failed on gr105 at the initial NCCL barrier, before checkpoint loading,
W&B initialization or training steps. Each exited 1 after 11:27; combined failed allocation cost was
0.7633 RTX GPU-hours. Callback19804212 completed and its registration was retired. Thus the initial
running scheduler state did not establish successful training bring-up.

The launcher now explicitly exports `NCCL_P2P_DISABLE=1` and forwards it through `APPTAINERENV_`
into the clean container, as recommended by `docs/torch.md` for RTX NCCL hangs. No training/configuration
changes. This is a documented mitigation, not proof of the cause; scheduler-selected nodes also differ.
Retry F19804782 was observed running on gr102 and M19804783 on gr103. CPU callback19804785 depends
on both via `afterany`. Webhook `055806e5-b121-4f5d-9abd-c3a743a08c9f` and the same quiet monitor
cover a ten-minute bring-up check, then hourly monitoring. First-step verification remains pending.
