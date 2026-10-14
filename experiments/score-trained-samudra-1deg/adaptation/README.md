<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Forecast adaptation: diagnostic sweep and stronger recovery

Authorized October 14, 2026: proposals 1 and 2 from the audit discussion. The larger exposure-control
three-arm experiment is not authorized/submitted in this wave.

## Saved-checkpoint diagnostic

Each arm's original epoch-70 EMA is adaptation epoch zero. Evaluate its existing cooldown raw
checkpoints at epochs 2,4,6,8,10 on the fixed validation origin, eight members, seed26000000 matching
the annual validation stream. Record per-channel metrics at 5,10,20,40,90,180,360 days and compare
raw curves against the logged EMA annual-validation curves. Completion retention at epochs0,4,10
uses sixteen members and four fixed quarterly validation origins, identical masks across checkpoints.
Epoch-zero completion uses the parent EMA; epochs4/10 use cooldown raw weights. This is explicitly
validation-only; no intermediate checkpoint selection on held-out dates.

## Matched stronger adaptation

Start both F/M again from their original epoch-70 EMA, not from the completed cooldown. All windows
are forecasting, all parameters trainable. Thirty epochs, fresh Adam, fixed learning rate1e-4 for twenty
epochs followed by ten epochs of cosine decay to zero. Implemented with the existing cosine-with-warmup
scheduler using equal warmup/base rates; tests verify the actual rate sequence.

Reset EMA after checkpoint loading: initial weights equal loaded model, update count zero, fixed decay
0.99 and faster-start disabled. EMA half-life is about69 updates, or0.8epoch at88updates/epoch.
The reset applies only when starting a fine-tune; ordinary resume restores saved EMA/optimizer state.
This explicitly fixes the old fine-tune's inherited EMA age. Both raw and EMA annual validation are
logged each epoch with identical random streams. Final EMA is primary; raw is sensitivity. Validation-best
selection remains the original EMA CRPS selection rule, separately reported if evaluated.

Retain Rust loading, data/splits/normalization, no hfds anomalies, native pointwise workaround,
NCCL_P2P_DISABLE=1 inside the container, two RTX GPUs/arm, batch4/rank×accumulation4=effective32,
seed15, four ensemble members and four autoregressive calls during training. Online W&B required.
Save raw every two epochs plus latest/EMA/best; 14-hour job ceiling, no automatic requeue. Recovery
must set finetune=false and resume the new run's checkpoint so it does not reset EMA or Adam again.
Expected training cost approximately40 RTX GPU-hours combined plus diagnostics/evaluation.

Resolved configs [F](F.yaml) and [M](M.yaml), [training harness](train_torch.sbatch),
[diagnostic harness](diagnose_torch.sbatch). Remote campaign remains
`/scratch/jr7309/runs/score-multitask-1deg-20261012`; production outputs
`outputs/adaptation-F-forecast` and `outputs/adaptation-M-forecast`. Scratch preflight:4.71TB/5TB
(94.20%), approximately0.29TB available; data/SIF/W&B credentials checked without printing secrets.

Before production, qualify the modified EMA loading and raw/EMA annual validation on real data.
Local checks cover resetting EMA only on fresh fine-tune and restoring it on resume, the LR sequence,
and existing multitask masks/gradients. After training, use the established fixed held-out forecasts,
completion retention and continuous salinity diagnostics to compare F/M with original/cooldown results.
Interpret short-lead temperature/salinity separately from annual all-channel aggregate scores.
