# Four-arm temporal-memory experiment

All four arms completed 70 epochs and final evaluations on Engaging. One paired random seed (15), fixed normalized wet-cell MSE, 1° OM4, four autoregressive calls with two timestamps per call. EMA checkpoint selection used the lowest 360-day validation MSE; held-out evaluation used six origins and 72 calls.

## Exact training treatments

These are four **new models trained from matched random initialization**, not fine-tunes of the official checkpoint. The official v2 1° checkpoint is used only in the separate intervention/gradient diagnostics. The implementation is in [common.py](../../common.py) (`OceanModel._features`, `forward`, and `feedback`) and [train.py](../../train.py) (the four-call training loop).

### Shared forward task and objective

Each sample contains ten consecutive five-day timestamps. Let `p[j]` denote the normalized 77-channel physical state at timestamp `j`, and `b[j]` the four prescribed forcing channels. Each network call consumes two physical states and the forcings at those same two timestamps, and predicts the next two physical states jointly:

| Call | Physical input | Prescribed forcing input | Supervised target |
|---|---|---|---|
| 1 | Truth `p[0], p[1]` | `b[0], b[1]` | `p[2], p[3]` |
| 2 | Call 1 predictions | `b[2], b[3]` | `p[4], p[5]` |
| 3 | Call 2 predictions | `b[4], b[5]` | `p[6], p[7]` |
| 4 | Call 3 predictions | `b[6], b[7]` | `p[8], p[9]` |

Thus training covers leads 5–40 days from the last initial input. Both predicted timestamps are fed back together. Calls 2–4 consume model predictions in **every** arm: detachment does not replace them with truth. There is no recurrent transition between the two outputs within a call, and no teacher forcing after the initial pair. Weights are shared across all four calls; the decoder predicts absolute normalized states rather than state increments.

The physical channels are U, V, temperature, and salinity at each of 19 depths (2.5–6000 m), plus SSH. Forcings are `tauuo`, `tauvo`, `hfds`, and `hfds_anomalies`. The input has 162 channels (154 physical + 8 forcing); the output has 154 physical channels. Depth-dependent dry cells are zero-filled and output-masked; masks are not additional input channels. Frozen archive means/standard deviations are used for normalization.

For each call, squared normalized errors are averaged over wet cells separately for each physical channel and output timestamp, then equally over those 154 channel/timestamp entries and the batch. The training loss is the arithmetic mean of the four call losses. There are no adaptive channel weights, latent targets, latent penalties, or texture penalties. Each call's physical output receives its own direct supervised loss, including in the detached arms.

### What differs between arms

`stopgrad(x)` preserves the values of `x` but blocks derivatives through that tensor. It is applied **only when handing a physical prediction to the next call**, after computing the current call's loss.

| Arm | Physical input passed to the next call | Latent state passed to the next call |
|---|---|---|
| `full` | `prediction` (gradient connected) | None |
| `detach` | `stopgrad(prediction)` | None |
| `latent_full` | `prediction` (gradient connected) | `h` (gradient connected) |
| `latent_only` | `stopgrad(prediction)` | `h` (gradient connected) |

In `detach`, later losses cannot backpropagate into earlier calls through their physical predictions. They still train the shared network parameters through their own calls. The four losses are summed before one backward pass per microbatch; this is not four independent optimizer updates or single-step training.

In `latent_only`, **only the cross-call physical connection is detached**. Gradients within each call, including the encoder, decoder, latent update, and latent readout, remain enabled. A later loss can reach an earlier call's features and parameters through the carried latent state. This does not mean only latent parameters are trained.

### Exact latent mechanism

Both latent arms add the same recurrent module immediately after the U-Net middle block, before the first upsampling. Let `z[k]` be that call's 520-channel bottleneck features and `h[k-1]` the previous 16-channel latent state. At the 180×360 input resolution their spatial grid is 11×22. The update and readout are:

```text
h[k]     = tanh(Conv1x1_update(concat(z[k], h[k-1])))
z_aug[k] = z[k] + Conv1x1_read(h[k])
```

The update maps 536 channels to 16 and has a bias; the readout maps 16 to 520 without a bias. The decoder consumes `z_aug[k]`, and the next call receives `h[k]`. There is one latent state per two-timestamp call, with no explicit ocean mask on the bottleneck state. The module adds 16,912 trainable parameters. Its readout weights start at zero; the update uses PyTorch's default convolution initialization. Physical model parameters are initialized first and are identical across all four arms, so adding latents leaves the initial physical forward function unchanged.

Latent state starts at zero for each training window and each independent validation/test origin. It is carried without detachment through all four training calls and, at inference, through the entire forecast. It is never carried between unrelated training samples or batches. The separate **reset control** changes inference only: it zeros the latent state before every call while retaining autoregressive physical predictions and the trained weights. It is not a fifth training treatment.

### Other matched settings

All arms use seed 15, the same chronological data split, identical epoch-specific shuffled windows, 70 epochs, and the archived v2 1° ConvNeXt U-Net with widths 280/380/480/520 and BatchNorm. Effective batch size is 32, accumulated from sixteen microbatches of two; BatchNorm therefore sees two samples at a time. Training uses FP32 with TF32 disabled, Adam (default betas/epsilon, no weight decay), and no gradient clipping. The epoch-indexed learning rate is `0.0006 * (1 + cos(pi * epoch / 70)) / 2`, for zero-based epochs 0–69: no warmup or separate cooldown, and the last trained epoch has a small positive learning rate.

EMA uses `min(0.999, (1 + updates)/(10 + updates))` as its decay, including floating-point BatchNorm buffers. Activation-checkpoint recomputation restores BatchNorm buffers to avoid counting the replay as a second statistics update. Each arm independently selects its EMA checkpoint by the same 360-day validation criterion; the selected epoch is not forced to match across arms. See the [protocol](../../PROTOCOL.md) and accompanying per-arm configurations for data bounds and the evaluation schedule.

## Follow-up experiments

Two full-resolution latent arms are scheduled. The [selected-versus-epoch-70 EMA map comparison](../checkpoint-comparison/REPORT.md) is complete for the original four arms. Their exact definitions and evaluation plan are in the [protocol extension](../../PROTOCOL.md#extension-full-resolution-physical-like-latent-channels-october-19). The results below remain the completed original four-arm comparison; no fine-latent result is claimed yet.

## Held-out results

Each number averages all 77 channels and both timestamps in the indicated block, then six origins. Day 720 denotes the days 715/720 block, not a cumulative loss.

| Arm | Selected epoch | Days 5/10 | Days 355/360 | Days 715/720 | Change vs full at 720 |
|---|---:|---:|---:|---:|---:|
| Full temporal gradients | 29 | 0.051337 | 0.200620 | 0.212812 | +0.00% |
| Detach physical feedback | 30 | 0.051144 | 0.162619 | 0.168480 | -20.83% |
| Latents + full gradients | 31 | 0.050895 | 0.199886 | 0.212149 | -0.31% |
| Latents-only temporal gradients | 31 | 0.050549 | 0.161990 | 0.167137 | -21.46% |

![Comparison and validation curves](comparison.png)

## First-block high-frequency error

Percentage change in normalized high-pass error MSE versus full gradients, averaged across group channels and six origins. The stencil is center minus its four-neighbor mean on all-wet interiors; it is not a sharp spectral band.

| Arm | Deep temperature | Deep salinity | SSH |
|---|---:|---:|---:|
| Full temporal gradients | +0.0% | +0.0% | +0.0% |
| Detach physical feedback | -28.0% | -33.2% | -34.3% |
| Latents + full gradients | -5.3% | -5.5% | -5.0% |
| Latents-only temporal gradients | -29.9% | -35.5% | -36.6% |

## Latent reset control

Resetting latent state to zero before every forecast call leaves the first call unchanged, but increases day-720 MSE:

- latent_full: 0.212149 → 0.218937 (+3.20%).
- latent_only: 0.167137 → 0.213829 (+27.94%).

## Interpretation and limits

Detaching physical feedback beats full gradients at day720 in 6/6 paired origins. Latent-only temporal gradients improve mean day720 MSE by 0.80% relative to detachment alone. That small difference needs replication; six correlated ocean origins do not substitute for independent training seeds.

The results support a harmful contribution from temporal gradients through physical outputs in this setup. They do not prove that the network deliberately stores historical information in texture. The single-example experiment produced residual texture without temporal feedback, so BPTT is not necessary for all such artifacts. The reset control shows functional dependence on carried latent state, but resetting it also creates an out-of-training-distribution intervention and does not establish what information it encodes.

Checkpoints selected at epochs29–31, although every arm completed70epochs; this is the predeclared validation-selected comparison, not final-epoch weights. Latent arms have extra parameters and no matched nonpersistent-parameter control. Batch size/normalization and fixed-MSE training differ from the official dynamic-loss training. No claim of statistical significance or generality to quarter-degree models is made.

## Provenance

Original array26067267 and scratch continuation26280150 all completed. Runtime Python3.12/PyTorch2.9.1+cu128, one L40S per arm, FP32. Code snapshot code-v4; scratch root ~/orcd/scratch/om4-temporal-memory-20261016. Per-arm configs,70validation records, heldout metrics, and selected-checkpoint SHA256 values accompany this report. No GPU execution was used to generate this report.

- [full W&B](https://wandb.ai/ocean_emulators/default/runs/om4mem-full-15-20261016); selected checkpoint SHA256 `b8b7ba147316f470c277dd550ff4354ef5c649cf405380a5cc0985bfe188d2df`.
- [detach W&B](https://wandb.ai/ocean_emulators/default/runs/om4mem-detach-15-20261016); selected checkpoint SHA256 `1520f529f46bedb32b88feccb029dfcc6ffdda3b06472ead9b4558c143c65d59`.
- [latent_full W&B](https://wandb.ai/ocean_emulators/default/runs/om4mem-latent_full-15-20261016); selected checkpoint SHA256 `7b3206ecd33ae55e87c5829ca8695905e153ce469eae1a999022fc9b1a819cf8`.
- [latent_only W&B](https://wandb.ai/ocean_emulators/default/runs/om4mem-latent_only-15-20261016); selected checkpoint SHA256 `535fb8faeb04299476bc8a4f8d6b4343a10c763d8c516f65debc8db350b9d262`.
