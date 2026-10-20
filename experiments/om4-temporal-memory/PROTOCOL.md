# OM4 temporal-memory experiment

Requested: October 16, 2026. Four new runs start from matched random initialization.
No scientific result is claimed until the corresponding job and artifacts are verified.

## Provenance

- Official evaluation: `uw49dzuq`, training run `3o112ot4`.
- Checkpoint: `nyu-osn-emulators:emulators/jr7309/outputs/2026-01-29-samudra_om4_onedeg_20_cap/saved_nets/ema_ckpt.pt`.
- Checkpoint SHA-256: `ff6160cb7e2a2b5b8ee2c5731c16bd114a1bb5a267245cc5ed766fb00c7f797a`.
- Official source revision: `a217a1c20117d7992375fd6d511581dd1a3b00ea`.
  Three architecture modules in `legacy/` are copied from that revision, with imports adapted.
- Engaging data: `/orcd/data/abodner/002/jrusak/om4_onedeg_v3`.
  Raw first-input U/T/S/SSH/hfds samples match archive SHA-256 bytes. Metadata matches the official evaluation's November 26, 2025 conservative, unfiltered Gaussian 180×360 source.
- Physical channels: U, V, temperature, salinity at 19 depths, then SSH (77).
- Forcings: tauuo, tauvo, hfds, hfds_anomalies. The last is reconstructed over the entire
  original source using the archived evaluator's climatology and normalization definition.
- FP32, BatchNorm, two input and two output timestamps per call; four-call training horizon.
  These settings deliberately follow the official **1°** model, not the different ¼° configuration.

Engaging local normalization files differ from the archive. The first CPU replay rejected them
(normalized RMSE 0.3602 in local units). A controlled local CPU replay using frozen archive
statistics reduced disagreement to 0.000372 in archive-normalized units. These statistics
are saved as `archive-normalization.json`; local shared stores are not modified.
The remaining cross-environment discrepancy is reported, not assumed to be bit equality.

## Qualification and diagnostics

1. Strict checkpoint loading, followed by numerical comparison of both first-call outputs
   against all 77 archived physical prediction fields. Cross-device normalized RMSE must be <0.001.
   Report the actual errors; threshold passage does not imply bit equality.
2. Six seasonal held-out initializations. Intervene on the first predicted two-timestamp block.
   Remove 0.25, 0.5, or 1 times the local high-frequency **error** component, separately in
   surface/deep T, surface/deep S, SSH, deep U and deep V. Deep means depths >=1050 m.
   Continue up to 12 calls (120 days from the last initial input).
3. Controls: unmodified forecasts, three independently phase-randomized equal-plane-norm
   perturbations per group, truth restart, and applying the same error-derived perturbation to truth.
   Longitude wraps; latitude edges and any stencil touching land are excluded.
4. At the first predicted block, differentiate its own loss and the sum of the next three
   calls' losses separately. Report projections onto error texture and gradient cosines,
   under fixed MSE and the checkpoint's archived adaptive loss weights.
   Use evaluation mode: these are checkpoint sensitivities, not recorded training gradients.
5. Run one actual optimizer update of each training arm, then verify online W&B access.
   Production training depends on successful qualification and full data preparation.

Opposed gradient signs and correction worsening later skill support a useful-distortion
mechanism. They do not alone establish storage of historical information or its origin in BPTT.

## Matched training

| Arm | Persistent latent state | Temporal gradient through physical feedback | Through latent feedback |
|---|---|---|---|
| full | No | Yes | N/A |
| detach | No | No | N/A |
| latent_full | Yes | Yes | Yes |
| latent_only | Yes | No | Yes |

- One paired seed (15), 70 epochs, identical shuffled windows and effective batch 32.
- Physical weights and buffers are exactly equal at initialization across all arms.
- Fixed normalized wet-cell MSE: average wet cells within each channel, then equal channel
  weighting and equal weighting of the four calls. No adaptive channel weights.
  This wet-cell reduction differs from the old zero-filled full-grid reduction; it is shared
  by all new arms and recorded explicitly. The official checkpoint remains a diagnostic reference.
- Adam, LR 0.0006, epoch-wise cosine decay, EMA 0.999 with faster startup.
- Two-sample microbatches with gradient accumulation. BatchNorm sees that microbatch,
  which differs from the original distributed training; all new arms match each other.
- Training windows entirely inside 1975-01-03 to 2013-10-05 (exclusive upper bound).
- Select EMA checkpoint by 360-day normalized validation MSE from the 2013-10-05
  initialization. The full forecast stays within the validation year ending 2014-10-05.
  Record per-field RMSE and local roughness at calls 1, 3, 9, and 36.
- Latents: 16 spatial channels at the bottleneck; tanh recurrent update from bottleneck
  features and previous latent state, projected additively back into bottleneck features.
  Zero initial readout preserves the physical-only initial forward function.
- Latent state is passed functionally, reset between training windows, and retained for the
  full validation rollout. Checkpoint recomputation cannot mutate it. BatchNorm running
  statistics are restored after checkpoint replay to avoid duplicate updates.
- Save resumable optimizer/model/EMA checkpoints every 20 updates and at epoch boundaries.
  Deadline exits preserve the next effective-batch index. All arms use the same completed
  epoch exposure; partial runs are never treated as a final matched comparison.

## Reporting and limitations

`report.py` produces diagnostic tables, CSVs, PNG/PDF plots, and a Markdown report.
The training comparison must wait for all four requested arms, use the same held-out
origins and horizons, and report both forecast error and texture. One seed is exploratory.
Latent arms add parameters; this four-arm experiment does not include a nonpersistent
equal-parameter control. Improvement alone therefore does not prove a memory mechanism.

CPU tests cover exact archived-backbone equivalence, paired initialization, real gradient
connectivity in every arm, checkpoint/BatchNorm equivalence, and masking of the texture operator.
