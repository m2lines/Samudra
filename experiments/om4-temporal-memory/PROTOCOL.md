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

## Extension: full-resolution physical-like latent channels (October 19)

User-requested arms `fine_latent` (display: fine-latent) and `fine_latent_detach`
(display: fine-latent-detach) retain the same seed, data, four-call exposure,
physical normalization/masking, loss, optimizer, batch size, 70-epoch schedule,
and EMA selection rule as the original four arms. These start from random
initialization, not from a coarse-latent or official checkpoint.

The state contains **16 latent channels per timestamp at 180×360**, or 32 channels
for the two-timestamp pair. It is concatenated after the physical/forcing inputs
before the first ConvNeXt block. Only that block's initial 3×3 convolution and
1×1 skip projection are widened; their original weights/biases and all other
physical weights are copied from the identically seeded physical model. The new
input slices start at zero, preserving the initial physical prediction function.
Hidden feature widths stay unchanged. A separate 3×3 head on the final decoder
features emits 32 latent channels; `tanh` bounds them to [-1,1], and the surface
wet mask zeros land in every latent channel. This head uses default convolution
initialization. It is unsupervised: there is no latent target or regularizer.
There is no direct latent skip or bottleneck recurrent update in these two arms.
The extra convolutions add 182,944 parameters. Both inputs and outputs have two
latent timestamps, while the earlier coarse treatment carried one 16-channel
11×22 state per call; this is therefore a physical-like state design comparison,
not an isolated spatial-resolution-only ablation or parameter-matched control.

At each call, the two latent outputs become the next call's two latent inputs,
alongside the two physical predictions. State starts at zero per training window
and evaluation origin and is retained for the whole rollout. In `fine_latent`,
both physical and latent feedback retain gradients. In `fine_latent_detach`,
only physical feedback is detached. All physical outputs still receive their
own loss, and every model parameter remains trainable. GPU qualification must
pass a full-size optimizer update, checkpoint/optimizer reload, and another
update for both arms before the production jobs can run.

## Extension: checkpoint comparison and spatial illustrations

Evaluate **both** `best-ema.pt` (`model` entry) and epoch-70 `latest.pt` (`ema`
entry, not its raw `model` weights) for every arm. Require the latter checkpoint
to have epoch=70 and next_group=0. Use the same six held-out origins, forcings,
72-call horizon, and metrics; keep these evaluations in separate directories
so the original endpoint is preserved. Report checkpoint file hashes and state
keys. Repeat the latent reset control for latent-bearing arms.

Save map subsets at calls 1,36,72 for origin 0. Initial figures focus on the first
predicted timestamp to inspect the immediate texture: T at 2.5/1850/4000m, S at
2.5/1850m, SSH, U and V at1850m. Show physical predictions, signed errors, and
high-pass errors for selected and final EMA weights on shared scales. Use global
maps plus fixed South Pacific (140–110°W,40–20°S) and North Pacific
(160–130°W,20–40°N) interior-gyre cutouts. These geographical boxes are chosen
before inspecting treatment results, not selected for a favorable difference;
report their measured truth variation/roughness and errors rather than asserting
they are uniformly quiet. No spatial regridding or display smoothing. Plot scales
use pooled 1st/99th percentiles across all shown arms/checkpoints, with saturation
explicitly marked. Single-origin maps supplement, not replace, six-origin scores.

## Standard observation-metrics and spectra extension (October 20)

For each original arm (`full`, `detach`, `latent_full`, `latent_only`) and both
selected/final EMA checkpoints, export one continuous physical rollout from the
October 10/15, 2014 input pair through the last available 2022 timestamp. This
extension uses the standard suite window **2015-01-01 through 2022-12-31**, eight
complete calendar years. It is separate from the earlier six-origin, 720-day
model-truth comparison. The standard window was stated as the default while an
optional user question about retaining the shorter multi-origin protocol was
pending; later user steering takes precedence.

`export_suite_rollout.py` writes the 77 normalized physical channels and retains
latent state throughout each rollout. The final odd timestamp is kept while the
second output beyond the source end is discarded. Checkpoint hashes, state keys,
initial timestamps, and output timestamps are recorded. Before marking an export
complete, the first 720 days' sampled MSE values must exactly reproduce the
previous same-checkpoint/origin evaluation. No latent-reset intervention is used.

`run_standard_suite.py` calls the repository's unchanged
`samudra.metrics.report.compute_observation_metrics` and
`samudra.viz.observations.spectra_figures`, from a staged copy of this checkout.
The adapter restores physical units, depth-dependent missing/dry cells as NaN,
native OM4 depth centers/thicknesses, and spherical cell areas computed with the
suite's native-coordinate fallback. The same OM4 truth trajectory is included as
a baseline. Standard metrics include geostrophic velocity/eddy energy, SST, OHC,
and residual-variance diagnostics; standard spectra cover surface EKE, SST
anomalies, and temporal geostrophic KE, including interannual bands. These are
observation comparisons, not replacements for the original normalized MSE or
local high-pass error. They do not by themselves provide deep-temperature spectra.

Required observation variables are staged once from the anonymously accessible
`m2lines-pubs/Samudra/v2026-07/obs` release on NYU OSN: DUACS `ugos/vgos`, OISST
`sst`, ARGO-IAP `temp`, restricted to October 2014–December 2022. Scoring and
spectral inputs are then restricted to 2015–2022. Record observation manifests,
source checksums, finite data coverage, metric tables, and spectral outputs.
Bootstrap uncertainty describes calendar-year variability, not training-seed
uncertainty. All scratch artifacts remain under the user's personal space.
