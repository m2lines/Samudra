# Current-preset OM4 structure-loss experiment

This is an experiment/results record, not intended to merge.

Base: Samudra main `f1e713220`. All three arms use the literal current
`samudra_om4/train.yaml` preset, seed 15, 1-degree v3 data, and the same
initialization and training schedule. Differences are confined to the loss:

- Local: MSE weighted by frozen local priorities bounded to [0.95, 1.10],
  with wet-cell mean one separately for every variable/depth.
- Spatial: MSE plus 100 times interior curvature-of-error MSE.
- Gradient: MSE plus 0.1 times horizontal forward-difference L1 error,
  following Cody's `llc_cpu_working` implementation at `2651acfff`.
  This uses MSE alone as the base, and omits the vertical term.

All objectives return per-output-channel values for the standard trainer.
Two output times repeat the same 77-channel mask/weight ordering.
MSE and curvature retain main's full-grid denominator. The gradient term
uses wet starting cells, includes wet-to-land differences, wraps longitude,
and excludes the final latitude difference. Local weights alter loss only,
not input/output normalization. `check_losses.py` verifies independent
loop-based references, exact-target zero loss, land gradients, repeated
output channels, and finite-difference gradients on CPU.

Local scales are regenerated from the training store's own means/stds and
32 sampled adjacent five-day pairs strictly within the training period.
They combine spatial variance and temporal change MSE using their maximum,
with a 0.02 normalized floor and approximately 300 km Gaussian smoothing.
The inverse standard-deviation weights are bounded to [0.5, 2] with a
per-channel mean-one solve, then shrunk 90% toward one. The archive records
selected dates, normalization, weights, masks, and checksum.

The current-model single-example qualification (Torch 20437457, all six
arms COMPLETED 0:0) showed relative state/high-pass MSE versus ordinary MSE:
local mild10 1.059/1.039; curvature100 0.374/0.0123; horizontal gradient
0.655/0.546. Local weighting is stable but has no demonstrated improvement.
These are memorization results, not forecast-skill claims.

## Execution

Set ROOT to a new personal-scratch run directory, DATA_ROOT to the selected
1-degree data root, and SIF to the verified runtime. Stage code under ROOT/code
and write a manifest.sha256 covering that snapshot. Submit prepare.sbatch;
it computes scales and validates six generated configs.

Pilot configs run one short training epoch with normal four-call
backpropagation, two output times, online W&B, and 360-day validation.
Full configs retain 70 epochs. All use checkpointing each epoch and
preemptible resume. Submit train.sbatch with MODE=pilot first. Full jobs
require an explicit PILOTS_VERIFIED file after checking finite training,
validation, online W&B, and checkpoints. No full jobs have been submitted.

2026-10-20: CPU preparation job 20439512 submitted to Torch; dependent
callback 20439565. Durable follow-up c25ded47-860e-4350-a65e-e39bac965379
checks preparation and continues the pilot/full-training sequence.

Existing recent MSE and dynamic-MSE controls differ in data version,
gradient accumulation, forcing configuration, and loader; treat them as
contextual controls, not strictly matched loss-only comparisons.

Preparation 20439512 completed 0:0 in 1m47s. All 77 channel weight means
and all sampled training-date bounds verified. Pilot array 20441707
(local/spatial/gradient, one concurrent RTX6000) submitted with preemptible
routing and requeue enabled. Dependent callback 20441724 and durable
30-minute reminder 388bf327-13c4-4598-a543-751235619040 cover bring-up and
completion. Preparation callback's outbound HTTPS failed; the local
reminder recovered status. Full training remains gated on pilot results.
