<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation/model-data baseline

Start with [`baseline.py`](baseline.py). It constructs the model, optimizer and
sources directly, then runs an explicit training, replay, validation and cooldown
loop. Scientific choices are named Python constants beside that code. The CLI
supplies data paths, output paths and runtime controls.

## Model and learning problem

[U-global](../models/surface_initialized.py) infers physical ocean states from
surface observations and atmospheric history, then advances those states
with prescribed forcing. The state contains velocity, temperature, salinity and
sea-surface height. Observation examples provide surface and monthly profile
supervision; model examples provide dense physical-field targets. Reconstruction
and artificial-gap completion train the initializer alongside forecasting.
The loss composition is visible in `baseline.objective`.

State normalization uses a shared artifact fitted over **global finite wet
support** in the observation training cohort. Observation, OM4 and CM4 states all
use those same means/stds. Conventions for velocity and depths beyond observation
coverage are explicit in
[`state_statistics`](../../../data/prepare_observation_normalization.py).

Forcing has separate variables and units. ERA5 inputs use their observation-fitted
statistics. Both model sources use the OM4 statistics for their native wind stress
and heat flux (`tauuo`, `tauvo`, `hfds`). Each source retains its own calendar and
wet mask.

## Prepare data

From the repository root, use the existing **data environment**, which supplies
xESMF/ESMF and the preprocessing dependencies. `PYTHONPATH=src` exposes the
checkout's lightweight preparation modules.

The top-level scripts consume the existing OISST, DUACS, IAP and ERA5 stores and
produce the common grid, daily shards, monthly samples and normalization artifact:

```bash
PYTHONPATH=src python data/prepare_observation_baseline.py grid \
  --om4 /data/om4_onedeg_v3/OM4.zarr --output /data/obs/grid.npz
PYTHONPATH=src python data/prepare_observation_baseline.py daily \
  --source /data/products --grid /data/obs/grid.npz --output /data/obs/daily \
  --product sst --year 1993
# Prepare each product/year needed by the splits in archive.py.
PYTHONPATH=src python data/prepare_observation_baseline.py months --root /data/obs
PYTHONPATH=src python data/prepare_observation_normalization.py /data/obs
```

Normalization can be written separately with `--output` and supplied to training
with `--normalization`. Retain daily shards: replay reconstructs intervals on the
original forecast-origin timeline. The normalization manifest records the global
fitting policy and input hashes; regenerate it when that policy changes.

OM4 and CM4 readers enforce the source splits and interval eligibility defined in
[`model_data.py`](model_data.py). Model windows require contiguous five-day
support. When an archive supplies only timestamps, the reader interprets them as
centers of five-day intervals.

## Train and inspect

Use the Samudra environment for training. Select `obs`, `om4-obs` or
`om4-cm4-obs` through `--mix`, supplying model paths for the selected sources:

```bash
python -m samudra.observations.baseline train \
  --observations /data/obs --om4 /data/om4_onedeg_v3 --cm4 /data/CM4.zarr \
  --mix om4-cm4-obs --output /runs/baseline
```

The recipe uses a fixed source mixture followed by an observation-only cooldown.
Each source has its own replay pool of detached forecast states. Fresh examples
and replay continuations share optimizer updates; replay advances by complete
calendar months while preserving five-day overlap. Source ratios, update budgets,
learning rates and replay limits are defined beside their use in the code.

The entrypoint runs in one training process. It uses Samudra's device selection,
autocast, logging, EMA and spatial RMSE utilities. Logs include local scalar
records and W&B metrics. `--stop-after` supports short checks, and restarting the
same command restores `latest.pt`, including optimizer, EMA, random streams and
replay. Checkpoint boundaries are optimizer updates.

Annual validation follows a continuous forecast and combines surface and profile
RMSE using their observation support. It averages times within each channel and
then channels equally, dividing errors by the shared observation-training standard
deviations. See [`evaluate.py`](evaluate.py) for the scoring contract and
`baseline.py` for the validation origin, cadence and selection key. Raw-unit
channel metrics accompany the selection score.

`best.pt` records the selected annual-validation model; `main.pt` and `endpoint.pt`
record the phase endpoints. Use the `evaluate` subcommand with `--checkpoint` to
score a saved model.

## Verification

Focused tests exercise calendar overlap, missingness, global normalization,
source exposure, replay, input isolation, checkpoint restoration and annual RMSE.
[`qualify_observation_baseline.py`](../../../scripts/qualify_observation_baseline.py)
checks these paths with the full model and real archives under a short schedule.
Its receipt records code/data identity, runtime and checkpoint hashes.

Run manifests identify source code, normalization inputs, daily-archive inventories
and model metadata. Preserve versioned input stores alongside these manifests.
