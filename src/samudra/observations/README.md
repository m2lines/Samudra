<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation training

`samudra train-obs` trains an ocean model that starts each forecast from surface
observations. An initializer turns 95 days of surface history into a full ocean
state, and an autoregressive processor advances that state in five-day steps.

Observation training has its own entrypoint because its targets arrive at
different cadences: satellite surfaces daily and subsurface profiles monthly. It
still uses Samudra's YAML configuration, `!include`, bundled presets and dotted
command-line overrides.

```bash
samudra train-obs observations/train.yaml \
  --experiment.data_root /data --experiment.base_output_dir /runs \
  --experiment.name obs-baseline --experiment.wandb.mode online
# The module entrypoint accepts the same options:
python -m samudra.observations.train observations/train.yaml \
  --experiment.data_root /data --pretrain_mix om4-cm4-obs
```

The preset includes the [data paths](../configs/observations/data.yaml) and the
[model definition](../configs/observations/model.yaml). Data paths are relative to
`experiment.data_root`.

## How training works

Training draws examples from up to three sources: observations (`obs`), the OM4
ocean simulation (`om4`) and the CM4 climate model (`cm4`). Every example in an
optimizer update comes from the same source. `--pretrain_mix` sets the order in
which sources take turns during the main phase:

| `--pretrain_mix` | Main-phase cycle |
| --- | --- |
| `obs` | obs |
| `om4-obs` | obs, om4 |
| `om4-cm4-obs` | obs, om4, obs, cm4 |
| `om4-cm4` | om4, cm4 |

A post-training phase on observations alone follows the main phase in every recipe.

![One fresh observation example: the initializer turns 19 five-day frames of observed SST and SSH into two ocean states, and the processor advances them with ERA5 forcing through about six five-day steps. Surface losses compare each step with observations; the calendar-month mean of predicted T/S is compared with IAP.](../../../docs/static/assets/obs-training.png)

*One fresh observation example. OM4 and CM4 examples compare against full model fields instead, and replay examples start from saved states rather than the initializer.*

Each example covers one target month, so the effective batch size is
`gradient_accumulation_steps`. The first `fresh_examples` of each update are
*fresh*: the model initializes from surface history and forecasts the first month.
The rest are *replay* examples. A replay example takes the saved, detached state
from an earlier forecast and continues it through its next month. That lets
training reach long lead times without backpropagating through the whole year. If
no saved state is eligible, the example is fresh instead.

After every example, the state at the start of the following month goes back into
that source's replay pool. The pool keeps `replay.per_month` states for each lead
month, counted from the forecast's original initialization, and evicts the oldest
first. It drops states whose original initialization happened more than
`replay.max_age` updates ago.

The loss combines the forecast error with two initializer terms. One is
reconstruction of the ocean interior from surface history; the other is
completion of surface cells that were hidden from the initializer.
[training.py](training.py) has the weights and the update loop.

### Configuration

Update budgets, learning rates, gradient accumulation, the fresh/replay split,
replay capacity and age, EMA, validation cadence and checkpoint cadence are all
configurable. Budgets, cadences and `replay.max_age` count optimizer updates.
[config.py](config.py) lists defaults and constraints. Channel layout and
calendar operators are fixed in code.

```bash
samudra train-obs observations/train.yaml \
  --experiment.data_root /data --main_updates 4000 --post_training_updates 500 \
  --gradient_accumulation_steps 4 --fresh_examples 2 --replay.max_age 100
```

`scheduler` sets the learning-rate schedule for the main phase.
`post_training_scheduler` sets it for post-training, which keeps the optimizer's
moments but can follow its own curve after any main-phase schedule. Both use
Samudra's shared scheduler configs, step once per optimizer update, and receive
their phase's update budget. Post-training starts at `post_training_learning_rate`,
or at `learning_rate` if that is unset. Scheduler state is saved and resumed with
the optimizer.

### Outputs and resuming

A run writes to `experiment.base_output_dir/experiment.name`. That directory holds
the resolved `config.yaml`, a manifest of the code, data and recipe, and the
metrics as `metrics.jsonl` (also sent to W&B). Checkpoints go in `saved_nets/`:

- `ckpt.pt`: the latest state.
- `best_validation_ckpt.pt`: the best validation score so far.
- `main.pt` and `endpoint.pt`: the ends of the main phase and of post-training.

Running the same command again resumes from the latest checkpoint. Before
resuming, the run checks that the scientific configuration and the code and data
identities match the manifest. `--resume_ckpt_path` resumes from another trusted
checkpoint. `--stop_after` stops early without changing the planned number of
updates. SIGTERM and SIGUSR1 make the run save and exit at the next update
boundary.

### Validation

Each validation runs a one-year forecast from a single observation initialization,
with ERA5 forcing prescribed. Scores use Samudra's area-weighted RMSE kernel and
are divided by the observation-training standard deviations. Surface channels are
scored at five-day steps; profile channels are averaged into calendar months
first. Every observed channel then counts equally in the aggregate score.
Validation uses the EMA weights, and the checkpoint with the lowest score across
both phases is kept as the best.

## Data preparation

Use the [data environment](../../../data/README.md#usage-via-command-line-interface)
(Python 3.12) with Samudra installed from the same checkout. The environment file
installs both packages. To add them to an existing environment, run
`python -m pip install -e . -e "./data[observations]"` from the repository root.

The preparation scripts read OISST, DUACS, IAP and ERA5 stores. They write a
common grid, daily shards, monthly samples and a separate normalization artifact:

```bash
python data/prepare_observation_baseline.py grid \
  --om4 /data/om4_onedeg_v3/OM4.zarr --output /data/obs/grid.npz
python data/prepare_observation_baseline.py daily \
  --source /data/products --grid /data/obs/grid.npz --output /data/obs/daily \
  --product sst --year 1993
# Repeat for sst, adt, era5 and iap, for every year the observation splits need.
python data/prepare_observation_baseline.py months --root /data/obs
python data/prepare_observation_normalization.py /data/obs \
  --om4 /data/om4_onedeg_v3
```

[archive.py](archive.py) defines the splits. Keep the daily shards after building
the monthly samples: replay rebuilds its five-day forcing and surface targets from
them, counting from each forecast's original initialization date.

### Grid geometry

Grid export reads the `lat_b` and `lon_b` cell boundaries from the prepared OM4
store. If the store lacks them, pass `grid --grid-geometry /data/target-grid.nc`,
using the grid file from that store's preparation. Likewise, `daily` accepts
`--source-grid /data/product-grid.nc` when a product store lacks boundaries. A
geometry file contains `lat`, `lon`, `lat_b` and `lon_b`, and its cell centers
must match the corresponding data store. Model and observation preparation share
the same ESMF conservative-regridding weight builder.

### Normalization

Normalization writes `observation_means.zarr` and `observation_stds.zarr`, with a
mean and standard deviation for each named state channel, depth levels included.
All sources use these state statistics.

- Observed temperature, salinity and sea-surface height take their statistics
  from finite observations over ocean cells in the training split. OISST supplies
  the surface-temperature channel.
- Fields that are never observed take the matching values from `OM4_means.zarr`
  and `OM4_stds.zarr`. These are all velocities, and temperature and salinity
  below the depth that observations cover.
- Model wind stress and heat flux use shared OM4 forcing statistics.

`statistics.npz` holds the statistics for the atmosphere and profile inputs, plus
the surface climatology. `normalization.json` records whether each field's
statistics came from observations or OM4. It also hashes the observation samples,
the OM4 statistics and the generated artifacts.

### Calendars

Sources keep their own calendars and masks; CM4 keeps its no-leap calendar. OM4
initializer histories may span missing days, but forecast and replay steps must
be contiguous, including the step from the last history frame to the first
target.

## Running on Slurm

The existing Apptainer launcher runs this entrypoint when `SAMUDRA_MODULE` names it:

```bash
CONFIG=src/samudra/configs/observations/train.yaml \
NAME=obs-baseline DATA_ROOT=/data OUTPUT_BASE=/runs WANDB_MODE=online \
SAMUDRA_MODULE=samudra.observations.train GPUS_PER_NODE=1 \
sbatch --nodes=1 --gres=gpu:rtx6000:1 --cpus-per-task=16 --mem=64G \
  scripts/slurm_apptainer_train.sbatch
```

Set the container and site-specific resources as for other Samudra runs.
Observation training runs as a single process, and the launcher rejects
multi-node or multi-GPU requests for it before starting any workers.

## Shared components and remaining duplication

The entrypoint uses these Samudra components directly: `TopLevelConfig`,
`ExperimentConfig`, `WandBConfig`, `Location`, `init_train_backend`, logging and
warning setup, seed setup, `CheckpointPaths`, `WandBLogger`, `EMATracker`,
the scheduler configs, `UNetBackboneConfig` and `BlockConfig`. The model config
sets padding, and `core_block.instance_norm_affine` turns on affine InstanceNorm
parameters.

The following code stays local for now:

| Local code | Why sharing it needs more work |
| --- | --- |
| Source loading and the monthly training loop | `Trainer` consumes epoch-based loaders of dense fields. Observation training combines daily surface data, monthly profiles and detached replay states. |
| Checkpoint payload and resume | Replay pools, their original ages, source exposure counts and the sampling RNG must be saved together, but `Trainer` owns its checkpoint format. Replay across ranks would also need each rank to own its entries. |
| Annual validation | Existing aggregators expect dense fields at one cadence. The observation evaluator reuses the spatial RMSE kernel but does its own time reductions. |
| Optimizer construction | `Trainer` builds Adam inline; this path builds AdamW. Both use the shared scheduler configs. |
| W&B run setup | `WandBLogger.setup_run` needs a `DataBundle` and the standard trainer's checkpoint keys. Logging and the run lifecycle use the shared logger. |

Tests cover config loading and overrides, source exposure, replay, exact resume,
calendar gaps, normalization and annual scoring. The default model matches the
parameter layout and seeded initialization of the
[U-global prototype](https://github.com/m2lines/Samudra/blob/d67e0b523ea651b55447dc3a7a9b0010ffcae35f/src/samudra/models/surface_initialized.py).
