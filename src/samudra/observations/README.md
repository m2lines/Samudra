<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation training

`samudra train-obs` trains an ocean model whose forecast state is initialized from
surface observations.
It uses Samudra's YAML configuration, `!include`, bundled presets and dotted CLI
overrides. The model, data preparation and monthly replay logic live under a
separate entrypoint because observation targets have different sampling cadences.

```bash
samudra train-obs observations/train.yaml \
  --experiment.data_root /data --experiment.base_output_dir /runs \
  --experiment.name obs-baseline --experiment.wandb.mode online
# The module entrypoint accepts the same options:
python -m samudra.observations.train observations/train.yaml \
  --experiment.data_root /data --pretrain_mix om4-cm4-obs
```

The preset includes [data paths](../configs/observations/data.yaml) and a
[model definition](../configs/observations/model.yaml). Paths resolve through
`experiment.data_root`. Choose `obs`, `om4-obs`, `om4-cm4-obs` or `om4-cm4` with
`--pretrain_mix`. Each recipe has a main phase followed by observation-only post-training.
The `om4-cm4` main phase alternates the two model sources. `scheduler` controls
the main phase for every recipe; `post_training_scheduler` controls the observation-only post-training phase.
They are separate so post-training can use its own LR curve after any main-phase
schedule, while retaining optimizer moments. Both use the shared Samudra scheduler
configs. Scheduler steps count optimizer updates here; each receives its phase’s
update budget. Post-training starts from `post_training_learning_rate`, which uses
`learning_rate` when omitted. Scheduler state resumes with the optimizer.

Update budgets, learning rates, accumulation, fresh/replay counts, replay capacity
and age, EMA, validation cadence and checkpoint cadence are configurable. Budgets,
cadences and replay age count optimizer updates. Fresh/replay counts are examples
per update and sum to `gradient_accumulation_steps`. Replay capacity is saved state
pairs per forecast lead month, counted from the original initialization. Each microbatch contains
one monthly window; the effective batch is `gradient_accumulation_steps`. A fresh
example initializes from surface history; replay advances a detached saved state
through its next target month. Channel layout and calendar
operators remain code-level conventions. See [config.py](config.py) for defaults
and constraints and [training.py](training.py) for the loss and update loop.

```bash
samudra train-obs observations/train.yaml \
  --experiment.data_root /data --main_updates 4000 --post_training_updates 500 \
  --gradient_accumulation_steps 4 --fresh_examples 2 --replay.max_age 100
```

Outputs follow `experiment.base_output_dir/experiment.name`, with the resolved
`config.yaml`, a code/data/recipe manifest, local `metrics.jsonl` and W&B records.
Checkpoints use `saved_nets/ckpt.pt` and `saved_nets/best_validation_ckpt.pt`.
The phase endpoints are `main.pt` and `endpoint.pt` in that same directory.
An identical invocation resumes the latest checkpoint. `--resume_ckpt_path`
selects another trusted checkpoint. `--stop_after` limits execution without
changing the scientific update budget. SIGTERM and SIGUSR1 save at an update
boundary. Resume checks the scientific configuration and data/code identities.

Each validation evaluates a one-year forecast with one observation initialization, prescribed ERA5 forcing,
and Samudra's area-weighted RMSE kernel. Observation-training standard deviations
normalize the scores. Surface and monthly profile scores receive equal channel
weight for each individual observed channel after their respective time reductions.
The EMA checkpoint with the lowest aggregate score is selected across both phases.

## Data preparation

Use the [data environment](../../../data/README.md#usage-via-command-line-interface)
with `PYTHONPATH=src`. The top-level scripts
consume OISST, DUACS, IAP and ERA5 stores and write a common grid, daily shards,
monthly samples and a separate normalization artifact:

```bash
PYTHONPATH=src python data/prepare_observation_baseline.py grid \
  --om4 /data/om4_onedeg_v3/OM4.zarr --output /data/obs/grid.npz
PYTHONPATH=src python data/prepare_observation_baseline.py daily \
  --source /data/products --grid /data/obs/grid.npz --output /data/obs/daily \
  --product sst --year 1993
# Repeat for sst, adt, era5 and iap, and years needed by the observation splits.
PYTHONPATH=src python data/prepare_observation_baseline.py months --root /data/obs
PYTHONPATH=src python data/prepare_observation_normalization.py /data/obs \
  --om4 /data/om4_onedeg_v3
```

The splits are defined in [archive.py](archive.py). Grid export reads `lat_b` and
`lon_b` cell boundaries from the prepared OM4 store. For a store without those
coordinates, pass `grid --grid-geometry /data/target-grid.nc` using the grid file
from its preparation. Daily preparation similarly accepts
`--source-grid /data/product-grid.nc` when the product store lacks boundaries.
These geometry files contain `lat`, `lon`, `lat_b` and `lon_b`; their centers
must match the corresponding data store.

The model and observation preparation paths share the ESMF conservative-weight
builder. Normalization writes `observation_means.zarr` and
`observation_stds.zarr`, with one mean and standard deviation per named state
channel, including depth-specific channels. Observed
temperature, salinity and sea-surface height (T/S/SSH) statistics come from
finite-valued observations over ocean cells globally in the training split. Unobserved fields use the matching means/stds
from `OM4_means.zarr` and
`OM4_stds.zarr`: all 3-D velocities and T/S channels below observation coverage.
OISST supplies the mean and standard deviation for the surface-temperature
state channel. All sources use this state normalization. Model wind stress and heat flux use shared OM4 forcing statistics.
`statistics.npz` holds the atmosphere/profile input statistics and surface
climatology; `normalization.json` records each field’s observation or OM4 source
and hashes the observation samples, OM4 statistics and generated artifacts.
Retain daily shards so replay can reconstruct five-day forcing and surface-target
intervals from the forecast’s original initialization date.

OM4 initializer histories may span missing days. Forecast and replay transitions
must remain contiguous, including the final history frame to the first target.
CM4 retains its native no-leap calendar. Source calendars and masks are preserved.

## Existing launch scripts

The existing Apptainer launcher accepts this module through `SAMUDRA_MODULE`:

```bash
CONFIG=src/samudra/configs/observations/train.yaml \
NAME=obs-baseline DATA_ROOT=/data OUTPUT_BASE=/runs WANDB_MODE=online \
SAMUDRA_MODULE=samudra.observations.train GPUS_PER_NODE=1 \
sbatch --nodes=1 --gres=gpu:rtx6000:1 --cpus-per-task=16 --mem=64G \
  scripts/slurm_apptainer_train.sbatch
```

Set the container and site-specific resource settings as for other Samudra runs.
The observation entrypoint requires one process per run; the launcher rejects
multi-node or multi-GPU requests for this module before spawning workers.

## Shared components and remaining duplication

The entrypoint directly uses `TopLevelConfig`, `ExperimentConfig`, `WandBConfig`,
`Location`, `init_train_backend`, logging/warning setup, seed setup,
`CheckpointPaths`, `WandBLogger`, `EMATracker`, scheduler configs,
`UNetBackboneConfig` and `BlockConfig`. The model config controls padding;
`core_block.instance_norm_affine` controls affine InstanceNorm parameters.
The following code remains local for later discussion:

| Local code | Why sharing requires more work |
| --- | --- |
| Source loading and monthly training loop | `Trainer` consumes epoch-based dense-field loaders; observations combine daily surface data, monthly profiles and detached replay states. |
| Checkpoint payload and resume | Replay pools, original ages, source exposure and sampling RNG need to be saved together. `Trainer` owns its checkpoint format internally. Multi-rank replay would also need rank-specific ownership. |
| Annual validation orchestration | Existing aggregators expect dense fields at a common cadence. The observation evaluator shares the spatial RMSE kernel and implements the cadence reductions locally. |
| Optimizer construction | `Trainer` constructs Adam inline. This path constructs AdamW; both use the shared scheduler configs. |
| W&B run identity setup | `WandBLogger.setup_run` requires a `DataBundle` and the standard trainer's checkpoint keys. Logging and run lifecycle use the shared logger. |

Tests cover config loading/overrides, source exposure, replay, exact resume,
calendar gaps, normalization and annual scoring. The default model matches the parameter layout and seeded initialization of the
[U-global prototype](https://github.com/m2lines/Samudra/blob/d67e0b523ea651b55447dc3a7a9b0010ffcae35f/src/samudra/models/surface_initialized.py).
