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

Set the container and site-specific resource settings as for other Samudra runs.
The legacy sequential path requires one process. Set `update_prefetch: true`
for DDP with a global example count divisible by the number of ranks. Resume
requires the same world size. On Torch RTX6000 nodes, export
`NCCL_P2P_DISABLE=1` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, including
the corresponding `APPTAINERENV_` variables; see [Torch guidance](../../../docs/torch.md#nccl-gotcha-on-rtx6000-nodes).

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
| Source loading and monthly training loop | `Trainer` consumes epoch-based dense-field loaders; observations combine daily surface data, monthly profiles and detached replay states. |
| Checkpoint payload and resume | Replay pools, original ages, source exposure and sampling RNG need to be saved together. `Trainer` owns its checkpoint format internally. Rank zero owns sample planning and replay pools; checkpoints include per-rank RNG and pending plans. |
| Annual validation orchestration | Existing aggregators expect dense fields at a common cadence. The observation evaluator shares the spatial RMSE kernel and implements the cadence reductions locally. |
| Optimizer construction | `Trainer` constructs Adam inline. This path constructs AdamW; both use the shared scheduler configs. |
| W&B run identity setup | `WandBLogger.setup_run` requires a `DataBundle` and the standard trainer's checkpoint keys. Logging and run lifecycle use the shared logger. |

Tests cover config loading/overrides, source exposure, replay, exact resume,
calendar gaps, normalization and annual scoring. The default model matches the parameter layout and seeded initialization of the
[U-global prototype](https://github.com/m2lines/Samudra/blob/d67e0b523ea651b55447dc3a7a9b0010ffcae35f/src/samudra/models/surface_initialized.py).

## Full deterministic model

The [mixed large preset](../configs/observations/deterministic_mixed_large.yaml)
contains the full Samudra evolution and initializer U-Nets, direct eight-field
ERA5 conditioning, source-separated replay, and observation-only cooldown.
It alternates OM4 and observation updates for 4,000 updates, then runs 1,000
observation-only cooldown updates. The global batch is eight monthly windows:
four ranks accumulate two examples each, normally four fresh and four replayed.
The preset enables the Rust readers and one-update asynchronous prefetch.

Install the Rust extra (`uv sync --extra rust`) and prepare the observation
archives and normalization as above. The daily product names are `sst`, `adt`,
`era5` and `iap`; prepare all four for observation years 1993–2014, including
history before the first training month (May 1993). The default native interval
cache spans starts from 1993-01-26 through 2014-12-27, each with five days of
support. OM4 requires `OM4.zarr`, `OM4_means.zarr` and `OM4_stds.zarr` in its
directory. The daily ERA5 archive must additionally cover the OM4 intervals
selected by `shared_era5.required_indices`; the cache builder rejects missing
support. Then prepare the exact-interval caches:

```bash
PYTHONPATH=src python data/prepare_observation_era5.py /data/obs /data/obs/daily /data/om4_onedeg_v3 /data/obs/era5-cache
PYTHONPATH=src python data/prepare_observation_native.py /data/obs/daily /data/obs/native-observations-v1
PYTHONPATH=src python data/prepare_observation_om4_native.py /data/om4_onedeg_v3 /data/obs/grid.npz /data/obs/native-om4-v1
```

The ERA5 cache must cover all OM4 training intervals as well as observation
history; observation-only date coverage is insufficient. The preset resolves all three cache paths to the output directories shown above
when `experiment.data_root` is `/data`. Each preparation command writes `COMPLETE.json`. The preset records reference
input proof hashes; replace `data.era5_proof_sha256`,
`data.native_observations_proof_sha256` and `data.native_om4_proof_sha256`
with the SHA-256 of the corresponding prepared `COMPLETE.json` when using
newly prepared archives. Training verifies these identities and cache contents.

```bash
torchrun --standalone --nproc_per_node=4 -m samudra.observations.train \
  observations/deterministic_mixed_large.yaml \
  --experiment.data_root /data --experiment.base_output_dir /runs
```

Use the [campaign generator](campaign.py) with the
[small-initializer base](../configs/observations/deterministic.yaml) to generate
observation-only/mixed and small/large initializer controls plus the no-replay
control. Configure its input cache paths, proof hashes and loading settings first, then
run `python -m samudra.observations.campaign CONFIG /runs/screen-configs`.
Here `CONFIG` is your configured small-initializer base, and the output directory
receives five YAML files. Train each generated config with the observation
entrypoint.
For a selected screen config, `python -m samudra.observations.campaign CONFIG OUTPUT
--winner` generates continuation from the uncooled 4k checkpoint to 16k total main
updates and separate 1k cooldown branches at 8k and 12k. The screen supplies the
4k cooldown; the main continuation ends with the 16k cooldown. These branches
preserve optimizer, replay and RNG state without cooling the continuing trajectory.
The generator derives checkpoint paths from the selected config's output directory.
Run the main continuation first so it writes `main-8000.pt` and `main-12000.pt`,
then run the two generated cooldown configs from those checkpoints.

Use `python -m samudra.observations.campaign CONFIG OUTPUT --qualification`
to generate short configs for `python -m samudra.observations.qualify CONFIG`.
`python -m samudra.observations.qualify_scaling CONFIG` uses the supplied update
budget for distributed loading and numerical checks; supply a separate short-run
config rather than a full scientific budget. These tools produce their own receipts; no run results
are required by the implementation.

The harness uses normalization policy v3: observed channels use observation
statistics and unobserved channels use OM4 statistics. Rebuild normalization with
`prepare_observation_normalization.py --om4` before starting a new run. Checkpoints produced with policy v2 require their original code and data;
this branch does not convert them. The
[four-to-two-rank migration script](../../../scripts/migrate_observation_checkpoint.py)
is restricted to its recorded legacy runtime.

The harness uses the configuration fields `pretrain_mix`, `post_training_updates`,
`post_training_scheduler` and `post_training_learning_rate`. The deterministic
presets use constant main-phase learning rate and the shared linear scheduler for
post-training. Extending a main phase with an active scheduler is rejected because
it would change that scheduler's horizon; the supplied constant-rate continuation
configs support post-training branches after 4k, 8k, 12k and 16k total main-phase
optimizer updates.
