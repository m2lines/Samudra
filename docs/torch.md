# Torch (NYU HPC) Slurm Training With Apptainer

This repo includes Slurm harness scripts that run training inside the published PhysicsNeMo container via Apptainer.

## Prereqs

- You have access to the torch cluster and can submit Slurm jobs.
- `apptainer` (or `singularity`) is available on the cluster.
- Data is available on the cluster filesystem (no S3/OSN copying in this workflow).
- You have a container published to [GitHub Container Registry (GHCR)](https://ghcr.io) (or you use an existing tag).

## Container Build/Publish (GitHub Actions)

The container build workflow is `.github/workflows/container-physicsnemo.yml`.

- It builds `containers/Dockerfile.physicsnemo-25.11`.
- It publishes to GHCR on:
  - `push` to `main`, or
  - `workflow_dispatch` (manual run).

Recommended for branch work:

1. Push your code changes to your branch.
2. Run the workflow manually (Actions UI): `Container PhysicsNeMo 25.11` with `workflow_dispatch` on your branch.
3. Use the resulting image tag(s):
   - `ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-<git_sha>`
   - `ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-manual-<branch-name>`

On torch, `scripts/slurm_apptainer_train.sbatch` can pull by:

- `CONTAINER_HASH=<git_sha>` (expands to tag `25.11-<git_sha>`), or
- `CONTAINER_TAG=25.11-manual-...`, or
- `IMAGE_REF=ghcr.io/...:<tag>` (takes precedence over the two above)

## Training Harness

Main script:

- `scripts/slurm_apptainer_train.sbatch`

Typically you will have this repo cloned on a scratch space so you
can use this script. But note that the actual training run will use the
**code and configs baked into the container** (under `/workspace/src` and
`/workspace/configs`). It does **not** bind-mount your host checkout into the container
for training. This keeps runs pinned to a container tag (and avoids accidental
drift from host edits).
This means host-side config file edits are ignored unless they are baked into a new
container image; for quick, run-specific tweaks, use CLI overrides via `ARGS`.

It expects environment variables:

- `CONFIG` (required): config path inside the container image. Relative paths are
  resolved under `/workspace/`, e.g. `configs/samudra_om4/train.yaml`.
- `NAME_SUFFIX` (required): populates the run name by prepending the current date;
  you can also set `NAME` directly if you prefer.
- `DATA_ROOT` (optional): host data path passed to
  `--experiment.data_root` (default:
  `/scratch/<current_user>/data/om4_onedeg_v3`)
- `OUTPUT_BASE` (optional): host output base dir passed to
  `--experiment.base_output_dir` (default: `/scratch/<current_user>/runs`)
- `ARGS` (optional): extra CLI overrides, e.g. `--batch_size=1`
- `WANDB_API_KEY` (optional): if set and `WANDB_MODE` unset, defaults to W&B online
- `WANDB_MODE` (optional): `online` or `disabled` (if unset, defaults based on
  whether `WANDB_API_KEY` is present)

Key behavior:

- Refuses to run if `${OUTPUT_BASE}/$NAME` already exists (forces unique run names).
- Fails early if either `${DATA_ROOT}` or `${OUTPUT_BASE}` does not exist, with
  instructions to set the corresponding env var.
- Uses the container venv explicitly (`/workspace/.venv/bin/python`) to avoid missing deps.
- To change training code or YAML configs, rebuild/publish a new container tag and
  point the harness at it (e.g. via `CONTAINER_HASH=<git_sha>`).
- Caches the pulled SIF under `${REPO_DIR}/.apptainer-images/` by default.

### Example: 1 Node, 8x RTX6000 on the NYU Torch HPC

For Torch RTX6000 nodes, size CPU and memory proportionally to GPUs. If you
request all GPUs on a node, also request the node's full CPU and memory.

Current `gr102` capacity:
- `8` GPUs (`rtx6000`)
- `128` CPUs total
- `1,400G` memory available via SLURM

So, sizing rule for this node when using our sbatch script which spawns a process
per GPU within a task:
- `--cpus-per-task=16 * <num_gpus>`
- `--mem=175G * <num_gpus>`

ie for an 8-GPU run, use `--cpus-per-task=128 --mem=1400G`.

Partition guidance:
- Do not set `--partition` by default.
- Let Slurm place the job unless you have a specific partition requirement.

```bash
export CONFIG=configs/samudra_om4/train.yaml
export NAME_SUFFIX=om4_samudra_baseline
export ARGS="--batch_size=1"
# Optional overrides (defaults are /scratch/$USER/data/om4_onedeg_v3 and /scratch/$USER/runs)
# export DATA_ROOT=/scratch/$USER/data/om4_onedeg_v3
# export OUTPUT_BASE=/scratch/$USER/runs

# Container selection (pick one)
export CONTAINER_HASH=<git_sha>
# export CONTAINER_TAG=25.11-manual-<branch>
# export IMAGE_REF=ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-<git_sha>

sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  scripts/slurm_apptainer_train.sbatch
```

### Monitoring

After submission:

- Slurm stdout: `slurm-<jobid>.out` in the submission directory (usually the repo root on torch).
- Training log: `${OUTPUT_BASE}/${NAME:-$(date +%Y-%m-%d)-${NAME_SUFFIX}}/experiment.log`

Useful commands:

```bash
squeue -j <jobid> -o '%.18i %.2t %.10M %R'
tail -f slurm-<jobid>.out
tail -f "${OUTPUT_BASE}/${NAME:-$(date +%Y-%m-%d)-${NAME_SUFFIX}}/experiment.log"
```

## Batch-First Guidance On Torch

In general, interactive allocations and TTY-driven `srun` sessions are flaky on Torch.
For routine checks and probes, prefer short `sbatch` jobs and inspect their outputs.

Example hostname probe:

```bash
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks=1 \
  --time=00:01:00 \
  --output="$HOME/oe-hostname-%j.out" \
  --wrap="/bin/hostname"

sacct -j <jobid> --format=JobID,State,Partition,NodeList%40,Elapsed,ExitCode -n
cat "$HOME/oe-hostname-<jobid>.out"
```

GPU status inside an allocation:

```bash
srun --overlap --jobid=<jobid> -N1 -n1 nvidia-smi
```

## NCCL Gotcha On RTX6000 Nodes

On Torch RTX6000 nodes we observed NCCL hangs for 8-GPU single-node training unless P2P is disabled.

Recommended env vars:

```bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

Symptom without the above:

- job prints `distributed init ... world_size 8` and then stalls
- GPUs show high utilization but low memory usage
- eventually you may see an NCCL watchdog timeout

## Apptainer Caching / Pulling

By default the harness will cache pulled SIFs in SIF_DIR, which
defaults to `${REPO_DIR}/.apptainer-images`. using a unqiue name
based on the container you've specified.

You can also point it directly to a SIF_PATH. If the SIF_PATH does
not exist, the harness will pull your specified container from GHCR
to that path.

## Private GHCR Images

If the image is private, set:

```bash
export GHCR_USERNAME=...
export GHCR_TOKEN=...
```

The harness maps these to the environment variables Apptainer uses for registry auth.

## Submission Helper Script

`scripts/torch_submit.sh` is a thin wrapper that sources a `.env` file and submits
the sbatch job with the right env vars and SLURM flags. See `torch.env.example` for
the full list of supported variables.

```bash
# Preview what will be submitted:
./scripts/torch_submit.sh torch.env --dry-run

# Submit:
./scripts/torch_submit.sh torch.env
```

The helper handles the `DATA_SOURCES_JSON` → `--data.sources '<json>'` translation
automatically (see "CLI Override Syntax" below).

## First-Time Setup Checklist

### 1. Clone the repo on torch scratch

```bash
ssh torch
cd /scratch/$USER
git clone --depth=1 git@github.com:open-athena/Ocean_Emulator.git
cd Ocean_Emulator
```

You need an SSH key on torch registered with GitHub (`ssh-keyscan github.com >> ~/.ssh/known_hosts` first).
The clone is only needed for the sbatch script -- training code runs from the container.

### 2. Verify your SLURM account

```bash
sacctmgr show associations where user=$USER format=Account%30,Partition%20
```

The expected account is `torch_pr_347_courant`.

### 3. Stage data (symlinks)

Data lives under a shared scratch directory. Create symlinks:

```bash
mkdir -p /scratch/$USER/data
ln -s /scratch/jr7309/data/om4_onedeg_v3 /scratch/$USER/data/om4_onedeg_v3
ln -s /scratch/jr7309/data/om4_halfdeg_v4 /scratch/$USER/data/om4_halfdeg_v4
ln -s /scratch/jr7309/data/om4_quarterdeg_v2 /scratch/$USER/data/om4_quarterdeg_v2
```

Verify:

```bash
ls /scratch/$USER/data/om4_onedeg_v3/OM4.zarr/.zmetadata
```

### 4. Build the container

PR builds do NOT push to GHCR. Trigger `workflow_dispatch` manually:

```bash
gh workflow run "Container PhysicsNeMo 25.11" --ref <your-branch>
# Wait for the build-and-smoke job to complete (publishes the image).
gh run list --workflow=container-physicsnemo.yml --limit=3
```

Use the git SHA or branch tag in your `.env`:
- `CONTAINER_HASH=<sha>` (expands to `25.11-<sha>`)
- `CONTAINER_TAG=25.11-manual-<branch-name>`

### 5. Create your .env file

```bash
cp torch.env.example torch.env
# Edit torch.env with your secrets and paths
```

### 6. Create output directory

```bash
ssh torch "mkdir -p /scratch/$USER/runs"
```

## Data Layout on Torch

```
/scratch/$USER/data/              # DATA_ROOT for multi-scale
├── om4_onedeg_v3/                # 1-degree
│   ├── OM4.zarr/
│   ├── OM4_means.zarr/
│   └── OM4_stds.zarr/
├── om4_halfdeg_v4/               # 1/2-degree
│   ├── OM4.zarr/
│   ├── OM4_means.zarr/
│   └── OM4_stds.zarr/
└── om4_quarterdeg_v2/            # 1/4-degree
    ├── OM4.zarr/
    ├── OM4_means.zarr/
    └── OM4_stds.zarr/
```

The config's `data.sources[].data_location` paths are resolved relative to `DATA_ROOT`
(passed as `--experiment.data_root`). For example, with `DATA_ROOT=/scratch/$USER/data`
and `data_location=om4_halfdeg_v4/OM4.zarr`, the resolved path is
`/scratch/$USER/data/om4_halfdeg_v4/OM4.zarr`.

## CLI Override Syntax for data.sources

The dot-indexed syntax does **not** work for list items:

```bash
# BROKEN:
--data.sources.0.data_location=om4_halfdeg_v4/OM4.zarr

# CORRECT -- pass the entire list as a JSON blob:
--data.sources '[{"data_location":"om4_halfdeg_v4/OM4.zarr","data_means_location":"om4_halfdeg_v4/OM4_means.zarr","data_stds_location":"om4_halfdeg_v4/OM4_stds.zarr"},{"data_location":"om4_onedeg_v3/OM4.zarr","data_means_location":"om4_onedeg_v3/OM4_means.zarr","data_stds_location":"om4_onedeg_v3/OM4_stds.zarr"}]'
```

The `scripts/torch_submit.sh` helper handles this automatically via the
`DATA_SOURCES_JSON` variable in your `.env` file.

## Troubleshooting

### OOM / DataLoader workers killed (signal 9)

**Symptom**: Job exits with `RuntimeError: DataLoader worker is killed by signal: Killed`
or `sacct` shows `OUT_OF_MEMO+`.

**Fix**: Request full-node memory. The sizing rule is 175G per GPU (1400G for 8 GPUs).
If you requested less, increase `--mem` or set `MEM=1400G` in your `.env`.

### W&B runs not tracked

**Symptom**: Training completes but no run appears in W&B.

**Fix**: Set `WANDB_API_KEY` in your `.env`. The harness auto-sets `WANDB_MODE=online`
when the key is present.

### Container pull fails (403 / unauthorized)

**Symptom**: `apptainer pull` fails with `unable to retrieve auth token: unauthorized`.

**Fix**: The GHCR image is private. Set `GHCR_USERNAME` and `GHCR_TOKEN` in your `.env`.
Generate a PAT at https://github.com/settings/tokens with `read:packages` scope.

### Partition unavailable / nodes down

**Symptom**: Job stays pending with reason `Nodes required for job are DOWN/DRAINED`.

**Fix**: By default, don't set `--partition` and let SLURM place the job. If you
pinned a partition, check its status and consider removing the constraint:

```bash
sinfo -N -o '%.15N %.6t %.10e %.10m %.20P'

# Remove PARTITION from your .env to let SLURM auto-place.
# Or set a specific fallback: PARTITION=l40s_courant  GPUS=gpu:l40s:4
```

### Run directory already exists

**Symptom**: `Refusing to run: output dir already exists`.

**Fix**: Pick a unique `NAME_SUFFIX` or remove the old directory:
```bash
ssh torch "rm -rf /scratch/$USER/runs/<old-run-name>"
```

### Config not found inside container

**Symptom**: `ERROR: config not found inside container: /workspace/configs/...`

**Fix**: The container bakes in configs at build time. Rebuild the container with
your config changes, or adjust `CONFIG` to match what's in the image.
