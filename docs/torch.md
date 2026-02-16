# Torch (NYU HPC) Slurm Training With Apptainer

This repo includes Slurm harness scripts that run training inside the published PhysicsNeMo container via Apptainer.

## Prereqs

- You have access to the torch cluster and can submit Slurm jobs.
- `apptainer` (or `singularity`) is available on the cluster.
- Data is available on the cluster filesystem (no S3/OSN copying in this workflow).
- You have a container published to GHCR (or you use an existing tag).

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
- `IMAGE_REF=ghcr.io/...:<tag>`

## Training Harness

Main script:

- `scripts/slurm_apptainer_train.sbatch`

By default, the harness runs against the **code and configs baked into the container**
(under `/workspace/src` and `/workspace/configs`). It does **not** bind-mount your
host checkout into the container for training. This keeps runs pinned to a container
tag (and avoids accidental drift from host edits).

It expects environment variables:

- `CONFIG` (required): config path inside the container image. Relative paths are
  resolved under `/workspace/`, e.g. `configs/samudra_om4/train.yaml`.
- `NAME` (required): run name (used as `--experiment.name` and output folder name)
- `ARGS` (optional): extra CLI overrides, e.g. `--batch_size=1`
- `WANDB_API_KEY` (optional): if set and `WANDB_MODE` unset, defaults to W&B online

Key behavior:

- Refuses to run if `/scratch/.../runs/$NAME` already exists (forces unique run names).
- Uses the container venv explicitly (`/workspace/.venv/bin/python`) to avoid missing deps.
- To change training code or YAML configs, rebuild/publish a new container tag and
  point the harness at it (e.g. via `CONTAINER_HASH=<git_sha>`).
- Caches the pulled SIF under `${REPO_DIR}/.apptainer-images/` by default.

### Name Convention (Date Prefix)

We recommend prefixing run names with a date, similar to the legacy workflow:

```bash
export NAME="$(date +%Y-%m-%d)-om4_samudra_baseline"
```

Because the harness bails if the output directory exists, the date prefix helps keep names unique and searchable.

### Example: 1 Node, 8x RTX6000

```bash
export CONFIG=configs/samudra_om4/train.yaml
export NAME="$(date +%Y-%m-%d)-om4_samudra_baseline"
export ARGS="--batch_size=1"
export DATA_ROOT=/scratch/jr7309/data/om4_onedeg_v3
export OUTPUT_BASE=/scratch/jr7309/runs

# Container selection (pick one)
export CONTAINER_HASH=<git_sha>
# export CONTAINER_TAG=25.11-manual-<branch>
# export IMAGE_REF=ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-<git_sha>

sbatch \
  --account=torch_pr_347_courant \
  --partition=rtx6000_lzanna \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --mem=1000GB \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  scripts/slurm_apptainer_train.sbatch
```

### Monitoring

After submission:

- Slurm stdout: `slurm-<jobid>.out` in the submission directory (usually the repo root on torch).
- Training log: `${OUTPUT_BASE}/${NAME}/experiment.log`

Useful commands:

```bash
squeue -j <jobid> -o '%.18i %.2t %.10M %R'
tail -f slurm-<jobid>.out
tail -f /scratch/jr7309/runs/$NAME/experiment.log
```

GPU status inside an allocation:

```bash
srun --overlap --jobid=<jobid> -N1 -n1 nvidia-smi
```

## W&B

The training config only accepts:

- `--experiment.wandb.mode=online`
- `--experiment.wandb.mode=disabled`

Notes:

- If `WANDB_API_KEY` is set and `WANDB_MODE` is unset, the harness chooses `online`.
- If `WANDB_MODE=offline`, the harness maps it to `disabled` (there is no `offline` mode in config).
- If you set `WANDB_DIR`, the harness will `mkdir -p` it on the host before running.

Example:

```bash
export WANDB_MODE=online
export WANDB_API_KEY=...
export WANDB_DIR="/scratch/jr7309/runs/$NAME/wandb"
```

## NCCL Gotcha On RTX6000 Nodes

On `rtx6000_lzanna` we observed NCCL hangs for 8-GPU single-node training unless P2P is disabled.

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

Defaults in the harness:

- SIF directory: `${REPO_DIR}/.apptainer-images`
- SIF path: `${SIF_DIR}/ocean-emulator-physicsnemo-<tag>.sif`

You can override:

- `SIF_DIR` or `SIF_PATH`

The harness will pull from GHCR if the SIF is missing or empty.

## Private GHCR Images

If the image is private, set:

```bash
export GHCR_USERNAME=...
export GHCR_TOKEN=...
```

The harness maps these to the environment variables Apptainer uses for registry auth.
