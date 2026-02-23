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
  --cpus-per-task=16 \
  --mem=1000GB \
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
