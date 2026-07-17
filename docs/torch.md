<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Torch (NYU HPC) Slurm Training/Eval With Apptainer

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
- `NSYS_ARGS` (optional): if set, wrap the training launch with `nsys profile`
- `REQUEUE_ON_USR1` (optional): set to `1` when submitting with
  `--requeue --signal=B:USR1@300`; the harness requeues the job when Slurm sends
  the warning signal.
- `WANDB_API_KEY` (optional): if set and `WANDB_MODE` unset, defaults to W&B online
- `WANDB_MODE` (optional): `online` or `disabled` (if unset, defaults based on
  whether `WANDB_API_KEY` is present)

Key behavior:

- Refuses to run if `${OUTPUT_BASE}/$NAME` already exists, except when
  `SLURM_RESTART_COUNT` indicates that Slurm restarted the same requeued job.
- Fails early if either `${DATA_ROOT}` or `${OUTPUT_BASE}` does not exist, with
  instructions to set the corresponding env var.
- Uses the container venv explicitly (`/workspace/.venv/bin/python`) to avoid missing deps.
- To change training code or YAML configs, rebuild/publish a new container tag and
  point the harness at it (e.g. via `CONTAINER_HASH=<git_sha>`).
- Keeps the repository working directory, pulled SIF, Apptainer caches, data
  cache, and Slurm logs under `/scratch/$USER` by default.
- If `NSYS_ARGS` is set and does not include `-o`/`--output`, reports are written under
  `${OUTPUT_BASE}/${NAME}/nsys/`.
- Defaults to the validated 4-GPU multi-resolution request on
  `rtx6000_lzanna`: 24 CPUs, 800G memory, and a 48-hour walltime. Override the
  `#SBATCH` defaults at submission time for other model or node sizes.

When `preemptible: true`, training detects the latest checkpoint in an existing
run directory and resumes from it after a Slurm requeue. The requeue hook does
not create a checkpoint at signal time, so checkpoint frequently enough that
restarting from the most recent completed epoch is acceptable.

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

- Use `--account=torch_pr_347_lzanna --partition=rtx6000_lzanna` for RTX6000
  training unless another allocation is explicitly required.

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
  --account=torch_pr_347_lzanna \
  --partition=rtx6000_lzanna \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  scripts/slurm_apptainer_train.sbatch
```

### Validated 4x RTX6000 Multi-Resolution Training

The 1, 1/2, and 1/4 degree configuration is
`configs/samudra_multi_om4/train.yaml`. It uses a matching-resolution schedule,
concurrent CPU loading, pinned memory, and two persistent loader workers per
rank.

On `rtx6000_lzanna`, the validated request is 4 GPUs, 24 CPUs, and 800G host
memory. The current submit policy rejects a 900G request for 4 GPUs; a 700G run
exhausted host memory. The 800G run can still touch its cgroup limit under peak
loading, so monitor `memory.events` and Slurm `MaxRSS`. After both training and
validation loaders have been used, persistent loading produces 16 worker
processes: two workers x two loaders x four ranks.

The 24-CPU request was sufficient to keep observed data wait near zero and can
start when CPU-only work prevents a 64-CPU request from being placed. These
resources, scratch-backed execution paths, and append-mode logging are defaults
in the training harness. The submit command only needs to opt into deadline
requeueing:

```bash
export CONFIG=configs/samudra_multi_om4/train.yaml
export NAME="$(date +%F)-samudra-multi-om4-multires-4gpu"
export DATA_ROOT=/scratch/$USER/data
export OUTPUT_BASE=/scratch/$USER/runs
export WANDB_MODE=online
export REQUEUE_ON_USR1=1

# Pin the exact image rather than depending on mutable wrapper defaults.
export IMAGE_REF=ghcr.io/<owner>/ocean-emulator-physicsnemo:<pinned-tag>

sbatch \
  --requeue \
  --signal=B:USR1@300 \
  scripts/slurm_apptainer_train.sbatch
```

The harness uses append mode because a requeued job keeps the same job ID and
log paths. The batch-level (`B:`) USR1 signal reaches the harness, which calls
`scontrol requeue`; on restart, the preemptible training configuration selects
the run's latest checkpoint. The harness also defaults `NCCL_P2P_DISABLE=1` and
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` for RTX6000 jobs; export different values
before submission when another platform requires them.

To enable profiling for a run, you typically want something like this:

```bash
export NSYS_ARGS="--trace=cuda,nvtx,osrt,nccl --sample=cpu --delay=300 --duration=120"
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

## Interactive And Batch Checks On Torch

Interactive allocations and TTY-driven `srun` sessions are available on Torch.
For quick probes, use a short interactive `srun` command. For reproducible checks
with saved logs, prefer short `sbatch` jobs and inspect their outputs.

Example interactive hostname probe:

```bash
srun \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks=1 \
  --time=00:02:00 \
  --pty bash -lc 'hostname'
```

Equivalent batch hostname probe:

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

## Evaluation Harness

Main script:

- `scripts/slurm_apptainer_eval.sbatch`

The eval harness runs one process (single-node, single-GPU by default) inside
the PhysicsNeMo container and executes:

```bash
python -m samudra.eval <CONFIG> ...
```

It expects environment variables:

- `CONFIG` (required): eval config path inside the container image. Relative paths
  resolve under `/workspace/`, e.g. `configs/samudra_om4/eval.yaml`.
- `NAME_SUFFIX` (required): populates the eval run name by prepending the current date;
  you can also set `NAME` directly if you prefer.
- One checkpoint selector (required):
  - `TARGET_CHECKPOINT`: checkpoint path relative to `${OUTPUT_BASE}`, or
  - `CKPT_PATH`: absolute checkpoint path on host (relative host paths are also accepted).
- `DATA_ROOT` (optional): host data path passed to `--experiment.data_root` (default:
  `/scratch/<current_user>/data/om4_onedeg_v3`)
- `OUTPUT_BASE` (optional): host output base dir passed to `--experiment.base_output_dir`
  (default: `/scratch/<current_user>/runs`)
- `ARGS` (optional): extra CLI overrides
- `WANDB_API_KEY` (optional): if set and `WANDB_MODE` unset, defaults to W&B online
- `WANDB_MODE` (optional): `online` or `disabled` (if unset, defaults based on
  whether `WANDB_API_KEY` is present)
- `BACKEND` (optional): eval backend (default `cuda`)

Key behavior:

- Refuses to run if `${OUTPUT_BASE}/$NAME` already exists (forces unique run names).
- Fails early if either `${DATA_ROOT}` or `${OUTPUT_BASE}` does not exist, with
  instructions to set the corresponding env var.
- Verifies the checkpoint exists before launching.
- Binds checkpoint parent paths automatically when checkpoint files live outside
  `${DATA_ROOT}`/`${OUTPUT_BASE}`.

### Example: 1 Node, 1x RTX6000 Eval

```bash
export CONFIG=configs/samudra_om4/eval.yaml
export NAME_SUFFIX=om4_samudra_baseline_eval
export TARGET_CHECKPOINT=2026-02-22-om4_samudra_baseline/saved_nets/ema_ckpt.pt
# Optional overrides (defaults are /scratch/$USER/data/om4_onedeg_v3 and /scratch/$USER/runs)
# export DATA_ROOT=/scratch/$USER/data/om4_onedeg_v3
# export OUTPUT_BASE=/scratch/$USER/runs
# export WANDB_MODE=online
# export WANDB_API_KEY=...

# Container selection (pick one)
export CONTAINER_HASH=<git_sha>
# export CONTAINER_TAG=25.11-manual-<branch>
# export IMAGE_REF=ghcr.io/<owner>/ocean-emulator-physicsnemo:25.11-<git_sha>

sbatch \
  --account=torch_pr_347_lzanna \
  --partition=rtx6000_lzanna \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --mem=128GB \
  --gres=gpu:rtx6000:1 \
  --time=04:00:00 \
  scripts/slurm_apptainer_eval.sbatch
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
