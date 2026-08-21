<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Empire AI Alpha Slurm Training With Apptainer

This is the project runbook for training Samudra on Empire AI Alpha. It uses
the repository's existing Slurm/Apptainer harness, stages the public OSN data
on Lustre, and keeps images, caches, runs, and large logs out of the home
filesystem.

The commands below were validated on Alpha with the NYU account and the public
1° OM4 dataset. Replace `nyu` with the account reported for your user.

## Choose Alpha, Grace, or Beta

- Use **Alpha H100** as the default for Samudra development and training. Alpha
  is `x86_64`; an H100 node has eight 80 GB GPUs, 96 CPU cores, and about 1.9 TB
  of host memory.
- Use **Alpha H200** when the additional GPU memory is useful. An H200 has
  141 GB. Request it by its full GRES name.
- Do not submit GPU training to **Grace**. Grace is CPU-only `arm64` and is
  intended for CPU-heavy work. It needs separate environments and compiled
  artifacts.
- **Beta NVL72/B200** uses a separate Slurm environment and is outside this
  Alpha guide.

An arm64 Samudra image is therefore not needed for Alpha. Use the repository's
`x86_64` PhysicsNeMo image.

Useful live checks:

```bash
uname -m
sacctmgr show assoc user="$USER" format=User,Account,Partition,QOS
sinfo -o '%P|%a|%l|%D|%G|%f'
scontrol show node alphagpu01
```

Empire AI is transitioning from institutional Alpha partitions to an `alpha`
hardware-tier partition. Always trust the live association and `sinfo` output
over a copied command. At the time of validation, the account and partition
were both `nyu`.

## Filesystem layout

Home is mounted at `/mnt/home/$USER`. Keep the repository, small scripts, and
small configuration files there. Do not put datasets or container images in
home.

Alpha Lustre scratch is mounted at:

```text
/mnt/lustre/<institution>/$USER
```

For NYU, use this layout:

```text
/mnt/lustre/nyu/$USER/
├── .apptainer-cache/
├── .apptainer-images/
├── .data_cache/
├── code/
├── data/
│   └── om4_onedeg/
├── logs/
└── runs/
```

Scratch is high-performance, transient storage. Empire AI may introduce a
purge policy, so copy important checkpoints and results elsewhere.

## Get the code

Use a pushed commit or branch so the source is reproducible. For the native
SDPA Perceiver work in PR 842:

```bash
ssh eai
git clone --branch feature/native-sdpa-perceiver --single-branch \
  https://github.com/m2lines/Samudra.git ~/Samudra
cd ~/Samudra
git rev-parse HEAD
```

The native implementation uses PyTorch
`scaled_dot_product_attention`; it does not require external `flash-attn` or
`flash-perceiver` wheels. `perceiver_implementation: auto` lets PyTorch select
an available SDPA backend, including its FlashAttention kernel when eligible.

## Stage public OSN data through a transfer node

Alpha has two data-transfer nodes, `alpha-dtn1` and `alpha-dtn2`. Start bulk
transfers there rather than on a login or GPU node:

```bash
ssh eai
ssh alpha-dtn1
```

At the time of validation, rclone was not installed system-wide. Install the
official static `x86_64` binary once in an architecture-labeled home path:

```bash
install_root="$HOME/software/alpha-x86/rclone"
tmp_dir="$(mktemp -d)"
curl --fail --location --silent --show-error \
  https://downloads.rclone.org/rclone-current-linux-amd64.zip \
  --output "$tmp_dir/rclone.zip"
unzip -q "$tmp_dir/rclone.zip" -d "$tmp_dir/unpacked"
rclone_src="$(find "$tmp_dir/unpacked" -type f -name rclone -perm -u+x -print -quit)"
mkdir -p "$install_root/bin"
install -m 0755 "$rclone_src" "$install_root/bin/rclone"
rm -rf "$tmp_dir"
```

Configure anonymous access to the public NYU OSN pod:

```bash
RCLONE="$HOME/software/alpha-x86/rclone/bin/rclone"
"$RCLONE" config create nyu-osn-public s3 \
  provider Other \
  endpoint https://nyu1.osn.mghpcc.org/ \
  env_auth false
```

Copy the complete 1° dataset, including the mean and standard-deviation Zarr
stores required by training:

```bash
SCRATCH="/mnt/lustre/nyu/$USER"
mkdir -p "$SCRATCH/data/om4_onedeg" "$SCRATCH/logs"

nohup "$RCLONE" copy \
  nyu-osn-public:m2lines-pubs/Samudra/v2026-07/om4_onedeg \
  "$SCRATCH/data/om4_onedeg" \
  --ignore-existing \
  --transfers=32 \
  --checkers=64 \
  --stats=1m \
  --stats-one-line \
  --log-level NOTICE \
  --stats-log-level NOTICE \
  >"$SCRATCH/logs/rclone-om4-onedeg.log" 2>&1 </dev/null &
```

The source is about 92 GiB. The command is restart-safe: rerunning it with
`--ignore-existing` fills in missing objects without replacing completed ones.
Monitor and verify it on the transfer node:

```bash
tail -f "$SCRATCH/logs/rclone-om4-onedeg.log"
pgrep -a -u "$USER" -f '[r]clone copy.*om4_onedeg'
du -sh "$SCRATCH/data/om4_onedeg"

"$RCLONE" check \
  nyu-osn-public:m2lines-pubs/Samudra/v2026-07/om4_onedeg \
  "$SCRATCH/data/om4_onedeg" \
  --size-only --one-way
```

Exit the transfer node after the copy and verification finish.

## Build and publish the exact container

The container workflow is `.github/workflows/container-physicsnemo.yml`.
Dependency or lockfile changes require an exact container rebuild. PR 842
changes `uv.lock`, so dispatch its container workflow before using it:

```bash
gh workflow run container-physicsnemo.yml \
  --repo m2lines/Samudra \
  --ref feature/native-sdpa-perceiver
```

After the workflow's `build-and-smoke` job completes its `Publish x86_64
image` step, its manual x86 image tag is:

```text
ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-manual-feature-native-sdpa-perceiver
```

Check the x86 build and publication jobs directly rather than relying only on
the workflow's overall conclusion: architecture-specific builds and later test
jobs can fail independently. During the 2026-08-21 validation, the x86 image
published successfully, while a later container CPU-test job failed collection
because `scripts.build_quickstart_notebook` was not present in the image. That
packaging issue did not affect training; the published image completed the EAI
pilot below.

For other refs, replace slashes in the branch name with hyphens. A production
run should prefer the immutable SHA tag when the same image has been published
from `main`.

## Pull the SIF once

The training harness can pull an absent SIF, but doing this once before a GPU
job makes startup predictable. Do not do this on a transfer node: although an
Apptainer module is visible there, its global configuration is not usable. The
Alpha CPU partition can also have a long queue. Use a short, one-GPU `test` QoS
job and the repository's atomic pull helper. The resulting SIF is shared
through Lustre and reused by later jobs:

```bash
ssh eai
SCRATCH="/mnt/lustre/nyu/$USER"
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer-cache"
export SCRATCH_DIR="$SCRATCH"
export SIF_DIR="$SCRATCH/.apptainer-images"
export SIF_PATH="$SIF_DIR/physicsnemo-26.05-native-sdpa.sif"
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-manual-feature-native-sdpa-perceiver
mkdir -p "$APPTAINER_CACHEDIR" "$SIF_DIR" "$SCRATCH/logs"

sbatch \
  --account=nyu \
  --partition=nyu \
  --qos=test \
  --constraint=h100 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --time=00:30:00 \
  --chdir="$SCRATCH" \
  --output="$SCRATCH/logs/pull-sif-%j.out" \
  --error="$SCRATCH/logs/pull-sif-%j.err" \
  --export=ALL \
  ~/Samudra/scripts/slurm_apptainer_pull.sbatch
```

The helper uses node-local `/tmp` for conversion, avoiding filesystem feature
mismatches on shared scratch. Wait for `COMPLETED` and inspect the log before
submitting training:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail "$SCRATCH/logs/pull-sif-JOB_ID.out"
ls -lh "$SIF_PATH"
```

## Choose a reproducible code source

Use one of these supported paths on Alpha:

1. Build and use an exact SIF for the experiment commit. This is required when
   `uv.lock` or `pyproject.toml` changes and is the strongest isolation.
2. For source/config-only changes, mount a clean, commit-pinned Git checkout
   read-only with the harness's `CODE_DIR` support.

Do **not** use the Torch EXT3 `CODE_LAYER` mechanism on Alpha. `apptainer
overlay create` can succeed, which is misleading, but the later `apptainer exec
--overlay` fails for ordinary users while attaching `/dev/loop0`:

```text
failed to find loop device
could not open /dev/loop0: permission denied
```

This limitation applies on Alpha compute nodes as well as login nodes.

To prepare the read-only-bind option, resolve a pushed ref to a full commit and
create a detached worktree. Do not edit this checkout after submitting jobs:

```bash
ssh eai
export SCRATCH="/mnt/lustre/nyu/$USER"
export CODE_REF=feature/native-sdpa-perceiver

git -C ~/Samudra fetch origin "$CODE_REF"
export CODE_COMMIT="$(git -C ~/Samudra rev-parse FETCH_HEAD)"
export CODE_DIR="$SCRATCH/code/Samudra-$CODE_COMMIT"
mkdir -p "$SCRATCH/code"
if [[ ! -d "$CODE_DIR" ]]; then
  git -C ~/Samudra worktree add --detach "$CODE_DIR" "$CODE_COMMIT"
fi

test "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git -C "$CODE_DIR" status --porcelain)"
```

Export both `CODE_DIR` and `CODE_COMMIT` when submitting training and eval.
The shared harness repeats those two Git checks on the compute node, requires
a full 40-character commit, and mounts the checkout at
`/opt/samudra-code:ro`. In the same container preflight used to read image
metadata, it verifies:

```bash
cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock
cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml
```

A mismatch fails before creating the run directory. Successful jobs pass the
code commit and repository to W&B and record the bind path, commit, dependency
hashes, container identity, and SIF path in `source-manifest.json` and
`run-provenance.json`.

Alpha lacks `squashfuse` and `fuse2fs`, so every `apptainer exec` may unpack the
12 GiB SIF into a temporary sandbox. The harness combines container metadata
and `CODE_DIR` compatibility into one preflight, followed by the actual run,
instead of using a separate invocation for every check. Startup can still take
several minutes.

## Shell conveniences and secrets

It is reasonable to add non-secret cluster paths to `~/.bashrc`:

```bash
export EAI_ACCOUNT=nyu
export EAI_SCRATCH="/mnt/lustre/nyu/$USER"
export PATH="$HOME/software/alpha-x86/rclone/bin:$PATH"
```

If every process in the account is trusted and experiment-related, keeping
`WANDB_API_KEY` in `~/.bashrc` is convenient for interactive shells. Set
`chmod 600 ~/.bashrc`, and never print or commit the file:

```bash
export WANDB_API_KEY='...'
```

Do not assume that `ssh eai 'command'`, a non-interactive shell, or a future
shell configuration will source `.bashrc`. For automation, a dedicated secret
file is a clearer interface:

```bash
mkdir -p ~/.config/samudra
chmod 700 ~/.config/samudra
# Create ~/.config/samudra/secrets.env without committing it:
# export WANDB_API_KEY='...'
chmod 600 ~/.config/samudra/secrets.env
```

Source that file explicitly before an online W&B submission and test that the
key exists without printing it:

```bash
source ~/.config/samudra/secrets.env
test -n "${WANDB_API_KEY:-}"
export WANDB_MODE=online
```

If the key is kept only in `.bashrc`, explicitly run `source "$HOME/.bashrc"`
and the same `test -n` check instead. Place the export before any
interactive-shell early return in that file. Slurm receives exported
submission-shell variables through `--export=ALL`. The pilot below disables
W&B so it does not require a credential.

## Submit the 1° SamudraMini pilot

Use Alpha H100 and the `test` QoS for the first end-to-end validation. Debug
mode performs four real training batches and four validation batches, then
exits cleanly. This exercises data loading, native SDPA forward/backward,
metrics, checkpointing, and output writing without spending a full epoch.

```bash
ssh eai
cd ~/Samudra

ACCOUNT=nyu
SCRATCH="/mnt/lustre/$ACCOUNT/$USER"
NAME="$(date +%F)-samudra-mini-onedeg-eai-pilot"

mkdir -p "$SCRATCH/runs" "$SCRATCH/logs"

export CONFIG=src/samudra/configs/samudra_mini_om4/train.yaml
export NAME
export DATA_ROOT="$SCRATCH/data/om4_onedeg"
export OUTPUT_BASE="$SCRATCH/runs"
export SCRATCH_DIR="$SCRATCH"
export SIF_DIR="$SCRATCH/.apptainer-images"
export SIF_PATH="$SIF_DIR/physicsnemo-26.05-native-sdpa.sif"
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-manual-feature-native-sdpa-perceiver
export DATA_CACHE_DIR="$SCRATCH/.data_cache/$NAME"
export WANDB_MODE=disabled
export ARGS='--debug=true --epochs=1 --save_freq=1 --batch_size=1 --data.loading.num_workers=2'

sbatch \
  --account="$ACCOUNT" \
  --partition="$ACCOUNT" \
  --qos=test \
  --constraint=h100 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --time=00:30:00 \
  --chdir="$SCRATCH" \
  --output="$SCRATCH/logs/samudra-eai-%j.out" \
  --error="$SCRATCH/logs/samudra-eai-%j.err" \
  --export=ALL \
  ~/Samudra/scripts/slurm_apptainer_train.sbatch
```

Command-line `sbatch` options override the Torch-specific defaults embedded in
the shared harness. Keep every resource override above: in particular, the
account, GRES, CPU count, memory, working directory, and logs must not fall
back to Torch values.

### Validated result

The pilot above was validated on 2026-08-21 with these results:

- source/container commit
  `b7f94a312a0d261ce9f65ea2d1d0d86654b1155e` from PR 842;
- public dataset copy: 91.639 GiB and 380,460 matching files according to
  `rclone check --size-only --one-way`;
- SIF preparation job `40324`: `COMPLETED` with exit code `0:0` in 16m49s;
- H100 debug training job `40380`: `COMPLETED` with exit code `0:0` in 5m26s;
- internal train/validation work: 54 seconds, train loss 2.420, validation loss
  0.636, and about 2.35 GB peak GPU memory during training;
- four checkpoints written: best validation, latest, epoch 1, and EMA; and
- `run-provenance.json` recorded the same commit for source and container.

Alpha currently lacks `squashfuse` and `fuse2fs` on compute nodes. Apptainer
therefore prints warnings and expands the SIF to node-local temporary
sandboxes. The original harness performed three container invocations during
startup, adding roughly two minutes in this pilot; the current harness combines
metadata and compatibility checks into one preflight. The warnings are
non-fatal; the final job state and exit code are the authoritative result.

## Compare queue-aware resource shapes

A full node has the greatest instantaneous throughput, but it may wait longer
than a smaller request. Before a production submission, compare several valid
shapes with `sbatch --test-only` using the same QoS, wall time, constraints, and
roughly proportional CPU and memory requests as the real job:

```bash
for shape in 1:16:200G 4:48:800G 8:96:1600G; do
  IFS=: read -r gpus cpus memory <<<"$shape"
  echo "--- ${gpus}x H100 ---"
  sbatch --test-only \
    --account=nyu \
    --partition=nyu \
    --qos=standard \
    --constraint=h100 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --gres="gpu:nvidia_h100_80gb_hbm3:$gpus" \
    --time=2-00:00:00 \
    --wrap=true
done
```

The projected start time is a volatile scheduler snapshot, not a reservation
or guarantee; it can move by hours within minutes. A submitted job may also
temporarily report `StartTime=Unknown`. Balance queue state against expected
scaling efficiency and runtime: four GPUs can finish sooner than eight when the
smaller request starts much earlier.

`sbatch --test-only` prints job-like IDs as part of its estimate, but it does
**not** enqueue those jobs. Confirm with `squeue -u "$USER"` if there is any
doubt. A `QOSMinGRES` result is a rejected shape, not a queued job.

## Scale to a full H100 node

After the pilot succeeds, a full-node 8-GPU run uses all 96 CPU cores. Do not
copy Torch's 128-CPU request; Alpha GPU nodes expose 96 schedulable CPU cores.
For example:

```bash
export NAME="$(date +%F)-samudra-onedeg-eai-8gpu"
export DATA_CACHE_DIR="$SCRATCH/.data_cache/$NAME"
export CONFIG=src/samudra/configs/samudra_om4/train.yaml
export WANDB_MODE=online
export ARGS='--data.loading.num_workers=2 --preemptible=true'
export REQUEUE_ON_USR1=1

TRAIN_JOB_ID="$(sbatch --parsable \
  --account="$ACCOUNT" \
  --partition="$ACCOUNT" \
  --qos=long \
  --constraint=h100 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=96 \
  --mem=1600G \
  --gres=gpu:nvidia_h100_80gb_hbm3:8 \
  --time=7-00:00:00 \
  --requeue \
  --signal=B:USR1@300 \
  --chdir="$SCRATCH" \
  --output="$SCRATCH/logs/samudra-eai-%j.out" \
  --error="$SCRATCH/logs/samudra-eai-%j.err" \
  --export=ALL \
  ~/Samudra/scripts/slurm_apptainer_train.sbatch)"
echo "Training job: $TRAIN_JOB_ID"
```

The harness traps `USR1`, requeues the job, and resumes from the latest
completed checkpoint when `preemptible: true`. The signal itself does not
create a checkpoint, so retain a suitable checkpoint frequency. EAI's
published `SIGTERM` example is not interchangeable with this harness's `USR1`
trap.

For a normal run of no more than 48 hours, use `--qos=standard`. The `long` QoS
allows up to seven days at a lower SU factor but may start more slowly. Check
the current limits before submitting because scheduler policy can change.

To target H200 instead, replace the constraint and GRES:

```bash
--constraint=h200 --gres=gpu:nvidia_h200:8
```

## Chain evaluation and visualization

For normal experiments, submit the stages as one failure-safe dependency chain:

```text
training ──afterok──> evaluation ──afterok──> visualization
```

Use `--kill-on-invalid-dep=yes` at every edge. If an upstream job fails or is
cancelled, Slurm then cancels the dependent job instead of leaving it pending
forever or producing artifacts from stale inputs.

The validated `samudra_mini_om4` pilot does not currently ship a matching eval
preset. The concrete example below applies when training with
`samudra_om4/train.yaml`; use the matching eval and visualization configs for
any other model.

After submitting the training command above, submit a one-GPU evaluation:

```bash
export TRAIN_NAME="$NAME"
export EVAL_NAME="${TRAIN_NAME}-eval"
export CONFIG=src/samudra/configs/samudra_om4/eval.yaml
export NAME="$EVAL_NAME"
export TARGET_CHECKPOINT="$TRAIN_NAME/saved_nets/ema_ckpt.pt"
export ARGS='--num_model_steps_forward=25'
export DATA_CACHE_DIR="$SCRATCH/.data_cache/$EVAL_NAME"

EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB_ID" \
  --kill-on-invalid-dep=yes \
  --account="$ACCOUNT" \
  --partition="$ACCOUNT" \
  --qos=standard \
  --constraint=h100 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --time=04:00:00 \
  --chdir="$SCRATCH" \
  --output="$SCRATCH/logs/samudra-eval-%j.out" \
  --error="$SCRATCH/logs/samudra-eval-%j.err" \
  --export=ALL \
  ~/Samudra/scripts/slurm_apptainer_eval.sbatch)"
echo "Evaluation job: $EVAL_JOB_ID"
```

Visualization does not yet have a dedicated Slurm harness, so invoke the same
SIF and optional read-only code checkout explicitly. The
`samudra_om4/viz.yaml` preset
reads its basin mask anonymously from public OSN:

```bash
export VIZ_CONFIG=src/samudra/configs/samudra_om4/viz.yaml
export VIZ_NAME="${TRAIN_NAME}-viz"
export RUNS="[{\"name\":\"$EVAL_NAME\",\"location\":\"$OUTPUT_BASE/$EVAL_NAME/predictions.zarr\"}]"

VIZ_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$EVAL_JOB_ID" \
  --kill-on-invalid-dep=yes \
  --account="$ACCOUNT" \
  --partition="$ACCOUNT" \
  --qos=standard \
  --constraint=h100 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --time=04:00:00 \
  --chdir="$SCRATCH" \
  --output="$SCRATCH/logs/samudra-viz-%j.out" \
  --error="$SCRATCH/logs/samudra-viz-%j.err" \
  --export=ALL \
  --wrap='set -euo pipefail
    source /etc/profile.d/modules.sh
    module load apptainer
    code_root=/workspace
    code_bind_args=()
    if [[ -n "${CODE_DIR:-}" ]]; then
      test "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
      test -z "$(git -C "$CODE_DIR" status --porcelain)"
      code_root=/opt/samudra-code
      code_bind_args=(--bind "$CODE_DIR:$code_root:ro")
    fi
    apptainer exec --nv \
      "${code_bind_args[@]}" \
      --bind "$DATA_ROOT:$DATA_ROOT,$OUTPUT_BASE:$OUTPUT_BASE" \
      --pwd "$code_root" \
      "$SIF_PATH" \
      bash -c "
        set -euo pipefail
        if [[ \"\$1\" == /opt/samudra-code ]]; then
          cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock
          cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml
        fi
        export PYTHONPATH=\"\$1/src\"
        exec /workspace/.venv/bin/python -m samudra.viz \
          \"\$1/\$2\" \
          --data_root=\"\$3\" \
          --base_output_dir=\"\$4\" \
          --name=\"\$5\" \
          --runs=\"\$6\"
      " bash "$code_root" "$VIZ_CONFIG" "$DATA_ROOT" "$OUTPUT_BASE" "$VIZ_NAME" "$RUNS"')"
echo "Visualization job: $VIZ_JOB_ID"
```

Although visualization is CPU-oriented, an otherwise valid CPU-only Alpha
request currently fails with `QOSMinGRES`. The practical Alpha workaround is
to reserve one H100 as above. Alternatives are running visualization on
another compatible system, or maintaining a separate `arm64` environment for
the CPU-only Grace system. Do not reuse an Alpha x86 SIF or checkout artifacts
on Grace.

The dependency policy has been validated in failure as well as success: when
an earlier EXT3 overlay-preparation experiment failed, `afterok` with
`--kill-on-invalid-dep=yes` cancelled its training, evaluation, and
visualization dependents immediately. No downstream stage ran with incorrect
code or stale artifacts.

## Monitor and diagnose

```bash
squeue -u "$USER" -o '%.18i %.12q %.9T %.10M %.6D %R'
sacct -u "$USER" -S today \
  --format=JobID,JobName,QOS,AllocTRES,Elapsed,State,ExitCode
sprio -j JOB_ID
tail -f "/mnt/lustre/nyu/$USER/logs/samudra-eai-JOB_ID.out"
```

The run output is under `$OUTPUT_BASE/$NAME`. The harness also records
`run-provenance.json`, including the exact container commit and SIF path.

Common failures:

- `Invalid account or account/partition combination`: rerun `sacctmgr show
  assoc` and use the institution associated with your user.
- `Requested node configuration is not available`: compare the request with
  `scontrol show node`. A full Alpha GPU node has 96, not 128, CPU cores.
- `DATA_ROOT directory does not exist`: finish or resume the OSN copy and use
  the directory containing `OM4.zarr`, `OM4_means.zarr`, and `OM4_stds.zarr`.
- Architecture or `exec format` errors: confirm `uname -m` is `x86_64` and use
  the x86 image on Alpha, not an arm64 image intended for Grace.
- `/dev/loop0: permission denied`: an EXT3 `CODE_LAYER` was selected. It is not
  supported on Alpha; use an exact SIF or the clean `CODE_DIR` read-only bind.
- A `CODE_DIR` commit, cleanliness, or lockfile failure: recreate the detached
  checkout from the pushed commit. Rebuild the container if `uv.lock` or
  `pyproject.toml` changed.
- A run name already exists: choose a new `NAME`. The harness refuses to mix a
  new job with an existing run except for a Slurm requeue.

## Empire AI references

- [Getting Started: Alpha, Grace, and Beta](https://empireai.freshdesk.com/support/solutions/articles/157000374441-empire-ai-getting-started-alpha-grace-beta-)
- [Alpha job submission and QoS](https://empireai.freshdesk.com/support/solutions/articles/157000374474-alpha-job-submission-and-qos-overview)
- [Submitting jobs](https://empireai.freshdesk.com/support/solutions/articles/157000010768-how-do-i-submit-jobs-)
- [Alpha storage](https://empireai.freshdesk.com/support/solutions/articles/157000175046-empire-ai-alpha-storage)
- [Alpha hardware](https://empireai.freshdesk.com/support/solutions/articles/157000007946-alpha-hardware)
- [Alpha and Grace mixed-architecture guidance](https://empireai.freshdesk.com/support/solutions/articles/157000373787-alpha-and-grace-mixed-architecture-guidance)
