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
Source/config-only changes can normally use the code-overlay workflow described
in [the Torch guide](torch.md), but dependency or lockfile changes require an
exact container rebuild. PR 842 changes `uv.lock`, so dispatch its container
workflow before using it:

```bash
gh workflow run container-physicsnemo.yml \
  --repo m2lines/Samudra \
  --ref feature/native-sdpa-perceiver
```

After the workflow succeeds, its manual x86 image tag is:

```text
ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-manual-feature-native-sdpa-perceiver
```

For other refs, replace slashes in the branch name with hyphens. A production
run should prefer the immutable SHA tag when the same image has been published
from `main`.

## Pull the SIF once

The training harness can pull an absent SIF, but doing this once before a GPU
job makes startup predictable. The SIF is shared through Lustre, so it can be
prepared on a transfer node without waiting for an Alpha CPU allocation:

```bash
ssh eai
ssh alpha-dtn1

source /etc/profile.d/modules.sh
module load apptainer

SCRATCH="/mnt/lustre/nyu/$USER"
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer-cache"
export APPTAINER_TMPDIR=/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$SCRATCH/.apptainer-images"

IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-manual-feature-native-sdpa-perceiver
SIF_PATH="$SCRATCH/.apptainer-images/physicsnemo-26.05-native-sdpa.sif"
apptainer pull "$SIF_PATH" "docker://$IMAGE_REF"
chmod 0444 "$SIF_PATH"
```

`APPTAINER_TMPDIR=/tmp` is intentional: image conversion uses node-local space
and avoids filesystem feature mismatches on shared scratch. Alpha's transfer
nodes had more than 800 GB free in `/tmp` during validation.

## Submit the 1° SamudraMini pilot

Use Alpha H100 and the `test` QoS for the first end-to-end validation. This
performs a real one-epoch training run but uses only one GPU and keeps W&B
disabled unless a key is deliberately supplied.

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
export DATA_CACHE_DIR="$SCRATCH/.data_cache/$NAME"
export WANDB_MODE=disabled
export ARGS='--epochs=1 --save_freq=1 --batch_size=1 --data.loading.num_workers=2'

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
  --time=02:00:00 \
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

## Scale to a full H100 node

After the pilot succeeds, a full-node 8-GPU run uses all 96 CPU cores. Do not
copy Torch's 128-CPU request; Alpha GPU nodes expose 96 schedulable CPU cores.
For example:

```bash
export NAME="$(date +%F)-samudra-mini-onedeg-eai-8gpu"
export DATA_CACHE_DIR="$SCRATCH/.data_cache/$NAME"
export WANDB_MODE=online
export ARGS='--data.loading.num_workers=2 --preemptible=true'
export REQUEUE_ON_USR1=1

sbatch \
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
  ~/Samudra/scripts/slurm_apptainer_train.sbatch
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

## Monitor and diagnose

```bash
squeue -u "$USER" -o '%.18i %.12q %.9T %.10M %.6D %R'
sacct -u "$USER" -S today \
  --format=JobID,JobName,QOS,AllocTRES,Elapsed,State,ExitCode
sprio -j JOB_ID
ssh alpha-dtn1 tail -f "/mnt/lustre/nyu/$USER/logs/samudra-eai-JOB_ID.out"
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
- An import or lockfile mismatch with a code overlay: rebuild the container
  when `uv.lock` or `pyproject.toml` changes; do not bypass the overlay
  builder's dependency check.
- A run name already exists: choose a new `NAME`. The harness refuses to mix a
  new job with an existing run except for a Slurm requeue.

## Empire AI references

- [Getting Started: Alpha, Grace, and Beta](https://empireai.freshdesk.com/support/solutions/articles/157000374441-empire-ai-getting-started-alpha-grace-beta-)
- [Alpha job submission and QoS](https://empireai.freshdesk.com/support/solutions/articles/157000374474-alpha-job-submission-and-qos-overview)
- [Submitting jobs](https://empireai.freshdesk.com/support/solutions/articles/157000010768-how-do-i-submit-jobs-)
- [Alpha storage](https://empireai.freshdesk.com/support/solutions/articles/157000175046-empire-ai-alpha-storage)
- [Alpha hardware](https://empireai.freshdesk.com/support/solutions/articles/157000007946-alpha-hardware)
- [Alpha and Grace mixed-architecture guidance](https://empireai.freshdesk.com/support/solutions/articles/157000373787-alpha-and-grace-mixed-architecture-guidance)
