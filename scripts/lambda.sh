#!/bin/bash

ROOT_DIR=$(pwd)
BRANCH=u/jder/experiments/2025-05-20-halfdeg
CONFIG=configs/train_om4_halfdeg.yaml
SHARED_FS_NAME=jder-fs-1

curl -LsSf https://astral.sh/uv/install.sh | sh

curl https://rclone.org/install.sh | sudo bash
rclone config create nyu-osn s3 provider=Other endpoint=https://nyu1.osn.mghpcc.org/ --all # will prompt for auth

mkdir data
rclone copy --progress nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr.zarr/ data/om4_sample_halfdeg_10yr/OM4.zarr
rclone copy --progress nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr_means.zarr/ data/om4_sample_halfdeg_10yr/OM4_means.zarr
rclone copy --progress nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr_stds.zarr/ data/om4_sample_halfdeg_10yr/OM4_stds.zarr

git clone git@github.com:suryadheeshjith/Ocean_Emulator.git
cd Ocean_Emulator
git switch $BRANCH

echo "Run in a tmux: uv run torchrun --standalone --nnodes=1 --nproc-per-node=auto -m ocean_emulators.train $CONFIG --experiment.cluster_data_dir=$ROOT_DIR/data/om4_sample_halfdeg_10yr --experiment.base_output_dir=$ROOT_DIR/$SHARED_FS_NAME"