#!/bin/bash

ROOT_DIR=$(pwd)
BRANCH=u/jder/experiments/2025-05-30
CONFIG=configs/train_om4.yaml
SHARED_FS_NAME=jder-fs-1

curl -LsSf https://astral.sh/uv/install.sh | sh

curl https://rclone.org/install.sh | sudo bash
rclone config create nyu-osn s3 provider=Other endpoint=https://nyu1.osn.mghpcc.org/ --all # will prompt for auth

mkdir data
rclone copy --progress nyu-osn:m2lines-pubs/Samudra/OM4_means data/public/OM4_means.zarr
rclone copy --progress nyu-osn:m2lines-pubs/Samudra/OM4_stds data/public/OM4_stds.zarr
rclone copy --progress nyu-osn:m2lines-pubs/Samudra/OM4 data/public/OM4.zarr

git clone git@github.com:suryadheeshjith/Ocean_Emulator.git
cd Ocean_Emulator
git switch $BRANCH

echo "Run in a tmux: uv run torchrun --standalone --nnodes=1 --nproc-per-node=auto -m ocean_emulators.train $CONFIG --experiment.cluster_data_dir=$ROOT_DIR/data/public --experiment.base_output_dir=$ROOT_DIR/$SHARED_FS_NAME --experiment.wandb.mode=online"