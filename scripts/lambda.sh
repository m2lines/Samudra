#!/bin/bash

ROOT_DIR=$(pwd)

sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config nyu-osn s3 --s3-provider Other --s3-endpoint https://nyu1.osn.mghpcc.org/ --all --s3-env-auth false # will prompt for auth

mkdir data
rclone copy nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr.zarr/ data/om4_sample_halfdeg_10yr/OM4.zarr
rclone copy nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr_means.zarr/ data/om4_sample_halfdeg_10yr/OM4_means.zarr
rclone copy nyu-osn:emulators/sd5313/test/om4_sample_halfdeg_10yr_std.zarr/ data/om4_sample_halfdeg_10yr/OM4_stds.zarr

git clone git@github.com:suryadheeshjith/Ocean_Emulator.git
cd Ocean_Emulator

echo "Run: uv run -m ocean_emulator.train configs/train_om4_halfdeg.yaml --cluster_data_dir=$ROOT_DIR/data/om4_sample_halfdeg_10yr --base_output_dir=$ROOT_DIR/jder-fs-1"