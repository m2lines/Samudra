#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=test_train_om4_samudra_10-24
#SBATCH -N 1
#SBATCH --mem=900GB
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=00-23:00:00

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# activate uv environment for ocean_emulator
uv sync --dev

echo "======== train ocean_emulator w/ 4 gpus on OM4 data ========"
# run for 200 epochs and save checkpoints every 10 epochs
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 -m ocean_emulators.train configs/train_om4.yaml \
    --save_freq 10 \
    --epochs 200 \
    --experiment.data_root "/orcd/data/abodner/002/cody/"