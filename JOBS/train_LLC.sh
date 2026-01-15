#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=test_train_LLC_Samudra
#SBATCH -N 1
#SBATCH --mem=400GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00-23:00:00
#SBATCH -o /orcd/home/002/codycruz/LLC_ocean_emulator/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/LLC_ocean_emulator/Ocean_Emulator/logs/%x-%j.out

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# cd to correct directory
cd /orcd/home/002/codycruz/LLC_ocean_emulator/Ocean_Emulator

# activate uv environment for ocean_emulator
uv sync --dev

echo "======== train ocean_emulator samudra w/ 1 gpu on LLC4320 data ========"
# run for 2 epochs and save checkpoints every epoch
uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  -m ocean_emulators.train configs/train_LLC.yaml \
  --save_freq 1 \
  --epochs 2 \
  --experiment.data_root "/orcd/data/abodner/"