#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=eval_OM4_samudra_10-27
#SBATCH -N 1
#SBATCH --mem=900GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=00-23:00:00

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# activate uv environment for ocean_emulator
uv sync --dev

echo "======== evaluate ocean_emulator w/ 1 gpu on OM4 data ========"

uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 -m ocean_emulators.eval configs/test_eval_om4.yaml \
    --ckpt_path /home/codycruz/Ocean_Emulator/.LOCAL/train_om4_samudra_10-24/saved_nets/ema_ckpt.pt \
    --experiment.data_root /orcd/data/abodner/002/cody/ \