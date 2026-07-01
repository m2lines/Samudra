#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-01-30-eval:samudra_llc:scaling_tests_k=:10
#SBATCH -N 1
#SBATCH --mem=96GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --time=00-23:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# cd to correct directory
cd /orcd/home/002/codycruz/Ocean_Emulator

# activate uv environment for ocean_emulator
uv sync --dev

# reduce data fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======== evaluate ocean_emulator samudra w/ 3 gpu on LLC4320 data ========"
# run for 2 epochs and save checkpoints every epoch
uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 \
  -m ocean_emulators.eval configs/samudra_llc/eval.yaml \
  --ckpt_path ".LOCAL/2026-01-29-samudra_llc:scaling_tests_k=:10/saved_nets/best_validation_ckpt.pt" \
  --experiment.data_root "/orcd/data/abodner/"