#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=110GB
#SBATCH --job-name=long_rollout
#SBATCH --output=sbatch_logs/%j/job_%j.out
#SBATCH --error=sbatch_logs/%j/job_err_%j.out

mkdir -p sbatch_logs/${SLURM_JOB_ID}

module purge
singularity exec --nv \
             --overlay /scratch/sd5313/M2Lines/overlays/emulator.ext3:ro \
             /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate emulator; cd notebooks; python3 -u 3D_Disk_chunks_emulation.py"
