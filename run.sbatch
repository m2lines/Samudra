#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=2:00:00
#SBATCH --mem=150GB
#SBATCH --job-name=emu
#SBATCH --output=job_%j.out


module purge


singularity exec --nv \
             --overlay /scratch/sd5313/M2Lines/overlays/emulator.ext3:ro \
             /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate emulator; python3 src/save_data.py"