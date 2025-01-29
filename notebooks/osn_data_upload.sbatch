#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=upload_data
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge


singularity exec --nv \
             --overlay /scratch/sd5313/M2Lines/overlays/emulator.ext3:ro \
             /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate emulator; python3 osn_data_upload.py"
