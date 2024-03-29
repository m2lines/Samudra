#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=12
#SBATCH --error=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.submitit/%j/%j_0_log.err
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=2024-03-26-1GPU_save_data_tensor_16step
#SBATCH --mem=150GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=120
#SBATCH --wckey=submitit

# setup
cd /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.snapshot
export DATA_OVERLAY=
export RESUBMIT_COUNT=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)-ib0
export MASTER_PORT=$(for port in $(shuf -i 30000-65500 -n 20); do if [[ $(netstat -tupln 2>&1 | grep $port | wc -l) -eq 0 ]] ; then echo $port; break; fi; done;)

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.submitit/%j/%j_%t_log.out --error /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.submitit/%j/%j_%t_log.err --cpu-bind=verbose \
 /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.resubmit.sh \
 /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/.python-greene \
 -u -m submitit.core._submit /scratch/sd5313/M2Lines/emulator/Ocean_Emulator/save_data_tensor/2024-03-26-1GPU_save_data_tensor_16step/.submitit/%j
