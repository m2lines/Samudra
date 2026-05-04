#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

preempt_handler()
{
    #place here: commands to run when preempt signal (SIGTERM) arrives from slurm
    kill -TERM ${1} #forward SIGTERM signal to the user application
    #if --requeue was used, slurm will automatically do so here
}

#
set -e

# Load conda environment
source .venv/bin/activate

# Set master node information
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400

# Debug info
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"

# Run the training script
python src/ocean_emulators/train.py \
     configs/slurm_perlmutter_train_om4.yaml \
     --experiment.name $SLURM_JOB_NAME

pid=$!
trap "preempt_handler '$pid'" SIGTERM #this catches preempt SIGTERM from slurm
wait
sleep 120 #keep the job step alive until slurm sends SIGKILL