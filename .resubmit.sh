#!/bin/bash

# This script is only expected to be called on a worker.
# If this script is used, the submitted command will be wrapped to
# enable resubmission.

# Somewhat hacky, this expects SUBMITTED_COMMAND to not be modified between
# the submission of the SBATCH call and the execution of .resubmit.sh. 
# Ideally 
export SOURCE_SLURM_COMMAND=$SUBMITTED_COMMAND
command=$@

if [[ "$RESUBMIT_COUNT" == "" ]]; then
    export RESUBMIT_COUNT=0
fi
echo "RESUBMIT: resubmits remaining -- $RESUBMIT_COUNT"

resubmit () 
{ 
    echo "RESUBMIT: USR2 signal received. Job is timing out."
    if [[ "$RESUBMIT_COUNT" -le 0 ]]; then
        echo "RESUBMIT_COUNT reached $RESUBMIT_COUNT. Exiting"
        exit
    fi

    if [[ "$SLURM_PROCID" -ge 1 ]]; then
        echo "SLURM_PROCID:$SLURM_PROCID exiting as it is not the master process."
        exit
    fi

    # Remove previously added values
    SOURCE_SLURM_COMMAND=${SOURCE_SLURM_COMMAND/--mem=* }
    SOURCE_SLURM_COMMAND=${SOURCE_SLURM_COMMAND/--array=* }
    SOURCE_SLURM_COMMAND=${SOURCE_SLURM_COMMAND/--cpus-per-task=* }
    SOURCE_SLURM_COMMAND=$(printf '%s' "$SOURCE_SLURM_COMMAND" | sed 's/compute.resubmit=[0-9]\+ \?//g')

    local SLURM_RESUBMIT_ARGS=""
    if [ "${SLURM_ARRAY_TASK_ID}" != "" ]; then
        SLURM_RESUBMIT_ARGS="--array=${SLURM_ARRAY_TASK_ID} "
    fi

    SLURM_RESUBMIT_ARGS="--cpus-per-task=$SLURM_CPUS_PER_TASK $SLURM_RESUBMIT_ARGS "
    SLURM_RESUBMIT_ARGS="--mem=$SLURM_MEM_PER_NODE $SLURM_RESUBMIT_ARGS "
    export SOURCED=""

    export RESUBMIT_COUNT=$((RESUBMIT_COUNT-1))
    echo "RESUBMIT: resubmitting with command -- $SOURCE_SLURM_COMMAND compute.resubmit=$RESUBMIT_COUNT $SLURM_RESUBMIT_ARGS"

    $SOURCE_SLURM_COMMAND compute.resubmit=$RESUBMIT_COUNT $SLURM_RESUBMIT_ARGS &

    sleep 30
    scancel $SLURM_JOB_ID
    exit
}

trap resubmit USR2
$command &
wait