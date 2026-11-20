#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
source /scratch/jr7309/skyline-small-lr-sweep-20261120/env.sh
python3 "$HOME/.config/codex-followup/send.py" --host torch --event-id "$SKYLINE_CALLBACK" --check
for arm in 1x 2x 4x; do
    export SKYLINE_CONFIG="$SKYLINE_ROOT/$arm-qualify.json"
    export SKYLINE_OUTPUT="$SKYLINE_ROOT/$arm-qualification"
    bash "$SKYLINE_ROOT/stage.sbatch"
done
