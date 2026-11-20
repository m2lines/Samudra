#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /scratch/jr7309/skyline-large-high-lr-20261120/env.sh
export SKYLINE_CONFIG="$SKYLINE_ROOT/train.json"
export SKYLINE_OUTPUT="$SKYLINE_ROOT/all-fields-60k"
exec bash "$SKYLINE_ROOT/stage.sbatch"
