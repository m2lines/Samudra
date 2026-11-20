#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /scratch/jr7309/skyline-large-classifier-20261112/env.sh
export SKYLINE_CONFIG="$SKYLINE_ROOT/train.json"
export SKYLINE_OUTPUT="$SKYLINE_ROOT/all-fields-25k"
cd "$SKYLINE_ROOT"
sha256sum -c configs.sha256
exec bash "$SKYLINE_ROOT/stage.sbatch"
