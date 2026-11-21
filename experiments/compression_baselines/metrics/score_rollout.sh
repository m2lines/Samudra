#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Score an existing rollout against observations with Samudra's own metrics,
# exactly as for a Samudra eval: python -m samudra.metrics reads
# BASE/NAME/predictions.zarr and writes its frame, scalars and figures there.
# Usage: bash score_rollout.sh NAME PREDICTIONS_ZARR BASE [PYTHON]
set -euo pipefail
NAME=$1; PRED=$2; BASE=$3; PY=${4:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
mkdir -p "$BASE/$NAME"
[[ -e "$BASE/$NAME/predictions.zarr" ]] || ln -s "$(realpath "$PRED")" "$BASE/$NAME/predictions.zarr"
"$PY" -m samudra.metrics samudra_om4_v2/eval.yaml \
  --experiment.name="$NAME" \
  --experiment.base_output_dir="$BASE" \
  --data=@"$HERE/data_official_onedeg.yaml" \
  --observations=@"$REPO/src/samudra/configs/data/obs.yaml"
