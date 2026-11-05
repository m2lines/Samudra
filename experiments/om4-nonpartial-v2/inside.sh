#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
PYTHON=/workspace/.venv/bin/python
EXPERIMENT="$RUN_ROOT/source/experiments/om4-nonpartial-v2"
cd "$RUN_ROOT"
# Both queued production tasks and the short test share a qualification gate.
# A failure leaves no pass marker and exits before training.
(
  flock -x 9
  if [[ ! -f preparation.json ]]; then
    "$PYTHON" "$EXPERIMENT/prepare.py" "$RUN_ROOT" "$SOURCE_DATA"
  fi
  fingerprint=$(cat commit.txt source.sha256 geometry/mask_manifest.json baseline/train.yaml nonpartial/train.yaml | sha256sum | cut -d' ' -f1)
  if [[ ! -f qualified.sha256 ]] || [[ $(cat qualified.sha256) != "$fingerprint" ]]; then
    "$PYTHON" -m pytest "$RUN_ROOT/source/tests/test_rollout_validation.py" -q
    for candidate in baseline nonpartial; do
      "$PYTHON" "$EXPERIMENT/qualify.py" "$RUN_ROOT/$candidate/train.yaml" "$RUN_ROOT/$candidate/qualification.json"
      "$PYTHON" "$EXPERIMENT/smoke.py" "$RUN_ROOT" "$candidate"
    done
    "$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('.')
a=json.loads((root/'baseline/qualification.json').read_text())
b=json.loads((root/'nonpartial/qualification.json').read_text())
assert a['initial_model_sha256']==b['initial_model_sha256'], 'Initialization differs'
assert a['parameters']==b['parameters']
assert b['empty_channels']==[18,37,56,75]
PY
    printf '%s\n' "$fingerprint" > qualified.sha256.tmp
    mv qualified.sha256.tmp qualified.sha256
  fi
) 9>"$RUN_ROOT/qualification.lock"
if [[ ${QUALIFY_ONLY:-0} == 1 ]]; then exit 0; fi
# Duplicate/recovery submission must not start another training process for an arm.
exec 8>"$RUN_ROOT/$ARM/training.lock"
flock -n -x 8
if [[ ! -f "$RUN_ROOT/$ARM/training.complete" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 -m samudra.train "$RUN_ROOT/$ARM/train.yaml" --backend=nccl
  touch "$RUN_ROOT/$ARM/training.complete"
fi
"$PYTHON" "$EXPERIMENT/run_evals.py" "$RUN_ROOT" "$ARM"
touch "$RUN_ROOT/$ARM/workflow.complete"
