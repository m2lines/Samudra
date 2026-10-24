#!/bin/bash
set -euo pipefail
cd /home/jder/.codex/worktrees/233b/Ocean_Emulator
PY=/home/jder/Ocean_Emulator/.venv/bin/python
ROOT=experiments/om4-one-example
trap 'python3 /home/jder/.codex/skills/async-job-followup/scripts/send.py --credential-file /tmp/om4-tiny-sweep-callback.json' EXIT
for variant in baseline final_dilation1 decoder_dilation1 all_dilation1 uncapped exact_upsample; do
  "$PY" -u "$ROOT/tiny.py" --data /tmp/om4-overfit-torch-input --output "$ROOT/results/tiny-seed15/$variant" --variant "$variant" --steps 5000
  test -f "$ROOT/results/tiny-seed15/$variant/COMPLETE"
done
