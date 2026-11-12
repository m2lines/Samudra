#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# End-to-end: cache 2° OM4 (train 2003-04, val 2010, test 2015), fit the ladder on
# all 77 channels, run the NCD ablation on two channels, render the report.
# Usage: bash run_all.sh [CACHE_DIR] [OUT_DIR]      (add SYNTHETIC=1 for the smoke test)
set -euo pipefail
cd "$(dirname "$0")"
CACHE=${1:-./cache_om4_twodeg}
OUT=${2:-./results_om4_twodeg}
if [[ "${SYNTHETIC:-0}" == 1 ]]; then
  python data.py "$CACHE" --synthetic
  NCD_CHANNELS="thetao_5"
else
  [[ -f "$CACHE/meta.json" ]] || python data.py "$CACHE" --res om4_twodeg
  NCD_CHANNELS="thetao_0 thetao_7"
fi
python run.py "$CACHE" "$OUT" --bins 8192 --n-eval 8000 | tee "$OUT.log"
for c in $NCD_CHANNELS; do python ncd_ablation.py "$CACHE" "$OUT" --channel "$c"; done
python report.py "$OUT" > /dev/null
git rev-parse HEAD > "$OUT/git_hash.txt" 2>/dev/null || true
echo "done: $OUT/tables.md"
