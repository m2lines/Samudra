#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Protocol v2, end to end: cache 2-degree OM4 on Samudra's 2-degree splits, sweep
# the bin tolerance eps, pick it by the rule in select_eps.py, fit the ladder on
# all 77 channels, render the report, run the sanity checks.
# Usage: bash run_all.sh [CACHE_DIR] [OUT_DIR]      (add SYNTHETIC=1 for the smoke test)
set -euo pipefail
cd "$(dirname "$0")"
CACHE=${1:-./cache_om4_twodeg_v2}
OUT=${2:-./results_om4_twodeg_v2}
SWEEP_CHANNELS=thetao_0,so_0,uo_0,vo_0,zos,thetao_10,so_10,uo_10,vo_10
mkdir -p "$OUT"
# Per-stage wall-clock seconds, appended to $OUT/timings.tsv for the report.
stage() { local t0=$SECONDS; "${@:2}"; printf "%s\t%d\n" "$1" $((SECONDS - t0)) >> "$OUT/timings.tsv"; }
: > "$OUT/timings.tsv"
if [[ "${SYNTHETIC:-0}" == 1 ]]; then
  stage data python data.py "$CACHE" --synthetic
else
  [[ -f "$CACHE/meta.json" ]] || stage data python data.py "$CACHE" --res om4_twodeg
fi
sweep() {
  for e in 0.5 0.25 0.125 0.0625 0.03125 0.015625; do
    python run.py "$CACHE" "$OUT/eps_sweep/eps_$e" --eps "$e" --channels "$SWEEP_CHANNELS" --no-diagnostics
  done
}
stage eps_sweep sweep
EPS=$(python select_eps.py "$OUT/eps_sweep")
echo "chosen eps: $EPS"
stage run python run.py "$CACHE" "$OUT" --eps "$EPS" --n-eval 8000 > >(tee "$OUT.log")
stage report python report.py "$OUT" > /dev/null
stage sanity python sanity.py "$CACHE" "$OUT/results.json" "$OUT/sanity.json"
git rev-parse HEAD > "$OUT/git_hash.txt" 2>/dev/null || true
echo "done: $OUT/tables.md"
