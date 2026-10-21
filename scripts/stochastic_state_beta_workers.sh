#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
arms=(deterministic initial fresh ar1)
pids=()
run_arm() {
  local arm="$1" gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"
  unset RANK LOCAL_RANK WORLD_SIZE
  export TORCHINDUCTOR_CACHE_DIR="$CAMPAIGN_ROOT/cache/$arm"
  local output="$CAMPAIGN_ROOT/runs/$arm"
  local args=(--root "$CAMPAIGN_ROOT" --arm "$arm"
    --reference "$CAMPAIGN_ROOT/checkpoints/selection-reference.json"
    --updates-per-task "${UPDATES_PER_TASK:-1000}" --decoder-width 64
    --checkpoint-every 250 --wandb-project default --wandb-entity ocean_emulators)
  evaluate() {
    local count="$1" cohort="$2" checkpoint="$3"
    local extra=()
    if [[ "$cohort" == screen ]]; then extra=(--origins 2018-01-01 --skip-validation); fi
    /workspace/.venv/bin/python -m samudra.experiments.stochastic_state_evaluate \
      --data "$CAMPAIGN_ROOT/data/observations" \
      --annual-data "$CAMPAIGN_ROOT/data/annual_observations" \
      --reference "$CAMPAIGN_ROOT/checkpoints/selection-reference.json" \
      --checkpoint "$checkpoint" --arm "$arm" \
      --expected-observation-updates "$count" --expected-om4-updates "$count" \
      --output "$output/evaluation/$cohort-obs$count" "${extra[@]}"
  }
  if [[ "$MODE" == qualify ]]; then
    output="$CAMPAIGN_ROOT/smoke/$arm"
    /workspace/.venv/bin/python -m samudra.experiments.stochastic_state_train \
      "${args[@]}" --output "$output" --smoke --max-new-updates 4 --hours 0.8
    /workspace/.venv/bin/python -m samudra.experiments.stochastic_state_train \
      "${args[@]}" --output "$output" --smoke --resume --max-new-updates 4 --hours 0.8
    evaluate 4 screen "$output/last.pt"
  elif [[ "$MODE" == train ]]; then
    local total="${UPDATES_PER_TASK:-1000}"
    local count=0 previous=0 limit=0 completed=0 cohort=screen
    for count in 125 250 "$total"; do
      if (( count <= previous || count > total )); then continue; fi
      local resume=()
      if [[ -f "$output/last.pt" ]]; then
        resume=(--resume)
        completed=$(/workspace/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$output/state.json")
      fi
      limit=$(( 2 * count - completed ))
      if (( limit > 0 )); then
        /workspace/.venv/bin/python -m samudra.experiments.stochastic_state_train \
          "${args[@]}" --output "$output" --hours 7.5 --max-new-updates "$limit" "${resume[@]}"
      fi
      if (( count == total )); then cohort=final; fi
      printf -v checkpoint '%s/step-%05d.pt' "$output" "$(( 2 * count ))"
      evaluate "$count" "$cohort" "$checkpoint"
      previous="$count"
    done
  else
    echo "Unsupported mode: $MODE" >&2
    return 2
  fi
}
for gpu in 0 1 2 3; do
  run_arm "${arms[$gpu]}" "$gpu" > "$CAMPAIGN_ROOT/logs/$MODE-${arms[$gpu]}-$SLURM_JOB_ID.log" 2>&1 &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else result=1; fi
done
if [[ "$MODE" == qualify && "$result" == 0 ]]; then
  /workspace/.venv/bin/python /pilot-code/scripts/qualify_stochastic_state.py \
    --root "$CAMPAIGN_ROOT" --code-id "$CODE_ID"
fi
exit "$result"
