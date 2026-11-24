#!/bin/bash
#SBATCH --job-name=om4-memory-diagnostics
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT=/orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="$ROOT/matplotlib"
cd "$ROOT/code-v4"
sha256sum -c "$ROOT/checkpoint.sha256"
"$PY" -u diagnose.py --data "$ROOT/diagnostic-data-archive" \
  --checkpoint "$ROOT/om4-official-onedeg-ema.pt" \
  --reference "$ROOT/om4-official-first-prediction.npz" \
  --output "$ROOT/diagnostics"
"$PY" report.py --results "$ROOT/diagnostics"
for arm in full detach latent_full latent_only; do
  "$PY" -u train.py --data "$ROOT/diagnostic-data-archive" --arm "$arm" \
    --output "$ROOT/smoke/$arm" --smoke --batch-size=2 --microbatch=2 --workers=0
done
"$PY" - <<'PY'
import wandb
run = wandb.init(entity='ocean_emulators', project='default',
    group='om4-temporal-memory-20261016', name='om4-memory-qualification', mode='online')
assert run.url, 'Online W&B preflight failed'
print('WANDB_QUALIFICATION_URL', run.url, flush=True)
run.log({'qualification/passed': 1})
run.finish()
PY
touch "$ROOT/QUALIFIED"
