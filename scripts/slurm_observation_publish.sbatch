#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-D-publish
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --exclude=alphagh01
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
root=/mnt/home/jrusak/data/obs_full_range/d-observation-pilot
export PATH="/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin:$HOME/bin:$PATH"
export PYTHONUNBUFFERED=1 RCLONE_S3_NO_CHECK_BUCKET=true
source_dir="$root/samples"
destination=nyu-osn:emulators/jr7309/data/observation_d_pilot/2026-09-22
python - "$source_dir" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'COMPLETE.json').is_file()
expected={'train':243,'validation':9,'test':96}
for split,count in expected.items():
    assert len(list((root/split).glob('*.npz')))==count,split
with (root/'SHA256SUMS').open('w') as out:
    for path in sorted(root.rglob('*.npz')):
        with path.open('rb') as stream:
            value=hashlib.file_digest(stream,'sha256').hexdigest()
        out.write(value+'  '+str(path.relative_to(root))+'\n')
print(json.dumps({'event':'source-hashed','files':len(list(root.rglob('*.npz')))}),flush=True)
PY
rclone copy "$source_dir" "$destination" --transfers=16 --checkers=8 --stats=30s --stats-one-line --log-level=INFO
rclone check "$source_dir" "$destination" --download --checkers=8 --one-way --stats=30s --stats-one-line --log-level=INFO
python - "$source_dir" "$root/PUBLISHED.json" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[1])
files=list(source.rglob('*.npz'))
proof={'published_utc':datetime.datetime.now(datetime.UTC).isoformat(),
       'npz_count':len(files),'npz_bytes':sum(p.stat().st_size for p in files),
       'sha256sums_sha256':hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest(),
       'verification':'rclone check --download --one-way succeeded'}
Path(sys.argv[2]).write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
PY
rclone copyto "$root/PUBLISHED.json" "$destination/PUBLISHED.json"
