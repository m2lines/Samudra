#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Run on Torch DTN after sourcing the rclone module.
set -euo pipefail
source_remote=nyu-osn:emulators/jr7309/data/observation_d_pilot/2026-09-22
target=/scratch/jr7309/data/obs-d-pilot
mkdir -p "$target"
proof="$target/publication-proof.pending"
ready=0
for ((attempt=0; attempt<180; attempt++)); do
    if rclone cat "$source_remote/PUBLISHED.json" --retries=1 --contimeout=10s --timeout=30s > "$proof" 2>/dev/null && test -s "$proof"; then
        python3 - "$proof" <<'PY'
import json,sys
record=json.load(open(sys.argv[1]))
assert record['npz_count']==350,record
assert record['verification']=='rclone check --download --one-way succeeded'
PY
        ready=1
        break
    fi
    sleep 120
done
if ((ready != 1)); then
    echo 'Publication proof did not appear within the six-hour staging window.' >&2
    exit 1
fi
rclone copy "$source_remote" "$target" --transfers=32 --checkers=16 --progress --stats=30s --stats-one-line
cd "$target"
sha256sum --check --status SHA256SUMS
python3 - "$target" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
proof=json.loads((root/'PUBLISHED.json').read_text())
assert hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest()==proof['sha256sums_sha256']
for split,count in [('train',243),('validation',9),('test',96)]:
    assert len(list((root/split).glob('*.npz')))==count,split
record={'ready_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification':'all 350 NPZ files checked against source SHA256SUMS','source_publication':proof}
temp=root/'DATA_READY.tmp'
temp.write_text(json.dumps(record,indent=2)+'\n')
temp.replace(root/'DATA_READY.json')
print(json.dumps(record),flush=True)
PY
