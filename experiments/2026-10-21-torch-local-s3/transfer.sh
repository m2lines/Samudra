#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
base=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021
src=/scratch/jr7309/data/om4_quarterdeg_v2
dst=nyu-osn-emulators:emulators/jr7309/data/testing/om4_quarterdeg_v2_zstd_20261021
rclone="$base/bin/rclone"
date -Is
hostname
test ! -e "$src/OM4.zarr/.COMPRESSION_IN_PROGRESS"
"$rclone" copy "$src" "$dst" --transfers=32 --checkers=16 --progress --stats-one-line --stats=60s
"$rclone" check "$src" "$dst" --download --checkers=16 --stats=60s --combined "$base/transfer-check.txt"
"$rclone" size "$src" --json > "$base/local-size.json"
"$rclone" size "$dst" --json > "$base/remote-size.json"
python3 - "$base" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'local-size.json').read_text()); b=json.loads((p/'remote-size.json').read_text())
assert a['count']==b['count'] and a['bytes']==b['bytes'], (a,b)
(p/'TRANSFER_COMPLETE.json').write_text(json.dumps({'local':a,'remote':b,'full_download_check':True})+'\n')
PY
date -Is
