#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Run on a transfer node or Grace allocation AFTER preparation succeeds.
set -euo pipefail
: "${PRODUCT:?Set PRODUCT=duacs|oisst|argo-iap}"
: "${REPO_DIR:?Set REPO_DIR to the checkout used for preparation}"
case "$PRODUCT" in duacs|oisst|argo-iap) ;; *) echo "Invalid PRODUCT" >&2; exit 2 ;; esac
WORK_ROOT="${WORK_ROOT:-/scratch/${USER}/data/obs_full_range}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/prepared}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifests/${PRODUCT}.json}"
OBS_PYTHON="${OBS_PYTHON:-${WORK_ROOT}/venv/bin/python}"
DEST="${DEST:-nyu-osn:emulators/jr7309/data/full_range}"
STORE="${OUTPUT_ROOT}/${PRODUCT}.zarr"
export PYTHONPATH="${REPO_DIR}/data${PYTHONPATH:+:${PYTHONPATH}}"
exec 9>"${WORK_ROOT}/.publish-${PRODUCT}.lock"
flock -n 9 || { echo "Another publication for $PRODUCT is active" >&2; exit 1; }
# Torch uses a module; Grace supplies its native rclone on PATH.
RCLONE_MODULE="${RCLONE_MODULE-rclone/1.72.1}"
if [[ -n "$RCLONE_MODULE" ]]; then module load "$RCLONE_MODULE"; fi
command -v rclone >/dev/null
# The existing OSN bucket permits object access without CreateBucket access.
export RCLONE_S3_NO_CHECK_BUCKET="${RCLONE_S3_NO_CHECK_BUCKET:-true}"
"$OBS_PYTHON" -m ocean_preprocessing.obs_preprocessing full_range validate \
    --manifest_path="$MANIFEST_PATH" --store="$STORE"
scratch=$(mktemp -d "${WORK_ROOT}/.publish-${PRODUCT}.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
rclone lsf "${DEST}/" --max-depth 1 > "$scratch/listing"
# A remote manifest fixes ownership before the first chunk is copied. Refuse
# to mix inventories, including a destination left by a different workflow.
if grep -Fxq "${PRODUCT}.inventory.json" "$scratch/listing"; then
    rclone cat "${DEST}/${PRODUCT}.inventory.json" > "$scratch/inventory.json"
    cmp "$MANIFEST_PATH" "$scratch/inventory.json" || { echo "Remote inventory differs; use another destination" >&2; exit 1; }
elif grep -Fxq "${PRODUCT}.zarr/" "$scratch/listing"; then
    echo "Unmanaged destination already exists; refusing to overwrite" >&2
    exit 1
else
    rclone copyto "$MANIFEST_PATH" "${DEST}/${PRODUCT}.inventory.json"
fi
flags=(--transfers=32 --checkers=16 --progress --stats-one-line)
# Publish consolidated metadata last, after a full read-back comparison.
rclone copy "$STORE" "${DEST}/${PRODUCT}.zarr" --exclude '/.zmetadata' "${flags[@]}"
rclone check "$STORE" "${DEST}/${PRODUCT}.zarr" --exclude '/.zmetadata' --download --checkers=16
rclone copyto "$STORE/.zmetadata" "${DEST}/${PRODUCT}.zarr/.zmetadata"
rclone cat "${DEST}/${PRODUCT}.zarr/.zmetadata" > "$scratch/zmetadata"
cmp "$STORE/.zmetadata" "$scratch/zmetadata"
"$OBS_PYTHON" - "$MANIFEST_PATH" "$scratch/success.json" <<'PY'
import datetime, hashlib, json, pathlib, sys
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
record = dict(product=plan['product'], inventory_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), published_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), verification='rclone check --download; consolidated metadata byte-compared', time_count=plan['time_count'])
pathlib.Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + '\n')
PY
rclone copyto "$scratch/success.json" "${DEST}/${PRODUCT}.SUCCESS.json"
rclone cat "${DEST}/${PRODUCT}.SUCCESS.json" > "$scratch/success.readback.json"
cmp "$scratch/success.json" "$scratch/success.readback.json"
printf 'Published and verified: %s/%s.zarr\n' "$DEST" "$PRODUCT"
