#!/bin/bash
# Copy a directory tree (e.g. a zarr store) with N parallel rsync streams.
#
# Usage: copy_tree_parallel.sh SRC DST [STREAMS]
#
# Restartable: rsync skips files whose size+mtime already match, and writes each
# file to a temp name before renaming, so a killed job never leaves a truncated
# chunk behind. Never deletes anything in DST. Rerun the same command to resume.

set -euo pipefail

SRC="${1:?usage: $0 SRC DST [STREAMS]}"
DST="${2:?usage: $0 SRC DST [STREAMS]}"
STREAMS="${3:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-300}"
SRC="${SRC%/}"
DST="${DST%/}"

[[ -d "${SRC}" ]] || { echo "ERROR: source ${SRC} does not exist" >&2; exit 1; }
mkdir -p "${DST}"

# -rlt, not -a: files inherit the destination's group (setgid dir / group quota)
# instead of carrying over the source group. Times are kept for the resume check.
RSYNC=(rsync -rlt --whole-file)

LIST_DIR="$(mktemp -d "${DST}.copy_lists.XXXXXX")"
trap 'kill "${MONITOR_PID:-}" 2>/dev/null; rm -rf "${LIST_DIR}"' EXIT

echo "$(date +%T) listing ${SRC}"
(cd "${SRC}" && find . -type f) > "${LIST_DIR}/all"
TOTAL_FILES="$(wc -l < "${LIST_DIR}/all")"
echo "$(date +%T) ${TOTAL_FILES} files, ${STREAMS} streams"

# Round-robin so each stream gets a similar mix of chunk sizes.
split -n "r/${STREAMS}" -d -a 3 "${LIST_DIR}/all" "${LIST_DIR}/shard."

# Each stream logs the size of every file it copies; the monitor sums those.
# Counts only files copied this run (already-present files are skipped silently).
START="$(date +%s)"
(
  while sleep "${PROGRESS_EVERY}"; do
    cat "${LIST_DIR}"/shard.*.log 2>/dev/null | awk -v total="${TOTAL_FILES}" -v t="$(( $(date +%s) - START ))" '
      { n++; b += $1 }
      END { gbs = b / 1e9 / (t > 0 ? t : 1)
            eta = (n > 0) ? (total - n) * t / n / 3600 : 0
            printf "%s progress: %d/%d files (%.1f%%), %.2f TB, %.2f GB/s, ETA %.1f h\n",
                   strftime("%T"), n, total, 100 * n / total, b / 1e12, gbs, eta }'
  done
) &
MONITOR_PID=$!; disown

ls "${LIST_DIR}"/shard.??? | xargs -P "${STREAMS}" -I{} \
  bash -c '"${@:4}" --out-format=%l --files-from="$1" "$2/" "$3/" > "$1.log"' _ {} "${SRC}" "${DST}" "${RSYNC[@]}"
echo "$(date +%T) parallel pass done"

# Serial catch-up/verify pass: copies anything missed (e.g. empty dirs) and
# should report ~0 files transferred.
"${RSYNC[@]}" --stats "${SRC}/" "${DST}/" | grep -E "Number of (regular )?files( transferred)?:|Total transferred"
echo "$(date +%T) done: ${DST}"
