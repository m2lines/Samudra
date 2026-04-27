#!/usr/bin/env bash

# Migrate FOMO datasets from nyu-osn:emulators (private) to nyu-osn:m2lines-pubs
# (public read), rewriting the layout to the FOMO/ publication structure.

set -euo pipefail

SOURCE_REMOTE="nyu-osn:emulators"
DEST_REMOTE="nyu-osn-public:m2lines-pubs/FOMO"

# Each entry is "<source-subpath>|<destination-subpath>", relative to the
# remotes above. `rclone copy SRC DST` copies the *contents* of SRC into DST,
# so point both sides at directories (or both at single objects).
MIGRATIONS=(
    "jbusecke/ocean_emulators/OM4/OM4_raw_test.zarr|raw/om4_5daily.zarr"
    "am16581/ocean_static_no_mask_table.zarr|raw/ocean_static_no_mask_table.zarr"
    "am16581/grids|raw/grids"
    "am16581/data/2025-11/om4_onedeg_blur_v8|v2025-11/om4_onedeg_filter"
    "am16581/data/2025-11/om4_onedeg_v3|v2025-11/om4_onedeg"
    "am16581/data/2025-11/om4_halfdeg_v4|v2025-11/om4_halfdeg"
    "am16581/data/2025-11/om4_quarterdeg_v2|v2025-11/om4_quarterdeg"
)

show_usage() {
    cat <<EOF
Usage: $0 [--dry-run] [-- <extra rclone args>...]

Copy each configured dataset from '${SOURCE_REMOTE}' to '${DEST_REMOTE}'
using 'rclone copy'. Runs for real by default.

Options:
  --dry-run   Pass --dry-run to rclone (no data is transferred).
  -h, --help  Show this message.

Any arguments after '--' are forwarded to every rclone invocation, e.g.:
  $0 -- --transfers 32 --checkers 32
EOF
}

DRY_RUN=0
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: 'rclone' not found on PATH." >&2
    exit 1
fi

RCLONE_ARGS=(--progress --fast-list)
if [ "$DRY_RUN" -eq 1 ]; then
    RCLONE_ARGS+=(--dry-run)
    echo "DRY RUN: no data will be transferred."
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    RCLONE_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Source:      ${SOURCE_REMOTE}"
echo "Destination: ${DEST_REMOTE}"
echo "Transfers:   ${#MIGRATIONS[@]}"
echo ""

FAILED=()
for entry in "${MIGRATIONS[@]}"; do
    src_sub="${entry%%|*}"
    dst_sub="${entry##*|}"
    src="${SOURCE_REMOTE}/${src_sub}"
    dst="${DEST_REMOTE}/${dst_sub}"

    echo "=== ${src_sub}"
    echo "  -> ${dst_sub}"
    if rclone copy "$src" "$dst" "${RCLONE_ARGS[@]}"; then
        echo "  ✅  Done"
    else
        echo "  ❌  Failed"
        FAILED+=("$src_sub")
    fi
    echo ""
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Migration finished with ${#FAILED[@]} failure(s):"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo "Migration complete."
