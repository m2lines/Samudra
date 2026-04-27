#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Annotate staged files with SPDX copyright + license headers using `reuse`.
# Code defaults to Apache-2.0; prose (.md, .rst) defaults to CC-BY-4.0.
# Files that already carry a SPDX header are left alone.
# After annotation, modified files are re-staged so the headers land in the
# same commit.

set -euo pipefail

COPYRIGHT="Ocean Emulator Authors"

STAGED=()
while IFS= read -r -d '' f; do
    STAGED+=("$f")
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if (( ${#STAGED[@]} == 0 )); then
    echo "No staged files to annotate."
    exit 0
fi

DOCS=()
CODE=()
for f in "${STAGED[@]}"; do
    case "$f" in
        *.md|*.rst) DOCS+=("$f") ;;
        *)          CODE+=("$f") ;;
    esac
done

annotate() {
    local license=$1; shift
    (( $# == 0 )) && return 0
    uvx reuse annotate \
        --copyright "$COPYRIGHT" \
        --license "$license" \
        --skip-existing \
        --skip-unrecognised \
        "$@"
}

annotate Apache-2.0 "${CODE[@]}"
annotate CC-BY-4.0  "${DOCS[@]}"

git add -- "${STAGED[@]}"
