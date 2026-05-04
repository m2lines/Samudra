#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Annotate staged files with SPDX copyright + license headers using `reuse`.
# Defaults by file class:
#   - prose (.md, .rst, .html, .bib) → CC-BY-4.0
#   - non-creative metadata (lockfiles, version pins, ignore files,
#     auto-generated baselines, GitHub Pages markers) → CC0-1.0
#   - everything else (source, configs, scripts, CI, Docker, ...)  → Apache-2.0
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
DATA=()
CODE=()
for f in "${STAGED[@]}"; do
    base=$(basename -- "$f")
    case "$f" in
        *.md|*.rst|*.html|*.bib)
            DOCS+=("$f") ;;
        *.lock|*.lock.license|.gitignore|.dockerignore|.python-version|.nojekyll*)
            DATA+=("$f") ;;
        *)
            case "$base" in
                .secrets.baseline*) DATA+=("$f") ;;
                *)                  CODE+=("$f") ;;
            esac ;;
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
annotate CC0-1.0    "${DATA[@]}"

git add -- "${STAGED[@]}"
