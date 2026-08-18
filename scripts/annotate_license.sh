#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Annotate files with SPDX copyright + license headers using `reuse`.
#
# Usage:
#   scripts/annotate_license.sh                # staged files only (default)
#   scripts/annotate_license.sh data/          # every tracked / non-ignored file under data/
#   scripts/annotate_license.sh src/ docs/     # multiple paths
#   scripts/annotate_license.sh --all          # every tracked / non-ignored file in the repo
#
# Defaults by file class:
#   - prose (.md, .rst, .html, .bib) → CC-BY-4.0
#   - non-creative metadata (lockfiles, version pins, ignore files,
#     auto-generated baselines, GitHub Pages markers) → CC0-1.0
#   - everything else (source, configs, scripts, CI, Docker, ...)  → Apache-2.0
# Files that already carry a SPDX header are left alone.
# When run on staged files, modified files are re-staged so the headers land
# in the same commit; when run on paths, no staging is performed.

set -euo pipefail

COPYRIGHT="Samudra Authors"

FILES=()
RESTAGE=0
if (( $# == 0 )); then
    # Default mode: staged files. Re-stage after annotation.
    RESTAGE=1
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(git diff --cached --name-only --diff-filter=ACM -z)
else
    # Path mode: annotate everything tracked / not-ignored under the given paths.
    # `--all` means the whole repo.
    paths=("$@")
    if [[ "${paths[0]}" == "--all" ]]; then
        paths=(".")
    fi
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(git ls-files --cached --others --exclude-standard -z -- "${paths[@]}")
fi

if (( ${#FILES[@]} == 0 )); then
    echo "No files to annotate."
    exit 0
fi

DOCS=()
DATA=()
CODE=()
for f in "${FILES[@]}"; do
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

if (( RESTAGE )); then
    git add -- "${FILES[@]}"
fi
