#!/usr/bin/env bash

# Merge the full history of m2lines/ocean_emulators into this repo under data/.
#
# Subcommands:
#   migrate   Clone the source repo, rewrite paths under data/, and merge into
#             the current branch with a single merge commit.
#   rollback  Undo the most recent migrate on the current branch using the SHA
#             saved to .git/merge-ocean-emulators-backup.
#   status    Report whether a backup SHA exists and what it points at.

set -euo pipefail

UPSTREAM_URL="git@github.com:m2lines/ocean_emulators.git"
SUBTREE="data"
REMOTE_NAME="oe-src"
BACKUP_REL=".git/merge-ocean-emulators-backup"

usage() {
    cat <<EOF
Usage: $0 <migrate|rollback|status>

Run from inside the Ocean_Emulator repository on a dedicated feature branch
(not 'main' or 'master'). Requires a clean working tree. 'git-filter-repo' is
invoked via 'uvx'.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

info() {
    echo "==> $*"
}

ensure_in_repo() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "not inside a git repository."
    local toplevel
    toplevel="$(git rev-parse --show-toplevel)"
    cd "$toplevel"
}

current_branch() {
    git rev-parse --abbrev-ref HEAD
}

refuse_protected_branch() {
    local branch
    branch="$(current_branch)"
    case "$branch" in
        main|master)
            die "refusing to run on '${branch}'. Switch to a dedicated feature branch first (this import should land on main only via a reviewed PR)."
            ;;
        HEAD)
            die "detached HEAD; check out a feature branch before running."
            ;;
    esac
}

require_clean_tree() {
    if [ -n "$(git status --porcelain)" ]; then
        die "working tree is not clean. Run 'git status' and commit, stash, or remove changes first."
    fi
}

require_filter_repo() {
    if ! command -v uvx >/dev/null 2>&1; then
        die "'uvx' not found on PATH. Install uv (https://docs.astral.sh/uv/) or install git-filter-repo another way and edit this script."
    fi
    uvx --quiet git-filter-repo --version >/dev/null 2>&1 \
        || die "'uvx git-filter-repo --version' failed. Check network access and uv install."
}

cmd_status() {
    ensure_in_repo
    if [ -f "$BACKUP_REL" ]; then
        local sha
        sha="$(cat "$BACKUP_REL")"
        echo "Backup SHA: $sha"
        echo "Commit summary:"
        git --no-pager log -1 --oneline "$sha" 2>/dev/null || echo "  (SHA not found in this repo)"
    else
        echo "No backup present at $BACKUP_REL"
    fi
    if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
        echo "Leftover remote '$REMOTE_NAME' present: $(git remote get-url "$REMOTE_NAME")"
    fi
}

cmd_migrate() {
    ensure_in_repo
    refuse_protected_branch
    require_clean_tree

    if [ -e "$SUBTREE" ]; then
        die "'${SUBTREE}/' already exists at repo root. Resolve collision before merging."
    fi
    if [ -f "$BACKUP_REL" ]; then
        die "backup already exists at ${BACKUP_REL}. Run '$0 rollback' first, or delete the file to force a new run."
    fi
    if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
        die "git remote '${REMOTE_NAME}' already exists. Remove it with 'git remote remove ${REMOTE_NAME}' before running."
    fi

    require_filter_repo

    local pre_sha
    pre_sha="$(git rev-parse HEAD)"
    info "Current HEAD: $pre_sha"

    local scratch
    scratch="$(mktemp -d -t ocean_emulators_merge.XXXXXX)"
    info "Scratch dir: $scratch"

    # Cleanup on any exit path. The backup file is preserved so rollback works
    # if we failed partway through after the merge commit landed.
    cleanup() {
        if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
            git remote remove "$REMOTE_NAME" || true
        fi
        rm -rf "$scratch"
    }
    trap cleanup EXIT

    info "Cloning $UPSTREAM_URL into scratch"
    git clone --no-local "$UPSTREAM_URL" "$scratch"

    local upstream_default
    upstream_default="$(git -C "$scratch" symbolic-ref --short HEAD)"
    info "Upstream default branch: $upstream_default"

    info "Rewriting paths in scratch clone so all files live under '${SUBTREE}/'"
    ( cd "$scratch" && uvx --quiet git-filter-repo --to-subdirectory-filter "$SUBTREE" )

    info "Saving pre-merge SHA to ${BACKUP_REL}"
    echo "$pre_sha" > "$BACKUP_REL"

    info "Adding scratch clone as remote '${REMOTE_NAME}' and fetching"
    git remote add "$REMOTE_NAME" "$scratch"
    git fetch "$REMOTE_NAME"

    info "Merging ${REMOTE_NAME}/${upstream_default} with --allow-unrelated-histories"
    git merge \
        --allow-unrelated-histories \
        --no-ff \
        -m "Merge m2lines/ocean_emulators history under ${SUBTREE}/" \
        -m "Imports the full history of git@github.com:m2lines/ocean_emulators.git, rewritten so every file lives under ${SUBTREE}/." \
        "${REMOTE_NAME}/${upstream_default}"

    info "Merge complete. New HEAD: $(git rev-parse HEAD)"
    echo
    echo "Next steps:"
    echo "  1. Inspect: git log --oneline -- ${SUBTREE}/ | tail"
    echo "  2. Confirm nothing outside ${SUBTREE}/ changed:"
    echo "       git diff HEAD^1 HEAD -- . ':!${SUBTREE}'"
    echo "  3. When satisfied: git push -u origin $(current_branch)"
    echo "  4. Open PR; merge via 'Create a merge commit' (NOT squash/rebase)."
    echo
    echo "If something looks wrong before pushing, run: $0 rollback"
}

cmd_rollback() {
    ensure_in_repo

    if [ ! -f "$BACKUP_REL" ]; then
        die "no backup found at ${BACKUP_REL}; nothing to roll back."
    fi

    local backup_sha
    backup_sha="$(cat "$BACKUP_REL")"
    if ! git cat-file -e "${backup_sha}^{commit}" 2>/dev/null; then
        die "backup SHA ${backup_sha} is not a commit in this repo. Refusing to reset."
    fi

    require_clean_tree

    local branch
    branch="$(current_branch)"
    local head_sha
    head_sha="$(git rev-parse HEAD)"
    if [ "$head_sha" = "$backup_sha" ]; then
        info "HEAD already at backup SHA. Cleaning up artifacts only."
    else
        # Refuse to roll back if the remote branch has moved past the backup —
        # that would require a force-push the operator should do explicitly.
        local remote_ref="origin/${branch}"
        if git rev-parse --verify --quiet "$remote_ref" >/dev/null; then
            local remote_sha
            remote_sha="$(git rev-parse "$remote_ref")"
            if [ "$remote_sha" != "$backup_sha" ] && [ "$remote_sha" != "$head_sha" ]; then
                info "Remote ${remote_ref} is at ${remote_sha}, neither the backup nor local HEAD."
            fi
            if [ "$remote_sha" != "$backup_sha" ]; then
                echo
                echo "WARNING: '${remote_ref}' is at ${remote_sha}, not ${backup_sha}."
                echo "Local rollback will leave the remote branch ahead of local; you will need"
                echo "to force-push ('git push --force-with-lease') to match. Make sure this is"
                echo "a personal branch (e.g. '${EXPECTED_BRANCH}') that no one else is working"
                echo "on before force-pushing."
                echo
            fi
        fi

        echo "About to: git reset --hard ${backup_sha} on branch '${branch}'."
        echo "Current HEAD (${head_sha}) and any uncommitted work will be discarded."
        read -r -p "Type 'rollback' to proceed: " confirm
        if [ "$confirm" != "rollback" ]; then
            die "aborted by user."
        fi

        info "Resetting ${branch} to ${backup_sha}"
        git reset --hard "$backup_sha"
    fi

    if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
        info "Removing leftover remote '${REMOTE_NAME}'"
        git remote remove "$REMOTE_NAME"
    fi

    info "Removing backup file ${BACKUP_REL}"
    rm -f "$BACKUP_REL"

    info "Rollback complete. HEAD: $(git rev-parse HEAD)"
}

main() {
    if [ $# -lt 1 ]; then
        usage
        exit 1
    fi
    case "$1" in
        migrate) shift; cmd_migrate "$@" ;;
        rollback) shift; cmd_rollback "$@" ;;
        status) shift; cmd_status "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
