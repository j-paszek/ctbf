#!/usr/bin/env bash
# Shared shell mechanics only. No scientific modules or sealed files changed.
set -Eeuo pipefail
FREEZE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$FREEZE_SCRIPT_DIR/config.sh"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_file() { [[ -s "$1" && ! -L "$1" ]] || die "Missing, empty or symlinked file: $1"; }
require_new() { [[ ! -e "$1" && ! -L "$1" ]] || die "Refusing to overwrite: $1. Keep it and use a fresh attempt/workspace."; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }
require_flag() { [[ "${1:-}" == "$2" ]] || die "This step requires explicit approval: $2"; }

# These obsolete root views embed Plotly.js. Keep working copies, but do not
# carry them into the new release's tracked Git tree. No history rewriting.
CTBF_LEGACY_EMBEDDED_HTML=(nj.html reconstructed.html simulated_tree.html true_tree.html)

exclude_legacy_embedded_html_from_index() {
    local file
    for file in "${CTBF_LEGACY_EMBEDDED_HTML[@]}"; do
        if git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
            # Individual skip-worktree bits can exist even without an active
            # sparse checkout. Authorize only this approved path's index
            # removal; do not change global sparsity rules or working files.
            git rm --cached --sparse -- "$file"
            printf 'Excluded from the next Git snapshot; local copy preserved: %s\n' "$file"
        fi
    done
}

require_no_tracked_embedded_html() {
    local file
    for file in "${CTBF_LEGACY_EMBEDDED_HTML[@]}"; do
        if git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
            die "Release still tracks obsolete third-party-JavaScript view: $file. Review step7 exclusions."
        fi
    done
}

release_identity() {
    printf '%s\t%s\n' CTBF_ROOT "$CTBF_ROOT" CTBFBIO_ROOT "$CTBFBIO_ROOT" \
        CTBF_RELEASE_TAG "$CTBF_RELEASE_TAG" CTBFBIO_RELEASE_TAG "$CTBFBIO_RELEASE_TAG" \
        CTBF_GITHUB "$CTBF_GITHUB" CTBFBIO_GITHUB "$CTBFBIO_GITHUB"
}

start_step() {
    FREEZE_STEP="$1"
    shift
    local name value
    for name in CTBF_ROOT CTBFBIO_ROOT CTBF_RELEASE_WORK; do
        value="${!name}"
        [[ "$value" == /* && "$value" != / && "$value" != *$'\n'* && "$value" != *$'\t'* ]] \
            || die "$name must be an absolute, non-root path without tabs/newlines."
    done
    [[ -d "$CTBF_ROOT" && -d "$CTBFBIO_ROOT" ]] || die 'Check repository paths in scripts/freeze_release/config.sh.'
    CTBF_ROOT="$(cd "$CTBF_ROOT" && pwd -P)"
    CTBFBIO_ROOT="$(cd "$CTBFBIO_ROOT" && pwd -P)"
    # Reject overlap before creating a directory, and again after resolving it.
    case "$CTBF_RELEASE_WORK/" in
        "$CTBF_ROOT/"*|"$CTBFBIO_ROOT/"*) die 'Release workspace must be outside both repositories.';;
    esac
    [[ ! -L "$CTBF_RELEASE_WORK" ]] || die 'Release workspace must not be a symlink.'
    if [[ "$FREEZE_STEP" != step1 ]]; then
        require_file "$CTBF_RELEASE_WORK/release-config.tsv"
    fi
    mkdir -p "$CTBF_RELEASE_WORK"
    CTBF_RELEASE_WORK="$(cd "$CTBF_RELEASE_WORK" && pwd -P)"
    case "$CTBF_RELEASE_WORK/" in
        "$CTBF_ROOT/"*|"$CTBFBIO_ROOT/"*) die 'Resolved release workspace overlaps a repository.';;
    esac
    case "$CTBF_ROOT/|$CTBFBIO_ROOT/" in
        "$CTBF_RELEASE_WORK/"*|*"|$CTBF_RELEASE_WORK/"*) die 'Release workspace must not contain a repository.';;
    esac
    local folder
    for folder in audit assets licence-candidates logs; do
        [[ ! -L "$CTBF_RELEASE_WORK/$folder" ]] || die "Symlinked workspace folder: $folder"
        mkdir -p "$CTBF_RELEASE_WORK/$folder"
    done
    # A real child + ordinary pipeline works on macOS Bash 3.2. In that Bash,
    # waiting on the PID of an exec/process-substitution logger is unreliable.
    # Checking PPID also prevents inherited logging state affecting nested tests.
    if [[ "${CTBF_RELEASE_LOG_PARENT:-}" != "$PPID" ]]; then
        FREEZE_LOG="$(mktemp "$CTBF_RELEASE_WORK/logs/$FREEZE_STEP-$(date +%Y%m%d-%H%M%S).log.XXXXXX")"
        printf 'Starting %s\nLog: %s\n' "$FREEZE_STEP" "$FREEZE_LOG"
        set +e
        CTBF_RELEASE_LOG_PARENT="$$" "$BASH" "$FREEZE_SCRIPT_DIR/$FREEZE_STEP.sh" "$@" 2>&1 | tee "$FREEZE_LOG"
        local statuses=("${PIPESTATUS[@]}") status
        set -e
        status="${statuses[0]}"
        if [[ "${statuses[1]}" != 0 ]]; then
            printf 'ERROR: logger failed for %s\n' "$FREEZE_LOG" >&2
            [[ "$status" != 0 ]] || status="${statuses[1]}"
        fi
        if [[ "$status" == 0 ]]; then
            printf '\ncomplete: %s\nLog: %s\n' "$FREEZE_STEP" "$FREEZE_LOG" | tee -a "$FREEZE_LOG"
        else
            printf '\nFAILED: %s (exit %s)\nLog: %s\n' "$FREEZE_STEP" "$status" "$FREEZE_LOG" | tee -a "$FREEZE_LOG" >&2
        fi
        exit "$status"
    fi
    trap 'printf "ERROR: %s failed at %s:%s (exit %s).\n" "$FREEZE_STEP" "${BASH_SOURCE[0]}" "$LINENO" "$?" >&2' ERR
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ -e "$CTBF_RELEASE_WORK/release-config.tsv" ]]; then
        require_file "$CTBF_RELEASE_WORK/release-config.tsv"
        release_identity | cmp -s - "$CTBF_RELEASE_WORK/release-config.tsv" \
            || die 'Workspace belongs to different paths/tags/repositories. Restore config.sh or use a new workspace.'
    else
        (set -o noclobber; release_identity > "$CTBF_RELEASE_WORK/release-config.tsv")
        printf 'Initialized workspace identity; existing files were preserved.\n'
    fi
    cd "$CTBF_ROOT"
    export CTBF_ROOT CTBFBIO_ROOT CTBF_RELEASE_WORK CTBF_PYTHON
}

require_python() {
    command -v "$CTBF_PYTHON" >/dev/null 2>&1 \
        || die 'Configured Python is unavailable. Edit CTBF_PYTHON in scripts/freeze_release/config.sh.'
    "$CTBF_PYTHON" --version
}

require_preparation() {
    local file
    for file in README.md requirements-release.txt CITATION.cff LICENSE LICENSE-DATA LICENSESCOPE.md \
        DATA_ATTRIBUTION.md THIRD_PARTY_NOTICES.md reproducibility/REPRODUCE.md reproducibility/evidence-map.tsv \
        reproducibility/RELEASE_NOTES_1.2.0.md; do
        require_file "$CTBF_ROOT/$file"
    done
    for file in README.md LICENSE LICENSESCOPE.md THIRD_PARTY_NOTICES.md CITATION.cff RELEASE_NOTES_1.0.0.md; do
        require_file "$CTBFBIO_ROOT/$file"
    done
}

require_publication_rights() {
    require_python
    "$CTBF_PYTHON" -m reproducibility.bundle verify \
        --bundle "$CTBF_RELEASE_WORK/ctbf-v5-reproduction" --public
    require_file "$CTBF_RELEASE_WORK/audit/empirical-rights-review.json"
    "$CTBF_PYTHON" -m reproducibility.bundle verify \
        --bundle "$CTBF_RELEASE_WORK/ctbfbio-ktn152-evidence" --public \
        --rights-review "$CTBF_RELEASE_WORK/audit/empirical-rights-review.json"
}

load_commits() {
    require_file "$CTBF_RELEASE_WORK/audit/selected-commits.tsv"
    local key value
    CTBF_RELEASE_COMMIT= CTBFBIO_RELEASE_COMMIT= CTBFBIO_RELEASE_BRANCH=
    while IFS=$'\t' read -r key value; do
        case "$key" in
            CTBF_RELEASE_COMMIT) CTBF_RELEASE_COMMIT="$value";;
            CTBFBIO_RELEASE_COMMIT) CTBFBIO_RELEASE_COMMIT="$value";;
            CTBFBIO_RELEASE_BRANCH) CTBFBIO_RELEASE_BRANCH="$value";;
            *) die 'Unrecognized selected-commits.tsv field.';;
        esac
    done < "$CTBF_RELEASE_WORK/audit/selected-commits.tsv"
    [[ "$CTBF_RELEASE_COMMIT" =~ ^[a-f0-9]{40,64}$ && "$CTBFBIO_RELEASE_COMMIT" =~ ^[a-f0-9]{40,64}$ ]] \
        || die 'Invalid selected commit identities.'
    [[ -n "$CTBFBIO_RELEASE_BRANCH" ]] || die 'Missing ctbfbio release branch.'
    git check-ref-format --branch "$CTBFBIO_RELEASE_BRANCH" >/dev/null
}

check_commits() {
    load_commits
    require_no_tracked_embedded_html
    [[ "$(git rev-parse HEAD)" == "$CTBF_RELEASE_COMMIT" && "$(git rev-parse refs/heads/main)" == "$CTBF_RELEASE_COMMIT" ]] \
        || die 'CTBF HEAD/main changed since step8. Select and validate a new release candidate.'
    [[ "$(git -C "$CTBFBIO_ROOT" rev-parse HEAD)" == "$CTBFBIO_RELEASE_COMMIT" \
        && "$(git -C "$CTBFBIO_ROOT" rev-parse "refs/heads/$CTBFBIO_RELEASE_BRANCH")" == "$CTBFBIO_RELEASE_COMMIT" ]] \
        || die 'ctbfbio HEAD/release branch changed since step8.'
    git diff --exit-code
    git diff --cached --exit-code
    git -C "$CTBFBIO_ROOT" diff --exit-code
    git -C "$CTBFBIO_ROOT" diff --cached --exit-code
}

verify_assets() {
    require_file "$CTBF_RELEASE_WORK/assets/CTBF-SHA256SUMS"
    require_file "$CTBF_RELEASE_WORK/assets/CTBFBIO-SHA256SUMS"
    (cd "$CTBF_RELEASE_WORK/assets"; shasum -a 256 -c CTBF-SHA256SUMS; shasum -a 256 -c CTBFBIO-SHA256SUMS)
}

release_fingerprint() {
    shasum -a 256 "$CTBF_RELEASE_WORK/release-config.tsv" \
        "$CTBF_RELEASE_WORK/audit/selected-commits.tsv" \
        "$CTBF_RELEASE_WORK/assets/CTBF-SHA256SUMS" \
        "$CTBF_RELEASE_WORK/assets/CTBFBIO-SHA256SUMS" \
        "$CTBF_RELEASE_WORK/audit/empirical-rights-review.json"
}
