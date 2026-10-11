#!/usr/bin/env bash
set -Eeuo pipefail
# Explicit publication after reviewing drafts; verify downloaded asset bytes.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step13 "$@"
[[ $# == 1 ]] || die 'Usage: step13.sh --approve-publication'
require_flag "$1" --approve-publication
require_command gh
require_publication_rights
check_commits
verify_assets
require_file "$CTBF_RELEASE_WORK/audit/drafts-created.sha256"
release_fingerprint | cmp -s - "$CTBF_RELEASE_WORK/audit/drafts-created.sha256" \
    || die 'Release identity changed since draft creation.'
download_and_check() {
    local directory="$1"
    mkdir "$directory"
    gh release download "$CTBF_RELEASE_TAG" --repo "$CTBF_GITHUB" --dir "$directory"
    gh release download "$CTBFBIO_RELEASE_TAG" --repo "$CTBFBIO_GITHUB" --dir "$directory"
    cmp "$CTBF_RELEASE_WORK/assets/CTBF-SHA256SUMS" "$directory/CTBF-SHA256SUMS"
    cmp "$CTBF_RELEASE_WORK/assets/CTBFBIO-SHA256SUMS" "$directory/CTBFBIO-SHA256SUMS"
    (cd "$directory"; shasum -a 256 -c CTBF-SHA256SUMS; shasum -a 256 -c CTBFBIO-SHA256SUMS)
}
attempt="$(mktemp -d "$CTBF_RELEASE_WORK/download-check.XXXXXX")"
download_and_check "$attempt/before-publication"
gh release edit "$CTBF_RELEASE_TAG" --repo "$CTBF_GITHUB" --draft=false
gh release edit "$CTBFBIO_RELEASE_TAG" --repo "$CTBFBIO_GITHUB" --draft=false
download_and_check "$attempt/after-publication"
printf 'Published releases and verified downloads. Now update manuscript availability.\n'
