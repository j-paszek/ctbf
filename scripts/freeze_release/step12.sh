#!/usr/bin/env bash
set -Eeuo pipefail
# Tag/push and upload draft releases, only after an explicit owner approval.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step12 "$@"
[[ $# == 1 ]] || die 'Usage: step12.sh --approve-reviewed-drafts'
require_flag "$1" --approve-reviewed-drafts
require_command gh
require_preparation
require_publication_rights
check_commits
verify_assets
require_file "$CTBF_RELEASE_WORK/audit/portable-review-approved.sha256"
release_fingerprint | cmp -s - "$CTBF_RELEASE_WORK/audit/portable-review-approved.sha256" \
    || die 'Missing/currently invalid portable-reproduction owner approval from step10.'
check_remote() {
    local directory="$1" repository="$2" url
    url="$(git -C "$directory" remote get-url --push origin)"
    case "$url" in
        "https://github.com/$repository"|"https://github.com/$repository.git"|\
        "git@github.com:$repository"|"git@github.com:$repository.git"|\
        "ssh://git@github.com/$repository"|"ssh://git@github.com/$repository.git") ;;
        *) die "Unexpected push remote in $directory; expected GitHub $repository.";;
    esac
}
check_remote "$CTBF_ROOT" "$CTBF_GITHUB"
check_remote "$CTBFBIO_ROOT" "$CTBFBIO_GITHUB"
gh auth status
ensure_tag() {
    local directory="$1" tag="$2" commit="$3" message="$4"
    git check-ref-format "refs/tags/$tag"
    [[ "$tag" != -* ]] || die 'Tag names must not begin with a dash.'
    if git -C "$directory" show-ref --verify --quiet "refs/tags/$tag"; then
        [[ "$(git -C "$directory" rev-parse "refs/tags/$tag^{commit}")" == "$commit" ]] \
            || die "Existing tag $tag points elsewhere. It will not be moved."
        printf 'Preserving existing matching tag: %s\n' "$tag"
    else
        git -C "$directory" tag -a "$tag" "$commit" -m "$message"
    fi
}
ensure_tag "$CTBF_ROOT" "$CTBF_RELEASE_TAG" "$CTBF_RELEASE_COMMIT" 'CTBF 1.2.0: CTBF v5 paper code and reproducibility release'
ensure_tag "$CTBFBIO_ROOT" "$CTBFBIO_RELEASE_TAG" "$CTBFBIO_RELEASE_COMMIT" 'ctbfbio 1.0.0: frozen KTN152 feasibility inputs and workflows'
git push origin main
git push origin "$CTBF_RELEASE_TAG"
git -C "$CTBFBIO_ROOT" push -u origin "$CTBFBIO_RELEASE_BRANCH"
git -C "$CTBFBIO_ROOT" push origin "$CTBFBIO_RELEASE_TAG"
gh release create "$CTBF_RELEASE_TAG" --repo "$CTBF_GITHUB" --verify-tag --draft \
    --title 'CTBF 1.2.0 — CTBF v5 paper release' \
    --notes-file "$CTBF_ROOT/reproducibility/RELEASE_NOTES_1.2.0.md" \
    "$CTBF_RELEASE_WORK/assets/ctbf-v5-reproduction-1.2.0.tar.gz" "$CTBF_RELEASE_WORK/assets/CTBF-SHA256SUMS"
gh release create "$CTBFBIO_RELEASE_TAG" --repo "$CTBFBIO_GITHUB" --verify-tag --draft \
    --title 'ctbfbio 1.0.0 — KTN152 feasibility release' \
    --notes-file "$CTBFBIO_ROOT/RELEASE_NOTES_1.0.0.md" \
    "$CTBF_RELEASE_WORK/assets/ctbfbio-ktn152-evidence-1.0.0.tar.gz" "$CTBF_RELEASE_WORK/assets/CTBFBIO-SHA256SUMS" \
    "$CTBF_RELEASE_WORK/assets/CTBFBIO-RIGHTS-REVIEW.json"
release_fingerprint > "$CTBF_RELEASE_WORK/audit/drafts-created.sha256"
gh release view "$CTBF_RELEASE_TAG" --repo "$CTBF_GITHUB"
gh release view "$CTBFBIO_RELEASE_TAG" --repo "$CTBFBIO_GITHUB"
printf 'Drafts created. Review both on GitHub before step13. No release was published.\n'
