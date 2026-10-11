#!/usr/bin/env bash
set -Eeuo pipefail
# Archive only the two curated directories after explicit content review.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step9 "$@"
[[ $# == 1 ]] || die 'Usage: step9.sh --approve-bundle-contents'
require_flag "$1" --approve-bundle-contents
require_preparation
require_publication_rights
check_commits
for spec in ctbf-v5-reproduction:1.2.0 ctbfbio-ktn152-evidence:1.0.0; do
    bundle="${spec%:*}"; version="${spec#*:}"
    require_file "$CTBF_RELEASE_WORK/$bundle/README.md"
    require_file "$CTBF_RELEASE_WORK/$bundle/LICENSESCOPE.md"
    require_file "$CTBF_RELEASE_WORK/$bundle/THIRD_PARTY_NOTICES.md"
    require_file "$CTBF_RELEASE_WORK/$bundle/file-rights.tsv"
    [[ ! -L "$CTBF_RELEASE_WORK/$bundle" ]] || die "Symlinked bundle root: $bundle"
    [[ -z "$(find "$CTBF_RELEASE_WORK/$bundle" -type l -print -quit)" ]] || die "Symlinks in bundle: $bundle"
    require_new "$CTBF_RELEASE_WORK/assets/$bundle-$version.tar.gz"
done
require_file "$CTBF_RELEASE_WORK/ctbf-v5-reproduction/ctbf-generated/LICENSE-DATA"
require_file "$CTBF_RELEASE_WORK/ctbf-v5-reproduction/ctbf-generated/DATA_ATTRIBUTION.md"
cmp "$CTBF_ROOT/LICENSE-DATA" "$CTBF_RELEASE_WORK/ctbf-v5-reproduction/ctbf-generated/LICENSE-DATA"
cmp "$CTBF_ROOT/DATA_ATTRIBUTION.md" "$CTBF_RELEASE_WORK/ctbf-v5-reproduction/ctbf-generated/DATA_ATTRIBUTION.md"
printf 'Scope files found. Content/privacy/no-upstream-code review remains part of your explicit approval.\n'
require_new "$CTBF_RELEASE_WORK/assets/CTBF-SHA256SUMS"
require_new "$CTBF_RELEASE_WORK/assets/CTBFBIO-SHA256SUMS"
require_new "$CTBF_RELEASE_WORK/assets/CTBFBIO-RIGHTS-REVIEW.json"
cp "$CTBF_RELEASE_WORK/audit/empirical-rights-review.json" "$CTBF_RELEASE_WORK/assets/CTBFBIO-RIGHTS-REVIEW.json"
for spec in ctbf-v5-reproduction:1.2.0 ctbfbio-ktn152-evidence:1.0.0; do
    bundle="${spec%:*}"; version="${spec#*:}"
    (
        cd "$CTBF_RELEASE_WORK/$bundle"
        manifest="$(mktemp "$CTBF_RELEASE_WORK/audit/$bundle-checksums.XXXXXX")"
        find . -type f ! -path './checksums.sha256' -print0 | xargs -0 shasum -a 256 > "$manifest"
        if [[ -e checksums.sha256 ]]; then
            require_file checksums.sha256
            LC_ALL=C sort checksums.sha256 > "$manifest.previous-sorted"
            LC_ALL=C sort "$manifest" | cmp -s - "$manifest.previous-sorted" \
                || die "Existing bundle manifest omits/changes files: $bundle. Preserve it and review a fresh candidate."
            shasum -a 256 -c checksums.sha256
        else
            ln "$manifest" checksums.sha256
            shasum -a 256 -c checksums.sha256
        fi
    )
    archive="$CTBF_RELEASE_WORK/assets/$bundle-$version.tar.gz"
    COPYFILE_DISABLE=1 tar -czf "$archive" -C "$CTBF_RELEASE_WORK" "$bundle"
    bytes="$(wc -c < "$archive")"
    [[ "$bytes" -lt 2147483648 ]] || die "$archive is too large for a GitHub asset; partition explicitly."
    inventory="$(mktemp "$CTBF_RELEASE_WORK/audit/$bundle-inventory.XXXXXX")"
    tar -tzf "$archive" > "$inventory"
    printf 'Archive inventory: %s\n' "$inventory"
done
(
    cd "$CTBF_RELEASE_WORK/assets"
    shasum -a 256 ctbf-v5-reproduction-1.2.0.tar.gz > CTBF-SHA256SUMS
    shasum -a 256 ctbfbio-ktn152-evidence-1.0.0.tar.gz CTBFBIO-RIGHTS-REVIEW.json > CTBFBIO-SHA256SUMS
)
verify_assets
ls -lh "$CTBF_RELEASE_WORK/assets"
