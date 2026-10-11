#!/usr/bin/env bash
set -Eeuo pipefail
# Prepare a fresh clean-clone test, then separately record owner review.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step10 "$@"
require_preparation
check_commits
verify_assets
if [[ "${1:-}" == --accept-portable-review ]]; then
    [[ $# == 2 ]] || die 'Usage: step10.sh --accept-portable-review /absolute/path/to/review.txt'
    require_file "$2"
    require_file "$CTBF_RELEASE_WORK/audit/clean-clone-passed.sha256"
    release_fingerprint | cmp -s - "$CTBF_RELEASE_WORK/audit/clean-clone-passed.sha256" \
        || die 'Commit/asset identity changed since the clean-clone test.'
    require_new "$CTBF_RELEASE_WORK/audit/portable-review-approved.sha256"
    release_fingerprint > "$CTBF_RELEASE_WORK/audit/portable-review-approved.sha256"
    shasum -a 256 "$2" > "$CTBF_RELEASE_WORK/audit/portable-review-evidence.sha256"
    printf 'Recorded YOUR approval of the portable all-table/figure reproduction.\n'
    printf 'This is an owner attestation, not an automatic check of the review contents.\n'
    exit 0
fi
[[ $# == 0 ]] || die 'Usage: step10.sh [--accept-portable-review /absolute/path/to/review.txt]'
require_python
attempt="$(mktemp -d "$CTBF_RELEASE_WORK/clean-check.XXXXXX")"
printf 'Fresh clean-clone attempt: %s\n' "$attempt"
git clone --no-hardlinks "$CTBF_ROOT" "$attempt/ctbf"
git -C "$attempt/ctbf" checkout --detach "$CTBF_RELEASE_COMMIT"
"$CTBF_PYTHON" -m venv "$attempt/env"
clean_python="$attempt/env/bin/python"
"$clean_python" -m pip install -r "$attempt/ctbf/requirements-release.txt"
"$clean_python" -m pip check
export MPLBACKEND=Agg
export MPLCONFIGDIR="$attempt/matplotlib"
export XDG_CACHE_HOME="$attempt/cache"
(
    unset PYTHONPATH PYTHONHOME
    cd "$attempt/ctbf"
    "$clean_python" -m pytest -q test/test_reconstructor_classical_nj.py \
        test/test_paper_confirmation_figure_v6.py test/test_paper_empirical_figure_v3.py \
        test/test_freezing_release_scripts.py test/test_release_reproduction.py
)
mkdir "$attempt/unpacked"
tar -xzf "$CTBF_RELEASE_WORK/assets/ctbf-v5-reproduction-1.2.0.tar.gz" -C "$attempt/unpacked"
tar -xzf "$CTBF_RELEASE_WORK/assets/ctbfbio-ktn152-evidence-1.0.0.tar.gz" -C "$attempt/unpacked"
for bundle in ctbf-v5-reproduction ctbfbio-ktn152-evidence; do
    (cd "$attempt/unpacked/$bundle"; shasum -a 256 -c checksums.sha256)
done
(
    unset PYTHONPATH PYTHONHOME
    cd "$attempt/ctbf"
    "$clean_python" -m reproducibility.example
    "$clean_python" -m reproducibility.reproduce \
        --bundle "$attempt/unpacked/ctbf-v5-reproduction" \
        --empirical-bundle "$attempt/unpacked/ctbfbio-ktn152-evidence" \
        --output "$attempt/reproduced" --figures --vectors
)
if [[ -e "$CTBF_RELEASE_WORK/audit/clean-clone-passed.sha256" ]]; then
    release_fingerprint | cmp -s - "$CTBF_RELEASE_WORK/audit/clean-clone-passed.sha256" \
        || die 'Existing clean-clone receipt differs. Use a fresh release workspace.'
else
    release_fingerprint > "$CTBF_RELEASE_WORK/audit/clean-clone-passed.sha256"
fi
printf '\nClean-clone smoke, asset checks and portable exports passed.\n'
printf 'Review all outputs in %s/reproduced; owner visual/portability approval is STILL REQUIRED.\n' "$attempt"
