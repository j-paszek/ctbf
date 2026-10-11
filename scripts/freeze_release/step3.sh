#!/usr/bin/env bash
set -Eeuo pipefail
# Local preparation only. No packaging, Git mutation or publication.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step3 "$@"
[[ $# == 0 || ( $# == 1 && "$1" == --prepare-bundles ) ]] || die 'Usage: step3.sh [--prepare-bundles]'
require_preparation
printf 'Required metadata files exist. Clean-clone and rights reviews remain release gates.\n'
if [[ $# == 0 ]]; then
    printf 'Next: ./scripts/freeze_release/step3.sh --prepare-bundles\n'
    exit 0
fi
require_python
for bundle in ctbf-v5-reproduction ctbfbio-ktn152-evidence; do
    target="$CTBF_RELEASE_WORK/$bundle"
    if [[ -e "$target" || -L "$target" ]]; then
        "$CTBF_PYTHON" -m reproducibility.bundle verify --bundle "$target"
        printf 'Preserved existing verified bundle: %s\n' "$target"
    elif [[ "$bundle" == ctbfbio-ktn152-evidence ]]; then
        "$CTBF_PYTHON" -m reproducibility.bundle prepare --ctbf-root "$CTBF_ROOT" \
            --bio-root "$CTBFBIO_ROOT" --output "$target" --empirical-draft
    else
        "$CTBF_PYTHON" -m reproducibility.bundle prepare --ctbf-root "$CTBF_ROOT" \
            --bio-root "$CTBFBIO_ROOT" --output "$target"
    fi
done
attempt="$(mktemp -d "$CTBF_RELEASE_WORK/portable-local.XXXXXX")"
"$CTBF_PYTHON" -m reproducibility.reproduce \
    --bundle "$CTBF_RELEASE_WORK/ctbf-v5-reproduction" \
    --empirical-bundle "$CTBF_RELEASE_WORK/ctbfbio-ktn152-evidence" \
    --output "$attempt/reproduced" --figures \
    --check-supplement "$CTBF_ROOT/paper/supplement/supplement.tex"
printf 'Review local results: %s/reproduced\n' "$attempt"
printf 'Empirical bundle remains LOCAL ONLY pending a recorded rights review. No archives or uploads.\n'
