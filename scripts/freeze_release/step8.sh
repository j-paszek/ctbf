#!/usr/bin/env bash
set -Eeuo pipefail
# Persist selected commits for subsequent scripts, without changing Git.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step8 "$@"
[[ $# == 0 ]] || die 'Usage: step8.sh'
require_preparation
require_no_tracked_embedded_html
[[ "$(git branch --show-current)" == main ]] || die 'Manually select/merge the reviewed release on main before step8.'
git diff --exit-code
git diff --cached --exit-code
git -C "$CTBFBIO_ROOT" diff --exit-code
git -C "$CTBFBIO_ROOT" diff --cached --exit-code
git status --short
git -C "$CTBFBIO_ROOT" status --short
git ls-files --error-unmatch algorithm_evaluation/v5_confirmation_run.py \
    algorithm_evaluation/v5_partial_comparator_extension_2m_nested.py trimmed_truth_evaluation.py \
    paper/figures/build_confirmation_figure_v6.py paper/figures/build_ktn152_empirical_figure_v3.py \
    experimental_description/g2_01a_v5_source_lock.json requirements-release.txt CITATION.cff LICENSE \
    LICENSE-DATA LICENSESCOPE.md DATA_ATTRIBUTION.md THIRD_PARTY_NOTICES.md \
    reproducibility/REPRODUCE.md reproducibility/evidence-map.tsv reproducibility/source-pins.json \
    reproducibility/bundle.py reproducibility/reproduce.py reproducibility/tables.py \
    reproducibility/method-map.tsv test/test_release_reproduction.py scripts/freeze_release/common.sh \
    examples/__init__.py examples/quickstart.py examples/tree_gallery.py examples/quickstart.json \
    test/test_quickstart.py presentation/README.md presentation/simulator_manual.md \
    presentation/quickstart-v2/index.html presentation/quickstart-v2/true_tree.html \
    presentation/quickstart-v2/trimmed_true_tree.html presentation/quickstart-v2/M0.html \
    presentation/quickstart-v2/M1.html presentation/quickstart-v2/M2.html presentation/quickstart-v2/M3.html \
    presentation/quickstart-v2/M4.html presentation/quickstart-v2/M5.html \
    presentation/quickstart-v2/M6.html presentation/quickstart-v2/M7.html \
    presentation/quickstart-v2/config.json presentation/quickstart-v2/input.json \
    presentation/quickstart-v2/trees.json presentation/quickstart-v2/summary.json
CTBF_RELEASE_COMMIT="$(git rev-parse HEAD)"
CTBFBIO_RELEASE_COMMIT="$(git -C "$CTBFBIO_ROOT" rev-parse HEAD)"
CTBFBIO_RELEASE_BRANCH="$(git -C "$CTBFBIO_ROOT" branch --show-current)"
[[ -n "$CTBFBIO_RELEASE_BRANCH" ]] || die 'ctbfbio must be on the intended release branch.'
candidate="$(mktemp "$CTBF_RELEASE_WORK/audit/selected-commits-candidate.XXXXXX")"
printf '%s\t%s\n' CTBF_RELEASE_COMMIT "$CTBF_RELEASE_COMMIT" \
    CTBFBIO_RELEASE_COMMIT "$CTBFBIO_RELEASE_COMMIT" CTBFBIO_RELEASE_BRANCH "$CTBFBIO_RELEASE_BRANCH" > "$candidate"
if [[ -e "$CTBF_RELEASE_WORK/audit/selected-commits.tsv" ]]; then
    cmp -s "$candidate" "$CTBF_RELEASE_WORK/audit/selected-commits.tsv" \
        || die 'A different candidate is already selected. Preserve it and use a fresh release workspace.'
else
    ln "$candidate" "$CTBF_RELEASE_WORK/audit/selected-commits.tsv"
fi
printf 'Selected commits are saved in audit/selected-commits.tsv; no exported shell state is needed.\n'
