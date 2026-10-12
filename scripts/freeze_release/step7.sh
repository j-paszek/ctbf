#!/usr/bin/env bash
set -Eeuo pipefail
# Preview an explicit CTBF source selection; stage only with owner opt-in.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step7 "$@"
[[ $# == 0 || ( $# == 1 && "$1" == --stage-reviewed-files ) ]] \
    || die 'Usage: step7.sh [--stage-reviewed-files]'
sources=(LICENSE LICENSE-DATA CITATION.cff requirements-release.txt
    examples/__init__.py examples/quickstart.py examples/tree_gallery.py examples/quickstart.json
    reproducibility/*.py reproducibility/source-pins.json reproducibility/method-map.tsv reproducibility/evidence-map.tsv
    trimmed_truth_evaluation.py algorithm_evaluation/medicc2_reconstruction_preflight.py
    algorithm_evaluation/v5_*.py test/test_v5_*.py test/test_paper_*.py
    test/test_trimmed_truth_evaluation.py test/test_medicc2_reconstruction_preflight.py
    test/test_freezing_release_scripts.py test/test_release_reproduction.py test/test_quickstart.py scripts/freeze_release/*.sh
    simulator_examples/paper_v5/confirmatory_standard.json)
ignored=(README.md LICENSESCOPE.md DATA_ATTRIBUTION.md THIRD_PARTY_NOTICES.md
    presentation/README.md presentation/simulator_manual.md algorithm_evaluation/README.md
    presentation/quickstart-v2/index.html presentation/quickstart-v2/true_tree.html
    presentation/quickstart-v2/trimmed_true_tree.html presentation/quickstart-v2/M0.html
    presentation/quickstart-v2/M1.html presentation/quickstart-v2/M2.html presentation/quickstart-v2/M3.html
    presentation/quickstart-v2/M4.html presentation/quickstart-v2/M5.html
    presentation/quickstart-v2/M6.html presentation/quickstart-v2/M7.html
    presentation/quickstart-v2/config.json presentation/quickstart-v2/input.json
    presentation/quickstart-v2/trees.json presentation/quickstart-v2/summary.json
    reproducibility/REPRODUCE.md reproducibility/SCHEMA.md reproducibility/BUNDLE_README.md
    reproducibility/RELEASE_NOTES_1.2.0.md
    paper/Makefile paper/figures/build_confirmation_figure_v*.py
    paper/figures/build_ktn152_empirical_figure*.py
    paper/figures/connect-then-complete-owner-v2.tex paper/figures/nonidentifiability-v2.tex
    paper/figures/trimmed-true-tree-v1.tex
    paper/review/sections/supplementary-methods/iteration-24/derive_m4_placement.py
    experimental_description/g2_01a_v5_confirmatory_manifest.json
    experimental_description/g2_01a_v5_source_lock.json
    experimental_description/v5_confirmatory_protocol.md
    experimental_description/v5_confirmatory_runbook.md
    experimental_description/v5_partial_comparator_extension_runbook.md
    experimental_description/v5_trimmed_truth_runbook.md freezing_release.md)
printf 'Proposed CTBF source selection (not a dependency-completeness guarantee):\n'
printf '  %s\n' "${sources[@]}" "${ignored[@]}"
printf 'Empirical SVG/data are excluded from code staging; they remain in the gated evidence draft.\n'
printf 'Proposed exclusions from the Git index only (local files preserved):\n'
printf '  %s\n' "${CTBF_LEGACY_EMBEDDED_HTML[@]}"
if [[ $# == 0 ]]; then
    printf 'Preview only. Add reviewed metadata/fixtures separately. No Git index changed.\n'
    exit 0
fi
require_preparation
for file in "${sources[@]}" "${ignored[@]}"; do require_file "$CTBF_ROOT/$file"; done
exclude_legacy_embedded_html_from_index
git add -- "${sources[@]}"
git add -f -- "${ignored[@]}"
git --no-pager diff --cached --stat
git --no-pager diff --cached --name-only
printf 'Review the index; commit/merge remains your manual operation. No commit or push.\n'
