#!/usr/bin/env bash
set -Eeuo pipefail
# Rebuild stored-score derivatives in a new attempt, never in paper/results.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step5 "$@"
[[ $# == 0 ]] || die 'Usage: step5.sh'
require_python
export MPLBACKEND=Agg
export MPLCONFIGDIR="$CTBF_RELEASE_WORK/audit/matplotlib"
export XDG_CACHE_HOME="$CTBF_RELEASE_WORK/audit/cache"
attempt="$(mktemp -d "$CTBF_RELEASE_WORK/derivatives.XXXXXX")"
confirmation="$CTBF_ROOT/algorithm_evaluation/results/v5_confirmation/confirmation_v1"
extension="$CTBF_ROOT/algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_2m_v1"
printf 'Fresh derivative outputs: %s\n' "$attempt"
"$CTBF_PYTHON" -m algorithm_evaluation.v5_confirmation_publication \
    --bank-root "$confirmation/bank" --run-root "$confirmation/internal" \
    --medicc2-root "$confirmation/medicc2" --report-root "$confirmation/report" \
    --output-root "$attempt/publication-v1"
"$CTBF_PYTHON" paper/figures/build_confirmation_figure_v3.py \
    --run-root "$confirmation/internal" --source paper/figures/v5-confirmation-v1 \
    --output-root "$attempt/figure-v3"
"$CTBF_PYTHON" paper/figures/build_confirmation_figure_v4.py \
    --run-root "$confirmation/internal" --source paper/figures/v5-confirmation-v1 \
    --previous paper/figures/v5-confirmation-v3 --output-root "$attempt/figure-v4"
"$CTBF_PYTHON" paper/figures/build_confirmation_figure_v5.py \
    --extension-root "$extension" --output-root "$attempt/figure-v5"
"$CTBF_PYTHON" paper/figures/build_confirmation_figure_v6.py \
    --extension-root "$extension" --figures-root paper/figures --output-root "$attempt/figure-v6"
compare_csv() {
    cmp "$1" "$2"
    printf 'Exact CSV match: %s\n' "$1"
}
compare_csv paper/figures/v5-confirmation-v1/main-figure-data.csv "$attempt/publication-v1/main-figure-data.csv"
compare_csv paper/figures/v5-confirmation-v3/biopsy-count-curves.csv "$attempt/figure-v3/biopsy-count-curves.csv"
compare_csv paper/figures/v5-confirmation-v4/closed-state-grf-curves.csv "$attempt/figure-v4/closed-state-grf-curves.csv"
compare_csv paper/figures/v5-confirmation-v5/partial-matched-curves.csv "$attempt/figure-v5/partial-matched-curves.csv"
for csv in native-biopsy-count-curves partial-trimmed-biopsy-count-curves closed-trimmed-height-curves; do
    compare_csv "paper/figures/v5-confirmation-v6/$csv.csv" "$attempt/figure-v6/$csv.csv"
done
printf 'Original-workspace derivative check only; all-table/portable verification remains separate.\n'
