#!/usr/bin/env bash
set -Eeuo pipefail
# Bounded local verification; the broad regression suite is explicit opt-in.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step4 "$@"
[[ $# == 0 || ( $# == 1 && "$1" == --full ) ]] || die 'Usage: step4.sh [--full]'
require_python
export MPLBACKEND=Agg
export MPLCONFIGDIR="$CTBF_RELEASE_WORK/audit/matplotlib"
export XDG_CACHE_HOME="$CTBF_RELEASE_WORK/audit/cache"
"$CTBF_PYTHON" -m pip check
"$CTBF_PYTHON" -c 'import sys, numpy, networkx, matplotlib, plotly, psutil; print(sys.executable); print(sys.version)'
environment="$(mktemp "$CTBF_RELEASE_WORK/audit/local-environment.XXXXXX")"
"$CTBF_PYTHON" -m pip freeze --all > "$environment"
printf 'Local environment (review private paths before sharing): %s\n' "$environment"
"$CTBF_PYTHON" - <<'PY'
import networkx as nx
import numpy as np
from simulator import Genotype
from reconstructor_nj import neighbor_joining_classical
cells = [Genotype([2, i], node_id=i, cell_id=label)
         for i, label in enumerate(("A", "B", "C", "D"))]
distance = np.array([[0, 5, 9, 9], [5, 0, 10, 10],
                     [9, 10, 0, 8], [9, 10, 8, 0]], dtype=float)
tree, _, root = neighbor_joining_classical(distance, cells, max_id=3)
assert nx.is_arborescence(tree) and tree.in_degree(root) == 0
print(f"complete: toy reconstruction; {tree.number_of_nodes()} nodes; {tree.number_of_edges()} edges")
PY
"$CTBF_PYTHON" -m pytest -q test/test_reconstructor_classical_nj.py \
    test/test_paper_confirmation_figure_v6.py test/test_paper_empirical_figure_v3.py \
    test/test_freezing_release_scripts.py test/test_release_reproduction.py test/test_quickstart.py
"$CTBF_PYTHON" -m algorithm_evaluation.v5_confirmation_common --require-approved
if [[ "${1:-}" == --full ]]; then
    "$CTBF_PYTHON" -m pytest -q test -m 'not slow and not json_full'
fi
