#!/usr/bin/env bash
set -Eeuo pipefail
# Render existing editable figures without changing the accepted artifacts.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step6 "$@"
[[ $# == 0 ]] || die 'Usage: step6.sh'
require_command latexmk
require_command inkscape
attempt="$(mktemp -d "$CTBF_RELEASE_WORK/figure-render.XXXXXX")"
printf 'Fresh render outputs: %s\n' "$attempt"
cd "$CTBF_ROOT/paper"
for figure in connect-then-complete-owner-v2 nonidentifiability-v2 trimmed-true-tree-v1; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$attempt" "figures/$figure.tex"
done
inkscape figures/ktn152-empirical-v4.svg --export-filename="$attempt/ktn152-empirical-v4.pdf"
