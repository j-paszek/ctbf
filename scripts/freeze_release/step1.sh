#!/usr/bin/env bash
set -Eeuo pipefail
# Create/reuse the workspace and inspect Git; never stage/commit anything.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step1 "$@"
[[ $# == 0 ]] || die 'Usage: ./scripts/freeze_release/step1.sh'
require_python
printf 'CTBF: %s\nctbfbio: %s\nWorkspace: %s\nPython: %s\n' \
    "$CTBF_ROOT" "$CTBFBIO_ROOT" "$CTBF_RELEASE_WORK" "$CTBF_PYTHON"
git -c core.pager=cat status --short --branch
git remote -v
git -c core.pager=cat branch --all
git -c core.pager=cat tag --list
git -C "$CTBFBIO_ROOT" -c core.pager=cat status --short --branch
git -C "$CTBFBIO_ROOT" remote -v
git -C "$CTBFBIO_ROOT" -c core.pager=cat tag --list
inventory="$(mktemp "$CTBF_RELEASE_WORK/audit/untracked-code.XXXXXX")"
git ls-files --others --exclude-standard -- '*.py' simulator_examples/paper_v5 test/fixtures scripts/freeze_release > "$inventory"
printf '\nUntracked-source inventory: %s\n' "$inventory"
ignore_status=0
git check-ignore -v paper/Makefile paper/figures/build_confirmation_figure_v6.py \
    experimental_description/g2_01a_v5_confirmatory_manifest.json \
    zzzz_verify_ctbf.sh freezing_release.md || ignore_status=$?
[[ "$ignore_status" -le 1 ]] || die "git check-ignore failed: $ignore_status"
printf '\nSetup is safe to rerun. Step2 copies the adopted MIT / CC BY 4.0 licence packet.\n'
