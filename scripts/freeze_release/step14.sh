#!/usr/bin/env bash
set -Eeuo pipefail
# Compile only, after the owner edits paper availability and integrates changes.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step14 "$@"
[[ $# == 0 ]] || die 'Usage: step14.sh'
make -C "$CTBF_ROOT/paper" main_a
printf 'Paper compiled; final content review/freeze remains with the owner.\n'
