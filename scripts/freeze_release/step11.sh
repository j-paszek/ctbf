#!/usr/bin/env bash
set -Eeuo pipefail
# Explicit optional GitHub setup; no pushes or releases in this step.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step11 "$@"
[[ $# -le 1 ]] || die 'Usage: step11.sh [--configure-github|--create-public-ctbfbio]'
case "${1:-}" in
    --configure-github)
        if ! command -v gh >/dev/null 2>&1; then
            require_command brew
            brew install gh
        fi
        gh auth login --web
        ;;
    --create-public-ctbfbio)
        require_command gh
        [[ -z "$(git -C "$CTBFBIO_ROOT" remote)" ]] || die 'ctbfbio already has a remote; review it manually.'
        gh repo create "$CTBFBIO_GITHUB" --public --source "$CTBFBIO_ROOT" --remote origin
        ;;
    '')
        require_command gh
        gh auth status
        git remote -v
        git -C "$CTBFBIO_ROOT" remote -v
        ;;
    *) die 'Usage: step11.sh [--configure-github|--create-public-ctbfbio]';;
esac
