#!/usr/bin/env bash
set -Eeuo pipefail
# Copy the adopted licence packet, without downloading or adopting candidates.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"
start_step step2 "$@"
[[ $# == 0 ]] || die 'Usage: step2.sh (MIT and CC BY 4.0 are already adopted; candidate-install options are retired).'
files=(LICENSE LICENSE-DATA LICENSESCOPE.md DATA_ATTRIBUTION.md THIRD_PARTY_NOTICES.md)
for file in "${files[@]}"; do require_file "$CTBF_ROOT/$file"; done
grep -Fq 'MIT License' "$CTBF_ROOT/LICENSE" || die 'Expected the adopted MIT software licence.'
grep -Fq 'Attribution 4.0 International' "$CTBF_ROOT/LICENSE-DATA" || die 'Expected the adopted CC BY 4.0 data licence.'
packet="$CTBF_RELEASE_WORK/approved-licences"
[[ ! -L "$packet" ]] || die 'Approved-licence directory must not be a symlink.'
mkdir -p "$packet"
for file in "${files[@]}"; do
    if [[ -e "$packet/$file" || -L "$packet/$file" ]]; then
        require_file "$packet/$file"
        cmp -s "$CTBF_ROOT/$file" "$packet/$file" \
            || die "Existing approved packet differs: $file. Preserve it and review a fresh release candidate."
        printf 'Preserved identical adopted notice: %s\n' "$file"
    else
        temporary="$(mktemp "$packet/.copy.XXXXXX")"
        cp "$CTBF_ROOT/$file" "$temporary"
        ln "$temporary" "$packet/$file"
        rm -- "$temporary"
        printf 'Copied adopted notice: %s\n' "$file"
    fi
done
printf '\nApproved licence packet: %s\n' "$packet"
printf 'Code: MIT. CTBF-generated benchmark data: CC BY 4.0.\n'
printf 'External outputs/empirical data retain separate scopes. Third-party software is not bundled.\n'
printf 'Old licence-candidates (including CC0) are untouched and are NOT release licences.\n'
printf 'Licensing is prepared. Next: the remaining section-3 metadata/reproduction work; do not tag yet.\n'
