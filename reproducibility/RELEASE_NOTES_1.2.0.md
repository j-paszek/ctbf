# CTBF 1.2.0 release candidate

Software version 1.2.0 implements the unchanged CTBF v5 scientific contract.
This preparation record does not assert that a public tag or DOI exists.

## Scope

- Representative-state simulation, chromosome-aware copy-number events,
  biopsy collection, reconstruction and rooted-tree evaluation.
- Pooled and adaptive/deferred biopsy-guided methods M0–M7. The publication
  identifier M3 denotes `rooted_labeled_nj`, historically named Pooled-Q in
  stored reports. Historical method IDs and records are unchanged.
- Read-only reproduction of the current numerical result tables and figures
  from explicitly mapped frozen summaries. Original campaign/audit runners
  retain their source and input checks.
- Separate full-truth and state-trimmed GRF targets. Native and fixed-label
  AD-F1 remain distinct endpoints. No inference, tree evaluation or new
  statistical draws are part of lightweight table/figure export.

The confirmation evidence comprises 100 histories, 800 conditions, 6,400
internal method records, 300 MEDICC2 records and 400 declared NestedBD-Long
runs (397 successful). NestedBD contributes two-run mean scores only when
both runs succeed. The matched partial figure uses 100/98/100 histories at
H14/H24/H34, with no NestedBD H34 point. Retain those failures and denominators.

## Reproduction boundaries

`REPRODUCE.md` distinguishes exporting verified stored numerical summaries
from rerunning the expensive campaigns. Rendering an existing empirical SVG
is not redoing copy-number calling or reconstruction. The lightweight assets
do not claim to contain all raw MCMC traces or external software.

The KTN152 evidence bundle is a **local draft pending its data-rights review**.
Its absence from public distribution must be reported explicitly. It is not
covered by the synthetic-data licence or a claim of complete public empirical
reproducibility. Packaging/publication guards enforce this boundary.

## Installation and licensing

The pinned Python environment in `requirements-release.txt` targets CPython
3.13. Independently installed TeX/latexmk and Inkscape are needed only for
vector schematic/empirical PDF rendering. External cnp2cnp, MEDICC2 and
NestedBD/BEAST installations are not required to export stored tables or
render the confirmation plots.

Original code: MIT. Original CTBF-generated benchmark data: CC BY 4.0.
See the scope and third-party notices. No upstream source, templates,
binaries, installed environments or container images are distributed.

Earlier releases and rejected-paper outputs are historical context, not
CTBF v5 reproduction targets. No algorithm, frozen result or source approval
was changed to prepare this release.
