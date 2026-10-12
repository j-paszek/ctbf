# CTBF 1.2.0 release candidate

Software version 1.2.0 implements the unchanged CTBF v5 scientific contract.
Release status: candidate. Public release identifiers are pending.

## Scope

- A one-JSON runner for all eight CTBF paper methods, with complete and
  state-trimmed truth, optional interactive visualization, fixed/fractional
  biopsy sampling, single-chromosome BED input and topology-only Newick
  export. The supplied ten-tree gallery uses CDN-based Plotly views.
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
H14/H24/H34. NestedBD was evaluated at H14/H24 only.

## Reproduction boundaries

`REPRODUCE.md` distinguishes exporting verified stored numerical summaries
from rerunning the expensive campaigns. The lightweight assets contain the
stored numerical evidence and editable figures. Full input banks, raw MCMC
traces and external software are excluded.

The KTN152 evidence bundle is a **local draft pending its data-rights review**.
It is outside the synthetic-data licence. Public distribution requires
clearance of the asset's documented rights review.

## Installation and licensing

The pinned Python environment in `requirements-release.txt` targets CPython
3.13. Independently installed TeX/latexmk and Inkscape are needed only for
vector schematic/empirical PDF rendering. External cnp2cnp, MEDICC2 and
NestedBD/BEAST installations are not required to export stored tables or
render the confirmation plots.

Original code: MIT. Original CTBF-generated benchmark data: CC BY 4.0.
See the scope and third-party notices. No upstream source, templates,
binaries, installed environments or container images are distributed.

Configurations and results from releases 1.0/1.1 are not backward-compatible
with release 1.2.0. The tutorial is separate from the frozen paper experiments.
