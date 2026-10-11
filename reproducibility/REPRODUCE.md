# Reproducing the release's tables and figures

Software release 1.2.0 uses the unchanged CTBF v5 scientific contract.
These commands read explicit bundle-relative files, verify hashes, and write
only to a **new** output directory. They do not access the original owner's
home directory. They do not infer trees, call external reconstruction tools,
re-evaluate scores, or make new bootstrap draws. Stored means and intervals
are preserved, including the original placement-family M4 intervals.

## Install and check the code

Use CPython 3.13 and run from the CTBF checkout:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-release.txt
.venv/bin/python -m pip check
.venv/bin/python -m reproducibility.example
```

The pins describe the release reproduction environment, not a retroactive
lock of every historical external-tool run. A clean-clone installation is
an owner-run release gate. The tiny example takes a supplied four-state
matrix and needs no cnp2cnp installation or files.

## Stored results: portable commands

Obtain the matching `ctbf-v5-reproduction` asset once released. No public
asset or DOI is asserted by this candidate documentation. Run:

```bash
.venv/bin/python -m reproducibility.reproduce --bundle /path/to/ctbf-v5-reproduction --output /path/to/new-reproduction --figures
```

This exports all **12 synthetic numerical tables**, the complete method map,
simulation configuration, and three numerical figures: the current main
confirmation plot, supplementary controls, and primary block effects.
The current paper's M4 results and NestedBD-Long H14/H24 coverage are retained.
There is no missing NestedBD H34 value to impute. Figure curves use the frozen
100/98/100 matched-history sets, while all-history tables retain 100.

The two additional empirical numerical tables require the separate
`ctbfbio-ktn152-evidence` bundle. It is currently a **local-only draft**, pending
the approved rights review. On the author's machine, the complete command is:

```bash
.venv/bin/python -m reproducibility.reproduce --bundle /path/to/ctbf-v5-reproduction --empirical-bundle /path/to/ctbfbio-ktn152-evidence --output /path/to/new-complete-reproduction --figures --vectors
```

`--vectors` also compiles the three existing TikZ schematics with `latexmk`
and converts the accepted empirical SVG with Inkscape. Install these tools
independently. Omitting this option needs only the pinned Python packages.
The SVG remains editable. Rendering it is not a new CNP-calling or tree run.
Outputs are numerical CSV/JSON and publication figures; assembling the final
manuscript remains an author operation. See `evidence-map.tsv` and `SCHEMA.md`.

An optional `--check-supplement /path/to/paper/supplement/supplement.tex`
cross-checks printed decimal table values against exports. It never uses TeX
as the source of those values. Record counts/method coverage are separately
validated. No manuscript is needed by a public release download.

## Prepare local bundles (owner only)

After release steps 1–2 and metadata preparation:

```bash
./scripts/freeze_release/step3.sh --prepare-bundles
```

This uses the explicit catalogue and source pins, copies only the listed
files, verifies them, exports the 14 numerical tables and numerical figures
in a new attempt directory, and checks table decimals against the current
supplement. Existing verified bundles are reused without being overwritten.
Incomplete/differing bundles stop the run and are left for review. No package,
commit, tag, download or upload occurs. The empirical publication gate remains
closed. Step 6/`--vectors` handles optional vector rendering.

## Expensive campaign reproduction

The original audit/generation scripts remain unchanged. Consult
`experimental_description/v5_confirmatory_runbook.md`,
`v5_partial_comparator_extension_runbook.md` and `v5_trimmed_truth_runbook.md`,
and ctbfbio's feasibility/NestedBD runbooks. They need independent external
installations, the declared pins and original campaign evidence. Existing
source approvals are tied to their historical snapshots, not automatically
to the eventual release commit. Do not bypass their source-lock guards.

The lightweight assets are deliberately **not a complete campaign archive**:
they omit full input banks, raw chain logs/trees and third-party snapshots.
Diagnostics are exported from frozen reports, not recomputed from traces.
Full simulation/inference/ESS replay requires additional archived evidence
and is not certified by a successful numerical export. Original evidence is
preserved locally. No public empirical reproduction claim is appropriate
until that asset's review and actual distribution are complete.

## Release gates

Run focused verification with `./scripts/freeze_release/step4.sh`. Broad
tests remain opt-in (`--full`). Steps 9/12/13 require both explicit owner
approval and the manifest-bound empirical rights receipt described in
`SCHEMA.md`. Step 10 tests the actual selected code commit and downloaded
asset layout in a fresh environment, then requires owner visual/portability
review. A locally successful export alone is not a published release.
