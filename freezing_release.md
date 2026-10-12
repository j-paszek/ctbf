# Freezing CTBF release 1.2 before freezing the paper

Prepared 2026-10-09 and updated for portable preparation and publication
gates on 2026-10-11.
The release scope follows the owner's request and `legacy/rev.txt`.

## Decision and scope

Freeze **CTBF software release 1.2.0**, preferably tagged `v1.2.0`,
and a separate **ctbfbio release 1.0.0**, then cite those releases in the
paper. “Release 1.2.0” is the public software version. **CTBF v5** remains
the scientific simulator/reconstruction contract. Do not rename schemas or
change scientific behaviour to reconcile these two version numbers.

GitHub releases with versioned data assets are a practical first publication
step. A Git tag alone does not contain ignored inputs, results or figures.
Add Zenodo later as agreed, archiving the same release bytes and recording
the DOI. Do not claim a Zenodo deposit or DOI before it exists.

This file is a runbook, **not a declaration that the release is ready**.
Run one section at a time. Section 3 is implemented and awaits the owner's
local preparation run. Empirical public packaging is blocked pending the
recorded rights review. Later commands must not bypass either gate.
No commit, tag, upload, inference or full test run was performed while
preparing this document.

### Initial inspection and current preparation status

- The local CTBF checkout is still on `development`, not `main`.
  Your merge may already exist remotely. Check it before selecting a release
  commit. Do not switch branches while trying to preserve uncommitted work.
- Many confirmation, MEDICC2, NestedBD and trimmed-evaluation modules/tests
  are untracked. A release of the existing commit would omit them.
- `paper/`, most Markdown, experimental manifests and the owner-local
  launcher are ignored. The current figure builders are therefore absent
  from an ordinary Git source archive.
- MIT software and scoped CC BY 4.0 CTBF-data licences, updated READMEs,
  citation files, release notes, pinned Python dependencies and a headless
  example are prepared. Clean-clone installation remains an owner-run gate.
- Existing v5/v6 figure builders check original absolute paths, source hashes
  and campaign artifacts. Their successful execution on the original machine
  is **not yet proof of portable reproduction from a release download**.
- The new exporter covers all 14 current numerical tables. Its values have
  been checked against the current supplement (446 printed decimal values).
  The original source-specific intervals are retained without new draws.
- ctbfbio is on `master`, has no configured remote in this checkout and
  contains untracked processed results and local IDE/scratch files.
- Do not use ctbfbio's existing `scripts/package_zenodo.py` as the final
  publication packager yet. It recursively selects the workspace, does not
  exclude every IDE/SAM-temporary file, and can replace its output archive.
  Its older checklist also names a shell wrapper not verified here.

## 1. Decide where the material belongs

| Material | Code repository | Data/artifact location |
|---|---|---|
| CTBF simulation, reconstruction, evaluation, comparator adapters and tests | CTBF | CTBF release |
| Scripts producing **all paper tables and figures**, including the empirical figure | CTBF | CTBF release |
| Simulation design, seeds, method mapping and source-lock manifests | CTBF, explicitly versioned | Also identify them in the evidence bundle |
| Simulated inputs, stored reconstruction scores, paired summaries and figure data | Not ordinary Git history | CTBF release assets |
| KTN152 download/calling/QC/conversion and external-runtime setup | ctbfbio | ctbfbio release |
| KTN152 retained integer CNPs, before/after collapse, mappings, QC and stored empirical reconstruction outputs | Not ordinary Git history by default | ctbfbio release assets |
| Raw FASTQ/BAM/reference genomes and third-party installations | Neither release | Upstream accessions/download instructions and exact pins |
| Paper draft and review iterations | Keep the current owner workflow | Freeze the accepted paper separately, **after** the two releases |

Keep the existing figure builders at their current CTBF paths for this release.
Do not move them into ctbfbio, or duplicate the CTBF inference implementation
there. Publish a small additional portable reproduction entry point outside
the source-locked scientific modules.

Two reproduction levels should be explicit:

1. **Stored paper-result reproduction:** read frozen summaries and selected
   per-history evidence, export the reported tables/statistics and render
   figures. No new bootstrap draws are made. No cnp2cnp, MEDICC2,
   BEAST, Docker or new inference should be needed.
2. **Campaign reproduction/audit:** the generator/runner commands, exact input
   banks, external pins, original configurations, failures, original CTBF
   source snapshots and, where relevant, MCMC traces. Third-party source
   snapshots are excluded. This is substantially more expensive.

The allowlist covers level 1. Full synthetic input banks, saved external
trees and raw chains are not in the lightweight candidate. Keep the
expensive campaign archive separate and do not claim a complete end-to-end
rerun from the lightweight assets alone.
Preserve all original audit material locally; do not delete it when preparing
a lightweight bundle. If posterior traces are not in the first release, state
that the bundle reproduces the reported diagnostic summaries, not their
calculation from raw chains. Archive that additional evidence before making
a stronger reproducibility claim.

## 2. Release workspace and adopted licences

**Current status:** the owner has completed steps 1 and 2. The licensing
decision is implemented: MIT for original code and CC BY 4.0 for original
CTBF-generated benchmark data. No further licence-selection command is
needed. The next owner action is the local preparation command in section 3,
not tagging or publication. The updated step 2 below can optionally
save the adopted licence packet in the existing release workspace.

Run the short commands below in **PyCharm Terminal or macOS Terminal**, not
with the Markdown Run button. No preliminary `bash` command, activation or
persistent shell session is needed. From the CTBF repository directory:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
```

```bash
./scripts/freeze_release/step1.sh
```

Each step reads `scripts/freeze_release/config.sh` independently. Edit that
one file if the paths, release tags, GitHub repositories or Python executable
need changing. Its Python default is the interpreter that passed the audit,
not whatever an IDE happens to call `python`.

Step 1 safely creates **or reuses** the workspace. It preserves files left by
the old Markdown commands. Every invocation prints its log path, saves output
in `/Users/voronwe/Work/ctbf_release_1.2.0/logs/`, and ends with
`complete: step1` or `FAILED: step1 (exit ...)`. Repeated runs get separate
logs and source inventories. If a command fails, stop and inspect its log.
Do not remove the workspace to bypass a guard.

For a fresh workspace, run step 1 and then step 2. An existing successful
step 1 need not be repeated. Stop for section-3 preparation before later steps.
The scripts do not automatically execute the next step. Script numbers and
this document's section numbers are different:

| Script | Operation |
|---|---|
| `step1.sh` | Setup and read-only Git inventory; safe to repeat |
| `step2.sh` | Copy the adopted licence packet offline; preserve old candidates and identical existing copies |
| `step3.sh` | Check metadata; `--prepare-bundles` prepares/verifies allowlisted local bundles and exports current tables/plots |
| `step4.sh` | Local smoke/tests; broad suite only with `--full` |
| `step5.sh` | Rebuild numerical derivatives in a fresh attempt |
| `step6.sh` | Render existing figures in a fresh attempt |
| `step7.sh` | Preview source selection; staging requires a flag |
| `step8.sh` | Record manually committed release candidates |
| `step9.sh` | Package reviewed bundles; requires approval and the empirical rights receipt |
| `step10.sh` | Clean-clone checks, then separate owner portability approval |
| `step11.sh` | Optional GitHub installation/authentication/repository setup |
| `step12.sh` | Approved tags, pushes and draft releases |
| `step13.sh` | Approved publication and download verification |
| `step14.sh` | Compile the owner-updated manuscript |

Logs can contain local paths and repository metadata. Review them before
sharing. The log does not depend on the IDE retaining a run window.

### Adopted licences and their scope

The owner approved **MIT for original CTBF/ctbfbio code** and **CC BY 4.0
for original CTBF-generated benchmark data** on 10 October 2026. The prepared
files are:

| File | Purpose |
|---|---|
| `LICENSE` | Standard MIT licence for original software |
| `LICENSE-DATA` | Full standard CC BY 4.0 licence for covered CTBF benchmark data |
| `LICENSESCOPE.md` | Defines covered code/data and excluded material, including the release-copy layout |
| `DATA_ATTRIBUTION.md` | Creator, source, licence and version-attribution guidance for CTBF data |
| `THIRD_PARTY_NOTICES.md` | Upstream tools, software terms, output/data distinctions and publication boundaries |

ctbfbio has its own MIT `LICENSE`, `LICENSESCOPE.md` and
`THIRD_PARTY_NOTICES.md`. Its empirical CNPs and derived trees are **not**
automatically covered by the CTBF synthetic-data grant. Licence grants cover
only rights the contributors hold. Using MIT-licensed CTBF does not require
other users to license their own output under CC BY 4.0.

Direct MEDICC2, NestedBD-Long, cnp2cnp and other tool outputs belong in
separate per-tool directories with file coverage, provenance and a reviewed
redistribution basis. A program's software licence does not automatically
license its output. Do not label all MEDICC2 trees GPL or all files in a
mixed archive CC BY 4.0. See [the scope](LICENSESCOPE.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) for the precise boundaries.

**No third-party source code or binaries may be included** in either code
release or an evidence archive. This includes copied/adapted implementations,
upstream templates, JARs, environments and software snapshots embedded in
results. Retain upstream links, exact pins and installation instructions.
Review both the tagged tree and any committed history being made public.
No local tool installation or frozen result is deleted by this policy.

NCN's current policy accepts CC0 or CC BY 4.0, or equivalent openness, for
research data. Its 2025 changes also cover specified ongoing earlier grants.
The grant number alone does not establish which provisions apply to your
agreement. Confirm this with the university/grant office. I found no
NCN certification of a particular **software** licence, so do not call MIT
“NCN-approved”. See the [NCN data policy](https://www.ncn.gov.pl/sites/default/files/pliki/zarzadzenia-dyrektora/zarzadzenieDyr-39_2025_ang.pdf)
and [letter covering ongoing grants](https://www.ncn.gov.pl/sites/default/files/pliki/2025-06-pismo-dyrektora-ncn-open-access_ang.pdf).

To save the five adopted CTBF licence/notice files in
`/Users/voronwe/Work/ctbf_release_1.2.0/approved-licences/`, optionally run:

```bash
./scripts/freeze_release/step2.sh
```

This is an offline copy, not another licence decision. Identical copies are
preserved. A differing existing packet causes a stop without replacement.
The previous `--install-approved-mit` option is retired.
Any earlier `licence-candidates/LICENSE-DATA-CC0.txt` is left untouched as
an **unused candidate**. It is not the adopted data licence and must not enter
the public bundle. The adopted texts are the standard
[MIT licence](https://opensource.org/license/mit) and
[CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## 3. Required preparation before staging or tagging

**Implemented; owner-run verification is next.** The approved split permits
code/synthetic preparation and a local empirical draft. Public empirical
packaging remains gated by the recorded rights review. Run:

```bash
./scripts/freeze_release/step3.sh --prepare-bundles
```

The command copies 38 explicitly pinned evidence files across two separate
bundles, verifies their checksums/rights inventories, and exports all 14
numerical tables plus the three numerical figures to a new attempt directory.
It checks printed table decimals and W/T/L against the current supplement.
Existing verified bundles are reused without replacement. It does not run
experiments, create archives, stage, commit, tag, upload or publish.
The final line is `complete: step3`. Keep the log and review the printed
output directory before proceeding. `step3.sh` without a flag only checks
metadata prerequisites.

The following preparation items are now implemented. Remaining human gates
are the empirical rights review, source/content review, clean-clone run and
visual review of the reproduced figures:

- Updated CTBF README: release 1.2.0 / CTBF v5, supported Python, installation,
  a tiny headless example, current method names, both GRF targets, release
  data links and separate “quick reproduction” versus “expensive rerun”
  instructions. Mark old versions/experiments as historical context.
- Added `requirements-release.txt` (working-environment pins; clean-install
  verification remains step 10), supported Python version and
  `CITATION.cff`. The licences/scope/notices/attribution are already prepared.
  Include them in the reviewed source selection and verify that each final
  release file has the correct scope.
- Added `reproducibility/REPRODUCE.md`,
  `reproducibility/evidence-map.tsv` and
  `reproducibility/RELEASE_NOTES_1.2.0.md`.
  The evidence map must associate every active numerical table/panel with its
  exact input file, statistic, generator, method roster, denominator, rounding
  and accepted output. Method-definition/parameter tables need source mapping,
  not invented experimental records.
- Added the **portable table exporter and figure entry point**. They
  consume a reviewed release bundle, expose input/output paths, refuse to
  overwrite frozen artifacts and work without the original user directory.
  Include all M0–M7 methods in their applicable output class and external
  comparators only in their actual coverage. Preserve the established
  stored intervals, including the two harmonized duplicate intervals.
- The existing original-workspace audit must remain usable. Do not remove
  checksum checks, edit frozen manifests or regenerate approvals to make a
  moved archive pass. A portable derivative should preserve original hashes
  and declare an explicit original-to-bundle path mapping, rather than
  pretending its relocated files were the original execution environment.
- Step 3 prepares two allowlisted data-bundle directories in the release workspace:
  `ctbf-v5-reproduction/` and `ctbfbio-ktn152-evidence/`.
  Each needs README, schema/column descriptions, provenance, licence scope,
  expected outputs and a complete file inventory. Both require
  `LICENSESCOPE.md`, `THIRD_PARTY_NOTICES.md` and `file-rights.tsv`, mapping
  exact paths to the applicable notices/redistribution basis. Follow the
  `ctbf-generated/` versus `external-outputs/<tool>/` separation in
  `LICENSESCOPE.md`. The CTBF-generated subtree includes `LICENSE-DATA` and
  `DATA_ATTRIBUTION.md`. Unresolved external/empirical material stays out of
  public assets until its status is resolved. Do not move sealed source
  evidence or include upstream software snapshots.
- Updated ctbfbio README and release metadata to the **completed** feasibility
  and NestedBD analyses. Keep old proposals/runbooks as historical records,
  not the default current instructions. The software citation and release
  notes are prepared, with explicit local scratch/IDE/log/result exclusions.
  Dataset/Zenodo templates remain drafts until data rights and identifiers
  are known. Contributor attribution remains part of owner metadata review.
- Superseded the broad ctbfbio packager for this release with the reviewed
  allowlisted bundle directories above. Archive only those.
  Do not upload `tools/`, FASTQs, BAMs, reference files, IDE files or
  the workspace's SAM temporary file.

The empirical bundle's public release requires
`audit/empirical-rights-review.json` in the release workspace. Its schema,
required basis statements and exact-file coverage are specified in
`reproducibility/SCHEMA.md`. The review must match the bundle-manifest hash.
Steps 9, 12 and 13 reject a missing, incomplete or stale review. Step 9 also
publishes the approved receipt as a separate checksummed companion notice,
so the preparation-time local-draft inventory is not mistaken for a final
rights clearance. Do not create an approval receipt before completing the
actual review. The agent has not supplied or assumed such approval.

Preserve source-locked scientific files byte-for-byte where possible. Put
release glue in new, unpinned files. If a necessary packaging change touches
a pinned source, retain the original snapshot and add a documented derivative
instead of changing the historical lock.

**Environment note:** the interpreter available as `python` during this
audit had Python 3.13.0, NumPy 2.2.6, NetworkX 3.4.2, Matplotlib 3.10.7,
Plotly 6.1.1, psutil 7.0.0 and pytest 8.3.5. These are a candidate
reproduction environment, **not a claim that every campaign used these
versions**. The inspected `python3` lacked NumPy/Matplotlib. Preserve
historical runtime manifests and test the release environment separately.

### Evidence that must not be overlooked

| Evidence | Existing source location |
|---|---|
| Registered bank, seven internal arms, MEDICC2 and primary report | `algorithm_evaluation/results/v5_confirmation/confirmation_v1/{bank,internal,medicc2,report}/` |
| Condition-specific distance-inclusive M2/MEDICC2 resources | `algorithm_evaluation/results/v5_confirmation/condition_specific_resource_v1/` |
| Independent full/trimmed score pairs | `algorithm_evaluation/results/v5_confirmation/trimmed_truth_v1/` |
| M4 and final uniform-2M NestedBD comparison | `algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_2m_v1/`, especially `reports/report_001/` and its completion/repair provenance |
| Current and prerequisite publication summaries | `paper/figures/v5-confirmation-v{1,3,4,5,6}/` |
| Exact simulation declaration and source locks | `experimental_description/g2_01a_v5_confirmatory_manifest.json`, `g2_01a_v5_source_lock.json`, `v5_confirmatory_protocol.md` and `simulator_examples/paper_v5/confirmatory_standard.json` |
| Accepted empirical input, collapse mapping and QC | ctbfbio `data/processed/ktn152_copykit_feasibility_v1/` and its declared caller-source provenance |
| Empirical distances, trees and summaries | ctbfbio `data/processed/ktn152_ctbf_feasibility_v1/` |
| Supplementary NestedBD pilot/budget/convergence claims | Include the specifically cited synthetic and empirical diagnostic summaries, contracts and pins in the evidence map; the final 2M comparison alone is not their source |

Expected coverage: 800 conditions; 5,600 original internal records plus 800 M4
records; 300 MEDICC2 records; 400 declared NestedBD runs, 397 successful.
The three NestedBD failures affect two H24 inputs, giving the matched
100/98/100 partial-comparison denominators. Do not impute missing results or
combine the separate 10M diagnostic with the uniform-2M accuracy comparison.

## 4. Verify the current local installation and rebuild derivatives

These commands use existing scripts on the **original owner workspace**.
They are not the clean-clone test in section 7.

### 4.1 Environment and a tiny reconstruction

```bash
./scripts/freeze_release/step4.sh
```

Expected: `complete: toy reconstruction; 7 nodes; 6 edges`.
This is a supplied-distance API smoke example, not a cnp2cnp validation or
paper experiment. Inspect the local environment export for local paths or
credentials before sharing it; it is not automatically a portable lockfile.

### 4.2 Tests and frozen contract

The default `step4.sh` invocation above already runs the focused tests and
frozen-contract check.

Before the release, also run the ordinary non-slow regression suite using the
release environment, after making its code/tests available in Git:

```bash
./scripts/freeze_release/step4.sh --full
```

Record passed/skipped/failed counts and explain optional-dependency skips.
Do not run historical/full-corpus suites or new inference to repair a
packaging problem. The public instructions use Python modules directly,
not the ignored owner-local `zzzz_verify_ctbf.sh`.

### 4.3 Numerical derivatives — new output directories only

These recompute summaries/plots from stored evidence, including fixed-seed
bootstrap summaries where the existing scripts require them. They do not
simulate histories, compute distances, reconstruct trees or run MCMC.
They can spend time reading/hash-checking the stored evidence.

```bash
./scripts/freeze_release/step5.sh
```

The script prints `Exact CSV match` after each successful comparison. If
anything differs, stop and inspect it.
Do not replace the accepted source data. Audit JSON paths and PDF metadata
may differ across output locations/tool versions, so they are not required
to be byte-identical. Numerical tolerances, if needed, must be explicitly
reviewed rather than silently accepted.

The first publication derivative contains **superseded block-wide resource
accounting**. Its accuracy rows remain useful, but export the final resource
table from `condition_specific_resource_v1`, never from that older
resource allocation. Step 3's portable exporter uses the condition-specific
summary and covers every current numerical table. Step 5 remains the separate
original-workspace statistical regeneration/audit, not a prerequisite for
every lightweight export.

### 4.4 Schematic and empirical figure rendering

These preserve current figure sources and write only to the release workspace.

```bash
./scripts/freeze_release/step6.sh
```

The empirical command renders the approved editable SVG. It does not recreate
integer calls or infer an empirical tree. Retain its underlying matrices,
tree, mapping and annotation provenance in the ctbfbio bundle. The earlier
empirical builders should remain available for that provenance chain.

## 5. Put the required source material under version control

You manage Git. Review paths and diffs before committing. Never use a blanket
`git add .` for either workspace.

The script lists the proposed CTBF source selection without changing the
index. Its source globs do not select generated result directories:

```bash
./scripts/freeze_release/step7.sh
```

After inspecting that list and completing section 3, explicitly stage it:

```bash
./scripts/freeze_release/step7.sh --stage-reviewed-files
```

This does not commit, merge or push. Also review/add remaining untracked tests
identified in section 2 and the necessary small
`test/fixtures/medicc2_total_cn/` fixtures after checking
their contents/licensing. Add the section-3 README/installation/citation/
reproduction files once prepared. The script selection is **not** a transitive
dependency guarantee: the clean-clone test detects overlooked sources.
Narrow ignore-file exceptions can later replace force-adds. Do not unignore
all of `paper/` or all results.

The current selection includes the all-method JSON runner and the 15 files
in `presentation/quickstart-v2/` (11 HTML views/index and four JSON files).
The earlier `presentation/quickstart/` example is left untouched locally.
Step 7 also previews four **index-only exclusions**: `nj.html`,
`reconstructed.html`, `simulated_tree.html` and `true_tree.html`. They embed
Plotly.js and therefore cannot enter the new release under the adopted
no-vendoring policy. Only `--stage-reviewed-files` applies these exclusions
with `git rm --cached --sparse`, preserving local files even when a target
has Git's `skip-worktree` flag. This does not change sparse-checkout settings
or unrelated index flags. The existing root-HTML ignore
rule prevents accidental re-addition. No Git history is rewritten. Step 8
and later publication checks refuse a candidate that still tracks them.

Review the staged README and notices again: earlier staged versions may
predate the latest refinements. The script refreshes those selected files.
Do not run it until all selected source/gallery changes are ready to stage.

After committing, publish `presentation/quickstart-v2/` through the intended
GitHub Pages source. For branch-based Pages, select the intended branch's
root in repository **Settings → Pages**, push using your usual workflow,
and wait for deployment. Existing Actions-based Pages workflows must include
that folder instead. No separate image upload is necessary. Local viewing
works immediately; online links work only after deployment. This is an owner
publication step, not performed by the freeze scripts.

For ctbfbio, stage reviewed workflow/scripts/tests/configuration, README,
completed citation/licence metadata and original manifests. Keep processed
data in release assets, not a blanket commit of `data/processed/`.
Exclude the visibly untracked IDE and temporary SAM material. Preserve
historical private files locally rather than deleting them for tidiness.

Commit the reviewed release changes and merge the intended CTBF release
commit into `main` using your normal Git workflow. Do not automatically
switch/merge a dirty worktree. Once that is done:

```bash
./scripts/freeze_release/step8.sh
```

Step 8 records the commits and ctbfbio branch in
`audit/selected-commits.tsv`. Later steps read that file, not shell exports.
An existing different selection is not overwritten: preserve that candidate
and use a new workspace for a different release commit.

Untracked/ignored owner documents may remain. Untracked **release dependencies**
may not. The clean-clone test is mandatory because a clean tracked diff alone
does not detect omitted files.

## 6. Package only the reviewed evidence bundles

Section 3 must first create and validate these two curated directories.
Do not substitute whole repository or results roots. Bundle manifests record
exact source hashes, logical original paths, scope and exclusions. Step 8
records the selected software commits, which the release tags must identify.
Step 9 requires the scope, third-party notice and `file-rights.tsv` files
in both bundles. It also checks that the CTBF-generated subtree contains
the adopted CC BY 4.0 licence and attribution unchanged. These checks do not
replace reviewing actual contents or resolve a missing redistribution right.

```bash
./scripts/freeze_release/step9.sh --approve-bundle-contents
```

Review both inventories for raw data, third-party source/templates/binaries,
unused CC0 candidates, tokens, private metadata, IDE files and unintended
intermediate outputs. Retain original
sealed provenance as read-only records; portable input resolution must not
depend on author-machine absolute paths.

GitHub release assets must each be under 2 GiB. If a reviewed evidence archive
is larger, partition it into documented smaller assets rather than committing
it to Git or dropping necessary data. See [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## 7. Validate from a clean clone before creating the tags

This tests the selected **committed sources**, not the current dirty filesystem.
It needs no public upload. The release dependency file must already exist.

```bash
./scripts/freeze_release/step10.sh
```

Step 10 prints its fresh `clean-check.*` directory. Its interpreter is
`env/bin/python`, the checkout is `ctbf/`, and the extracted inputs are in
`unpacked/` inside that directory. Step 10 now executes
`python -m reproducibility.reproduce` with both bundles and `--figures --vectors`
from that clean checkout. It writes `reproduced/` in the attempt directory.
It needs independently installed `latexmk` and Inkscape for the vector
assets, but no external scientific inference software. Success still
requires the separate owner visual/content/portability review below.

Acceptance:

- no reads from the original CTBF/ctbfbio checkouts, hidden local environments
  or external scientific inference tools for lightweight reproduction;
- every numerical table/panel matches the accepted values and coverage;
- all current figures render, including the supplementary controls and
  empirical SVG;
- all intentionally unavailable method/condition entries stay unavailable;
- both archive inventories and licence scopes have been reviewed;
- the same commands can be followed using only the released code and assets.

Use a second machine/container or a filesystem-restricted process to confirm
the first condition. Merely running in another directory on the same machine
could still allow an accidental absolute-path fallback.

Record that review in a text file in the release workspace. Only after the
actual portable reproduction and content/rights review pass, record your
explicit approval (the named review file must already exist):

```bash
./scripts/freeze_release/step10.sh --accept-portable-review /Users/voronwe/Work/ctbf_release_1.2.0/portable-review.txt
```

This is an owner attestation tied to the selected commits and asset hashes,
not an automated assessment of the text file. Step 12 checks that approval.
Do not retag if validation fails. Preserve the failed attempt, fix packaging,
make a new release-candidate commit and repeat in a fresh release workspace.

## 8. Create the releases — only after the gates pass

These commands perform external publication. They are for the owner to run
after checking the exact commits and assets. Do not force-push tags or use
upload `--clobber`. Prefer GitHub immutable releases: publish a draft
only after attaching all reviewed assets. GitHub then protects the tag and
assets. See [immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).

If GitHub CLI is absent, install it with Homebrew or use the GitHub website.
Authenticate through the browser, never by pasting tokens into this file.

```bash
./scripts/freeze_release/step11.sh --configure-github
```

ctbfbio has no remote in the inspected checkout. **Only if that remains true
and you have approved making the curated committed history public:**

```bash
./scripts/freeze_release/step11.sh --create-public-ctbfbio
```

Before either push, inspect the repository's full committed history for
sensitive files. Ignore rules do not remove already committed material.
If such material exists, stop for a separate history-cleanup decision.
Do not publish it and promise to remove it later.

```bash
./scripts/freeze_release/step12.sh --approve-reviewed-drafts
```

The approval flag also confirms that you reviewed both repositories' committed
history and the public-release licence/privacy scope. This step does not
commit or merge for you. It refuses mismatched commit identities, push remotes
or tags that point elsewhere. Matching existing tags are preserved. If
creating one draft succeeds and the other fails, retain
both the log and any created draft, and review that partial publication state
before retrying; no tag or asset is force-replaced.

Prepare/review ctbfbio's release-notes file before that step as well.
Verify the draft descriptions, both assets, method coverage, licence scope
and download links. Then publish explicitly:

```bash
./scripts/freeze_release/step13.sh --approve-publication
```

Step 13 verifies the draft asset bytes before publication, then downloads
and verifies them again after publication. Each download check has a fresh
directory. Step 12 uses `--verify-tag`, which prevents the CLI from silently
creating a tag on the wrong default-branch commit. See the official
[create](https://cli.github.com/manual/gh_release_create) and
[edit](https://cli.github.com/manual/gh_release_edit) command documentation.

## 9. Update availability in the paper, then freeze the paper

After the releases are public, add a short availability paragraph and software/
dataset citations identifying:

- CTBF **1.2.0**, scientific contract **v5**, release URL and full commit SHA;
- ctbfbio **1.0.0**, release URL and full commit SHA;
- the named reproduction/evidence assets and SHA-256 checksums;
- the location of the installation, tiny example and table/figure commands;
- empirical source accession PRJNA396019 / SRP114962, processed-input
  provenance and the separately documented absolute-scale limitation;
- the adopted software/data licences and NCN grant
  **2020/39/D/ST6/03321**.

Suggested availability wording, **only after the releases and portable
reproduction have passed the checks**:

> CTBF version 1.2.0, implementing the CTBF v5 model, is available at
> https://github.com/j-paszek/ctbf/releases/tag/v1.2.0. The release includes
> reconstruction and evaluation code, scripts for reproducing the reported
> tables and figures, and versioned reproduction data. The KTN152 preprocessing
> workflows and processed inputs are available in ctbfbio version 1.0.0 at
> https://github.com/j-paszek/ctbfbio/releases/tag/v1.0.0.

Replace the URLs if the approved repository/tag choices differ. Add the
actual licence names and commit identifiers to the citation/reproduction
record, not placeholders. Until level 2 is fully archived, describe the
precise supported reproduction scope rather than claiming that every raw
pipeline can be rerun from the lightweight bundle.

Update the abstract's availability field and the main Data/Code Availability
statement consistently. Do not imply that a GitHub release is already a
persistent DOI archive. Deposit the same frozen releases/data with Zenodo
before journal submission as planned, then add their version-specific DOIs.

Finally integrate accepted manuscript candidates, compile the preprint with:

```bash
./scripts/freeze_release/step14.sh
```

Freeze that accepted paper separately. Subsequent prose-only changes do not
require moving the software tag. Any numerical/code correction needs a new
version and a transparent correction record.

## Verification of this runbook

- Source inspection verified the listed current CLI arguments and the active
  figure paths. All four queried Python CLI help commands succeeded with the
  available `python` interpreter.
- The four-state supplied-distance example was executed successfully:
  seven nodes and six edges.
- All 16 shell files pass `bash -n`. The combined focused verification has
  97 passing tests, including release fixtures, NJ and current figure tests.
  It covers relocated bundles, pins, extra-file/symlink rejection, missing or
  stale empirical rights reviews, overwrite protection and shell approval
  boundaries. `pip check` reports no broken requirements in the working
  environment. No clean installation has been performed yet.
- Owner-approved MIT and CC BY 4.0 licences and scope notices are now in
  place. The old CC0 candidate is not adopted. No full suite, statistical
  regeneration, data packaging, owner-repository Git mutation, repository
  creation or publication was performed.
- Exported numerical values match 446 printed decimal entries and printed
  W/T/L across all 14 current tables. All 38 selected source hashes remain
  unchanged. The three numerical figures rendered successfully in a fresh
  temporary smoke-check directory. The main plot was visually inspected.
- Portable code, dependencies, citation metadata and allowlisted preparation
  are implemented. Actual owner bundles, their content/rights review, the
  clean-clone run and public releases remain outstanding. The empirical
  bundle stays local until its recorded rights gate passes.
