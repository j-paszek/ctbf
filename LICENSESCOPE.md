# CTBF licence scope

Original CTBF software is licensed under **MIT** ([LICENSE](LICENSE)).
Original CTBF-generated research data identified below are licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**
([LICENSE-DATA](LICENSE-DATA), [official licence](https://creativecommons.org/licenses/by/4.0/)).
These grants concern different material. They are not alternative licences
for every file in the repository or an attached release archive.
The data grant concerns original benchmark data released by the CTBF project.
Using the MIT-licensed program does not require other users to apply CC BY 4.0
to their own outputs.

## Scope by origin

| Material | Licence or treatment |
|---|---|
| Original CTBF simulation, reconstruction and evaluation code, tests, adapters, release/plotting scripts, original configurations and associated software documentation | MIT |
| CTBF-generated simulated histories, event records, integer CNPs, sampled biopsies, sampling metadata and synthetic input-format exports | CC BY 4.0 |
| CTBF-produced reconstructions of CTBF-simulated inputs, original CTBF evaluation scores, aggregate numerical tables and numerical plot data | CC BY 4.0, for the original contributions of CTBF's rights holders |
| Direct MEDICC2, NestedBD-Long, cnp2cnp or other external-tool output files, including converted copies preserving that content | Separate external-output scope. No automatic MIT or CC BY 4.0 grant is made by this notice |
| Public empirical reads, reference genomes, published datasets and empirical derivatives, including KTN152/CopyKit/scAbsolute CNPs and reconstructions based on them | Separate source/data-rights scope. Not covered by the CTBF-generated-data grant |
| Third-party source, copied/adapted upstream code, templates containing upstream content, packages, binaries, JARs, environments and container images | Excluded from CTBF release distribution. Original upstream rights are unchanged |
| Manuscript text, review iterations, posters, journal templates and third-party artwork | Outside this software/data grant unless separately marked |

"CTBF-generated data" here means original CTBF simulation-benchmark data,
not every file written by a script in this repository. File location, JSON
conversion, renaming or evaluation by CTBF does not transfer upstream rights.
Original score calculations about an external result can be included in a
CTBF-authored numerical comparison. The external tree, trace, matrix or other
source artifact retains its separate provenance and scope.

The grants extend only to copyright and related rights that the contributors
are entitled to license. No exclusive rights are asserted over unprotected
facts. Existing third-party notices take precedence for their material.
This document identifies covered material; it does not add restrictions to
the standard MIT or CC BY 4.0 terms.

## Attribution for CTBF data

Use the attribution record in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
Retain supplied creator/source and licence notices, link to CC BY 4.0, and
indicate changes, as required by that licence. Include the released version
and archive identifier when available. Citation of the accompanying research
paper is requested for scientific credit, not imposed as an additional
licence condition. Do not invent a release DOI before the deposit exists.

## Separation in release assets

The release packager must explicitly list which files each notice covers.
The existing frozen experiment directories must not be moved or relabeled
in place. Create reviewed release copies using a layout such as:

```text
ctbf-v5-reproduction/
  README.md                         # states that the archive has multiple scopes
  LICENSESCOPE.md
  THIRD_PARTY_NOTICES.md
  file-rights.tsv                   # exact archive paths and applicable notices
  ctbf-generated/
    LICENSE-DATA
    DATA_ATTRIBUTION.md
    inputs/                         # original simulated histories / biopsies
    results/                        # CTBF trees / original scores / tables
  external-outputs/
    medicc2/NOTICE.md
    nestedbd-long/NOTICE.md
    cnp2cnp/NOTICE.md
```

Only include the external-output directories actually needed. Each external
output notice must identify the tool/version or commit, the generating run,
input origin, file coverage, any supplied attribution and the reviewed
output-specific redistribution basis. The program's software licence alone
is not an output licence. An unresolved status must be stated explicitly and
resolved before that material is published. A disclaimer cannot create
missing redistribution rights.

Keep empirical material in a separately described ctbfbio evidence asset.
Do not put a blanket `CC-BY-4.0` label on a mixed synthetic/external/empirical
archive. Its metadata must point to these per-scope notices. A record hosting
only `ctbf-generated/` may use CC-BY-4.0 as its data licence.

## No third-party software redistribution

The releases may contain original CTBF wrappers, installation instructions,
upstream URLs, version pins, checksums and bibliographic references. They must
not contain upstream software source, adapted copies of that source, bundled
libraries, executable packages, JARs or container images. A copied example or
XML template is not made original merely by changing parameter values.
Link to any required upstream template and preserve its hash instead.

This rule applies to the Git tree published under the tag, source snapshots
inside evidence archives, submodule contents and ancillary assets. Check
committed history before making a previously private repository public.
Do not delete local installations or rewrite frozen scientific source locks
to satisfy the release policy. Keep exact upstream identity and independent
installation instructions available for reproducibility.

The dependency/output distinction is detailed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Superseded candidates

CC0 was an earlier proposal and was not adopted for this release. Any
`LICENSE-DATA-CC0.txt` in the owner's local release workspace is an unused
candidate, not a licence declaration for CTBF data. It must not be placed in
the public evidence asset. Do not remove licences from material that another
rights holder has already independently released under CC0.
