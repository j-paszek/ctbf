# Third-party tools, materials and outputs

This notice identifies external tools used with CTBF 1.2.0 and complements
the [licence scope](LICENSESCOPE.md). CTBF's MIT licence covers its original
implementation, adapters and associated software documentation. External
software retains its upstream licence and is installed separately.

## External software

External tools are installed separately. This release contains CTBF code
and its original integration wrappers, but does not redistribute third-party
software implementations or binaries. Installation links and version/checksum
records identify the dependencies. Copied or adapted upstream source and
templates, JARs, container images and installed environments are also excluded
from the code and data assets.

| Tool | Upstream source | Licence information for the version used |
|---|---|---|
| cnp2cnp | [AEVO-lab/cnp2cnp](https://github.com/AEVO-lab/cnp2cnp) | A package-wide licence was not verified in the inspected version. The CTBF release provides its adapter and source identity, not the upstream implementation. |
| MEDICC2 | [Schwarz lab repository](https://bitbucket.org/schwarzlab/medicc2), [project description](https://pypi.org/project/medicc2/) | The inspected LICENSE and README specify GPLv3. The README identifies modified OpenFST/pywrapfst code under Apache 2.0. These notices take precedence here over the conflicting PyPI MIT classifier. |
| NestedBD / NestedBD-Long | [Androstane/NestedBD](https://github.com/Androstane/NestedBD) | A package-wide licence was not verified in the inspected version. Java source, package archives, upstream XML templates and generator code are excluded from CTBF distribution. |
| BEAST 2 and its packages | [BEAST 2](https://www.beast2.org/), [CoupledMCMC](https://github.com/nicfel/CoupledMCMC), [Sampled Ancestors](https://github.com/CompEvol/sampled-ancestors) | Each independently installed runtime/package retains its own licence. Package versions are recorded in the run manifests. |
| CopyKit | [navinlabcode/copykit](https://github.com/navinlabcode/copykit) | The ctbfbio tool manifest records AGPL-3.0-or-later for the pinned version. Its implementation and bundled genomic resources are outside CTBF's licence grants. |
| scAbsolute and scDNAseq-workflow | [scAbsolute](https://github.com/markowetzlab/scAbsolute), [scDNAseq-workflow](https://github.com/markowetzlab/scDNAseq-workflow) | The ctbfbio tool manifest records GPL-3.0-or-later. Implementations and runtimes are installed separately. |
| Python/R libraries and calling/alignment dependencies | Sources and versions in the corresponding environment and tool manifests | Each dependency retains its own licence. Empirical-pipeline dependencies are listed in `ctbfbio/metadata/tools.tsv`. |

These records describe the inspected versions. The applicable upstream
notice accompanies each independently obtained version. The cnp2cnp hashes
used in the confirmation experiments are recorded in the design manifest.
MEDICC2 runtime identity is recorded with its resource checks and runs.
NestedBD/BEAST and empirical-calling identities are recorded in the
corresponding ctbfbio manifests.

The interactive tutorial loads Plotly.js from Plotly's CDN. Its HTML files
contain CTBF-generated plot data and configuration, without an embedded copy
of that library.

## External output files

Direct external-tool outputs have a separate rights scope from CTBF software
and synthetic data. Where distributed, they are identified by per-tool
notices, for example in `external-outputs/medicc2/`,
`external-outputs/nestedbd-long/` or `external-outputs/cnp2cnp/`.
Each notice specifies:

1. Covered files and their content, such as trees, traces, matrices or logs.
2. Generating tool, version/commit, upstream source and originating run.
3. Input-data provenance and any incorporated upstream material.
4. Output-specific reuse terms or redistribution basis and required attribution.

CTBF's MIT and CC BY 4.0 grants exclude these direct-output files and
content-preserving conversions. Their reuse terms are stated in the
corresponding output notice, separately from the generating program's
software licence. Material with unresolved redistribution status is excluded
from public release assets.

Original CTBF evaluation scores and aggregate tables may include comparisons
with external methods under the scope stated in [LICENSESCOPE.md](LICENSESCOPE.md).
The source trees, traces and matrices retain their separate notices.

## Empirical data

KTN152 originates from Kim et al., *Cell* (2018),
[DOI 10.1016/j.cell.2018.03.041](https://doi.org/10.1016/j.cell.2018.03.041),
ENA project [PRJNA396019](https://www.ebi.ac.uk/ena/browser/view/PRJNA396019) /
SRA study SRP114962. Raw reads and reference genomes are excluded from the
CTBF release.

Processed CopyKit/scAbsolute CNPs, bin coordinates and empirical
reconstructions are assigned to the separately documented ctbfbio evidence
asset. They are outside CTBF's synthetic-data licence. That asset records
source attribution, access/consent conditions and derived-data rights. Its
current status is a local draft pending the recorded data-rights review.
