# Interactive tree guide

Open the [all-method gallery](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/index.html)
or download this repository and open `presentation/quickstart-v2/index.html`
in a browser. Its plots load Plotly from a CDN, so an internet connection is
required. The [README](../README.md#reproduce-the-example) gives the command
and configuration needed to generate a new local copy.

## The ten views

All views use one simulated history and the same three biopsies.

- **True tree:** the complete history. Green vertices mark sampled occurrences.
- **Trimmed true tree:** all occurrences of sampled CNP states, with unlabeled branching vertices retained where needed to connect them.
- **M0:** partial pooled classical NJ.
- **M1:** full pooled labeled closest-pair.
- **M2:** partial biopsy-guided reconstruction with binary anticentral top completion.
- **M3:** full pooled rooted labeled-Q.
- **M4:** partial biopsy-guided reconstruction with classical NJ top completion.
- **M5:** full pooled binary anticentral reconstruction.
- **M6:** full biopsy-guided reconstruction with binary anticentral top completion.
- **M7:** full biopsy-guided reconstruction with rooted labeled-Q top completion.

M0/M4, M3/M7 and M5/M6 each share a completion method. The biopsy-guided member of
each pair additionally uses biopsy order, adaptive distance radii and
deferred parent ties. M2 and M6 have the same topology. M2 removes the labels
from newly inferred top vertices, whereas M6 retains them.

## Reading and exploring a tree

1. Hover over a vertex for its CNP label and copy numbers. Vertex identifiers distinguish occurrences; the same CNP label can appear more than once.
2. Hover over an edge midpoint for recorded events or reconstruction edge weight. Edges in trimmed truth may represent contracted paths.
3. Use the toolbar to zoom or pan, and double-click to reset the view.
4. Click a legend entry to hide or show a group of vertices.
5. Use the navigation links to compare trees. The overview also lists sampling counts and scores.

Light-blue vertices carry CNP states observed in at least one biopsy. In
truth views, green distinguishes the occurrences actually sampled from
other occurrences of the same states. Reconstruction vertices are not
matched to particular true occurrences. An en dash represents an unlabeled
vertex. Vertical position shows topological depth, not elapsed time.

The overview links the configuration, biopsy inputs, tree JSON and summary.
GRF similarity is shown against full and trimmed truth. AD-F1 is shown for
the full reconstructions. Optional partial-tree AD-F1 is marked as a
diagnostic. Higher values indicate closer agreement for these scores; a
single example is an illustration rather than a performance study.

Set `methods` in `examples/quickstart.json` to select any subset of M0–M7.
Set `visualize` to `false` for a headless run, or `write_newick` to `true`
for additional topology-only exports. The gallery always reflects the
selected methods. No Python editing is needed.

## Publishing the supplied gallery

The HTML files are ready to view locally. Online links require committing
and pushing `presentation/quickstart-v2/` to the configured GitHub Pages
publishing source. For branch-based deployment, use **Settings → Pages →
Build and deployment → Deploy from a branch**, then choose the intended
branch and its **root** directory. Preserve the folder path above. If the
repository already uses an Actions deployment, include this directory in its
published artifact instead. Wait for that deployment to finish before
checking the online links. See
[GitHub's publishing-source guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The owner release script `scripts/freeze_release/step7.sh` previews the exact
source/gallery list. Its explicit `--stage-reviewed-files` option stages
that selection, including ignored gallery files, and untracks four obsolete
root HTML files containing embedded Plotly.js while keeping local copies.
It does not commit, push or configure GitHub Pages. The new gallery loads
Plotly.js from a CDN and includes no copy of that library.
