# CTBF: Cancer Tree Biopsy Framework

Jarosław Paszek, Agnieszka Mykowiecka, Krzysztof Gogolewski

CTBF studies how a cancer's copy-number profiles change over time and how much
of that history can be reconstructed from biopsies. A **copy-number profile
(CNP)** is a sequence of integers: each entry gives the number of copies of a
genomic bin. CTBF can simulate a known evolutionary history, sample it at
several generations, reconstruct trees from the sampled CNPs, and compare the
reconstructions with the known history.

Reconstruction takes CNPs grouped by biopsy time and a matrix of pairwise
dissimilarities. **Partial trees** may have unlabeled inferred ancestors.
**Full, closed-state trees** label every vertex with one of the observed CNPs.
A CNP label can occur at several vertices in either representation.

This is the **1.2.0 release candidate**, using the CTBF v5 simulation model.
Configurations and results from releases 1.0/1.1 are not backward-compatible
with this release.

## Install

Use CPython 3.13. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-release.txt
```

For event-based CNP comparisons, install
[cnp2cnp](https://github.com/AEVO-lab/cnp2cnp) separately following its upstream
instructions. CTBF needs the path to its `cnp2cnp.py` file. The interactive
examples below can also be viewed without installing either program.

An optional installation check, requiring no external tools, is:

```bash
python -m reproducibility.example
```

It applies classical neighbor joining to four supplied CNPs and their
dissimilarity matrix. Expected output: `complete: toy reconstruction; 7 nodes; 6 edges`.

## One history, all eight CTBF paper methods

The example uses 100 bins, 14 generations and biopsies at generations 9, 12
and 14. It produces **one true tree, its state-trimmed view and eight
reconstructions**, all from the same history. Select any subset of M0–M7
in one JSON file, without editing Python.

| Tree | What it shows | Interactive view |
|---|---|---|
| True tree | Complete simulated history, with sampled occurrences highlighted | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/true_tree.html) |
| Trimmed true tree | Every occurrence of an observed CNP state, with necessary unlabeled branching vertices retained | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/trimmed_true_tree.html) |
| M0 — partial, pooled | Classical neighbor joining of pooled distinct CNPs | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M0.html) |
| M1 — full, pooled | Labeled closest-pair reconstruction | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M1.html) |
| M2 — partial, biopsy-guided | Binary anticentral top completion, with new top labels suppressed | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M2.html) |
| M3 — full, pooled | Rooted labeled-Q reconstruction | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M3.html) |
| M4 — partial, biopsy-guided | Classical NJ top completion | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M4.html) |
| M5 — full, pooled | Binary anticentral reconstruction | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M5.html) |
| M6 — full, biopsy-guided | Binary anticentral top completion, retaining new top labels | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M6.html) |
| M7 — full, biopsy-guided | Rooted labeled-Q top completion | [View](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/M7.html) |

[Open the gallery](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/index.html)
or open the supplied [index.html](presentation/quickstart-v2/index.html) in a
browser. The online links use GitHub Pages and become available when these
files are published there. Hover over vertices to inspect CNPs and biopsy
membership, and over edge midpoints for event or distance information.
[Viewing guide](presentation/README.md).

### Reproduce the example

Edit [examples/quickstart.json](examples/quickstart.json): replace
`"cnp2cnp": null` with the absolute path to your installed `cnp2cnp.py`.
With the environment activated, run:

```bash
python -m examples.quickstart --config examples/quickstart.json
```

Open `quickstart-output/index.html`. Expected final line:
`complete: 10 trees; open .../quickstart-output/index.html`.
The output directory must be new. No browser opens automatically.
HTML plots load Plotly from its CDN, so viewing them requires internet access.

All settings are in [examples/quickstart.json](examples/quickstart.json):

| Setting | Example value | Meaning |
|---|---|---|
| `seed` | `9` | Reproducible simulation, biopsy draws and reconstruction tie resolution |
| `simulator` | JSON object | Complete simulator configuration, including 100 bins, 14 generations and CNA initiation probability `0.001` per bin |
| `biopsy_generations` | `[9, 12, 14]` | Sampled generations, in chronological order; the founder is generation 0 |
| `biopsy_sampling` | `{"mode":"fraction","fraction":0.5,"lower_bound":6}` | Fractional sampling with a lower bound; alternatively use `{"mode":"fixed","size":6}` |
| `methods` | `["M0","M1","M2","M3","M4","M5","M6","M7"]` | Select any nonempty subset of the eight CTBF paper methods |
| `visualize` | `true` | Write the interactive HTML gallery; `false` produces headless JSON outputs only, plus optional Newick |
| `write_newick` | `false` | Also export each tree as topology-only `.nwk`, retaining labels and leaving unlabeled vertices empty |
| `partial_ad_f1` | `false` | Also report AD-F1 on partial trees as a labeled diagnostic, rather than their primary endpoint |
| `bedfile` | `null` | Optional single-chromosome BED-like CSV, described below |
| `cnp2cnp` | `null` | Set to the path of your independently installed `cnp2cnp.py`, or override with `--cnp2cnp` |
| `output` | `"../quickstart-output"` | New output directory, relative to the JSON file; optionally override with `--output` |
| `distance_workers` | `1` | One uses two ordered matrix calls; larger values use the bounded pairwise provider with up to this many workers |
| `max_distance_profiles` | `64` | Stop before distance computation if the number of distinct sampled CNPs exceeds this limit |
| `cnp2cnp_timeout_seconds` | `120` | Per-process time limit for an ordered matrix call, or for each directional pair call in parallel mode |

For a generation of size `N`, fractional sampling requests
`min(N, max(lower_bound, ceil(fraction * N)))` representatives. Fixed sampling
requests `min(N, size)`. Both rules accept small nonempty generations.
The simulator tracks genotype representatives, rather than population
abundances. Equal CNPs are collapsed within a biopsy and retained as separate
observations across biopsy times.

The selected methods share one matrix over exactly the sampled distinct CNPs.
Each entry is the minimum of the two directed cnp2cnp `d_any` costs, also in
parallel mode. M2/M4/M6/M7 use the adaptive median radius and defer ambiguous
parent ties. M0/M4 share classical NJ completion; M3/M7 share rooted labeled-Q;
M5/M6 share binary anticentral completion. M2 is M6's partial projection.

The output also contains `config.json`, `input.json` (biopsy CNPs and bin
coordinates), `trees.json` (all selected trees), and `summary.json` (method
identities, sampling counts, scores and simulator/distance-tool provenance).
The displayed scores describe this example only. JSON retains complete tree
attributes; optional Newick exports only labeled topology, without lengths.

To run only **M0**, set `"methods": ["M0"]` and keep `"visualize": true`.
This produces its Plotly tree and both GRF scores. Set `"partial_ad_f1": true`
if you also want its ancestry diagnostic. M0 labels only leaves and therefore
has no positive labeled ancestor–descendant pairs. Its diagnostic AD-F1 is
zero, including under the empty/empty convention; GRF is its meaningful
paper endpoint.

For all **partial methods**, select `["M0", "M2", "M4"]`. For all **full
methods**, select `["M1", "M3", "M5", "M6", "M7"]`. AD-F1 is reported for
every full method. MEDICC2 and NestedBD-Long use the separate external-tool
workflows linked below and are not bundled with this runner.

Relative paths in JSON are resolved against that JSON file's directory.
CLI overrides are relative to the current working directory. For example,
without editing the distributed configuration:

```bash
python -m examples.quickstart --config examples/quickstart.json \
  --cnp2cnp /path/to/cnp2cnp/cnp2cnp.py --output quickstart-output
```

`config.json` preserves the supplied configuration; environment-specific CLI
path overrides are not copied into it. To rerun, choose a fresh output path
and provide your local cnp2cnp path again.

With `bedfile`, remove `GENOME_LENGTH`, `NUMBER_OF_CHROMOSOMES`,
`INITIAL_COPY_NUMBER` and `CRUCIAL_BIN_INDICES` from `simulator`. BED rows
supply the bin layout, initial CNs and crucial flags. This runner uses
single-chromosome distances and rejects multi-chromosome inputs rather than
allowing CNA events across chromosome boundaries. See the
[simulator manual](presentation/simulator_manual.md) for the CSV format.

## Advanced driver: `ctbs.py`

The general simulation/reconstruction driver is:

```bash
python ctbs.py
```

Before running it, edit [ctbs_config.json](ctbs_config.json). Set
`cnp2cnp_FILE` to your installed `cnp2cnp.py` and `cnp2cnp_FOLDER` to the
installation directory. `RUN_SINGLE_TEST` controls the analysis:

| Option | Meaning |
|---|---|
| `seed` | Random seed |
| `config` | Simulator JSON file, for example `simulator_examples/default.json` |
| `bedfile` | Optional chromosome/bin CSV with per-bin settings; use `null` without one |
| `biopsy_generations` | Increasing generation numbers to sample |
| `biopsy_size` | Fixed sample size, capped by availability |
| `biopsy_size_scalable` | Fractional sampling: `min(N, max(1, floor(f*N)))` for nonempty generations; takes precedence over `biopsy_size` |
| `r_dist` | Fixed distance radius for biopsy connections |
| `biopsy_guided_strategy` | Bottom-connection strategy, e.g. `"deferred_tie"` |
| `reconstruction_algorithm` | Pairwise/top solver, specified below |
| `write_newick` | Print simulated and reconstructed trees in Newick format |
| `visualize` | Generate plots; static-image export additionally needs Plotly's optional image-export dependencies |

Use `"neighbor_joining_classical"` for partial reconstruction or
`"neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"` for
full anticentral reconstruction. The driver constructs both a biopsy-guided
tree and a pooled tree. Its fixed-radius settings support exploratory runs;
the JSON runner above supplies all eight exact paper compositions. Fixed
radius or alternative bottom settings constitute custom variants, not the
adaptive methods M2/M4/M6/M7.

The [simulator manual](presentation/simulator_manual.md) describes every
current JSON field, chromosome-aware intervals, offspring generation,
viability rules, resource limits and the optional BED-like CSV. For an
HTML-only workflow with no static-image dependency, use the JSON runner.

Console evaluation includes `ancestors_unique_restricted`, whose `F1` field
compares unique ancestor–descendant CNP-label pairs on the biopsy label set.
This is a restriction by **state label**, not by sampled vertex identity:
all occurrences of those labels may contribute. The example reports AD-F1
for full trees and GRF similarity against both truth views.

## Methods and evaluation

| Input | Partial trees | Full, closed-state trees |
|---|---|---|
| Pooled distinct CNPs | M0: classical NJ | M1: labeled closest-pair; M3: rooted labeled-Q; M5: binary anticentral |
| Biopsy-guided, adaptive-radius/deferred-tie bottom | M2: binary anticentral top; M4: classical NJ top | M6: binary anticentral top; M7: rooted labeled-Q top |

Pooled methods use the CNPs without biopsy order. Biopsy-guided methods first
connect adjacent biopsy levels and then complete the top of the resulting
forest. M2 and M6 share the same construction and differ in whether newly
created top vertices retain their CNP labels.

**Rooted GRF similarity** compares tree structure, including repeated labels.
CTBF reports it against the complete true tree and against state-trimmed
truth. **AD-F1** measures agreement between unique ancestor–descendant
state-label pairs for full reconstructions. Both reported scores range from
0 to 1, with larger values indicating closer agreement. For AD-F1, the
evaluation convention assigns 0 when both relation sets are empty.

## Reproduce paper results

[REPRODUCE.md](reproducibility/REPRODUCE.md) gives the portable commands,
[evidence-map.tsv](reproducibility/evidence-map.tsv) maps the tables and
figures to their inputs, and [SCHEMA.md](reproducibility/SCHEMA.md) defines
the exports. Once the matching synthetic evidence asset is available:

```bash
python -m reproducibility.reproduce \
  --bundle /path/to/ctbf-v5-reproduction \
  --output /path/to/new-output --figures
```

This exports 12 synthetic numerical tables and renders three numerical
figures from stored summaries and selected per-history evidence. It preserves
the original intervals and method-specific coverage. Rebuilding the vector
schematics additionally requires TeX/latexmk.

The separate KTN152 empirical evidence asset is a local draft pending its
data-rights review. It adds two empirical tables and the editable empirical
figure, whose PDF rendering requires Inkscape. This case study demonstrates
computational feasibility; its absolute copy-number scale remains unresolved.

Release assets and archive identifiers will be linked here when published.
The larger simulation and external-comparator workflows are documented in
the [confirmation](experimental_description/v5_confirmatory_runbook.md),
[partial-comparator](experimental_description/v5_partial_comparator_extension_runbook.md)
and [trimmed-truth](experimental_description/v5_trimmed_truth_runbook.md)
runbooks. Full reruns require their input banks and independently installed
external tools. The lightweight evidence asset supports stored-result
reproduction rather than the complete raw-data and inference pipelines.

## Verification, citation and licence

```bash
python -m pytest -q test/test_quickstart.py test/test_release_reproduction.py test/test_reconstructor_classical_nj.py
```

Use [CITATION.cff](CITATION.cff) for software attribution. Original CTBF code
is licensed under [MIT](LICENSE). Original CTBF-generated simulation and
benchmark data covered by [LICENSESCOPE.md](LICENSESCOPE.md) use
[CC BY 4.0](LICENSE-DATA), with [data attribution](DATA_ATTRIBUTION.md).
[Third-party notices](THIRD_PARTY_NOTICES.md) identify external dependencies
and the separate scope of their outputs and empirical data.

Funding: National Science Centre grant no. 2020/39/D/ST6/03321.

Contact: j.paszek@uw.edu.pl
