# Supplementary Figure 3

This directory contains the code and data needed to reproduce Supplementary
Figure 3.

## Prerequisites

- Python 3 with `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scipy`
- For exact reproduction from the frozen bundle, no external reconstruction
  tools are needed
- For re-running the study from scratch, `MEDICC2` must be installed and
  available in the environment
- For re-running the study from scratch, `cnp2cnp` must be installed in the
  location configured in `ctbs_config.json`

References:

- MEDICC2: <https://anaconda.org/bioconda/medicc2>
- cnp2cnp: <https://github.com/AEVO-lab/cnp2cnp>

The `measures_evaluation/supp_figure_3` code now resolves `cnp2cnp` from the same
`ctbs_config.json` file used by the main CTBF codebase.

Newly generated matrices use G0-03 semantics
`ctbf-cnp2cnp-any-min-bidirectional-v1`:

```text
min(d_any(u, v), d_any(v, u))
```

Both the direct optimized helper and subprocess helper evaluate both profile
orders. A cnp2cnp failure stops/skips that matrix; it is never replaced by L1
under the cnp2cnp label. New `metadata.csv` rows include the semantic version,
formula, construction path, command/API template, source revision, and source
hashes.

## Exact reproduction

Frozen input data is included in
[`published_figure_3_data`](./published_figure_3_data). To redraw the published
panels:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python measures_evaluation/supp_figure_3/generate_subfigures_figure_3.py
```

The generated files are written to:

- `measures_evaluation/supp_figure_3/figures/subfigure_a.png`
- `measures_evaluation/supp_figure_3/figures/subfigure_b.png`
- `measures_evaluation/supp_figure_3/figures/subfigure_c.png`

The frozen bundle predates G0-03. Redrawing it reproduces the historical
figures; it does not retroactively establish bidirectional-minimum semantics
or provenance for those stored matrices.

## Re-run from scratch

The same directory also contains the scripts needed to regenerate the study.

Example:

```bash
conda run --no-capture-output -n medicc2_src \
  python measures_evaluation/supp_figure_3/generate_subfigures_figure_3.py \
  --data-source generate \
  --results-dir measures_evaluation/supp_figure_3/results
```

Before running from scratch, make sure `ctbs_config.json` points to a working
`cnp2cnp` installation.

If you already have a results directory and only want to redraw the panels:

```bash
python measures_evaluation/supp_figure_3/generate_subfigures_figure_3.py \
  --data-source existing \
  --results-dir /path/to/results
```

## Files

- `generate_subfigures_figure_3.py`
  Main entry point.
- `generate_figure_3_data.py`
  Data-generation pipeline.
- `subfigure_a.py`, `subfigure_b.py`, `subfigure_c.py`
  Plotting scripts for the three panels.
- `published_figure_3_data/`
  Frozen minimal data bundle for exact reproduction.
