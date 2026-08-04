# Evaluation of NJ-derived Reconstruction Algorithms

## CTBF v2 paper runner

Current paper evidence uses the manifest-locked v2 modules
`v2_paper_contract.py`, `v2_paper_runner.py`, and `v2_paper_analysis.py`. Run
the read-only preflight with:

```bash
python -m algorithm_evaluation.v2_paper_runner validate \
  --manifest experimental_description/g0_05a_v2_preregistration_manifest.json
```

The `smoke` command is non-held-out and stamps its injected L1 matrix as
ineligible for paper evidence. Registered large runs require a new empty
output root and remain owner-run. The historical scripts and result folders
described below are CTBF v1/rejected-paper context, not v2 evidence.

## Historical workflow

The concept is described in Appendix E, below material shall enable to reconstruct results depicted in Figure 4.

This folder contains the workflow used for the appendix heatmap figure based on pairwise comparisons of NJ-like reconstruction algorithms.

The main scripts are:
- `tester.py` - generates raw per-algorithm results for each test variant.
- `analyzer.py` - compares algorithms and writes ranking CSV files.
- `visualizer.py` - generates the final heatmap figure from the ranking CSV files.

The seven test variants are:
- `r2bss025`
- `r2bss05`
- `r2bss075`
- `r4bss05`
- `r4bss075`
- `r4bss05high`
- `r4bss05highdm`

## To Generate The Figure From Already Computed Variants

If the variant folders in `algorithm_evaluation/results/` are already present and contain files such as `r2bss025`, `r2bss05`, and the other committed variants, run:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/analyzer.py
python algorithm_evaluation/visualizer.py
```

This regenerates the ranking CSV files and then writes:
- `algorithm_evaluation/heatmaps_side_by_side.png`

If the ranking CSV files are already present and you only want the final figure, run:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/visualizer.py
```

## To Regenerate Data In Variant Folders

The variant folders such as `algorithm_evaluation/results/r2bss05`, `algorithm_evaluation/results/r2bss025`, and the other committed variants can be generated directly from `tester.py`.

Examples:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/tester.py --r 2 --bss 0.5
```

This writes raw outputs into:

```bash
algorithm_evaluation/results/r2bss05/
```

Other examples:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/tester.py --r 2 --bss 0.25
python algorithm_evaluation/tester.py --r 4 --bss 0.75
python algorithm_evaluation/tester.py --r 4 --bss 0.5 --profile high
python algorithm_evaluation/tester.py --r 4 --bss 0.5 --profile highdm
```

These produce:

```bash
algorithm_evaluation/results/r2bss025/
algorithm_evaluation/results/r4bss075/
algorithm_evaluation/results/r4bss05high/
algorithm_evaluation/results/r4bss05highdm/
```

Useful optional arguments:

```bash
--seed 295
--seeds-file test/data/seeds.json
--algorithm-index 0
--config test/data/config_high_dm.json
--output-dir results/my_custom_variant
```

By default, `tester.py` reads seeds from:

```bash
test/data/seeds.json
```

Seed files can be provided in two formats:
- CSV file with a `seeds` column
- JSON file with a `seeds` array

Examples:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/tester.py --r 2 --bss 0.5 --seeds-file test/data/seeds.json
python algorithm_evaluation/tester.py --r 2 --bss 0.5 --seeds-file my_seeds.csv
```

For example, to rerun only one seed and one algorithm for `r2bss05`:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/tester.py --r 2 --bss 0.5 --seed 295 --algorithm-index 0
```

After raw outputs are written, regenerate the ranking tables and the final figure with:

```bash
cd /Users/voronwe/Work/PyCharmProjects/ctbf
python algorithm_evaluation/analyzer.py
python algorithm_evaluation/visualizer.py
```

## Notes

- `tester.py` depends on the main CTBF code and on a working `cnp2cnp` setup configured in `ctbs_config.json`.
- `tester.py` reads benchmark configs and the baseline seed file from `test/data/`.
- The committed `algorithm_evaluation/results/` folders already contain the data needed to rebuild the ranking tables and final figure.
- The reproduced files like `0nj.csv` in `algorithm_evaluation/results/r2bss05/` can differ from repository files in line order only; the metric rows are otherwise unchanged for matching seeds.
