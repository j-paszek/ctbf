# Evaluation of NJ-derived Reconstruction Algorithms

The concept is described in Appendix E, below material shall enable to reconstruct results depicted in Figure 4.

This folder contains the workflow used for the appendix heatmap figure based on pairwise comparisons of NJ-like reconstruction algorithms.

The main scripts are:
- `tester.py` - generates raw per-algorithm results for each test variant.
- `analyzer.py` - compares algorithms and writes ranking CSV files.
- `visualizer.py` - generates the final heatmap figure from the ranking CSV files.
- `test_ctbs.py` - unit and regression tests for the reconstruction code.

The seven test variants are:
- `r2bss025`
- `r2bss05`
- `r2bss075`
- `r4bss05`
- `r4bss075`
- `r4bss05high`
- `r4bss05highdm`

## To Generate The Figure From Already Computed Variants

If the variant folders in `test/results/` are already present and contain files such as `r2bss025`, `r2bss05`, and the other committed variants, run:

```bash
cd test
PYTHONPATH=.. python analyzer.py
python visualizer.py
```

This regenerates the ranking CSV files and then writes:
- `test/heatmaps_side_by_side.png`

If the ranking CSV files are already present and you only want the final figure, run:

```bash
cd test
python visualizer.py
```

## To Regenerate Data In Variant Folders

To regenerate the raw data inside folders such as `test/results/r2bss05`, `test/results/r2bss025`, and the other variants, run:

```bash
cd test
PYTHONPATH=.. python tester.py
```

Then regenerate the ranking tables and figure:

```bash
PYTHONPATH=.. python analyzer.py
python visualizer.py
```

## Notes

- `tester.py` depends on the main CTBF code and on a working `cnp2cnp` setup configured in `../ctbs_config.json`.
- `analyzer.py` and `visualizer.py` should be run from the `test/` directory.
- The committed `test/results/` folders already contain cached experiment outputs, so rerunning `tester.py` is not necessary unless you want to recompute the raw data.
