# Evaluation of NJ-derived Reconstruction Algorithms

## CTBF v5 paper pipeline

Reusable pipeline mechanics live in `paper_pipeline_contract.py`,
`paper_pipeline_runner.py`, and `paper_pipeline_analysis.py`. Registered and
smoke execution are currently disabled: CTBF v5 simulator parameters and a new
protocol/manifest must be owner-approved before the manifest hash lock is set.

The old G0-05-A CTBF-v2 manifest, commands, outputs, and exact-lock tests are
legacy evidence and are not accepted or replayed. Generic pipeline invariants
remain in `test/test_paper_pipeline.py`; new manifest-specific tests are added
only after the v5 freeze. The historical scripts and result folders described
below are CTBF v1/rejected-paper context, not v5 evidence.

## CTBF v5 non-paper reconstruction-intuition probe

`simulator_reconstruction_intuition_probe.py` implements the owner-approved
bounded design check that precedes selection of the paper height regime. It
runs 12 fresh common-seed blocks at H14/H24/H34, uses the Rule-Y generations
`[9,12,14]`, `[15,20,24]`, and `[21,28,34]`, and samples
`min(6, available distinct representative states)` at every nonempty biopsy.
It runs production minimum-bidirectional cnp2cnp once per case and passes that
same realized input to all six registered reconstruction arms.

The output distinguishes reconstruction problems: the two partial-output arms
declare GRF only, while fully labeled arms declare AD-F1 and GRF. It also
records truth-side sampled-ancestry, hidden-fork/path, recurrence,
representability, ambiguity, resource, runtime, and typed-failure summaries.
It writes no CNP profiles, trees, distance matrices, or simulator node
identities. The artifact is discovery-only and cannot freeze a paper height or
select simulator parameters from accuracy.

Run the complete owner handoff with:

```bash
./zzzz_verify_ctbf.sh simulator-reconstruction-probe \
  --base-config simulator_examples/default.json \
  --replicates 12 --base-seed 20260812 \
  --output /tmp/ctbf-v5-reconstruction-intuition-probe.json \
  --progress
```

## CTBF v5 truth-only sampling-fraction probe

`simulator_sampling_fraction_truth_probe.py` tests whether the sparse ancestry
seen above is driven by the capped-six rule. On the same truth blocks it
compares the nested hybrid rule `min(N,max(6,ceil(pN)))` for the capped-six
control, 5%, 10%, 25%, and 50%. Percentages refer to distinct representative
genotype states, not physical cells or clone abundance.

The compact probe records ancestry/hidden-branch diagnostics and projected
distance cost only. It runs no cnp2cnp, reconstruction, or evaluator. Run:

```bash
./zzzz_verify_ctbf.sh simulator-sampling-fraction-truth-probe \
  --base-config simulator_examples/default.json \
  --reference-report /tmp/ctbf-v5-reconstruction-intuition-probe.json \
  --replicates 12 --base-seed 20260812 \
  --output /tmp/ctbf-v5-sampling-fraction-truth-probe.json \
  --progress
```

## CTBF v5 dense-reconstruction operational preflight

`simulator_dense_reconstruction_preflight.py` runs only the registered largest
50%-sampling case from the completed truth probe: H34 replicate index 11, with
329 unique representative genotype states and 107,912 ordered distance
entries. It validates the regenerated truth and selection against both compact
reference artifacts, then runs production cnp2cnp and all six established
reconstruction/evaluation arms sequentially.

This is a technical feasibility gate, not an accuracy test. Even a passing
result requires owner review before any full dense reconstruction probe. Run:

```bash
./zzzz_verify_ctbf.sh simulator-dense-reconstruction-preflight \
  --base-config simulator_examples/default.json \
  --fraction-truth-report \
    /tmp/ctbf-v5-sampling-fraction-truth-probe.json \
  --sparse-reconstruction-report \
    /tmp/ctbf-v5-reconstruction-intuition-probe.json \
  --output /tmp/ctbf-v5-dense-reconstruction-preflight.json \
  --progress
```

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
