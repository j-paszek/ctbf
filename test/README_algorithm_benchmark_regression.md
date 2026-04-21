# Algorithm Benchmark Regression Tests

`test_algorithm_benchmark_regression.py` is an opt-in slow regression suite. It reruns the legacy algorithm benchmark pipeline and compares the new metrics against the committed CSV files in `algorithm_evaluation/results/`.

By default, the test is skipped:

```bash
pytest -q test/test_algorithm_benchmark_regression.py
```

Run the full benchmark regression with:

```bash
CTBF_RUN_SLOW_BENCHMARKS=1 pytest -q test/test_algorithm_benchmark_regression.py
```

The full run covers all committed benchmark variants, all legacy algorithm indexes from `algorithm_evaluation/tester.py`, and all seeds that have expected rows in both the matching `rec.csv` and `nj.csv`.

To run one targeted smoke test:

```bash
CTBF_RUN_SLOW_BENCHMARKS=1 \
CTBF_BENCHMARK_VARIANTS=r4bss05 \
CTBF_BENCHMARK_ALGORITHM_INDEXES=20 \
CTBF_BENCHMARK_SEEDS=295 \
pytest -q test/test_algorithm_benchmark_regression.py
```

This command means:

- `CTBF_RUN_SLOW_BENCHMARKS=1` enables the otherwise skipped slow suite.
- `CTBF_BENCHMARK_VARIANTS=r4bss05` limits the run to the `r4bss05` benchmark variant.
- `CTBF_BENCHMARK_ALGORITHM_INDEXES=20` limits the run to algorithm index `20`, currently `neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony`.
- `CTBF_BENCHMARK_SEEDS=295` limits the run to seed `295`.
- `pytest -q test/test_algorithm_benchmark_regression.py` runs only the benchmark regression test file in quiet mode.

Multiple variants, indexes, or seeds can be comma-separated:

```bash
CTBF_RUN_SLOW_BENCHMARKS=1 \
CTBF_BENCHMARK_VARIANTS=r4bss05,r2bss05 \
CTBF_BENCHMARK_ALGORITHM_INDEXES=17,20 \
CTBF_BENCHMARK_SEEDS=295,689 \
pytest -q test/test_algorithm_benchmark_regression.py
```

Algorithms can also be selected by name:

```bash
CTBF_RUN_SLOW_BENCHMARKS=1 \
CTBF_BENCHMARK_VARIANTS=r4bss05 \
CTBF_BENCHMARK_ALGORITHM_NAMES=neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
CTBF_BENCHMARK_SEEDS=295 \
pytest -q test/test_algorithm_benchmark_regression.py
```

The benchmark path uses the existing `cnp2cnp` file-based workflow configured by `ctbs_config.json`, so it must be able to write into the configured `cnp2cnp` examples/output location.

## Freezing New Algorithm Cases

Use `test/tools/freeze_algorithm_case.py` to generate a new frozen regression fixture from a simulator run. The script stores:

- the simulator tree in `test/data/tree_samples/`,
- exact biopsy node IDs per generation,
- the distance matrix used by reconstruction,
- grouped expected outputs for all legacy algorithms unless `--no-expectations` is passed.

Example:

```bash
python test/tools/freeze_algorithm_case.py \
  --seed 689 \
  --r 4 \
  --bss 0.5 \
  --profile base \
  --case-id seed689_r4bss05_new
```

The script refuses to overwrite existing files unless `--overwrite` is provided.

By default, distance matrices are computed serially to avoid multiprocessing restrictions in sandboxed environments. Pass `--parallel-distance` if you want to use the existing multiprocessing distance helper.

## Freezing Nested Variant Fixtures

Use `test/tools/freeze_algorithm_variant_cases.py` for the richer fixture layout used for fast algorithm iteration. By default it freezes all seven benchmark variants and the three reference algorithms:

- `neighbor_joining_baseline`
- `neighbor_joining_hybrid_anticentral_adaptive_v3`
- `neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony`

The output layout is:

```text
test/data/algorithm_cases/<variant>/<seed>/
  input.json
  full_cnp/<algorithm>.json
  biopsy_guided_top/<algorithm>.json
```

`input.json` stores the true tree, biopsy cells, the frozen `cnp2cnp` distance matrix, and the true-tree distance matrix. Each algorithm result stores the reconstructed tree, Newick string, root, ancestor-F1 metrics, and GRF score.

Preview a batch without writing files:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --variant r4bss05 \
  --seed 295 \
  --dry-run
```

Freeze one variant and one seed:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --variant r4bss05 \
  --seed 295
```

Freeze only inputs, without algorithm outputs:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --variant r4bss05 \
  --seed 295 \
  --input-only
```

The tool skips no failures silently: it reports failures and exits non-zero at the end. Use `--fail-fast` to stop at the first failing case. Existing files are not overwritten unless `--overwrite` is passed.

## Adding New Algorithm Variants

The committed benchmark CSV files use legacy algorithm indexes from `algorithm_evaluation/tester.py`. Do not reorder or insert into `get_legacy_algorithms_to_test()`, because that changes the meaning of files such as `20rec.csv` and `20nj.csv`.

For exploratory variants, add the callable to `get_experimental_algorithms_to_test()`. The public `get_algorithms_to_test()` function returns legacy algorithms first and experimental algorithms afterward, so old result indexes remain stable while new variants can still be selected by their appended index.

Once an experimental variant is accepted, freeze expected behavior for it deliberately by adding committed result files or a frozen case fixture rather than changing legacy indexes.

`algorithm_evaluation/tester.py` can select algorithms by either index or name. Name selection is useful for experimental variants whose appended index may change while you are developing:

```bash
python algorithm_evaluation/tester.py --r 4 --bss 0.5 --list-algorithms
```

Preview a benchmark run without running simulations:

```bash
python algorithm_evaluation/tester.py \
  --r 4 \
  --bss 0.5 \
  --seed 295 \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
  --dry-run
```

```bash
python algorithm_evaluation/tester.py \
  --r 4 \
  --bss 0.5 \
  --seed 295 \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony
```

`--algorithm-index` and `--algorithm-name` can be combined; duplicates are ignored while preserving the requested order.
