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

The benchmark path uses the existing `cnp2cnp` file-based workflow configured by `ctbs_config.json`, so it must be able to write into the configured `cnp2cnp` examples/output location.
