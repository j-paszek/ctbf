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

Use `test/tools/freeze_algorithm_variant_cases.py` for the richer fixture layout used for fast algorithm iteration. By default it freezes all seven benchmark variants and all 21 publication heatmap legacy algorithms. The legacy set includes:

- `neighbor_joining_baseline`
- `neighbor_joining_hybrid_anticentral_adaptive_v3`
- `neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony`

To add missing algorithm outputs for already frozen inputs without rerunning the simulator or `cnp2cnp`, run:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --existing-seeds \
  --results-only \
  --skip-existing
```

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

## Fast Biopsy Preset Benchmark

Use `test/tools/fast_biopsy_preset_benchmark.py` to compare biopsy-guided presets without rerunning simulations, biopsies, `cnp2cnp`, or existing algorithms. The tool reads the frozen nested fixture inputs and writes only additional `biopsy_guided_top` result JSON files for preset rows:

- `biopsy_preset_default`
- `biopsy_preset_anticentral_tie`
- `biopsy_preset_binarized`
- `biopsy_preset_anticentral_binarized`

Run all frozen variants and all preset rows:

```bash
python test/tools/fast_biopsy_preset_benchmark.py --overwrite
```

Run one smoke case:

```bash
python test/tools/fast_biopsy_preset_benchmark.py \
  --variant r4bss05 \
  --seed 1001 \
  --preset biopsy_preset_anticentral_binarized \
  --overwrite
```

Then regenerate rankings and the heatmap that includes the new biopsy-preset rows:

```bash
MPLCONFIGDIR=/tmp/ctbf-mplconfig \
XDG_CACHE_HOME=/tmp/ctbf-xdg-cache \
python test/tools/heatmaps_from_json.py \
  --output-file test/heatmaps_side_by_side_biopsy_presets.png \
  --rankings-dir test/data/results_from_json
```

The heatmap tool uses mode-specific algorithm lists: the `full_cnp` panels remain limited to legacy NJ-like algorithms, while `biopsy_guided_top` panels include those algorithms plus the biopsy-preset rows.

## JSON Fixture Workflows

The nested JSON fixtures support fast checks that do not rerun the simulator except where explicitly noted.

## Corrected-GRF Metric Refresh

After the exact multiset-GRF fix, result JSON metrics can be refreshed from
stored true/reconstructed trees without rerunning simulation, distance
computation, or reconstruction:

```bash
python test/tools/refresh_algorithm_case_metrics.py \
  --cases-root /tmp/ctbf_algorithm_cases_refreshed \
  --summary-file /tmp/ctbf_algorithm_cases_refreshed/metric_refresh_summary.csv
```

The refreshed metric schema keeps corrected `metrics["grf"]` as the
higher-is-better exact-multiset similarity, adds `metrics["ext_grf"]` as the
underlying distance, and preserves the old set-collapsed similarity at
`metrics["grf_legacy_set_similarity"]`.

Use the single GRF metric gate for evaluator exactness plus frozen JSON metric
consistency:

```bash
python test/tools/run_grf_metric_checks.py \
  --cases-root /tmp/ctbf_algorithm_cases_refreshed \
  --algorithm-name new_alg \
  --algorithm-name biopsy_preset_default \
  --algorithm-name biopsy_preset_anticentral_tie \
  --algorithm-name biopsy_preset_binarized \
  --algorithm-name biopsy_preset_anticentral_binarized
```

## Universal Legacy JSON Checker

`run_algorithm_case_json_checks.py` is now the canonical checker. It validates only the publication benchmark set committed in `PUBLICATION_HEATMAP_ALGORITHM_NAMES`, even if extra stored rows such as `new_alg` or `biopsy_preset_*` are present under `test/data/algorithm_cases/`.

Run the canonical checker from the repo root:

```bash
python test/tools/run_algorithm_case_json_checks.py
```

From inside the `test/` directory:

```bash
python tools/run_algorithm_case_json_checks.py
```

The default canonical checker runs:

- evaluator replay for every stored reconstructed tree,
- reconstruction determinism for every stored algorithm result,
- JSON heatmap and ranking CSV regeneration.

This is the recommended default after refactors because it verifies that the committed canonical benchmark set still scores correctly, reconstruction remains deterministic from frozen inputs, and the heatmap workflow can consume the canonical files.

## Experimental JSON Checker

Use `run_experimental_json_checks.py` for non-canonical stored rows such as:

- `new_alg`
- `biopsy_preset_default`
- `biopsy_preset_binarized`

From the repo root:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name new_alg
```

From inside the `test/` directory:

```bash
python tools/run_experimental_json_checks.py \
  --algorithm-name new_alg
```

That runs metrics replay and determinism replay only for the selected stored row names.

Validate several extra rows at once:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name new_alg \
  --algorithm-name biopsy_preset_default \
  --algorithm-name biopsy_preset_binarized
```

Also regenerate a filtered heatmap for the selected rows:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name new_alg \
  --algorithm-name neighbor_joining_baseline \
  --algorithm-name neighbor_joining_hybrid_opt \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
  --check heatmap \
  --heatmap-output-file test/heatmaps_side_by_side_new_alg.png
```

Or run all three checks together:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name new_alg \
  --algorithm-name neighbor_joining_baseline \
  --algorithm-name neighbor_joining_hybrid_opt \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
  --check metrics \
  --check determinism \
  --check heatmap \
  --heatmap-output-file test/heatmaps_side_by_side_new_alg.png
```

For the full infrastructure replay, run:

```bash
python test/tools/run_algorithm_case_json_checks.py --all
```

From inside the `test/` directory:

```bash
python tools/run_algorithm_case_json_checks.py --all
```

`--all` also recomputes both frozen distance matrices:

- true-tree distance matrices from `input.json.true_tree`,
- `cnp2cnp` distance matrices from `input.json.biopsies`.

Useful runner options:

```bash
python test/tools/run_algorithm_case_json_checks.py --dry-run
python test/tools/run_algorithm_case_json_checks.py --all --skip-heatmap
python test/tools/run_algorithm_case_json_checks.py --check metrics
python test/tools/run_algorithm_case_json_checks.py --check determinism --check heatmap
```

Available check names are:

- `metrics`
- `determinism`
- `true-tree-matrix`
- `cnp2cnp-matrix`
- `heatmap`

## Regenerating The Heatmap From JSON

Run:

```bash
python test/tools/heatmaps_from_json.py
```

or from this readme

```bash
python tools/heatmaps_from_json.py
```


This reads:

```text
test/data/algorithm_cases/
```

and writes:

```text
test/data/results_from_json/
test/heatmaps_side_by_side_from_json.png
```

If Matplotlib reports unwritable cache directories, use temporary cache paths:

```bash
MPLCONFIGDIR=/tmp/ctbf_mpl_config \
XDG_CACHE_HOME=/tmp/ctbf_xdg_cache \
MPLBACKEND=Agg \
python test/tools/heatmaps_from_json.py
```

To write outputs outside the repo:

```bash
MPLCONFIGDIR=/tmp/ctbf_mpl_config \
XDG_CACHE_HOME=/tmp/ctbf_xdg_cache \
MPLBACKEND=Agg \
python test/tools/heatmaps_from_json.py \
  --rankings-dir /tmp/ctbf_json_rankings \
  --output-file /tmp/ctbf_heatmaps_side_by_side_from_json.png
```

To regenerate only selected variants:

```bash
python test/tools/heatmaps_from_json.py \
  --variant r4bss05 \
  --variant r4bss05highdm
```

## Selecting Algorithms For The Heatmap

`test/tools/heatmaps_from_json.py` reads already stored result JSON files from:

```text
test/data/algorithm_cases/<variant>/<seed>/<mode>/<algorithm>.json
```

The y-axis rows are algorithm/result names. By default, the script uses every result row available for each mode:

- `full_cnp` currently has the legacy NJ-like algorithms plus any experimental algorithm generated for that mode.
- `biopsy_guided_top` has the same algorithm rows when generated for that mode, and may also have biopsy-preset rows.

To list available rows for one variant and mode:

```bash
PYTHONPATH=test python - <<'PY'
from json_case_results import algorithms_for_variant

for mode in ["full_cnp", "biopsy_guided_top"]:
    print(mode)
    for index, name in enumerate(algorithms_for_variant("test/data/algorithm_cases", "r4bss05", mode)):
        print(f"  {index}: {name}")
PY
```

To generate a heatmap with a specific subset, pass `--algorithm-name` once per row:

```bash
MPLCONFIGDIR=/tmp/ctbf_mpl_config \
XDG_CACHE_HOME=/tmp/ctbf_xdg_cache \
python test/tools/heatmaps_from_json.py \
  --algorithm-name neighbor_joining_baseline \
  --algorithm-name neighbor_joining_hybrid_opt \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
  --algorithm-name new_alg \
  --output-file test/heatmaps_side_by_side_new_alg.png
```

The script keeps only selected names that exist in each mode. For example, if a row exists in `biopsy_guided_top` but not in `full_cnp`, it is omitted from the `full_cnp` panels.

Common named subsets live in `reconstructor_algorithm_config.py` as `COMPARISON_GROUPS`. Current useful groups are:

```text
publication
recommended_core
biopsy_preset_comparison
new_alg_comparison
```

Use a group instead of spelling every algorithm:

```bash
MPLCONFIGDIR=/tmp/ctbf_mpl_config \
XDG_CACHE_HOME=/tmp/ctbf_xdg_cache \
python test/tools/heatmaps_from_json.py \
  --algorithm-group new_alg_comparison \
  --output-file test/heatmaps_side_by_side_new_alg.png
```

`new_alg_comparison` currently expands to:

```text
neighbor_joining_baseline
neighbor_joining_hybrid_opt
neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony
new_alg
```

You can combine groups and explicit names:

```bash
python test/tools/heatmaps_from_json.py \
  --algorithm-group recommended_core \
  --algorithm-name new_alg \
  --output-file /tmp/recommended_plus_new_alg.png
```

The heatmap highlights rows listed in `HIGHLIGHTED_HEATMAP_ALGORITHMS` in `reconstructor_algorithm_config.py`. At the moment this is:

```text
new_alg
```

## Creating A New Algorithm From Blocks

The intended pattern is:

1. Choose a pair selector from `reconstructor_pair_selection.py`.
2. Choose or add an ancestor selector in `reconstructor_ancestor_selection.py`.
3. Choose a merge strategy from `reconstructor_merge.py`.
4. Choose a distance update from `reconstructor_distance_update.py`.
5. Wire those blocks together in `reconstructor_algorithms.py`.
6. Register the algorithm in `reconstructor_algorithm_specs.py`.
7. Add display/explanation metadata and comparison groups in `reconstructor_algorithm_config.py`.
8. Generate JSON results from frozen cases.
9. Generate a heatmap using `--algorithm-name` or `--algorithm-group`.

`new_alg` is the example of this pattern.

Its ancestor selector is in `reconstructor_ancestor_selection.py`:

```python
def plausible_then_centrality_parent_selector(state, pair):
    i = pair.i
    j = pair.j
    a = state.node_list[i]
    b = state.node_list[j]

    can_a_parent_b = is_biologically_plausible_ancestor(a, b)
    can_b_parent_a = is_biologically_plausible_ancestor(b, a)

    if can_a_parent_b and not can_b_parent_a:
        return Orientation(i, j)

    if can_b_parent_a and not can_a_parent_b:
        return Orientation(j, i)

    centrality = _pair_centrality_metric(state, pair)
    parent_idx, child_idx = _choose_parent_by_larger_metric(centrality, i, j, state.rng)
    return Orientation(parent_idx, child_idx)
```

This means:

- if only `a -> b` is biologically plausible, use `a` as parent;
- if only `b -> a` is biologically plausible, use `b` as parent;
- if plausibility does not decide, use centrality.

The algorithm itself is in `reconstructor_algorithms.py`:

```python
def new_alg(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
):
    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=plausible_then_centrality_parent_selector,
    )
```

`_run_anticentral_v3_algorithm(...)` supplies the remaining shared blocks:

```text
merge_strategy = anticentral_weighted_copy_parent_node
distance_update = anticentral_v3_distance_update
configure_state = configure_anticentral_v3_state
```

Register it in `reconstructor_algorithm_specs.py`:

```python
EXPERIMENTAL_ALGORITHM_SPECS = [
    ReconstructionAlgorithmSpec("new_alg", _constant_algorithm(new_alg), legacy=False),
]
```

Add metadata in `reconstructor_algorithm_config.py`:

```python
AlgorithmDisplayConfig(
    name="new_alg",
    label="new_alg",
    summary="Experimental anticentral reconstruction: anticentral adaptive v3 pair selection, then plausible ancestor orientation, then centrality fallback.",
    procedure=AlgorithmProcedureConfig(
        pair_selection="anticentral adaptive v3 pair selection",
        ancestor_selection="plausible ancestor selector, then larger-centrality fallback",
        distance_update="anticentral_v3_distance_update",
        merge_strategy="anticentral weighted-copy parent node",
        plausibility="ancestor plausibility first; centrality if plausibility does not decide",
    ),
    groups=("experimental", "new_alg_comparison"),
    highlight_in_heatmap=True,
)
```

Then generate frozen JSON results for the new algorithm without rerunning simulations or `cnp2cnp`:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --existing-seeds \
  --results-only \
  --algorithm-name new_alg \
  --overwrite
```

Finally generate the focused comparison heatmap:

```bash
MPLCONFIGDIR=/tmp/ctbf_mpl_config \
XDG_CACHE_HOME=/tmp/ctbf_xdg_cache \
python test/tools/heatmaps_from_json.py \
  --algorithm-group new_alg_comparison \
  --output-file test/heatmaps_side_by_side_new_alg.png
```

## Adding `brand_new_algorithm`

Use this sequence.

### 1. Build the algorithm

If `brand_new_algorithm` is a real reconstruction algorithm, add it in `reconstructor_algorithms.py`.

You choose blocks:

- pair selector from `reconstructor_pair_selection.py`
- ancestor selector from `reconstructor_ancestor_selection.py`
- merge strategy from `reconstructor_merge.py`
- distance update from `reconstructor_distance_update.py`
- optional state setup from `reconstructor_anticentral.py` or elsewhere

Typical shape:

```python
def brand_new_algorithm(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=YOUR_PAIR_SELECTOR,
        ancestor_selector=YOUR_ANCESTOR_SELECTOR,
        merge_strategy=YOUR_MERGE_STRATEGY,
        distance_update=YOUR_DISTANCE_UPDATE,
        configure_state=YOUR_CONFIGURE_STATE,
    )
```

If it matches the anticentral-v3 family, usually use `_run_anticentral_v3_algorithm(...)` instead.

### 2. Register it

Add it to `reconstructor_algorithm_specs.py`.

If it is experimental:

```python
EXPERIMENTAL_ALGORITHM_SPECS = [
    ReconstructionAlgorithmSpec("brand_new_algorithm", _constant_algorithm(brand_new_algorithm), legacy=False),
]
```

After that it becomes available through the registry.

### 3. Describe it

Add metadata in `reconstructor_algorithm_config.py`.

That is where you document:

- label
- summary
- pair selection
- ancestor selection
- distance update
- merge strategy
- plausibility behavior
- comparison groups
- heatmap highlighting

Example:

```python
AlgorithmDisplayConfig(
    name="brand_new_algorithm",
    label="brand_new_algorithm",
    summary="Short explanation.",
    procedure=AlgorithmProcedureConfig(
        pair_selection="...",
        ancestor_selection="...",
        distance_update="...",
        merge_strategy="...",
        plausibility="...",
        biopsy_guided_preset=None,
    ),
    groups=("experimental", "my_comparison_group"),
    highlight_in_heatmap=True,
)
```

If you want it in a standard comparison set:

```python
COMPARISON_GROUPS["my_comparison_group"] = (
    "neighbor_joining_baseline",
    "neighbor_joining_hybrid_opt",
    "brand_new_algorithm",
)
```

### 4. If you want biopsy-guided inference

There are two different cases.

#### Case A: same reconstruction algorithm, but run inside biopsy-guided top-level inference

Then you do not create a biopsy preset. You just freeze results for `brand_new_algorithm`, and it will produce both:

- `full_cnp/brand_new_algorithm.json`
- `biopsy_guided_top/brand_new_algorithm.json`

using:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --existing-seeds \
  --results-only \
  --algorithm-name brand_new_algorithm \
  --overwrite
```

#### Case B: you want a special biopsy-guided heuristic config

For example:

- special parent choice between biopsy levels
- tie breaker
- binarized subgroup inference
- subtree pair selector / ancestor selector

Then define a preset in `reconstructor_biopsy_presets.py`, backed by blocks from:

- `reconstructor_biopsy_blocks.py`
- `reconstructor_pair_selection.py`
- `reconstructor_ancestor_selection.py`

That gives you a benchmark row like `biopsy_preset_xxx`, not a registry algorithm.

### 5. Test locally before freezing

Run focused tests first:

```bash
python -m pytest -q test/test_reconstructor_blocks.py test/test_reconstructor_biopsy_blocks.py test/test_reconstructor_algorithm_variants.py test/test_reconstructor_refactor_surfaces.py
```

If your algorithm is registry-based, also check it appears:

```bash
python -m pytest -q test/test_reconstructor_algorithm_variants.py::test_combined_algorithm_registry_keeps_legacy_prefix_and_unique_names
```

### 6. Freeze benchmark outputs

For a real algorithm:

```bash
python test/tools/freeze_algorithm_variant_cases.py \
  --existing-seeds \
  --results-only \
  --algorithm-name brand_new_algorithm \
  --overwrite
```

That writes frozen JSON results from existing inputs only. No new simulation, no new `cnp2cnp`.

### 7. Validate only the new algorithm

Do not use the canonical checker yet.

Use the experimental checker:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name brand_new_algorithm
```

If you also want a heatmap:

```bash
python test/tools/run_experimental_json_checks.py \
  --algorithm-name neighbor_joining_baseline \
  --algorithm-name neighbor_joining_hybrid_opt \
  --algorithm-name neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony \
  --algorithm-name brand_new_algorithm \
  --check metrics \
  --check determinism \
  --check heatmap \
  --heatmap-output-file test/heatmaps_side_by_side_brand_new_algorithm.png
```

### 8. Only after you accept it, make it canonical

If you decide it should join the canonical benchmark set:

- move it into the canonical list logic in `reconstructor_algorithm_specs.py` if appropriate
- or include it in `PUBLICATION_HEATMAP_ALGORITHM_NAMES` flow
- regenerate canonical frozen outputs as needed
- then `python tools/run_algorithm_case_json_checks.py` will include it automatically

### Rule of thumb

- `reconstructor_algorithms.py`: executable composition of blocks
- `reconstructor_algorithm_specs.py`: registry membership
- `reconstructor_algorithm_config.py`: explanation, labels, comparison groups, highlighting
- `reconstructor_biopsy_presets.py`: biopsy-guided heuristic presets
- `freeze_algorithm_variant_cases.py`: freeze results for registry algorithms
- `run_experimental_json_checks.py`: validate non-canonical additions
- `run_algorithm_case_json_checks.py`: canonical set only

The test suite also validates these JSON workflows on a representative fixture:

```bash
pytest -q test/test_algorithm_case_json_workflows.py
```

Those tests cover:

- loading `input.json.true_tree` and stored reconstructed trees to recompute evaluator metrics,
- recomputing the frozen `cnp2cnp` matrix from biopsy cells,
- recomputing the true-tree distance matrix,
- rerunning reconstruction from frozen biopsies and matrices,
- building pairwise rankings directly from JSON outputs.

## Testing Evaluator Functions From JSON Trees

To test evaluator behavior without rerunning simulation or reconstruction, load:

```text
test/data/algorithm_cases/<variant>/<seed>/input.json
test/data/algorithm_cases/<variant>/<seed>/full_cnp/<algorithm>.json
```

Use:

- `input.json["true_tree"]` as the true tree,
- `full_cnp/<algorithm>.json["reconstructed_tree"]` as the reconstructed tree,
- `full_cnp/<algorithm>.json["metrics"]` as the expected evaluator output.

The existing test that does this is:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_json_reconstructed_tree_metrics_match_stored_values
```

From inside the `test/` directory, omit the leading `test/`:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_json_reconstructed_tree_metrics_match_stored_values
```

To run evaluator replay for every frozen variant, seed, mode, and stored algorithm:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_all_json_reconstructed_tree_metrics_match_stored_values
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_all_json_reconstructed_tree_metrics_match_stored_values
```

That test:

1. loads the stored true tree from `input.json.true_tree`,
2. loads the stored reconstructed tree from `full_cnp/{algorithm}.json.reconstructed_tree` and `biopsy_guided_top/{algorithm}.json.reconstructed_tree`,
3. rebuilds NetworkX trees from the JSON node-link format,
4. reruns `evaluate_4(true_tree, reconstructed_tree, restrict_labels=...)`,
5. reruns `grf_tree(true_tree, reconstructed_tree)`,
6. compares the recomputed values to the metrics stored in the algorithm result JSON.

This is useful when changing evaluator code: if reconstruction has not changed, evaluator regressions show up immediately against frozen tree pairs.

## Testing Reconstruction Determinism From JSON

To test reconstruction determinism without rerunning simulation or `cnp2cnp`, load:

```text
test/data/algorithm_cases/<variant>/<seed>/input.json
test/data/algorithm_cases/<variant>/<seed>/<mode>/<algorithm>.json
```

Use:

- `input.json["biopsies"]` as the frozen biopsy cells,
- `input.json["distance_matrices"]["cnp2cnp"]` as the frozen reconstruction matrix,
- `<mode>/<algorithm>.json["reconstructed_tree"]` and `["newick"]` as the expected output.

For the representative fixture only:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_json_reconstruction_is_deterministic_against_stored_tree
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_json_reconstruction_is_deterministic_against_stored_tree
```

To run reconstruction determinism for every frozen variant, seed, mode, and stored algorithm:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_all_json_reconstruction_is_deterministic_against_stored_tree
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_all_json_reconstruction_is_deterministic_against_stored_tree
```

That test:

1. loads frozen biopsy cells from `input.json.biopsies`,
2. loads the frozen `cnp2cnp` matrix from `input.json.distance_matrices.cnp2cnp`,
3. reruns the selected reconstruction algorithm in `full_cnp` or `biopsy_guided_top` mode,
4. compares the regenerated root and Newick tree to the stored algorithm result JSON,
5. checks that the stored reconstructed tree serializes to the same Newick string.

This is useful when changing reconstruction code: any behavioral change in a stored algorithm shows up as a deterministic tree mismatch.

## Testing Frozen Distance Matrices From JSON

The frozen `input.json` file stores two distance matrices:

```text
input.json["distance_matrices"]["cnp2cnp"]
input.json["distance_matrices"]["true_tree"]
```

### cnp2cnp Matrix Replay

To test `cnp2cnp` matrix generation on the representative fixture:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_json_biopsies_recompute_cnp2cnp_matrix
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_json_biopsies_recompute_cnp2cnp_matrix
```

To run it for every frozen variant and seed:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_all_json_biopsies_recompute_cnp2cnp_matrix
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_all_json_biopsies_recompute_cnp2cnp_matrix
```

That test:

1. loads biopsy cells from `input.json.biopsies`,
2. runs `cnp2cnp` matrix generation,
3. compares the generated IDs and matrix to `input.json.distance_matrices.cnp2cnp`.

### True-Tree Distance Matrix Replay

To test true-tree distance matrix generation on the representative fixture:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_json_true_tree_recomputes_true_tree_distance_matrix
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_json_true_tree_recomputes_true_tree_distance_matrix
```

To run it for every frozen variant and seed:

```bash
pytest -q test/test_algorithm_case_json_workflows.py::test_all_json_true_tree_recomputes_true_tree_distance_matrix
```

From inside the `test/` directory:

```bash
pytest -q test_algorithm_case_json_workflows.py::test_all_json_true_tree_recomputes_true_tree_distance_matrix
```

That test:

1. loads the true tree from `input.json.true_tree`,
2. reads biopsy-cell IDs from `input.json.distance_matrices.true_tree.ids`,
3. recomputes distances on the stored true tree,
4. compares the generated IDs and matrix to `input.json.distance_matrices.true_tree`.

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
