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

## CTBF v5 algorithm-development bank

The v5 development workflow is separate from the later paper experiment. Its
active v2 bank creates 100 independent H34 truths and uses each truth's H14 and
H24 prefixes. It therefore provides 300 paired H14/H24/H34 condition outcomes
but only 100 independent truth blocks. The completed 50-block v1 bank remains
untouched historical development provenance and must not be mixed with v2
scores. The active bank stores simulation, observation, and production
cnp2cnp inputs once. Later algorithm variants rerun only reconstruction and
evaluation.

Production bank simulation and distance stages now use fresh spawned task
processes, and every development case-arm reconstruction/evaluation runs in a
fresh spawned process. Absolute process-tree RSS therefore includes the
current input and algorithm but cannot inherit allocator high-water marks or
truth caches from an earlier record. The development reporter exposes whether
every merged run has this qualification; resource comparisons from an older
long-lived execution boundary are labeled historical context, while their
successful accuracy records remain development provenance.

Generate the one-time bank (this is the large owner-run input command):

```bash
python -m algorithm_evaluation.v5_algorithm_development_bank \
  --base-config simulator_examples/default.json \
  --base-seed 20260813 \
  --output-root algorithm_evaluation/results/v5_development/bank_v2 \
  --progress
```

Run the initial 32-arm roster on those exact inputs:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --run-id initial-32-v2 \
  --arms all \
  --progress
```

If that exact run is interrupted by a technical runner failure, resume its
validated stored record prefix without replacing the result root:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --run-id initial-32-v2 \
  --arms all \
  --progress \
  --resume
```

Resume refuses a changed bank, run id, arm roster/order, resource declaration,
schema, or non-prefix record inventory. It records the interrupted status and
preserved record count before continuing only the missing entries.

Create the descriptive fixed-incumbent leaderboards and full within-family
pairwise tables:

```bash
python -m algorithm_evaluation.v5_algorithm_development_report \
  --result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --output-root algorithm_evaluation/results/v5_development/reports/initial_32_v2
```

Every new algorithm receives a stable id in the explicit v5 roster and is run
into a new output directory with `--arms <id>`. Regenerate the report into
another new directory, repeating `--result-root` for the initial run and each
added algorithm run. The reporter performs no significance test and declares
no automatic winner; H14/H24/H34 wins are descriptive, while rare-loss
summaries use the 100 paired block effects.

The first approved partial-top development extension fixes default biopsy
attachment at radius 2. The existing `biopsy_guided_classical_r2` result is its
classical-Q control; run only the three new top variants with:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/partial_top_r2_v2 \
  --run-id partial-top-r2-v2 \
  --arms \
    biopsy_guided_top_rooted_labeled_q_r2 \
    biopsy_guided_top_anticentral_binary_r2 \
    biopsy_guided_top_anticentral_parent_reuse_r2 \
  --progress
```

Then combine that non-overwriting run with the initial result:

```bash
python -m algorithm_evaluation.v5_algorithm_development_report \
  --result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_r2_v2 \
  --output-root algorithm_evaluation/results/v5_development/reports/partial_top_r2_v2
```

The partial-output projection clears the label and working genome only from
nodes created by the final top solver. It never deletes those nodes and never
changes biopsy observations or missing-parent copy-up occurrences. The merged
report adds a `Partial top-layer screen` section to `report.md` and writes the
same direct comparisons to `partial_top_layer_vs_r2_classical.csv`.

The expanded screen favors projected binary anticentral top reconstruction.
Complete the approved radius-by-top check by running the same top solver with
only the biopsy-guided bottom radius changed to 4:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/partial_top_binary_r4_v2 \
  --run-id partial-top-binary-r4-v2 \
  --arms biopsy_guided_top_anticentral_binary_r4 \
  --progress
```

After that run, create the final development report from all three result
roots:

```bash
python -m algorithm_evaluation.v5_algorithm_development_report \
  --result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_r2_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_binary_r4_v2 \
  --output-root algorithm_evaluation/results/v5_development/reports/partial_top_r2_r4_v3
```

The report adds top effects within each radius, radius effects within each top
method, and the paired difference-in-differences. Positive interaction means
that the binary-top advantage is larger at r2 than at r4. These are descriptive
development contrasts over 300 conditions and 100 independent truth blocks.

The approved seven-policy bottom-layer screen keeps radius 2 and the projected
binary anticentral plausible-parsimony top fixed. Its default policy already
exists as `biopsy_guided_top_anticentral_binary_r2`; run only the six missing
rows:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/partial_bottom_r2_v2 \
  --run-id partial-bottom-r2-v2 \
  --arms \
    biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie \
    biopsy_guided_top_anticentral_binary_r2_bottom_binarized \
    biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie_binarized \
    biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie \
    biopsy_guided_top_anticentral_binary_r2_bottom_central_tie \
    biopsy_guided_top_anticentral_binary_r2_bottom_diploid_parsimony_tie \
  --progress
```

A complete run writes 1,800 arm records. Then merge it with the three existing
result roots without rerunning simulations or the default control:

```bash
python -m algorithm_evaluation.v5_algorithm_development_report \
  --result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_r2_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_binary_r4_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_bottom_r2_v2 \
  --output-root algorithm_evaluation/results/v5_development/reports/partial_bottom_r2_v4
```

The merged report writes `partial_bottom_layer_vs_default_r2.csv` for the six
candidate-minus-default comparisons and `partial_bottom_mechanism_summary.csv`
for tie, deferral, copy-up, and shared-parent diagnostics. The historical
`anticentral_binarized` preset is not substituted for the clean combined row:
it also changes pair selection inside each local child subtree.

The seven-policy screen strongly favors exact-tie deferral under the binary
top. Complete the approved attribution factorial with only the missing
deferred-bottom/classical-top row:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/partial_bottom_deferred_classical_r2_v2 \
  --run-id partial-bottom-deferred-classical-r2-v2 \
  --arms biopsy_guided_classical_r2_bottom_deferred_tie \
  --progress
```

A complete run writes 300 arm records. Merge all five non-overlapping result
roots into the factorial summary:

```bash
python -m algorithm_evaluation.v5_algorithm_development_report \
  --result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_r2_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_top_binary_r4_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_bottom_r2_v2 \
  --result-root algorithm_evaluation/results/v5_development/runs/partial_bottom_deferred_classical_r2_v2 \
  --output-root algorithm_evaluation/results/v5_development/reports/partial_bottom_top_factorial_v5
```

The new report section and CSVs show the binary-minus-classical top effect
under each bottom policy, the deferred-minus-default bottom effect under each
top method, and their paired difference-in-differences. This is one bounded
attribution check, not a new broad candidate screen.

The completed fully labeled biopsy-guided extension adds ten unprojected arms;
its ordered incumbent is
`biopsy_guided_full_anticentral_binary_r2_bottom_deferred_tie`. After combining
that run with the initial pooled result, run the approved read-only attachment
diagnostic with:

```bash
python -m algorithm_evaluation.v5_algorithm_development_full_attachment_audit \
  --ordered-result-root algorithm_evaluation/results/v5_development/full_biopsy_guided_v1 \
  --pooled-result-root algorithm_evaluation/results/v5_development/runs/initial_32_v2 \
  --output-root algorithm_evaluation/results/v5_development/diagnostics/full_attachment_audit_v1
```

The command reads only the two target arms and their already stored counters.
It writes a compact JSON audit and Markdown report without rerunning simulation,
cnp2cnp, reconstruction, or evaluation. Absolute non-same-state hard-
attachment count is reported alongside its per-child-decision fraction so
tree-size growth is not mistaken for a changed attachment propensity. The
audit is descriptive development evidence and never promotes an algorithm.

The owner-approved missing fully labeled factorial cell combines the existing
deferred-tie bottom with the existing rooted-labeled Q top. Run only that arm
into a new result root:

```bash
python -m algorithm_evaluation.v5_algorithm_development_run \
  --bank-root algorithm_evaluation/results/v5_development/bank_v2 \
  --output-root algorithm_evaluation/results/v5_development/runs/full_rooted_q_deferred_r2_v2 \
  --run-id full-rooted-q-deferred-r2-v2 \
  --arms biopsy_guided_full_rooted_labeled_q_r2_bottom_deferred_tie \
  --progress
```

This writes 300 new records and does not change the completed ten-arm
`biopsy_guided_full` alias. Report schema v8 recognizes the resulting complete
`{default,deferred} x {rooted-labeled Q,binary anticentral}` full-output design
and writes its top effects, bottom effects, and difference-in-differences.

## CTBF v5 fully labeled shortlist robustness workflow

The final adaptive-development qualification is separate from the immutable
v2 bank above. It fixes four existing methods, denoted A--D:

- A: `biopsy_guided_full_anticentral_binary_r2_bottom_deferred_tie`;
- B: `biopsy_guided_full_rooted_labeled_q_r2_bottom_deferred_tie`;
- C: `biopsy_guided_full_anticentral_binary_r4`; and
- D: `neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony`.

The full bank contains 100 fresh H38 truths, paired H14/H24/H34/H38 prefixes,
and spread, late, and prospectively random three-biopsy conditions at every
height. It writes 1,200 declared condition inputs and, when every condition is
available, 4,800 arm records. The truth block remains the only independent
unit. The report never pools the 1,200 conditions into one winner and never
combines AD-F1 with GRF.

Before the full bank, run the disjoint five-block H38-late resource preflight.
Its input generator is resumable with the identical command plus `--resume`:

```bash
python -m algorithm_evaluation.v5_shortlist_robustness_bank \
  --base-config simulator_examples/default.json \
  --base-seed 20260816 \
  --block-count 5 \
  --heights 38 \
  --placement-policies late \
  --technical-preflight \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/preflight_h38_late_v1/bank \
  --progress
```

After reviewing that bank's availability and distance resources, run A--D and
write its compact report:

```bash
python -m algorithm_evaluation.v5_shortlist_robustness_run \
  --bank-root algorithm_evaluation/results/v5_development/shortlist_robustness/preflight_h38_late_v1/bank \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/preflight_h38_late_v1/run \
  --run-id shortlist-h38-late-preflight-v1 \
  --expected-block-count 5 \
  --progress

python -m algorithm_evaluation.v5_shortlist_robustness_report \
  --result-root algorithm_evaluation/results/v5_development/shortlist_robustness/preflight_h38_late_v1/run \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/preflight_h38_late_v1/report \
  --expected-block-count 5
```

Those `preflight_h38_late_v1` paths describe the already completed historical
preflight and must not be overwritten. Its scientific inputs remain useful,
but its old long-lived-process runtime/RSS records are not fresh-process
resource evidence.

Only after the preflight passes, generate the fresh full bank. This command is
also resumable by adding `--resume` without changing any scientific or
resource-limit argument:

```bash
python -m algorithm_evaluation.v5_shortlist_robustness_bank \
  --base-config simulator_examples/default.json \
  --base-seed 20260817 \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/bank_v1 \
  --distance-workers 6 \
  --progress
```

`--distance-workers` bounds concurrent condition-level worker processes. Each
fresh worker executes exactly one condition and still runs the unchanged
forward and reverse cnp2cnp matrix calls sequentially. Thus six workers mean
at most six active cnp2cnp processes rather than six threads inside one
distance calculation. Larger unique-profile
matrices are submitted first to avoid a serial H38 tail; results are restored
to declared condition order and checkpointed only after a complete
12-condition truth block. The manifest records requested/effective workers and
the submission policy for every new block; therefore a serial completed prefix
can be resumed with six workers without changing any stored input or distance.
`KeyboardInterrupt` and `SystemExit` abort the incomplete block and are never
recorded as scientific condition failures.

Dense truth diagnostics use exact arborescence preorder/postorder counts rather
than a generic NetworkX LCA call per occurrence pair. Candidate and four-cycle
summaries use aligned array operations, reuse the already computed radius-4
graph, and do not repeat parent-pair-by-child Python scans. With `--progress`,
simulation, preparation, distance, and post-distance summarization are printed
as separate phases so a serial prerequisite cannot be mistaken for an inactive
worker pool.

The historical `runs/abcd_v1` result is an implementation diagnostic only: a
long-lived runner caused two large D records to contaminate the absolute RSS
of later records, producing 1,902 false cascade failures. Do not report or
resume it. The completed `bank_v1` inputs and distances remain reusable and
are not regenerated; its older generation-time RSS metadata is explicitly
labeled unqualified historical context in the new report. First replay the two
trigger regions under the corrected boundary:

```bash
python -m algorithm_evaluation.v5_shortlist_resource_isolation_probe \
  --bank-root algorithm_evaluation/results/v5_development/shortlist_robustness/bank_v1 \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/isolation_probe_v1 \
  --expected-block-count 100 \
  --progress
```

The probe fixes the requested order to H38 late followed by random for blocks
18 and 67, runs all A--D records in fresh workers, and is explicitly not
accuracy evidence. It completed 14/16 records: pooled D alone exceeded 4 GiB
on both H38-late cases (`5.76` and `4.62` GB), while both following H38-random
pooled-D records and all ordered records succeeded. This validates isolation
and exposes a separate implementation defect: an unbounded cache retained the
two int64 upper-triangle index arrays for every decreasing matrix size, making
the pooled reconstruction retain O(N^3) memory. The semantics-preserving fix
keeps only the current size. Two four-arm fresh replays, including one with
299 pooled states, match the corresponding `abcd_v1` scientific payloads
exactly.

The cache-fix trigger probe is retained as the reproducible resource gate:

```bash
python -m algorithm_evaluation.v5_shortlist_resource_isolation_probe \
  --bank-root algorithm_evaluation/results/v5_development/shortlist_robustness/bank_v1 \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/isolation_probe_cache_fix_v1 \
  --expected-block-count 100 \
  --progress
```

It completed all `16/16` records. Pooled D on the formerly failing 1,336- and
1,168-state H38-late cases peaked at `743` and `656` MiB, respectively. All 14
scientific payloads with successful pre-fix probe references remained exactly
unchanged after excluding resource measurements. The unchanged 4-GiB gate is
therefore satisfied. Run the fixed shortlist once into a fresh non-overwriting
result and create the block-aware report:

```bash
python -m algorithm_evaluation.v5_shortlist_robustness_run \
  --bank-root algorithm_evaluation/results/v5_development/shortlist_robustness/bank_v1 \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/runs/abcd_v2 \
  --run-id shortlist-abcd-v2 \
  --expected-block-count 100 \
  --progress

python -m algorithm_evaluation.v5_shortlist_robustness_report \
  --result-root algorithm_evaluation/results/v5_development/shortlist_robustness/runs/abcd_v2 \
  --output-root algorithm_evaluation/results/v5_development/shortlist_robustness/reports/abcd_v2 \
  --expected-block-count 100
```

Both bank and reconstruction runs preserve exact stored prefixes on `--resume`
and reject changed seeds, contracts, arm order, resources, or record order.
The bank alone permits a changed distance-worker count at a block boundary and
records the mixed serial/parallel execution history explicitly.
Unavailable simulation, biopsy, distance, reconstruction, and evaluation
outcomes remain typed and are never replaced with another seed.
The shortlist reporter requires the recorded fresh-process case-arm contract;
it refuses `abcd_v1` and any other unqualified long-lived run.

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
result requires owner review before any full dense reconstruction probe. The
owner-reviewed compact-solver artifact is
`/tmp/ctbf-v5-dense-reconstruction-preflight-compact-edmonds.json`, SHA-256
`431fddca7d73d9da0ead156d1dd4cc1a1a2085d5b0a5b0bbf3a81fed8b7144e9`.
All six arms passed the unchanged 4 GiB stage bound. A fresh preflight run is
technical evidence only and must be reviewed and registered before it can
replace that authorization artifact. Run a fresh preflight with:

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

## CTBF v5 50%-sampling reconstruction probe

`simulator_dense_reconstruction_probe.py` implements the authorized bounded
follow-up. It regenerates the same 12 paired truth blocks at H14/H24/H34 and
uses only `min(N,max(6,ceil(0.5*N)))` representative-state sampling at the
Rule-Y generations. Each case computes production minimum-bidirectional
cnp2cnp once and supplies the same distance matrix to all six established
arms.

The primary comparison is paired within arm: each 50% result is compared with
that arm's completed capped-six result for the same seed and height. The probe
does not rank algorithms whose output problems differ, does not test 5%, 10%,
or 25% reconstruction conditions, and does not perform significance tests.
It records compact ancestry, reconstruction, declared-metric, runtime, and
typed-failure summaries without profiles, matrices, trees, or node identities.
This remains a non-paper design diagnostic; it cannot select simulator
parameters or freeze the paper height/sampling design by itself.

The production run checksum-gates both completed comparison artifacts and the
owner-reviewed passing preflight, and retains the 4 GiB per-stage limit. Run:

```bash
./zzzz_verify_ctbf.sh simulator-dense-reconstruction-probe \
  --base-config simulator_examples/default.json \
  --fraction-truth-report \
    /tmp/ctbf-v5-sampling-fraction-truth-probe.json \
  --sparse-reconstruction-report \
    /tmp/ctbf-v5-reconstruction-intuition-probe.json \
  --dense-preflight-report \
    /tmp/ctbf-v5-dense-reconstruction-preflight-compact-edmonds.json \
  --output /tmp/ctbf-v5-fraction50-reconstruction-probe.json \
  --progress
```

The completed non-paper artifact has SHA-256
`cee0803d4d818ea372a56992a4883e5d6064dc1568f93e68f8f1eaccf2c1966a`.
All 36 cases and 216 arms passed. The result supports 50%/lower-bound-six over
capped-six for this reconstruction diagnostic. The owner approved H24 as the
standard regime, H14 as the easy/low-depth control, and H34 as the stress
sensitivity. Other fractions and capped-six do not advance as paper
reconstruction conditions.

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
