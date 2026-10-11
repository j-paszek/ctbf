# Independent trimmed-true-tree evaluation

CTBF v5. Owner-approved complementary target, 2026-10-04.
The mathematical definition and two examples are in Experimental Evaluation
iteration 14 and Supplementary Methods iteration 18.

## Scientific boundary

Keep the published/frozen full-truth GRF values unchanged. Read the already
stored reconstructed trees and compute a separate `grf_trim` against the
state-restricted true tree. Never modify or trim a reconstruction. This is
not a sampled-occurrence-only projection: retain every true occurrence of a
state in the union of biopsy labels, including unsampled and repeated-state
occurrences. Keep necessary unlabeled branching vertices, prune unlabeled
terminal branches, suppress unlabeled nonroot unary nodes and promote an
unlabeled unary root's child. Labeled unary nodes always survive.

The new evaluator checks the exact retained AD relation set once per condition
and all AD-F1 fields against each frozen result. The AD-F1 target is unchanged.
Metric label normalization and the native GRF kernel remain unchanged. The
new versioned envelope declares the different truth target, rather than
silently altering `ctbf-tree-evaluation-result-v1` or a registered report.

No new simulation, input distance matrix, reconstruction, external inference,
posterior summary selection, figure or registered primary p-value is produced.
M4 and the remaining 90 NestedBD histories are later work, not part of this run.

## Owner command

Run from the CTBF repository root:

```bash
caffeinate -is bash ./zzzz_verify_ctbf.sh v5-trimmed-truth run \
  --workers 4 \
  --progress \
  --resume
```

The command defaults to the completed sources:

- `algorithm_evaluation/results/v5_confirmation/confirmation_v1/bank`
- `algorithm_evaluation/results/v5_confirmation/confirmation_v1/internal`
- `algorithm_evaluation/results/v5_confirmation/confirmation_v1/medicc2`
- `algorithm_evaluation/results/v5_nestedbd_long/benchmark_v1_report_v2`

The last source also references eight sealed summaries in the original
`benchmark_v1` directory. Both directories remain read-only. No BEAST runtime
or ctbfbio installation is needed for this reevaluation.

Expected with the current sources: **800 conditions, 5,940 records**:
5,600 internal records, 300 MEDICC2 records, and 40 NestedBD seed records from
20 inputs. The duplicated M0/M2 rows in the NestedBD report are not counted
again. If all reevaluations pass, the final line is:

```text
complete: 5940 trimmed-truth records; 0 audit failures; 0 source failures; full scores preserved; no reconstruction
```

Report only this line and any focused failure message. The agent can then
review the compact summaries. Do not replace the prior results.

## Outputs and interpretation

New root:
`algorithm_evaluation/results/v5_confirmation/trimmed_truth_v1/`

- `report.md`: full/trimmed mean scores, paired method effects and sampling
  contrasts, with explicit complete/planned denominators.
- `summary.json`: the same summaries, history-level contrasts, condition-level
  and block-level W/T/L, source-failure counts and scope metadata.
- `scores.csv`: one row per original method/seed result, retaining original
  full GRF, new trimmed GRF, AD-F1 and explicit statuses.
- `case_scores.csv`: paired input-level scores. NestedBD's two seed scores
  are averaged only when both are successful.
- `cases/*.json`: committed per-condition audit, metric-only trimmed tree,
  original/trimmed dimensions, input/record references, AD invariance checks
  and native evaluation on the trimmed target.
- `analysis_contract.json`: fixed selection, consumed-source digests, code,
  metric semantics, runtime, resource limits and analysis settings.
- `_COMPLETE.json`: hashes of all final audit artifacts.

Both columns of any full-versus-trimmed comparison use the same successful
records. Paired method contrasts use only matched conditions, retaining
failed outcomes in their declared denominators. Original scores are available
in `scores.csv` even if the new evaluation fails. Do not interpret an audit
resource/validation failure as a new reconstruction failure.

Bootstrap intervals use 100,000 deterministic draws of independent histories.
For a multi-height contrast, first average the paired height effects within
each complete history. Report condition-level W/T/L separately. NestedBD seeds
and nested heights are not extra independent histories. No automatic method
promotion, significance declaration or primary-test replacement is performed.
Convergence of NestedBD remains unestablished, independent of this evaluation.

For sampling contrasts the trimmed target changes with the condition's
observed-state union. These are not fixed-target recovery comparisons. Keep
this qualification when interpreting biopsy-number or placement effects.
The existing fixed-label AD-F1 analysis is untouched.

## Resume, resources and integrity

`--resume` works both on a new root and after an interruption. Each finished
condition is committed atomically with a digest and is reused unchanged.
An interrupted, uncommitted condition is evaluated again. Terminal source
failures and audit failures are retained, not automatically retried.
The output lock prevents concurrent controllers from writing the same root.

Four workers are the default, with **one fresh process per condition**. Each
worker loads that condition's truth once and evaluates its stored methods.
The default timeout is 300 seconds for the complete condition evaluation, not
300 seconds per algorithm. The outer worker deadline is 390 seconds. The
post-stage process-tree RSS guard is 4 GiB per worker, not an OS-enforced
allocation limit. Four workers can therefore approach 16 GiB plus controller
and system overhead. Reduce `--workers` if necessary. Worker count can change
on resume. Resource limits, sources, code and scientific settings cannot.

The per-case timing/memory fields describe this evaluation audit only. They
must not replace original whole-pipeline method costs or support comparative
speed claims.

Every consumed file is bound to its source inventory/seal and checked before
and after the audit. Unused matrices and inference traces are not opened.
The output must be disjoint from all protected roots. Completed outputs are
verified and returned without regeneration. To change settings or source
coverage, choose a new `--output-root`.

If all case files are committed but reporting was interrupted, either repeat
the run command or use:

```bash
bash ./zzzz_verify_ctbf.sh v5-trimmed-truth report --resume
```

`report` refuses missing case results and never schedules an evaluation.
`--case-id ID` (repeatable) creates an explicitly marked small verification
subset. `--skip-medicc2` and `--skip-nestedbd` are explicit coverage exclusions,
not silent fallbacks for unavailable sources.

## Focused verification

```bash
python -m pytest -q test/test_trimmed_truth_evaluation.py \
  test/test_v5_trimmed_truth_audit.py \
  test/test_evaluation_contract.py test/test_evaluator_grf.py \
  test/test_evaluator_full_contexts.py
```

Tests cover both owner examples, repeated labels, root/unary rules,
normalization, nonmutation, a 10,000-node path, exact AD invariance, native
metric stability, source/selection drift, failed cases, resumed reporting,
protected-output guards and correct history/seed aggregation.

Implementation verification (2026-10-04): 183 focused tests pass. One real
condition, `conf-b001-H14_rule_y`, passed ten stored CTBF/MEDICC2/NestedBD
records with unchanged consumed-source hashes and exact AD checks. A
completed report/resume reused those results without evaluation. The test
used 1,000 bootstrap draws solely as a small technical check. The owner run
above uses the default 100,000 draws. No full reevaluation has run yet.
