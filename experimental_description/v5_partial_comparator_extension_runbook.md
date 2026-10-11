# CTBF v5: five-method partial comparison with uniform 10M NestedBD runs

Owner-approved budget revision, 2026-10-05. This supersedes the prepared
2M extension's default configuration. Old 2M outputs and the completed
eight-run diagnostic remain immutable. Do not edit their runners or results.

## Owner command

From the CTBF repository, using the same Python environment as confirmation:

```bash
caffeinate -is bash ./zzzz_verify_ctbf.sh v5-partial-comparator-extension run \
  --bio-root /Users/voronwe/Work/ctbfbio \
  --output-root algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_10m_v1 \
  --workers 4 \
  --parallel-runs 2 \
  --resume
```

## Time-limited 2M fallback

If the full 10M extension cannot fit the available time, run a separate
uniform 2M series. This mode does not consume or mix the eight 10M diagnostic
runs. It starts all 400 NestedBD seed runs (100 histories, H14/H24, two seeds)
at the original 2M budget and writes to a distinct output root.

```bash
caffeinate -is bash ./zzzz_verify_ctbf.sh v5-partial-comparator-extension run \
  --nested-budget 2m \
  --bio-root /Users/voronwe/Work/ctbfbio \
  --output-root algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_2m_v1 \
  --workers 4 \
  --parallel-runs 2 \
  --resume
```

The 2M series reuses the frozen M0/M2/MEDICC2 scores, adds M4 on the same 800
conditions, and reports all NestedBD outcomes at 2M. Its contract and summary
schema are distinct from the 10M series. Do not resume a 10M output root with
`--nested-budget 2m`, and do not combine the two budgets in one table.

### Resuming the active 2M campaign

Owner-approved repair, 2026-10-06: use the **same 2M command above**, including
the existing output root. Do not delete any attempts or regenerate inputs.
The earlier adapter stopped on a previously sealed `failed` result, contrary
to the extension's fixed rule to retain terminal failures. Its local resume
selector now verifies and reuses both completed successes and failures.
Timeout, RSS and capture-limit failures also remain terminal. Only missing or
interrupted attempts start again, from zero, with the original seed and budget.
Unsealed reports, changed data/XML/checksums, duplicate terminal outcomes and
live JVMs still block unsafe reuse. No external numerical check is bypassed.

The controller may count from `1/400` again while verifying cached outcomes.
This is not repeated MCMC. A `start:` line identifies an actual new attempt.
The count includes failures, not only successfully reconstructed trees.
The 2M profile is now explicitly passed to fresh MCC/evaluation workers, and
the final completion record binds both technical and `pilot` attempt seals.

The two changed orchestration files were included in the original source
snapshot. Resume accepts only their exact audited predecessor hashes, with
all other sources, inputs, model rules, seeds and runtime unchanged. The
first owner resume adds `resume_repair_v1.json`, recording old/new source
hashes and the unchanged original contract's hash. It never rewrites
`analysis_contract.json` or saved inference. Subsequent resumes require the
same repair record. Do not delete it or edit source-pinned code mid-campaign.

Verification: 131 focused tests pass, including a real spawned-worker check
for each budget, mixed success/failure resume, and incomplete-attempt restart.
Read-only checks reused the actual b016/H24 seed 1711012498 failure and
b001/H14 seed 1946510383 success without execution or result changes.

## Uniform 10M alternative

No raw-data download, new simulation, new distance calculation or old-result
overwrite is performed. The script prepares its frozen declaration, reconstructs
M4, executes 392 new 10M NestedBD jobs, reuses eight completed 10M diagnostic
runs, and produces the combined report. All 400 NestedBD trees are summarized
and scored from 10M traces. No 2M score is included in the new series.
This completes the owner's intended comparator coverage. It is not a method or
case selection based on the first ten histories' accuracy.

New output root:
`algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_10m_v1/`.
The final Markdown report is normally `reports/report_001/report.md`.

Expected completion, if all inputs and evaluations succeed:

```text
complete: partial comparator extension 10M; 3100 records; 392 new + 8 reused NestedBD seed records; 0 incomplete case-method pairs; protected results unchanged; convergence not established
```

Completion means all declared outcomes are accounted for, not that every job
must succeed or that MCMC converged. Unsupported copy numbers, timeouts and
other terminal failures remain explicit. Report the final line and any concise
failure messages before deciding how to present the results.

## Fixed scope

| Method | New work | Reused work | Final coverage |
|---|---|---|---|
| M0 | None | 800 scores | 100 histories × eight conditions |
| M2 | None | 800 scores | 100 histories × eight conditions |
| M4 | 800 reconstructions with frozen aligned D | None | Same 800 conditions |
| MEDICC2 | None | 300 scores | H14/H24/H34, 100 histories |
| NestedBD-Long | Histories 003–100, H14/H24, two seeds: 392 runs at 10M | Eight completed 10M runs from histories 001–002 | 200 inputs, 400 newly scored seed results |

Reused inference comes from `budget_diagnostic_v1`, selected by the
`--budget-root` option. Reuse is based on declared identities and matching
inputs/model/seeds, not on ESS, agreement or accuracy. Keep that source directory
available throughout the run. The first ten histories' old 2M results remain
separate and unchanged. The old `partial_comparators_100_v1` output root cannot
be used for this series.

M4 is the existing
`biopsy_guided_classical_adaptive_median_prior_nn_bottom_deferred_tie`:
the same adaptive median prior-nearest-neighbor radius/deferred bottom as M2,
but classical NJ top. No algorithm variant or parameter search is introduced.
The original seven-arm confirmation roster and registered M2-minus-M0 analysis
remain unchanged. M4's additive records are not silently inserted into that
original roster or its report.

All eight original conditions are retained for M0/M2/M4: H14/H24/H34 standard
sampling and H24's two/four/five-biopsy, random-placement and late-placement
sensitivities. MEDICC2/NestedBD are absent from the biopsy sensitivities by
design. NestedBD is absent from H34. No missing value is imputed.

## Budgets and execution

The separate eight-run 2M/10M diagnostic improved posterior ESS >=200 from
0/8 to 7/8, although b001/H24 retained substantial clade-frequency disagreement
(maximum 0.251 at 10% burn-in and 0.324 at 50%). Unresolved convergence at 10M
does not justify retaining 2M. The owner approved uniform 10M, with no further
model/clock/burn-in tuning. See `v5_nestedbd_long_preextension_audit.md`.
This is still a finite-budget comparison, not convergence certification.

- Same released model/runtime, priors/heating, fixed nominal mean clock
  0.001061875 per generation, 100 original bins, integer values unchanged.
- Original two deterministic seed identities per input. The seed namespace
  retains its historical `ten-history` name deliberately so old identities do
  not change. New identifiers extend the same rule, with collision checking.
- A full-input 10k technical gate precedes each eligible new 10M run.
- At most two JVMs concurrently, each with four coupled chains: eight MCMC
  chains, not twelve JVMs. Heap 2 GiB and process-tree RSS ceiling 4 GiB per JVM.
- Per-seed technical timeout 300 s, 10M timeout 18000 s (five hours), capture ceiling 8 MiB.
  Failures are retained and do not stop unrelated inputs.
- M4 uses fresh processes, four concurrently by default, with 300 s each for
  reconstruction, native evaluation and trimmed evaluation, 4 GiB RSS per
  process, and a 1020 s outer task watchdog. Hard worker/controller faults stop
  safely for inspection/resume, rather than inventing numerical scores.
- The M4 and MCMC stages are sequential, not concurrent. MCC/evaluation uses
  at most two fresh workers, with 900 s per native TreeAnnotator invocation
  and a 1620 s outer summary/full/trimmed task watchdog. Parallel settings may
  be lowered on resume without changing scientific parameters.

This is a potentially multi-day owner run. The 392 × 5-hour inference
limits imply up to 980 hours at two-way concurrency if every run exhausted
its limit, plus gates and reporting. This is a conservative budget ceiling,
not a runtime prediction. On the eight diagnostic runs, 10M took 5.66--6.11
times the matched 2M runtime. Actual campaign time depends strongly on input size.

Ctrl+C during MCMC requests orderly cancellation. Run the same command to
resume. Terminal outcomes, including failures, are reused. Interrupted native
attempts are preserved and restarted **from zero**, never resumed from BEAST
checkpoints. Avoid force-killing unless necessary. If a JVM remains alive, the
runner refuses to start a duplicate. A partial M4 batch reuses saved case
bundles. An interrupted MCC/report can also resume without rerunning MCMC.
Do not edit source-pinned code or parameters while the campaign is running.

## Scoring and interpretation

The existing completed trimmed-truth audit supplies 1,900 immutable
M0/M2/MEDICC2 full/trimmed score pairs. There is no old-score replay
or roundoff-based equality trap. New M4 and all 400 NestedBD 10M trees are evaluated with
the unchanged native full-truth GRF and independently against trimmed truth.
The reconstruction is unchanged. Exact observed-state ancestry and all AD-F1
fields must agree between truth targets.

NestedBD uses the same native SA-aware MCC, 10% burn-in, original heights and
sampled-ancestor occurrence mapping as the initial comparison. A successful
10M trace has 10,001 logged samples; the first 1,000 are discarded, leaving
9,001. Both reused and new runs use these counts. Cached 2M summaries/scores
are rejected. No selection
by convergence, GRF, seed, input size or genotype values is introduced.

Each NestedBD input contributes the mean of **both successful** seed scores.
Combined-height contrasts average within each history, then resample whole
histories for 100,000-draw descriptive percentile 95% bootstrap intervals.
Report history W/T/L and condition W/T/L separately: 100 histories can provide
200 H14/H24 pairs or 300 H14/H24/H34 pairs. Seeds are not independent histories.
Pairwise comparisons use matched complete coverage with missing denominators
reported. A separate five-method table uses exactly the same successful
histories for all methods. H34 is not fabricated for NestedBD.

No new p-values, automatic ranking/promotion rule, converged-posterior claim
or primary-decision amendment is made. The goal is comparison of obtained
trees under the declared budget. Diagnostics remain alongside all results.

The trimmed target depends on the observed-state union. Biopsy-condition
effects do not hold that target fixed. Preserve this caveat in interpretation.
Runtime/RSS records are stage measurements. M4 uses cached D, so they cannot
establish end-to-end resource superiority. Any later pipeline comparison must
add the existing condition-specific distance audit, not the old union-matrix
batching cost.

## Files and recovery

- `analysis_contract.json`: source/runtime/rule identities and all declarations.
- `inputs/`: checksum-bound observable occurrence manifests for new inputs.
- `scores/M4/`: per-case trees, native evaluation, trimmed audit and resources.
- `inference/`: new native technical/long attempts and seals. Eight reused runs stay in the diagnostic root.
- `summaries/`: native MCC, mapped topology and native full-truth evaluation.
- `scores/NestedBD-Long/`: full/trimmed paired scores and summary references.
- `reports/report_*/`: report.md, report.json, scores.csv, case_scores.csv,
  diagnostics.json, and a seal.
- `_COMPLETE.json`: final report identity and new score-artifact closure.

The optional `prepare` command declares/checks inputs but runs no inference.
The `report` command refuses unfinished M4/MCMC jobs and can finish missing
native MCC, evaluation and reporting only. Both accept the same options.

`check` requires an explicit separate output root. It uses exactly
`conf-b003-H14_rule_y`, one M4 reconstruction and the first seed's 10k technical
run, with no 10M inference or main-report inclusion. Never promote its output
into the production root. This bounded check is for implementation validation,
not another owner preflight requirement.
