# CTBF V5 Confirmation Runbook

Status: prospectively locked run complete and checksum-closed  
Updated: 2026-09-13

## Completed Evidence

The owner approved the exact protocol, manifest, and source-lock hashes on
2026-09-10 and subsequently completed every stage below. The bank contains
800/800 available conditions. All 5,600 internal arm records and all 300
MEDICC2 records succeeded without dependency failure. The registered report
supports the sole primary comparison on all 100 truth blocks and is preserved
under `algorithm_evaluation/results/v5_confirmation/confirmation_v1/report`.
Do not rerun, replace, or retune this evidence after inspection.

## Evidential Gate

The confirmation commands intentionally fail while
`algorithm_evaluation/v5_confirmation_approval.py` contains unset hashes. The
only command permitted before approval is structural validation:

```bash
./zzzz_verify_ctbf.sh v5-confirmation-contract
```

The non-evidential technical smoke has already passed in a separately seeded
temporary namespace. Do not rerun it as paper evidence.

## Fresh Owner Run

After the owner approves the exact protocol and manifest bytes and their three
hashes are installed, run the following stages in order from the CTBF root.
The result root must not already exist.

```bash
caffeinate -is ./zzzz_verify_ctbf.sh v5-confirmation-bank \
  --output-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/bank \
  --cnp2cnp-file /Users/voronwe/Work/PyCharmProjects/cnp2cnp/cnp2cnp.py \
  --simulation-workers 4 \
  --distance-workers 6 \
  --progress

caffeinate -is ./zzzz_verify_ctbf.sh v5-confirmation-run \
  --bank-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/bank \
  --output-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/internal \
  --record-workers 8 \
  --progress

caffeinate -is ./zzzz_verify_ctbf.sh v5-confirmation-medicc2 resource-gate \
  --bank-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/bank \
  --output-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/medicc2_gate \
  --medicc2-executable /Users/voronwe/Work/Medicc-runtime-3f1b8a6/bin/medicc2 \
  --source-root /Users/voronwe/Work/Medicc

caffeinate -is ./zzzz_verify_ctbf.sh v5-confirmation-medicc2 run \
  --bank-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/bank \
  --output-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/medicc2 \
  --resource-gate algorithm_evaluation/results/v5_confirmation/confirmation_v1/medicc2_gate \
  --medicc2-executable /Users/voronwe/Work/Medicc-runtime-3f1b8a6/bin/medicc2 \
  --source-root /Users/voronwe/Work/Medicc \
  --workers 4 \
  --progress

./zzzz_verify_ctbf.sh v5-confirmation-report \
  --run-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/internal \
  --medicc2-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/medicc2 \
  --output-root algorithm_evaluation/results/v5_confirmation/confirmation_v1/report
```

The report command is the first authorized aggregate inspection of accuracy.
It refuses an incomplete or checksum-changed internal or MEDICC2 run.

## Publication Derivatives

The following command creates a new non-overwriting figure/data package after
revalidating the bank, internal-run, MEDICC2-run, and report checksum closures:

```bash
./zzzz_verify_ctbf.sh v5-confirmation-publication \
  --output-root paper/figures/v5-confirmation-v1
```

The established `v5-confirmation-v1` package was generated successfully. Use a
different output root for an independent reproduction. The derivative records
the validated MEDICC2 checksum explicitly because the immutable source report
serialized its MEDICC2 checksum field as null after validating the closure.
This provenance omission does not affect any reconstruction or score.

The resource rows in that first derivative used the complete block-wide
distance stage, which included states from all eight conditions. They are not
the final per-dataset M2--MEDICC2 resource comparison. The accuracy summaries
and figure remain valid.

## Condition-Specific M2 Resource Audit

The following non-overwriting, post-confirmation audit recomputes `D` from only
the observed CNPs in each of the 300 frozen standard inputs and then runs M2.
Every direct matrix and M2 topology summary must equal its frozen counterpart
before the record contributes to the resource summary. Truth and evaluation
are not loaded, and no registered accuracy artifact is changed.

```bash
caffeinate -is ./zzzz_verify_ctbf.sh v5-confirmation-resource-audit \
  --cnp2cnp-file /Users/voronwe/Work/PyCharmProjects/cnp2cnp/cnp2cnp.py \
  --output-root algorithm_evaluation/results/v5_confirmation/condition_specific_resource_v1 \
  --workers 4 \
  --progress
```

Four concurrent case workers match the frozen MEDICC2 run's concurrency. This
avoids introducing a systematic wall-time difference solely through a wider
M2 audit run.

If interrupted, repeat the same command with `--resume`. A successful full run
prints `300 cases; 300 exact matrices; 300 exact reconstructions` and writes a
checksum-closed `summary.json` plus `resources.csv`. Only then may its M2
distance-plus-reconstruction measurements replace the provisional block-wide
allocation in a newly named publication package and manuscript iteration.

## Resume

If the bank or internal run is interrupted, repeat its identical command with
`--resume`. Repeat the MEDICC2 run command with `--resume`. The resource gate
is short and non-resumable. If it is interrupted, retain its incomplete root
for provenance and use a newly named gate root. Do not delete, replace, or
repair individual scientific records.
