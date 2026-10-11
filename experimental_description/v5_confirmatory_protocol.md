# CTBF V5 Independent Confirmatory Protocol

Status: frozen design candidate awaiting exact-byte owner approval  
Prepared: 2026-09-10

## Purpose And Evidential Boundary

This protocol prospectively tests the frozen CTBF v5 reconstruction methods on
synthetic evolutionary histories that were not used to design, choose, or tune
the methods.  No confirmatory truth, biopsy, distance, reconstruction, or
accuracy result may be generated until the owner approves the exact bytes of
this protocol and its manifest and the approval hashes are installed in the
runner.

The experiment tests performance under the stated CTBF v5 data-generating
model.  It does not establish performance under another simulator, validate
the biological accuracy of that simulator, or turn the KTN152 feasibility case
into an accuracy benchmark.  Model-mismatch experiments remain separate.

## Frozen Methods

The internal portfolio contains exactly seven paper methods.  The identifiers
below resolve to the current CTBF v5 implementations and are frozen together
with the source inventory.

| Paper name | Stable implementation identifier | Output | Role |
| --- | --- | --- | --- |
| M0 | `classical_partial` | partial | pooled classical-NJ baseline |
| M2 | `biopsy_guided_top_anticentral_binary_adaptive_median_prior_nn_bottom_deferred_tie` | partial | principal biopsy-guided method |
| M1 | `neighbor_joining_baseline` | full | pooled labeled closest-pair baseline |
| Pooled-Q | `rooted_labeled_nj` | full | pooled rooted labeled-Q baseline |
| M5 | `neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony` | full | pooled binary-anticentral comparator |
| M6 | `biopsy_guided_full_anticentral_binary_adaptive_median_prior_nn_bottom_deferred_tie` | full | biopsy-guided binary-anticentral completion |
| M7 | `biopsy_guided_full_rooted_labeled_q_adaptive_median_prior_nn_bottom_deferred_tie` | full | biopsy-guided rooted labeled-Q completion |

The external MEDICC2 topology-only partial method is not inserted into this
shared-distance portfolio.  It is executed separately on the same standard
observations after its frozen runtime and resource gate pass.  It is a
contextual comparator, not part of the fully labeled ranking or the biopsy-
design sensitivity.

No method implementation, parameter, input order, tie rule, output projection,
or method roster may change after the held-out lock.  A change invalidates the
confirmation and requires a new manifest identifier and seed namespace.
The manifest also freezes the CPython version and the versions of NumPy,
NetworkX, Plotly, and psutil used by the internal pipeline. Executable and
platform paths are recorded as provenance but are not path locks.

## Simulator And Independent Unit

Each of 100 independent blocks simulates one CTBF v5 history through generation
34 from `simulator_examples/paper_v5/confirmatory_standard.json`.  H14 and H24
are prefixes of the same H34 history.  Consequently, a condition, biopsy,
profile, node, edge, pairwise distance, or algorithm record is not an
independent replicate.  The independent unit is the complete truth block.

The simulator uses 100 bins, one chromosome, ongoing CNA initiation probability
0.001 per bin and descendant attempt, 10% conditional interval events, the
80% unit/20% additive-2 interval-gain mixture, no WGD, and the fixed ten-bin
crucial-survival mask.  All remaining values are fixed by the referenced JSON.
Resource-limit outcomes are retained and never replaced.

## Observation Design

At a selected generation containing `N` distinct representative genotype
states, the biopsy contains

`min(N, max(6, ceil(0.5 N)))`

states.  A single deterministic permutation is keyed only by the held-out seed
namespace, block, and generation.  Thus every condition containing the same
generation receives exactly the same representatives.  Selection is without
replacement.  The fraction describes representative genotype states and not
physical cells, tissue volume, or clone abundance.

The primary standard conditions are:

| Condition | Generations |
| --- | --- |
| `H14_rule_y` | `[9,12,14]` |
| `H24_rule_y` | `[15,20,24]` |
| `H34_rule_y` | `[21,28,34]` |

The focused secondary H24 biopsy-count conditions are nested:

| Condition | Generations |
| --- | --- |
| `H24_L2` | `[15,24]` |
| `H24_rule_y` / `H24_L3` | `[15,20,24]` |
| `H24_L4` | `[15,18,20,24]` |
| `H24_L5` | `[15,18,20,22,24]` |

The secondary H24 placement conditions contain three biopsies:

- Rule Y: `[15,20,24]`;
- late: `[22,23,24]`; and
- random: two distinct generations drawn uniformly without replacement from
  `[15,23]`, sorted, followed by generation 24.

The random schedule has its own prospectively fixed seed stream.  An accidental
match to another policy is retained.  Placement comparisons are practical
policy comparisons because their realized sample sizes may differ.  They are
not fixed-budget causal effects.

There are eight unique conditions per block: the three standard-height
conditions, H24 L2/L4/L5, and H24 late/random.  `H24_rule_y` is stored once and
serves both the standard H24 and L3 roles.

## Distance And Reconstruction Inputs

For each block, cnp2cnp computes one maximum-observation matrix over all unique
states needed by its eight conditions.  The distance is
`min(d_any(u,v), d_any(v,u))`.  Every condition receives the exact label-aligned
submatrix.  No fallback, directed alternative, ordered-triangle shortcut,
truth distance, or simulator identity is available to reconstruction.
The cnp2cnp revision and the SHA-256 digests of its executable and four
transitive local source dependencies are frozen in the manifest and verified
before any held-out truth is generated and again before the bank is closed.

Reconstruction input contains only the ordered biopsy lists, their CNPs,
opaque state labels, and the condition distance matrix.  Within-level state
order is canonical and fixed before execution.  One reconstruction seed per
block and condition is shared by all methods.  Truth is loaded only by the
evaluator after reconstruction.

## Primary Confirmatory Hypothesis

The sole primary comparison is M2 minus M0 on rooted GRF in the three Rule-Y
conditions.  For truth block `b`, let the block effect be the unweighted mean
of the paired H14, H24, and H34 differences.  A block enters the primary
analysis only when both methods succeed at all three heights.  Missing values
are not imputed and blocks or seeds are not replaced.

The directional claim is supported only if all of the following hold:

1. at least 90 of 100 blocks have all three paired effects;
2. neither primary method has a failure rate above 10% at any height;
3. the mean block effect is at least 0.01 rooted-GRF units;
4. the lower endpoint of the two-sided 95% block-bootstrap percentile interval
   is greater than zero; and
5. a one-sided 100,000-draw block sign-flip test gives `p < 0.05` with the
   plus-one Monte Carlo correction.

This is one primary test and therefore has no multiplicity adjustment.  Report
the complete denominators, condition- and block-level wins/ties/losses with
tolerance `1e-12`, mean, median, range, worst-five-block mean, conditional mean
gain and loss, interval, p-value, failures, runtime, and peak memory regardless
of direction.  Failure of any support condition makes the primary result
inconclusive rather than negative evidence being discarded.

## Prespecified Secondary Analyses

Secondary analyses do not alter the primary decision and do not select or tune
a method.

1. Report M6 minus M5 and M7 minus Pooled-Q on AD-F1, averaged within truth
   block across the three Rule-Y heights.  These are matched top-completion
   comparisons of biopsy-guided and pooled reconstruction.
2. Report all seven methods separately by height.  Keep partial and full output
   families separate and do not construct an omnibus winner or composite
   metric.
3. For M6, compare H24 L5 with H24 L2 using AD-F1 restricted in both trees to
   the labels observed in L2.  Report the complete L2--L5 curve, native scores,
   failures, and realized observations.  Do not require monotonic adjacent
   steps.
4. Report Rule-Y minus late and Rule-Y minus random H24 effects within each
   method, together with realized observation counts.  Do not interpret these
   as isolated effects of temporal placement.
5. Report rooted GRF, AD precision, and AD recall where defined by the method's
   output contract, plus stored mechanism and resource diagnostics.  These are
   complementary and descriptive.
6. Run MEDICC2 separately on the 300 standard Rule-Y cases and report its
   rooted-GRF difference from M2, failures, wall time, and peak RSS.  Do not use
   MEDICC2 for full-label metrics or biopsy-time ablation.

All secondary intervals use truth-block resampling.  No secondary result may
be promoted after inspection into the sole primary claim.

## Failure, Resource, And Execution Contract

- 300 seconds per simulation block;
- 1,200 seconds per cnp2cnp distance stage and 540 seconds per cnp2cnp child;
- 300 seconds per reconstruction and 300 seconds per evaluation;
- 4 GiB peak process-tree RSS per stage, sampled every 10 ms;
- at most 1 MiB captured stdout or stderr per external process;
- at most six concurrent fresh distance workers and eight concurrent fresh
  case-arm workers; and
- one declared task per spawned process before process exit.

Every declared outcome is stored as success, typed failure, or dependency
failure.  Deterministic failures are never retried.  One byte-identical retry
is permitted only for a documented transient infrastructure or external-tool
failure, with both attempts retained.  The bank and reconstruction roots are
checksum-closed before aggregate accuracy is read.  Resume is permitted only
for an exact declared prefix under identical manifest, source, resource, and
input hashes.

## Staging And Blinding

1. Structurally validate the draft protocol, manifest, source inventory,
   simulator config, method registry, and seed separation without generating a
   held-out seed.
2. Run only a separately namespaced, non-evidential technical smoke/resource
   case.
3. Obtain explicit owner approval of the exact protocol and manifest bytes and
   install their SHA-256 locks.
4. Generate and checksum-close all held-out truth, biopsy, and distance inputs.
5. Run and checksum-close every internal reconstruction record without
   aggregate inspection.
6. Run the frozen MEDICC2 contextual stage.
7. Only after all declared raw statuses exist, generate the deterministic
   block-paired report and retain it irrespective of direction.

Any result-informed algorithm, endpoint, threshold, schedule, case-exclusion,
or seed change requires a new confirmation namespace.  Development outputs,
CTBF v1--v4 outputs, KTN152 outputs, and technical smoke outputs are prohibited
from the confirmatory aggregate.
