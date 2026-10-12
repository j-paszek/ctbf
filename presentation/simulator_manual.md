# CTBF simulator manual

This guide describes the CTBF v5 simulator included in software release 1.2.0.
A simulated vertex represents a CNP at a particular generation. The simulator
records its parent and copy-number events, providing a known tree against
which reconstructions can be evaluated.

## Start with a complete example

The [all-method tutorial](../README.md#one-history-all-eight-ctbf-paper-methods) supplies
all simulation, biopsy and reconstruction settings in
[examples/quickstart.json](../examples/quickstart.json). Its `simulator` object
contains the fields described below.

To run the simulator alone, use a standalone configuration:

```python
from simulator import CancerCellEvolutionSimulator

sim = CancerCellEvolutionSimulator("simulator_examples/default.json", seed=9)
sim.run_simulation()
biopsy = sim.perform_biopsy(generation=5, biopsy_size=6, seed=9)
print("Tree vertices:", len(sim.tree))
print("Sampled CNPs:", [cell.genome.tolist() for cell in biopsy])
```

`default.json` generates six generations after the founder. The tutorial
uses the same event settings with a 14-generation horizon. A Python mapping
can be supplied instead of a configuration path.

For the general `python ctbs.py` driver, `RUN_SINGLE_TEST.config` selects a
standalone simulator JSON file, and `RUN_SINGLE_TEST.bedfile` selects the
optional bin-coordinate CSV. These settings belong in `ctbs_config.json`.
The tutorial's outer configuration, which also contains biopsy and method
settings, is used by `python -m examples.quickstart` instead.

## JSON parameters

All fields below are required. With a BED-like CSV, omit `GENOME_LENGTH`,
`NUMBER_OF_CHROMOSOMES`, `INITIAL_COPY_NUMBER` and `CRUCIAL_BIN_INDICES`:
those values are supplied by its rows. Unknown fields and invalid values
are rejected before simulation. Booleans are JSON `true`/`false`, and
probabilities are finite numbers in `[0,1]`.

| Field | Accepted value | Meaning |
|---|---|---|
| `SIMULATOR_SEMANTIC_VERSION` | `"ctbf-cnp-state-simulator-v5"` | Configuration/model version |
| `GENOME_LENGTH` | Integer ≥ 1 | Number of genomic bins |
| `NUMBER_OF_CHROMOSOMES` | Integer from 1 to `GENOME_LENGTH` | Contiguous, nearly equal chromosome blocks; extra bins go to earlier blocks |
| `INITIAL_COPY_NUMBER` | Nonnegative integer | Founder copy number in every bin; the complete founder must contain at least one nonzero bin |
| `CRUCIAL_BIN_INDICES` | Sorted, unique zero-based indices | Bins whose loss to zero is lethal when crucial-bin survival is enabled |
| `NUMBER_OF_GENERATIONS` | Integer ≥ 0 | Number of generations after founder generation 0 |
| `REPRESENTATION_TYPE` | `"representative"` or `"full"` | Retain one occurrence per exact CNP per generation, or every viable occurrence |
| `OFFSPRING_MODEL` | `"constant"`, `"poisson"`, `"uniform_range"` | Distribution of additional descendant attempts |
| `OFFSPRING_PARAMETER` | Model-specific, below | Additional-attempt parameter |
| `BASELINE_DESCENDANT_ATTEMPTS` | Integer `1` | One baseline attempt per continuing parent |
| `CNA_EVENT_PROBABILITY` | Probability | Per-bin probability of initiating a segmental event in each attempted descendant |
| `CNA_INITIATION_SCHEDULE` | Object, below | Generation-specific multiplier of initiation probabilities |
| `GAIN_GIVEN_CNA_PROBABILITY` | Probability | Conditional gain probability; loss has probability `1 − gain` |
| `INTERVAL_CNA_PROBABILITY` | Probability | Conditional probability of interval mode instead of a point-unit event |
| `INTERVAL_GAIN_OPERATOR_PROBABILITIES` | Object with `unit`, `additive`, `multiplicative` | Nonnegative probabilities summing to 1 |
| `ADDITIVE_GAIN_LAMBDA` | Finite number ≥ 0 | Additive interval gain size is `2 + Poisson(lambda)` |
| `MULTIPLICATIVE_FACTOR_PROBABILITIES` | Object, e.g. `{"2": 1}` | Probabilities of integer multiplication factors ≥ 2, summing to 1 |
| `WGD_PROBABILITY` | Probability | Probability of one whole-genome-doubling proposal per attempted descendant |
| `STATE_LINEAGE_REGULATION` | Object, below | Optional regulation of continuing representatives |
| `RESOURCE_GUARD` | Object, below | Abort limits for retained vertices |
| `TELOMERIC_INSTABILITY_ENABLED` | Boolean | Enable increased initiation probabilities at chromosome ends |
| `TELOMERIC_INSTABILITY_INCREMENT` | Probability | Initiation increment for telomeric bins |
| `TELOMERIC_FRACTION` | Probability | Fraction at each chromosome end, rounded down to a bin count |
| `CRUCIAL_SURVIVAL_ENABLED` | Boolean | Reject an attempted descendant as soon as a crucial bin reaches zero |

The tutorial uses 100 initially diploid bins, one chromosome, CNA initiation
`0.001`, conditional gain probability `0.5`, interval probability `0.1`, no
WGD, and no telomeric increment. Conditional interval gains are +1 with
probability 0.8 and +2 with probability 0.2. The fixed crucial-bin list is
`[4,7,38,40,41,49,54,61,69,72]`.

These are explicit JSON settings. Changing `GENOME_LENGTH` does not rescale
`CNA_EVENT_PROBABILITY` automatically. To retain the example's expected 0.1
initiation proposals per attempted descendant, use `0.1 / GENOME_LENGTH`
when initiation is uniform and the schedule and telomeric multipliers are
inactive.

### Descendant attempts and representation

For each continuing parent, the number of attempts is `1 + S`:

| Model | Parameter | Additional attempts `S` |
|---|---|---|
| `constant` | Integer ≥ 0 | Exactly that integer |
| `poisson` | Finite mean ≥ 0 | Poisson draw with that mean |
| `uniform_range` | Two integers `[min, max]`, `0 ≤ min ≤ max` | Uniform integer from the inclusive range |

An attempt may be rejected by a viability rule. In `representative` mode,
only the first generated occurrence of each exact CNP in the new generation
is retained. Later identical occurrences are discarded, including those
from different parents. The retained tree contains genotype representatives
without abundance weights. In `full` mode, every viable attempt is retained.

Simulation representation is separate from reconstruction output: either
simulator mode can provide biopsies for partial or fully labeled reconstruction.

### CNA mechanics and viability

At each bin, an initiation draw is followed by footprint, gain/loss and
magnitude draws. Point-unit events change one bin by +1 or −1. An interval
starts at its initiation bin, chooses left or right with equal probability,
and selects an inclusive endpoint uniformly up to that chromosome boundary.
Singleton intervals are retained. Events never cross chromosome boundaries.

Interval losses subtract one. Interval gains select a unit increment,
`2 + Poisson(lambda)` additional copies, or a multiplication factor according
to the configured mixture. WGD separately doubles the profile.

Copy numbers stay nonnegative, and zero is absorbing under gain operations.
Proposed events can overlap. They are applied in sampled temporal order.
An all-zero founder is invalid. An attempted descendant is rejected
immediately if its entire profile reaches zero, or if a crucial bin reaches
zero while crucial survival is enabled. Enabling that rule requires a
nonempty mask whose founder copy numbers are positive.

### Time-dependent initiation

For a constant rate:

```json
"CNA_INITIATION_SCHEDULE": {"MODEL": "constant"}
```

For an early burst followed by decay:

```json
"CNA_INITIATION_SCHEDULE": {
  "MODEL": "early_burst_decay",
  "INITIAL_MULTIPLIER": 2.0,
  "FINAL_MULTIPLIER": 0.25,
  "DECAY_EXPONENT": 2.0
}
```

The initial multiplier must exceed the positive final multiplier, and the
exponent must be positive. For generation `g` in a horizon `H > 1`, the
multiplier is `final + (initial − final) * (1 − (g−1)/(H−1))^exponent`.
A one-generation run uses the initial multiplier. At each bin, the effective
initiation probability is `min(1, multiplier * (base + telomeric increment))`.
The schedule changes segmental initiation only. WGD and conditional event
properties retain their separate settings.

### Regulation and resource limits

The example leaves regulation off:

```json
"STATE_LINEAGE_REGULATION": {"MODEL": "none"}
```

The optional model, available only in `representative` mode, is:

```json
"STATE_LINEAGE_REGULATION": {
  "MODEL": "soft_capacity", "CAPACITY": 1000, "STEEPNESS": 2.0
}
```

`CAPACITY` is a positive integer and `STEEPNESS` is positive. With `G` parent
representatives and unregulated expected attempts `R0`, each parent
continues with probability `1 / (1 + (R0−1)*(G/CAPACITY)^STEEPNESS)`.
This regulates expected attempted descendants, rather than enforcing an
exact population size. Extinction is a recorded outcome.

The independent resource guard is required in both modes:

```json
"RESOURCE_GUARD": {
  "MAX_REPRESENTATIVES_PER_GENERATION": 2000,
  "MAX_TOTAL_NODES": 40000
}
```

Both limits are positive integers. The total includes the founder and all
retained historical vertices. A run raises `SimulationResourceLimitExceeded`
before retaining a vertex that would exceed a limit. That incomplete history
is unsuitable for reconstruction. The guard does not prune a tree to fit.

## Optional chromosome/bin CSV

Pass a BED-like CSV as the simulator's second argument:

```python
sim = CancerCellEvolutionSimulator("my_config.json", "my_bins.csv", seed=9)
```

Remove the four no-BED fields listed above from `my_config.json`. The exact
header and an example are:

```csv
ChromosomeNumber,Start,End,Parameters
1,0,100000,CN=2;CNA_EVENT_PROBABILITY=0.001
1,100000,200000,CN=2;GAIN_GIVEN_CNA_PROBABILITY=0.7;CRUCIAL=1
2,0,100000,CN=1;INTERVAL_CNA_PROBABILITY=0.2
```

Each row is one CNP bin. Chromosome names must be nonempty, and coordinates
must satisfy `0 ≤ Start < End`. Rows are sorted and non-overlapping within
each chromosome, which must occupy one contiguous block of rows. `CN` is
required in every row.

Optional semicolon-separated row settings are `CNA_EVENT_PROBABILITY`,
`GAIN_GIVEN_CNA_PROBABILITY`, `INTERVAL_CNA_PROBABILITY`,
`INTERVAL_GAIN_OPERATOR_PROBABILITIES`, `ADDITIVE_GAIN_LAMBDA`,
`MULTIPLICATIVE_FACTOR_PROBABILITIES`, `TELOMERIC_INSTABILITY` and `CRUCIAL`.
Distribution values use forms such as `unit:0.8,additive:0.2,multiplicative:0`.
CSV fields containing commas must be quoted. `CRUCIAL` is `0` or `1`.
Explicit `TELOMERIC_INSTABILITY` requires telomeric mode to be enabled.
Overrides are taken from the initiation row and apply over the event footprint.

## Biopsy sampling and CNP identity

`perform_biopsy(generation, biopsy_size=n)` samples without replacement,
capped by the available count. With `biopsy_size_scalable=f`, a nonempty
generation of size `N` contributes `min(N, max(1, floor(f*N)))` raw samples.
The tutorial uses an explicit size calculation with a configurable lower
bound and ceiling, described in its [configuration table](../README.md#reproduce-the-example).
Empty generations yield no sample.

`node_id` identifies a tree occurrence. `cell_id` identifies a CNP state.
Exact recurrent CNPs are canonicalized to the same state label at the biopsy
and truth-export boundaries. Identical states are collapsed within a biopsy,
while repeated observations across biopsy times remain separate. The raw
simulation tree is preserved.

## Inspecting results

- `sim.tree`: directed simulation tree, including typed edge-event records.
- `sim.canonicalized_tree_by_genome()`: a truth-tree copy with canonical CNP labels.
- `sim.provenance()`: seed, configuration hashes, versions and RNG information.
- `sim.diagnostics_snapshot()`: generation sizes, proposals, rejected attempts and retained events.
- `sim.create_bed_csv(path)`: resolved bin configuration.
- `sim.get_parameters_csv(path)`: resolved parameters as a wide table.

The tutorial calls the same `trim_true_tree` helper used in evaluation. It
retains every true occurrence of an observed state, clears other labels,
prunes branches without retained labels and suppresses unlabeled unary
vertices, promoting the child of an unlabeled unary root. Repeated labeled
vertices and unlabeled branching vertices remain.

The [interactive gallery](https://j-paszek.github.io/ctbf/presentation/quickstart-v2/index.html)
shows the complete and trimmed truth alongside all eight CTBF paper methods.
The simulator supports multiple chromosomes, but this lightweight runner
currently requires one chromosome because its distance provider operates
on one interval domain. It rejects multi-chromosome configurations explicitly.
