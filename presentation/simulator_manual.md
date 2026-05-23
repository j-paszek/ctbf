# CTBF Simulator Manual

This document describes the **current simulator implementation** in [`simulator.py`](../simulator.py).
It replaces the older [`simulator_parameters.csv`](./simulator_parameters.csv), which contains several parameters that are no longer implemented, are only partially implemented, or describe intended behavior rather than current behavior.

## Scope

The simulator is configured from:

1. a JSON configuration file passed as `config_path`,
2. optionally, a BED-like CSV file passed as `genome_csv`.

If `genome_csv` is provided, the founder genome and per-locus annotations are loaded from that file.

## Global JSON Parameters

| Parameter | Type | Required | Default | Current behavior |
| --- | --- | --- | --- | --- |
| `genome_length` | integer | Yes if `genome_csv` is not provided | none | Length of the founder genome. Ignored when `genome_csv` is used. |
| `initial_copies` | integer | Yes if `genome_csv` is not provided | none | Copy number assigned to every locus in the founder genome. Ignored when `genome_csv` is used. |
| `NUMBER_OF_GENERATIONS` | integer | Yes | none | Number of generations simulated. |
| `REPRESENTATION_TYPE` | string | No | `"representative"` | Must be `"full"` or `"representative"`. In `"representative"` mode, duplicate child genomes within the same generation are collapsed. |
| `OFFSPRING_MODEL` | string | Yes | none | Supported values: `"constant"`, `"uniform_range"`, `"poisson"`, `"fitness"`. |
| `OFFSPRING_PARAMETER` | number or string | Yes | none | Interpretation depends on `OFFSPRING_MODEL`; see below. |
| `GENERAL_EVENT_PROB` | float | Yes | none | Baseline probability that a CN event is attempted at a locus. |
| `GENERAL_DUPLICATION_PROB` | float | Yes | none | Probability that an event is a duplication rather than a loss. |
| `GENERAL_DUPLICATION_MULTIPLICITY` | float | Yes | none | Mean of the Poisson used for duplication size. Actual duplicated copy count is `Poisson(mean) + 1`. |
| `GENERAL_LOSS_PROB` | float | Yes in config, but not used in event generation | none | Loaded and stored, but the current implementation does not use it when choosing event type. Loss occurs whenever duplication is not chosen. |
| `GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB` | float | Yes | none | Probability that an event extends from locus `i` to a randomly chosen `j >= i`. |
| `MODEL_TELOMERIC_REGIONS` | string or bool | No | `"False"` | Enables telomeric instability logic. Values are interpreted case-insensitively through `str(...).lower() == "true"`. |
| `GENERAL_TELOMERIC_PERCENTAGE` | float | Required if `MODEL_TELOMERIC_REGIONS` is enabled | none | Fraction of each chromosome end considered telomeric. |
| `GENERAL_TELOMERIC_INSTABILITY` | float | Required if `MODEL_TELOMERIC_REGIONS` is enabled | none | Added to event probability inside telomeric regions, capped at 1. |
| `MODEL_CRUCIAL_FOR_SURVIVAL` | string or bool | No | `"False"` | If enabled, any event that reduces a locus marked `CRUCIAL=1` to copy number 0 causes that child genome to be discarded. |

## Offspring Models

| `OFFSPRING_MODEL` | `OFFSPRING_PARAMETER` format | Current behavior |
| --- | --- | --- |
| `"constant"` | integer-like | Each parent produces exactly `int(OFFSPRING_PARAMETER)` children, then the loop runs `num_children + 1` times in the current implementation. |
| `"uniform_range"` | string `"min,max"` | Samples an integer uniformly from the closed interval `[min, max]`, then the loop runs `num_children + 1` times. |
| `"poisson"` | numeric | Samples offspring count from `Poisson(parameter)`, then the loop runs `num_children + 1` times. |
| `"fitness"` | numeric | Uses the parameter as `MAX_N` in the logistic fitness-to-Poisson mean transform, then samples from `Poisson(lambda)`. The outer loop still runs `num_children + 1` times. |

### Important implementation note

The current code iterates over `range(num_children + 1)`, not `range(num_children)`.
As a result, the realized number of children is always **one larger** than the sampled offspring count.

## BED-like Genome CSV Format

The optional genome file is read with `csv.DictReader` and is expected to contain at least:

| Column | Required | Current behavior |
| --- | --- | --- |
| `ChromosomeNumber` | No | If present, telomeric regions are computed separately per chromosome. If absent, the whole genome is treated as a single chromosome. |
| `Start` | No | Ignored by the current implementation. Included for BED-like readability only. |
| `End` | No | Ignored by the current implementation. Included for BED-like readability only. |
| `Parameters` | Yes | Semicolon-separated `key=value` pairs, e.g. `CN=2;EVENT_PROB=0.8;CRUCIAL=1`. |

Example:

```csv
ChromosomeNumber,Start,End,Parameters
1,0,100000,CN=2;EVENT_PROB=0.8;TELOMERIC_INSTABILITY=0.5
1,100000,200000,CN=3
2,100000,200000,CN=1;CRUCIAL=1
2,200000,300000,CN=2;FITNESS_WEIGHT=0.1
```

## Per-Locus Parameters in `Parameters`

| Key | Type | Current behavior |
| --- | --- | --- |
| `CN` | integer | Founder copy number at this locus. Defaults to `2` if omitted. |
| `EVENT_PROB` | float | Overrides `GENERAL_EVENT_PROB` at that locus. This override is used. |
| `DUPLICATION_PROB` | float | Parsed and stored, but currently **not used** during event generation. |
| `DUPLICATION_MULTIPLICITY` | float | Parsed and stored, but currently **not used** during event generation. |
| `LOSS_PROB` | float | Parsed and stored, but currently **not used** during event generation. |
| `SINGLE_OR_MULTIPLE_EVENT_PROB` | float | Parsed and stored, but currently **not used** during event generation. The simulator uses the global `GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB`. |
| `TELOMERIC_INSTABILITY` | float | Overrides telomeric instability at that locus when telomeric modeling is enabled. |
| `CRUCIAL` | `0` or `1` | If `MODEL_CRUCIAL_FOR_SURVIVAL` is enabled, loss to copy number 0 at this locus invalidates the child genome. |
| `FITNESS_WEIGHT` | float | Per-locus weight used only by the `"fitness"` offspring model. |

## What the Simulator Actually Uses During Event Generation

For each locus `i`, the simulator:

1. starts from `EVENT_PROB` if provided, otherwise `GENERAL_EVENT_PROB`,
2. adds telomeric instability if telomeric modeling is enabled,
3. decides whether the event is single-locus or multi-locus using the **global** `GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB`,
4. decides duplication vs loss using the **global** `GENERAL_DUPLICATION_PROB`,
5. samples duplication size using the **global** `GENERAL_DUPLICATION_MULTIPLICITY`,
6. applies the change across `[i, j]`.

This means the following values are **loaded but currently inactive** in the mutation step:

- `GENERAL_LOSS_PROB`
- per-locus `DUPLICATION_PROB`
- per-locus `DUPLICATION_MULTIPLICITY`
- per-locus `LOSS_PROB`
- per-locus `SINGLE_OR_MULTIPLE_EVENT_PROB`

## Founder Genome Behavior

If `genome_csv` is not provided:

- the founder genome is `initial_copies` repeated `genome_length` times.

If `genome_csv` is provided:

- the founder genome is built from the ordered `CN` values in the file,
- `genome_length` is inferred from the CSV,
- optional chromosome, cruciality, telomeric, and fitness annotations are taken from the CSV.

## Telomeric Regions

If `MODEL_TELOMERIC_REGIONS` is enabled:

- both `GENERAL_TELOMERIC_INSTABILITY` and `GENERAL_TELOMERIC_PERCENTAGE` must be present,
- if chromosome labels exist, telomeric regions are computed separately for each chromosome,
- otherwise the first and last `int(length * percentage)` loci of the whole genome are treated as telomeric,
- per-locus `TELOMERIC_INSTABILITY` values override the automatically assigned telomeric values.

## Representation Modes

| Mode | Current behavior |
| --- | --- |
| `representative` | Within one spawning step, repeated child genomes are collapsed using genome equality. |
| `full` | All children are retained, even if they have identical genomes. |

## Biopsy Sampling

`CancerCellEvolutionSimulator.perform_biopsy` samples unique genotype
representatives from the requested generation.

The simulator does not currently store an explicit population size or abundance
weight for each genotype representative. A genotype node should therefore be
read as the observable copy-number state present at that generation, not as a
single explicitly counted biological cell. If a selected generation contains one
genotype representative, that means the simulated state available for biopsy at
that generation is represented by that one genotype. Scalable biopsy sampling
therefore treats genotype representatives as equally weighted and ensures that a
non-empty selected generation contributes at least one representative.

The method supports two sizing modes:

- `biopsy_size`: fixed number of cells, capped at the number available in the
  requested generation.
- `biopsy_size_scalable`: fraction of the requested generation. For non-empty
  generations this mode samples at least one cell, even when
  `int(biopsy_size_scalable * generation_size)` would be `0`. Empty generations
  still return an empty biopsy.

If both sizing modes are provided, `biopsy_size_scalable` controls the final
sample size.

## Output Helpers

The simulator currently exposes two helper exporters:

| Method | Output |
| --- | --- |
| `get_parameters_csv(file)` | Writes a flat per-locus table of the simulator state after initialization. |
| `create_bed_csv(file)` | Writes a minimal BED-like CSV with `ChromosomeNumber,Start,End,Parameters`, but currently outputs placeholder `Start=0`, `End=0`, and only `CN=...`. |

## Differences From the Old Manual

The old `simulator_parameters.csv` includes several entries that do not reflect the current codebase:

- `TELOMERIC_MASKING_FILE` is not implemented.
- `FITNESS_MODEL` is not implemented as a config option.
- `SELECTIVE_PRESSURE` is not implemented as a config option.
- `GENERAL_LOSS_PROB` is described as active, but is not currently used in event selection.
- several per-locus overrides are parsed but not actually applied in mutation generation.

## Recommended Usage Notes

- Use the JSON file to control global simulation behavior.
- Use the BED-like CSV only when founder-genome heterogeneity or per-locus annotations are needed.
- If precise biological interpretation matters, treat the inactive parameters listed above as **unsupported in the current implementation**, even if they appear in older documentation.
