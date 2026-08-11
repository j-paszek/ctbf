"""Strict immutable configuration and BED-like inputs for CTBF v3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


SIMULATOR_SEMANTIC_VERSION = "ctbf-cnp-state-simulator-v3"

_CONFIG_KEYS = frozenset(
    {
        "SIMULATOR_SEMANTIC_VERSION",
        "GENOME_LENGTH",
        "INITIAL_COPY_NUMBER",
        "NUMBER_OF_GENERATIONS",
        "OFFSPRING_MODEL",
        "OFFSPRING_PARAMETER",
        "BASELINE_DESCENDANT_ATTEMPTS",
        "CNA_EVENT_PROBABILITY",
        "CNA_INITIATION_SCHEDULE",
        "GAIN_GIVEN_CNA_PROBABILITY",
        "INTERVAL_CNA_PROBABILITY",
        "INTERVAL_GAIN_OPERATOR_PROBABILITIES",
        "ADDITIVE_GAIN_LAMBDA",
        "MULTIPLICATIVE_FACTOR_PROBABILITIES",
        "WGD_PROBABILITY",
        "REPRESENTATION_TYPE",
        "STATE_LINEAGE_REGULATION",
        "RESOURCE_GUARD",
        "TELOMERIC_INSTABILITY_ENABLED",
        "TELOMERIC_INSTABILITY_INCREMENT",
        "TELOMERIC_FRACTION",
        "CRUCIAL_SURVIVAL_ENABLED",
    }
)

_COMMON_REQUIRED_CONFIG_KEYS = _CONFIG_KEYS - {
    "GENOME_LENGTH",
    "INITIAL_COPY_NUMBER",
}

_BED_HEADERS = frozenset({"ChromosomeNumber", "Start", "End", "Parameters"})
_BED_PARAMETER_KEYS = frozenset(
    {
        "CN",
        "CNA_EVENT_PROBABILITY",
        "GAIN_GIVEN_CNA_PROBABILITY",
        "INTERVAL_CNA_PROBABILITY",
        "INTERVAL_GAIN_OPERATOR_PROBABILITIES",
        "ADDITIVE_GAIN_LAMBDA",
        "MULTIPLICATIVE_FACTOR_PROBABILITIES",
        "TELOMERIC_INSTABILITY",
        "CRUCIAL",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Simulator configuration must contain JSON values only.") from exc


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown!r}.")
    if missing:
        raise ValueError(f"Missing required {context} keys: {missing!r}.")


def _as_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def _as_int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _as_float(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _as_probability(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")
    return result


@dataclass(frozen=True)
class DiscreteDistribution:
    """An immutable, validated finite probability distribution."""

    values: Tuple[Union[str, int], ...]
    probabilities: Tuple[float, ...]

    def sample(self, rng: Any) -> Union[str, int]:
        index = int(rng.choice(len(self.values), p=self.probabilities))
        return self.values[index]

    def select(self, unit_interval_value: float) -> Union[str, int]:
        if unit_interval_value < 0.0 or unit_interval_value >= 1.0:
            raise ValueError("Distribution selection value must satisfy 0 <= value < 1.")
        cumulative = 0.0
        for value, probability in zip(self.values, self.probabilities):
            cumulative += probability
            if unit_interval_value < cumulative:
                return value
        return self.values[-1]

    def as_dict(self) -> dict[Union[str, int], float]:
        return dict(zip(self.values, self.probabilities))


def _probability_distribution(
    value: Any,
    name: str,
    *,
    exact_string_keys: Optional[Sequence[str]] = None,
    integer_keys: bool = False,
) -> DiscreteDistribution:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty probability object.")

    if exact_string_keys is not None:
        expected = set(exact_string_keys)
        actual = set(value)
        if actual != expected:
            raise ValueError(
                f"{name} must contain exactly {sorted(expected)!r}; got {sorted(actual)!r}."
            )

    parsed = []
    for raw_key, raw_probability in value.items():
        if integer_keys:
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} keys must be integer factors.") from exc
            if key < 2:
                raise ValueError(f"{name} factors must be >= 2.")
        else:
            key = str(raw_key)
        parsed.append((key, _as_probability(raw_probability, f"{name}[{raw_key!r}]")))

    parsed.sort(key=lambda item: item[0])
    total = sum(probability for _, probability in parsed)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} probabilities must sum to 1; got {total!r}.")

    return DiscreteDistribution(
        values=tuple(key for key, _ in parsed),
        probabilities=tuple(probability for _, probability in parsed),
    )


@dataclass(frozen=True)
class StateLineageRegulationConfig:
    """Validated neutral regulation of representative-state continuation."""

    model: str
    capacity: Optional[int]
    steepness: Optional[float]


@dataclass(frozen=True)
class ResourceGuardConfig:
    """Validated abort-only operational resource limits."""

    max_representatives_per_generation: int
    max_total_nodes: int


@dataclass(frozen=True)
class CnaInitiationScheduleConfig:
    """Validated generation-dependent multiplier for segmental CNA starts."""

    model: str
    initial_multiplier: float
    final_multiplier: float
    decay_exponent: Optional[float]


@dataclass(frozen=True)
class SimulatorConfig:
    semantic_version: str
    genome_length: Optional[int]
    initial_copy_number: Optional[int]
    number_of_generations: int
    offspring_model: str
    offspring_parameter: Union[int, float, Tuple[int, int]]
    baseline_descendant_attempts: int
    cna_event_probability: float
    cna_initiation_schedule: CnaInitiationScheduleConfig
    gain_given_cna_probability: float
    interval_cna_probability: float
    interval_gain_operators: DiscreteDistribution
    additive_gain_lambda: float
    multiplicative_factors: DiscreteDistribution
    wgd_probability: float
    representation_type: str
    state_lineage_regulation: StateLineageRegulationConfig
    resource_guard: ResourceGuardConfig
    telomeric_instability_enabled: bool
    telomeric_instability_increment: float
    telomeric_fraction: float
    crucial_survival_enabled: bool


@dataclass(frozen=True)
class GenomeBin:
    chromosome: str
    start: int
    end: int
    initial_copy_number: int
    cna_event_probability: float
    gain_given_cna_probability: float
    interval_cna_probability: float
    interval_gain_operators: DiscreteDistribution
    additive_gain_lambda: float
    multiplicative_factors: DiscreteDistribution
    telomeric_instability: float
    crucial: bool


@dataclass(frozen=True)
class _UnresolvedGenomeBin:
    chromosome: str
    start: int
    end: int
    initial_copy_number: int
    cna_event_probability: Optional[float]
    gain_given_cna_probability: Optional[float]
    interval_cna_probability: Optional[float]
    interval_gain_operators: Optional[DiscreteDistribution]
    additive_gain_lambda: Optional[float]
    multiplicative_factors: Optional[DiscreteDistribution]
    telomeric_instability: Optional[float]
    crucial: bool


@dataclass(frozen=True)
class LoadedSimulatorInputs:
    config: SimulatorConfig
    genome_bins: Tuple[GenomeBin, ...]
    config_sha256: str
    bed_sha256: Optional[str]
    config_source: str
    bed_source: Optional[str]


def _offspring_parameter(model: str, raw_value: Any) -> Union[int, float, Tuple[int, int]]:
    if model == "constant":
        return _as_int(raw_value, "OFFSPRING_PARAMETER", minimum=0)
    if model == "poisson":
        return _as_float(raw_value, "OFFSPRING_PARAMETER", minimum=0.0)
    if model == "uniform_range":
        if not isinstance(raw_value, list) or len(raw_value) != 2:
            raise ValueError(
                "OFFSPRING_PARAMETER must be [minimum, maximum] for uniform_range."
            )
        low = _as_int(raw_value[0], "OFFSPRING_PARAMETER[0]", minimum=0)
        high = _as_int(raw_value[1], "OFFSPRING_PARAMETER[1]", minimum=0)
        if low > high:
            raise ValueError("OFFSPRING_PARAMETER minimum must not exceed maximum.")
        return (low, high)
    raise ValueError(
        "OFFSPRING_MODEL must be one of 'constant', 'poisson', or 'uniform_range'; "
        "fitness is not part of CTBF v3."
    )


def _state_lineage_regulation(
    raw_value: Any,
) -> StateLineageRegulationConfig:
    if not isinstance(raw_value, Mapping):
        raise ValueError("STATE_LINEAGE_REGULATION must be a JSON object.")

    model = raw_value.get("MODEL")
    if model == "none":
        _require_exact_keys(
            raw_value,
            allowed=frozenset({"MODEL"}),
            required=frozenset({"MODEL"}),
            context="STATE_LINEAGE_REGULATION",
        )
        return StateLineageRegulationConfig(
            model="none",
            capacity=None,
            steepness=None,
        )

    if model == "soft_capacity":
        keys = frozenset({"MODEL", "CAPACITY", "STEEPNESS"})
        _require_exact_keys(
            raw_value,
            allowed=keys,
            required=keys,
            context="STATE_LINEAGE_REGULATION",
        )
        capacity = _as_int(
            raw_value["CAPACITY"],
            "STATE_LINEAGE_REGULATION.CAPACITY",
            minimum=1,
        )
        steepness = _as_float(
            raw_value["STEEPNESS"],
            "STATE_LINEAGE_REGULATION.STEEPNESS",
        )
        if steepness <= 0.0:
            raise ValueError("STATE_LINEAGE_REGULATION.STEEPNESS must be > 0.")
        return StateLineageRegulationConfig(
            model="soft_capacity",
            capacity=capacity,
            steepness=steepness,
        )

    raise ValueError(
        "STATE_LINEAGE_REGULATION.MODEL must be 'none' or 'soft_capacity'."
    )


def _resource_guard(raw_value: Any) -> ResourceGuardConfig:
    if not isinstance(raw_value, Mapping):
        raise ValueError("RESOURCE_GUARD must be a JSON object.")

    keys = frozenset(
        {
            "MAX_REPRESENTATIVES_PER_GENERATION",
            "MAX_TOTAL_NODES",
        }
    )
    _require_exact_keys(
        raw_value,
        allowed=keys,
        required=keys,
        context="RESOURCE_GUARD",
    )
    return ResourceGuardConfig(
        max_representatives_per_generation=_as_int(
            raw_value["MAX_REPRESENTATIVES_PER_GENERATION"],
            "RESOURCE_GUARD.MAX_REPRESENTATIVES_PER_GENERATION",
            minimum=1,
        ),
        max_total_nodes=_as_int(
            raw_value["MAX_TOTAL_NODES"],
            "RESOURCE_GUARD.MAX_TOTAL_NODES",
            minimum=1,
        ),
    )


def _cna_initiation_schedule(raw_value: Any) -> CnaInitiationScheduleConfig:
    if not isinstance(raw_value, Mapping):
        raise ValueError("CNA_INITIATION_SCHEDULE must be a JSON object.")

    model = raw_value.get("MODEL")
    if model == "constant":
        _require_exact_keys(
            raw_value,
            allowed=frozenset({"MODEL"}),
            required=frozenset({"MODEL"}),
            context="CNA_INITIATION_SCHEDULE",
        )
        return CnaInitiationScheduleConfig(
            model="constant",
            initial_multiplier=1.0,
            final_multiplier=1.0,
            decay_exponent=None,
        )

    if model == "early_burst_decay":
        keys = frozenset(
            {
                "MODEL",
                "INITIAL_MULTIPLIER",
                "FINAL_MULTIPLIER",
                "DECAY_EXPONENT",
            }
        )
        _require_exact_keys(
            raw_value,
            allowed=keys,
            required=keys,
            context="CNA_INITIATION_SCHEDULE",
        )
        initial_multiplier = _as_float(
            raw_value["INITIAL_MULTIPLIER"],
            "CNA_INITIATION_SCHEDULE.INITIAL_MULTIPLIER",
        )
        final_multiplier = _as_float(
            raw_value["FINAL_MULTIPLIER"],
            "CNA_INITIATION_SCHEDULE.FINAL_MULTIPLIER",
        )
        decay_exponent = _as_float(
            raw_value["DECAY_EXPONENT"],
            "CNA_INITIATION_SCHEDULE.DECAY_EXPONENT",
        )
        if final_multiplier <= 0.0:
            raise ValueError(
                "CNA_INITIATION_SCHEDULE.FINAL_MULTIPLIER must be > 0."
            )
        if initial_multiplier <= final_multiplier:
            raise ValueError(
                "CNA_INITIATION_SCHEDULE.INITIAL_MULTIPLIER must be greater "
                "than FINAL_MULTIPLIER."
            )
        if decay_exponent <= 0.0:
            raise ValueError(
                "CNA_INITIATION_SCHEDULE.DECAY_EXPONENT must be > 0."
            )
        return CnaInitiationScheduleConfig(
            model="early_burst_decay",
            initial_multiplier=initial_multiplier,
            final_multiplier=final_multiplier,
            decay_exponent=decay_exponent,
        )

    raise ValueError(
        "CNA_INITIATION_SCHEDULE.MODEL must be 'constant' or "
        "'early_burst_decay'."
    )


def cna_initiation_schedule_at_generation(
    schedule: CnaInitiationScheduleConfig,
    *,
    generation: int,
    number_of_generations: int,
) -> Tuple[float, float]:
    """Return ``(normalized_time, multiplier)`` for one generated generation."""
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("generation must be an integer.")
    if isinstance(number_of_generations, bool) or not isinstance(
        number_of_generations, int
    ):
        raise ValueError("number_of_generations must be an integer.")
    if number_of_generations < 1:
        raise ValueError("number_of_generations must be >= 1.")
    if generation < 1 or generation > number_of_generations:
        raise ValueError(
            "generation must be between 1 and number_of_generations inclusive."
        )

    normalized_time = (
        0.0
        if number_of_generations == 1
        else (generation - 1) / (number_of_generations - 1)
    )
    if schedule.model == "constant":
        return float(normalized_time), 1.0
    if schedule.model != "early_burst_decay":  # pragma: no cover - parser invariant
        raise ValueError(f"Unsupported CNA initiation schedule {schedule.model!r}.")
    if schedule.decay_exponent is None:  # pragma: no cover - parser invariant
        raise ValueError("Early-burst schedule decay exponent is missing.")

    multiplier = schedule.final_multiplier + (
        schedule.initial_multiplier - schedule.final_multiplier
    ) * (1.0 - normalized_time) ** schedule.decay_exponent
    return float(normalized_time), float(multiplier)


def parse_simulator_config(
    value: Mapping[str, Any],
    *,
    has_bed: bool,
) -> SimulatorConfig:
    required = _COMMON_REQUIRED_CONFIG_KEYS
    if not has_bed:
        required = required | {"GENOME_LENGTH", "INITIAL_COPY_NUMBER"}
    _require_exact_keys(
        value,
        allowed=_CONFIG_KEYS,
        required=frozenset(required),
        context="simulator configuration",
    )

    if has_bed and ({"GENOME_LENGTH", "INITIAL_COPY_NUMBER"} & set(value)):
        raise ValueError(
            "GENOME_LENGTH and INITIAL_COPY_NUMBER must be omitted when BED-like input is used."
        )

    semantic_version = value["SIMULATOR_SEMANTIC_VERSION"]
    if semantic_version != SIMULATOR_SEMANTIC_VERSION:
        raise ValueError(
            "SIMULATOR_SEMANTIC_VERSION must be "
            f"{SIMULATOR_SEMANTIC_VERSION!r}; got {semantic_version!r}."
        )

    offspring_model = value["OFFSPRING_MODEL"]
    if not isinstance(offspring_model, str):
        raise ValueError("OFFSPRING_MODEL must be a string.")
    representation_type = value["REPRESENTATION_TYPE"]
    if representation_type not in {"full", "representative"}:
        raise ValueError("REPRESENTATION_TYPE must be 'full' or 'representative'.")

    state_lineage_regulation = _state_lineage_regulation(
        value["STATE_LINEAGE_REGULATION"]
    )
    if (
        state_lineage_regulation.model == "soft_capacity"
        and representation_type != "representative"
    ):
        raise ValueError(
            "STATE_LINEAGE_REGULATION soft_capacity requires "
            "REPRESENTATION_TYPE='representative'."
        )

    baseline_attempts = _as_int(
        value["BASELINE_DESCENDANT_ATTEMPTS"],
        "BASELINE_DESCENDANT_ATTEMPTS",
        minimum=0,
    )
    if baseline_attempts != 1:
        raise ValueError(
            "CTBF v3 fixes BASELINE_DESCENDANT_ATTEMPTS at 1."
        )

    return SimulatorConfig(
        semantic_version=semantic_version,
        genome_length=(
            None
            if has_bed
            else _as_int(value["GENOME_LENGTH"], "GENOME_LENGTH", minimum=1)
        ),
        initial_copy_number=(
            None
            if has_bed
            else _as_int(
                value["INITIAL_COPY_NUMBER"],
                "INITIAL_COPY_NUMBER",
                minimum=0,
            )
        ),
        number_of_generations=_as_int(
            value["NUMBER_OF_GENERATIONS"],
            "NUMBER_OF_GENERATIONS",
            minimum=0,
        ),
        offspring_model=offspring_model,
        offspring_parameter=_offspring_parameter(
            offspring_model,
            value["OFFSPRING_PARAMETER"],
        ),
        baseline_descendant_attempts=baseline_attempts,
        cna_event_probability=_as_probability(
            value["CNA_EVENT_PROBABILITY"],
            "CNA_EVENT_PROBABILITY",
        ),
        cna_initiation_schedule=_cna_initiation_schedule(
            value["CNA_INITIATION_SCHEDULE"]
        ),
        gain_given_cna_probability=_as_probability(
            value["GAIN_GIVEN_CNA_PROBABILITY"],
            "GAIN_GIVEN_CNA_PROBABILITY",
        ),
        interval_cna_probability=_as_probability(
            value["INTERVAL_CNA_PROBABILITY"],
            "INTERVAL_CNA_PROBABILITY",
        ),
        interval_gain_operators=_probability_distribution(
            value["INTERVAL_GAIN_OPERATOR_PROBABILITIES"],
            "INTERVAL_GAIN_OPERATOR_PROBABILITIES",
            exact_string_keys=("unit", "additive", "multiplicative"),
        ),
        additive_gain_lambda=_as_float(
            value["ADDITIVE_GAIN_LAMBDA"],
            "ADDITIVE_GAIN_LAMBDA",
            minimum=0.0,
        ),
        multiplicative_factors=_probability_distribution(
            value["MULTIPLICATIVE_FACTOR_PROBABILITIES"],
            "MULTIPLICATIVE_FACTOR_PROBABILITIES",
            integer_keys=True,
        ),
        wgd_probability=_as_probability(value["WGD_PROBABILITY"], "WGD_PROBABILITY"),
        representation_type=representation_type,
        state_lineage_regulation=state_lineage_regulation,
        resource_guard=_resource_guard(value["RESOURCE_GUARD"]),
        telomeric_instability_enabled=_as_bool(
            value["TELOMERIC_INSTABILITY_ENABLED"],
            "TELOMERIC_INSTABILITY_ENABLED",
        ),
        telomeric_instability_increment=_as_probability(
            value["TELOMERIC_INSTABILITY_INCREMENT"],
            "TELOMERIC_INSTABILITY_INCREMENT",
        ),
        telomeric_fraction=_as_probability(
            value["TELOMERIC_FRACTION"],
            "TELOMERIC_FRACTION",
        ),
        crucial_survival_enabled=_as_bool(
            value["CRUCIAL_SURVIVAL_ENABLED"],
            "CRUCIAL_SURVIVAL_ENABLED",
        ),
    )


def _parse_probability_text(value: str, name: str) -> DiscreteDistribution:
    entries: dict[str, float] = {}
    for part in value.split(","):
        key, separator, raw_probability = part.partition(":")
        if not separator or not key.strip() or not raw_probability.strip():
            raise ValueError(f"{name} must use key:probability comma-separated entries.")
        key = key.strip()
        if key in entries:
            raise ValueError(f"Duplicate {name} key {key!r}.")
        try:
            entries[key] = float(raw_probability)
        except ValueError as exc:
            raise ValueError(f"Invalid probability in {name}: {raw_probability!r}.") from exc
    return _probability_distribution(
        entries,
        name,
        exact_string_keys=("unit", "additive", "multiplicative")
        if name == "INTERVAL_GAIN_OPERATOR_PROBABILITIES"
        else None,
        integer_keys=name == "MULTIPLICATIVE_FACTOR_PROBABILITIES",
    )


def _parse_bed_parameters(raw: str, row_number: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(
                f"BED-like row {row_number} parameters must use KEY=value entries."
            )
        if key not in _BED_PARAMETER_KEYS:
            raise ValueError(f"Unknown BED-like parameter {key!r} on row {row_number}.")
        if key in result:
            raise ValueError(f"Duplicate BED-like parameter {key!r} on row {row_number}.")
        result[key] = value
    if "CN" not in result:
        raise ValueError(f"BED-like row {row_number} must define CN.")
    return result


def _optional_probability(params: Mapping[str, str], key: str, row_number: int) -> Optional[float]:
    if key not in params:
        return None
    try:
        raw_value = float(params[key])
    except ValueError as exc:
        raise ValueError(f"BED-like row {row_number} {key} must be numeric.") from exc
    return _as_probability(raw_value, f"BED-like row {row_number} {key}")


def _optional_nonnegative_float(
    params: Mapping[str, str],
    key: str,
    row_number: int,
) -> Optional[float]:
    if key not in params:
        return None
    try:
        raw_value = float(params[key])
    except ValueError as exc:
        raise ValueError(f"BED-like row {row_number} {key} must be numeric.") from exc
    return _as_float(raw_value, f"BED-like row {row_number} {key}", minimum=0.0)


def _parse_bed_like(data: bytes, source: str) -> Tuple[_UnresolvedGenomeBin, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"BED-like input {source!r} must be UTF-8.") from exc

    reader = csv.DictReader(text.splitlines())
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(_BED_HEADERS)
        or set(reader.fieldnames) != _BED_HEADERS
    ):
        raise ValueError(
            "BED-like input must contain exactly ChromosomeNumber, Start, End, Parameters."
        )

    bins = []
    seen_chromosomes = set()
    current_chromosome = None
    previous_end = None
    for row_number, row in enumerate(reader, start=2):
        chromosome = str(row["ChromosomeNumber"]).strip()
        if not chromosome:
            raise ValueError(f"BED-like row {row_number} has an empty chromosome.")
        try:
            start = int(row["Start"])
            end = int(row["End"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BED-like row {row_number} Start and End must be integers."
            ) from exc
        if start < 0 or end <= start:
            raise ValueError(
                f"BED-like row {row_number} must satisfy 0 <= Start < End."
            )

        if chromosome != current_chromosome:
            if chromosome in seen_chromosomes:
                raise ValueError(
                    f"BED-like chromosome {chromosome!r} appears in multiple blocks."
                )
            seen_chromosomes.add(chromosome)
            current_chromosome = chromosome
            previous_end = None
        if previous_end is not None and start < previous_end:
            raise ValueError(
                f"BED-like row {row_number} is unsorted or overlaps the previous interval."
            )
        previous_end = end

        params = _parse_bed_parameters(row["Parameters"] or "", row_number)
        try:
            initial_copy_number = int(params["CN"])
        except ValueError as exc:
            raise ValueError(f"BED-like row {row_number} CN must be an integer.") from exc
        if initial_copy_number < 0:
            raise ValueError(f"BED-like row {row_number} CN must be >= 0.")

        crucial_raw = params.get("CRUCIAL", "0")
        if crucial_raw not in {"0", "1"}:
            raise ValueError(f"BED-like row {row_number} CRUCIAL must be 0 or 1.")

        bins.append(
            _UnresolvedGenomeBin(
                chromosome=chromosome,
                start=start,
                end=end,
                initial_copy_number=initial_copy_number,
                cna_event_probability=_optional_probability(
                    params, "CNA_EVENT_PROBABILITY", row_number
                ),
                gain_given_cna_probability=_optional_probability(
                    params, "GAIN_GIVEN_CNA_PROBABILITY", row_number
                ),
                interval_cna_probability=_optional_probability(
                    params, "INTERVAL_CNA_PROBABILITY", row_number
                ),
                interval_gain_operators=(
                    None
                    if "INTERVAL_GAIN_OPERATOR_PROBABILITIES" not in params
                    else _parse_probability_text(
                        params["INTERVAL_GAIN_OPERATOR_PROBABILITIES"],
                        "INTERVAL_GAIN_OPERATOR_PROBABILITIES",
                    )
                ),
                additive_gain_lambda=_optional_nonnegative_float(
                    params, "ADDITIVE_GAIN_LAMBDA", row_number
                ),
                multiplicative_factors=(
                    None
                    if "MULTIPLICATIVE_FACTOR_PROBABILITIES" not in params
                    else _parse_probability_text(
                        params["MULTIPLICATIVE_FACTOR_PROBABILITIES"],
                        "MULTIPLICATIVE_FACTOR_PROBABILITIES",
                    )
                ),
                telomeric_instability=_optional_probability(
                    params, "TELOMERIC_INSTABILITY", row_number
                ),
                crucial=crucial_raw == "1",
            )
        )

    if not bins:
        raise ValueError("BED-like input must contain at least one data row.")
    return tuple(bins)


def _resolve_bins(
    config: SimulatorConfig,
    unresolved: Sequence[_UnresolvedGenomeBin],
) -> Tuple[GenomeBin, ...]:
    telomeric_indices = set()
    by_chromosome: dict[str, list[int]] = {}
    for index, genome_bin in enumerate(unresolved):
        by_chromosome.setdefault(genome_bin.chromosome, []).append(index)

    if config.telomeric_instability_enabled:
        for indices in by_chromosome.values():
            telomeric_size = int(len(indices) * config.telomeric_fraction)
            if telomeric_size > 0:
                telomeric_indices.update(indices[:telomeric_size])
                telomeric_indices.update(indices[-telomeric_size:])

    resolved = []
    for index, genome_bin in enumerate(unresolved):
        if not config.telomeric_instability_enabled and genome_bin.telomeric_instability is not None:
            raise ValueError(
                "BED-like TELOMERIC_INSTABILITY requires TELOMERIC_INSTABILITY_ENABLED=true."
            )
        telomeric_instability = (
            genome_bin.telomeric_instability
            if genome_bin.telomeric_instability is not None
            else (
                config.telomeric_instability_increment
                if index in telomeric_indices
                else 0.0
            )
        )
        if (
            config.crucial_survival_enabled
            and genome_bin.crucial
            and genome_bin.initial_copy_number == 0
        ):
            raise ValueError("A crucial BED-like row cannot start at copy number zero.")
        resolved.append(
            GenomeBin(
                chromosome=genome_bin.chromosome,
                start=genome_bin.start,
                end=genome_bin.end,
                initial_copy_number=genome_bin.initial_copy_number,
                cna_event_probability=(
                    genome_bin.cna_event_probability
                    if genome_bin.cna_event_probability is not None
                    else config.cna_event_probability
                ),
                gain_given_cna_probability=(
                    genome_bin.gain_given_cna_probability
                    if genome_bin.gain_given_cna_probability is not None
                    else config.gain_given_cna_probability
                ),
                interval_cna_probability=(
                    genome_bin.interval_cna_probability
                    if genome_bin.interval_cna_probability is not None
                    else config.interval_cna_probability
                ),
                interval_gain_operators=(
                    genome_bin.interval_gain_operators
                    if genome_bin.interval_gain_operators is not None
                    else config.interval_gain_operators
                ),
                additive_gain_lambda=(
                    genome_bin.additive_gain_lambda
                    if genome_bin.additive_gain_lambda is not None
                    else config.additive_gain_lambda
                ),
                multiplicative_factors=(
                    genome_bin.multiplicative_factors
                    if genome_bin.multiplicative_factors is not None
                    else config.multiplicative_factors
                ),
                telomeric_instability=telomeric_instability,
                crucial=genome_bin.crucial,
            )
        )
    return tuple(resolved)


def _default_unresolved_bins(config: SimulatorConfig) -> Tuple[_UnresolvedGenomeBin, ...]:
    assert config.genome_length is not None
    assert config.initial_copy_number is not None
    return tuple(
        _UnresolvedGenomeBin(
            chromosome="1",
            start=index,
            end=index + 1,
            initial_copy_number=config.initial_copy_number,
            cna_event_probability=None,
            gain_given_cna_probability=None,
            interval_cna_probability=None,
            interval_gain_operators=None,
            additive_gain_lambda=None,
            multiplicative_factors=None,
            telomeric_instability=None,
            crucial=False,
        )
        for index in range(config.genome_length)
    )


def load_simulator_inputs(
    config_source: Union[str, Path, Mapping[str, Any]],
    bed_source: Optional[Union[str, Path]] = None,
) -> LoadedSimulatorInputs:
    if isinstance(config_source, Mapping):
        config_mapping = dict(config_source)
        config_bytes = _canonical_json_bytes(config_mapping)
        config_label = "<mapping>"
    else:
        config_path = Path(config_source)
        config_bytes = config_path.read_bytes()
        config_label = str(config_path)
        try:
            config_mapping = json.loads(config_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid simulator JSON configuration {config_label!r}.") from exc
        if not isinstance(config_mapping, Mapping):
            raise ValueError("Simulator JSON configuration must be an object.")

    config = parse_simulator_config(config_mapping, has_bed=bed_source is not None)

    if bed_source is None:
        bed_bytes = None
        bed_label = None
        unresolved = _default_unresolved_bins(config)
    else:
        bed_path = Path(bed_source)
        bed_bytes = bed_path.read_bytes()
        bed_label = str(bed_path)
        unresolved = _parse_bed_like(bed_bytes, bed_label)

    bins = _resolve_bins(config, unresolved)
    return LoadedSimulatorInputs(
        config=config,
        genome_bins=bins,
        config_sha256=_sha256_bytes(config_bytes),
        bed_sha256=None if bed_bytes is None else _sha256_bytes(bed_bytes),
        config_source=config_label,
        bed_source=bed_label,
    )


def choose_crucial_mask(
    number_of_positions: int,
    fraction: float,
    seed: int,
) -> Tuple[int, ...]:
    """Select an exact, reproducible fraction without replacement for tests."""
    if number_of_positions <= 0:
        raise ValueError("number_of_positions must be positive.")
    fraction = _as_probability(fraction, "fraction")
    requested = number_of_positions * fraction
    if not requested.is_integer():
        raise ValueError(
            "number_of_positions * fraction must be an integer; choose a test size "
            "that needs no rounding rule."
        )
    count = int(requested)
    import numpy as np

    rng = np.random.default_rng(seed)
    return tuple(sorted(int(value) for value in rng.choice(number_of_positions, count, replace=False)))
