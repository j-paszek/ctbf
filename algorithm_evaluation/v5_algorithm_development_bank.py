"""Generate the immutable CTBF v5 H14/H24/H34 development bank.

The command-line contract is intentionally fixed to 100 independent H34
simulations. Every H34 truth supplies its paired H14 and H24 prefixes. The
resulting truths, observable reconstruction inputs, cnp2cnp matrices, and seeds
can be reused while individual reconstruction algorithms are developed
manually.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.hypothesis_height_ambiguity_trend import (
    ambiguity_case_summary,
)
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _validate_production_distance,
    candidate_graph_summary,
    distance_matrix_summary,
)
from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.paper_pipeline_runner import (
    measured_stage,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    CONDITION_DISTANCE_WORKER_UNIT,
    TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    _canonical_cells_at_generation,
    _event_totals,
    _profile,
    truth_sampling_diagnostics,
)
from algorithm_evaluation.simulator_sampling_fraction_truth_probe import (
    hybrid_sample_size,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    BANK_CONFIG_NAME,
    BANK_MANIFEST_NAME,
    BANK_SCHEMA_VERSION,
    BIOPSY_LOWER_BOUND,
    CASE_INPUT_SCHEMA_VERSION,
    CASE_METADATA_SCHEMA_VERSION,
    DEFAULT_BANK_ID,
    DEFAULT_BASE_SEED,
    DEFAULT_BLOCK_COUNT,
    DEVELOPMENT_NAMESPACE,
    HEIGHT_SCHEDULES,
    INITIAL_ARM_SPECS,
    SAMPLING_RULE,
    TARGET_FRACTION,
    case_id,
    condition_paths,
    derived_seed,
    ensure_new_output_root,
    serialize_distance,
    serialize_truth,
    truth_path,
    write_json,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import stable_distance_label_key
from simulator import CancerCellEvolutionSimulator, Genotype
from simulator_config import load_simulator_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "simulator_examples" / "default.json"
DEFAULT_SIMULATION_TIMEOUT_SECONDS = 300
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 1200
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 540
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576

DistanceCompute = Callable[[Sequence[Genotype]], DistanceMatrix]
SimulatorFactory = Callable[[Mapping[str, Any], int], CancerCellEvolutionSimulator]
FRESH_WORKER_GRACE_SECONDS = 60


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _typed_error(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _resource_audit_error(
    resources: Mapping[str, Any],
    stage: str,
) -> RuntimeError | None:
    memory = resources.get("memory")
    if (
        not isinstance(memory, Mapping)
        or memory.get("peak_rss_bytes") is None
    ):
        return RuntimeError(
            f"{stage} has no auditable process-tree peak-RSS measurement."
        )
    return None


def _select_levels(
    simulator: CancerCellEvolutionSimulator,
    generations: Sequence[int],
    *,
    base_seed: int,
    block_index: int,
    height: int,
) -> tuple[list[list[Genotype]], list[dict[str, Any]]]:
    selected_levels: list[list[Genotype]] = []
    rows = []
    for generation in generations:
        available = _canonical_cells_at_generation(simulator, int(generation))
        if not available:
            raise ValueError(
                f"Required generation {generation} is empty in block {block_index + 1}."
            )
        realized = hybrid_sample_size(
            len(available),
            TARGET_FRACTION,
            BIOPSY_LOWER_BOUND,
        )
        seed = derived_seed(
            "sampling",
            base_seed,
            block_index,
            int(height),
            int(generation),
        )
        if realized == len(available):
            chosen = list(available)
            mode = "all_available"
            seed_used = False
        else:
            permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
                len(available)
            )
            chosen = [available[int(index)] for index in permutation[:realized]]
            mode = "seeded_without_replacement"
            seed_used = True
        chosen.sort(
            key=lambda cell: (
                _profile(cell),
                stable_distance_label_key(cell.cell_id),
            )
        )
        selected_levels.append(chosen)
        rows.append(
            {
                "generation": int(generation),
                "available_distinct_state_count": len(available),
                "target_fraction": TARGET_FRACTION,
                "fractional_count": int(math.ceil(TARGET_FRACTION * len(available))),
                "biopsy_lower_bound": BIOPSY_LOWER_BOUND,
                "realized_occurrence_count": len(chosen),
                "selection_mode": mode,
                "sampling_seed": int(seed),
                "sampling_seed_used": seed_used,
            }
        )
    return selected_levels, rows


def _reconstruction_input(
    case_id_value: str,
    height: int,
    generations: Sequence[int],
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    payload = {
        "schema_version": CASE_INPUT_SCHEMA_VERSION,
        "case_id": case_id_value,
        "condition_id": f"H{height}_rule_y_fraction_50_lower_6",
        "sampling_rule": SAMPLING_RULE,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": cell.cell_id,
                        "cnp": list(_profile(cell)),
                    }
                    for cell in cells
                ],
            }
            for level_index, (generation, cells) in enumerate(
                zip(generations, selected_levels)
            )
        ],
    }
    validate_reconstruction_input(payload)
    return payload


def _distance_cells(
    selected_levels: Sequence[Sequence[Genotype]],
) -> list[Genotype]:
    occurrences = [cell for level in selected_levels for cell in level]
    return sorted(
        unique_cells_by_cell_id(occurrences),
        key=lambda cell: stable_distance_label_key(cell.cell_id),
    )


def _build_case(
    *,
    simulator: CancerCellEvolutionSimulator,
    truth_tree,
    block_index: int,
    height: int,
    generations: Sequence[int],
    base_seed: int,
    distance_compute: DistanceCompute | None,
    runtime_config: CtbsRuntimeConfig | None,
    distance_timeout_seconds: int,
    rss_limit_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], DistanceMatrix]:
    case_id_value = case_id(block_index, height)
    selected_levels, sampling = _select_levels(
        simulator,
        generations,
        base_seed=base_seed,
        block_index=block_index,
        height=height,
    )
    distance_cells = _distance_cells(selected_levels)
    if not distance_cells:
        raise ValueError(f"{case_id_value} has no unique observed states.")
    if distance_compute is None:
        if runtime_config is None:
            raise RuntimeError("Production distance task has no runtime config.")
        with FreshSpawnPerTaskExecutor() as distance_executor:
            distance, distance_runtime, distance_error = distance_executor.run(
                _execute_isolated_distance_stage,
                tuple(distance_cells),
                runtime_config,
                distance_timeout_seconds,
                rss_limit_bytes,
                timeout_seconds=(
                    distance_timeout_seconds + FRESH_WORKER_GRACE_SECONDS
                ),
            )
    else:
        distance, distance_runtime, distance_error = measured_stage(
            lambda: distance_compute(distance_cells),
            timeout_seconds=distance_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
    if distance_error is None:
        distance_error = _resource_audit_error(
            distance_runtime,
            "distance computation",
        )
    if distance_error is not None:
        raise RuntimeError(
            f"Distance computation failed for {case_id_value}: {distance_error}"
        ) from distance_error
    if distance is None:
        raise RuntimeError(f"Distance computation returned no value for {case_id_value}.")
    _validate_production_distance(distance)

    input_payload = _reconstruction_input(
        case_id_value,
        height,
        generations,
        selected_levels,
    )
    reconstruction_seed = derived_seed(
        "reconstruction",
        base_seed,
        block_index,
        int(height),
    )
    occurrence_count = sum(len(level) for level in selected_levels)
    metadata = {
        "schema_version": CASE_METADATA_SCHEMA_VERSION,
        "case_id": case_id_value,
        "block_index": block_index,
        "height": int(height),
        "generations": [int(value) for value in generations],
        "sampling_rule": SAMPLING_RULE,
        "sampling": sampling,
        "reconstruction_seed": int(reconstruction_seed),
        "canonical_input_order": "profile_then_stable_state_label",
        "selected_occurrence_count": occurrence_count,
        "selected_unique_state_count": len(distance_cells),
        "repeated_state_across_biopsy_occurrence_count": (
            occurrence_count - len(distance_cells)
        ),
        "truth_sampling_diagnostics": truth_sampling_diagnostics(
            truth_tree,
            selected_levels,
        ),
        "distance_summary": distance_matrix_summary(distance),
        "distance_runtime": distance_runtime,
        "candidate_graph_diagnostics_r2_r4_r8": candidate_graph_summary(
            selected_levels,
            distance,
            radii=(2, 4, 8),
        ),
        "scale_free_and_fixed_r4_ambiguity": ambiguity_case_summary(
            selected_levels,
            generations,
            distance,
            fixed_radius=4,
        ),
    }
    return input_payload, metadata, distance


def generate_bank(
    *,
    output_root: Path | str,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    base_seed: int = DEFAULT_BASE_SEED,
    block_count: int = DEFAULT_BLOCK_COUNT,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    distance_compute: DistanceCompute | None = None,
    simulator_factory: SimulatorFactory | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
    allow_nonproduction_size: bool = False,
) -> dict[str, Any]:
    """Generate a bank; non-100 sizes exist only for focused technical tests."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    _validate_positive_integer(block_count, "block_count")
    if block_count != DEFAULT_BLOCK_COUNT and not allow_nonproduction_size:
        raise ValueError(
            f"The development-bank command requires exactly "
            f"{DEFAULT_BLOCK_COUNT} blocks."
        )
    for field, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        _validate_positive_integer(value, field)

    config_path = Path(base_config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Base simulator config is not a file: {config_path}.")
    base_config = read_json(config_path)
    _validate_standard_base_config(base_config)
    simulation_config = dict(base_config)
    simulation_config["NUMBER_OF_GENERATIONS"] = max(HEIGHT_SCHEDULES)
    load_simulator_inputs(simulation_config)

    root = ensure_new_output_root(output_root)
    write_json(root / BANK_CONFIG_NAME, simulation_config)
    started_at = created_at_utc or datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": DEFAULT_BANK_ID,
        "status": "in_progress",
        "created_at_utc": started_at,
        "scientific_role": "method_development_only_not_paper_accuracy_evidence",
        "contract_mode": (
            "production_100_block_three_height_development_bank"
            if block_count == DEFAULT_BLOCK_COUNT
            else "nonproduction_technical_smoke"
        ),
        "base_seed": int(base_seed),
        "seed_namespace": DEVELOPMENT_NAMESPACE,
        "seed_extension_contract": (
            "retain_block_indices_0_through_49_and_add_50_through_99"
            if block_count == DEFAULT_BLOCK_COUNT
            else "nonproduction_subset_of_active_seed_stream"
        ),
        "base_config_path": str(config_path),
        "base_config_sha256": _file_sha256(config_path),
        "simulator_config_path": BANK_CONFIG_NAME,
        "block_count": int(block_count),
        "condition_count": int(block_count * len(HEIGHT_SCHEDULES)),
        "height_schedules": {
            str(height): list(generations)
            for height, generations in HEIGHT_SCHEDULES.items()
        },
        "simulation_height": max(HEIGHT_SCHEDULES),
        "paired_condition_heights": sorted(HEIGHT_SCHEDULES),
        "sampling_rule": SAMPLING_RULE,
        "initial_arm_count": len(INITIAL_ARM_SPECS),
        "expected_initial_arm_status_count": (
            block_count * len(HEIGHT_SCHEDULES) * len(INITIAL_ARM_SPECS)
        ),
        "resource_contract": {
            "simulation_timeout_seconds_per_block": simulation_timeout_seconds,
            "distance_timeout_seconds_per_condition": distance_timeout_seconds,
            "rss_limit_bytes_per_stage": rss_limit_bytes,
            "simulation_execution": (
                fresh_process_contract(TRUTH_BLOCK_SIMULATION_WORKER_UNIT)
                if simulator_factory is None
                else {
                    "isolation": "inline_injected_test_double",
                    "worker_unit": TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
                }
            ),
            "distance_execution": (
                fresh_process_contract(CONDITION_DISTANCE_WORKER_UNIT)
                if distance_compute is None
                else {
                    "isolation": "inline_injected_test_double",
                    "worker_unit": CONDITION_DISTANCE_WORKER_UNIT,
                }
            ),
        },
        "cases": [],
        "failures": [],
    }
    write_json(root / BANK_MANIFEST_NAME, manifest)

    if distance_compute is None:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
        compute = None
    else:
        runtime_config = None
        compute = distance_compute
    factory = simulator_factory or _default_simulator_factory

    try:
        for block_index in range(block_count):
            simulation_seed = derived_seed(
                "simulation",
                base_seed,
                block_index,
            )
            if simulator_factory is None:
                with FreshSpawnPerTaskExecutor() as simulation_executor:
                    simulator, simulation_runtime, simulation_error = (
                        simulation_executor.run(
                            _execute_isolated_simulation_stage,
                            dict(simulation_config),
                            simulation_seed,
                            simulation_timeout_seconds,
                            rss_limit_bytes,
                            timeout_seconds=(
                                simulation_timeout_seconds
                                + FRESH_WORKER_GRACE_SECONDS
                            ),
                        )
                    )
            else:
                simulator, simulation_runtime, simulation_error = measured_stage(
                    lambda: _simulate(factory, simulation_config, simulation_seed),
                    timeout_seconds=simulation_timeout_seconds,
                    rss_limit_bytes=rss_limit_bytes,
                )
            if simulation_error is None:
                simulation_error = _resource_audit_error(
                    simulation_runtime,
                    "simulation",
                )
            if simulation_error is not None:
                raise RuntimeError(
                    f"Simulation failed in block {block_index + 1}: {simulation_error}"
                ) from simulation_error
            if simulator is None:
                raise RuntimeError(
                    f"Simulation returned no simulator in block {block_index + 1}."
                )
            truth_tree = simulator.canonicalized_tree_by_genome()
            truth_relative = truth_path(block_index)
            write_json(
                root / truth_relative,
                serialize_truth(block_index, truth_tree),
            )

            for height, generations in HEIGHT_SCHEDULES.items():
                input_payload, metadata, distance = _build_case(
                    simulator=simulator,
                    truth_tree=truth_tree,
                    block_index=block_index,
                    height=height,
                    generations=generations,
                    base_seed=base_seed,
                    distance_compute=compute,
                    runtime_config=runtime_config,
                    distance_timeout_seconds=distance_timeout_seconds,
                    rss_limit_bytes=rss_limit_bytes,
                )
                paths = condition_paths(block_index, height)
                metadata.update(
                    {
                        "simulation_seed": int(simulation_seed),
                        "simulation_runtime": simulation_runtime,
                        "simulator_event_totals": _event_totals(simulator),
                    }
                )
                write_json(root / paths["input"], input_payload)
                write_json(root / paths["distance"], serialize_distance(
                    case_id(block_index, height), distance
                ))
                write_json(root / paths["metadata"], metadata)
                manifest["cases"].append(
                    {
                        "case_id": case_id(block_index, height),
                        "block_index": block_index,
                        "height": int(height),
                        "metadata_path": paths["metadata"],
                        "input_path": paths["input"],
                        "distance_path": paths["distance"],
                        "truth_path": truth_relative,
                    }
                )
            write_json(root / BANK_MANIFEST_NAME, manifest)
            if progress:
                print(
                    f"completed block {block_index + 1}/{block_count}",
                    file=sys.stderr,
                    flush=True,
                )
            del simulator, truth_tree
            gc.collect()
    except BaseException as error:
        manifest["status"] = "failure"
        manifest["failures"].append(_typed_error(error))
        manifest["completed_condition_count"] = len(manifest["cases"])
        write_json(root / BANK_MANIFEST_NAME, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_condition_count"] = len(manifest["cases"])
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / BANK_MANIFEST_NAME, manifest)
    return manifest


def _simulate(
    factory: SimulatorFactory,
    simulation_config: Mapping[str, Any],
    simulation_seed: int,
) -> CancerCellEvolutionSimulator:
    simulator = factory(simulation_config, simulation_seed)
    simulator.run_simulation()
    return simulator


def _default_simulator_factory(
    config: Mapping[str, Any],
    seed: int,
) -> CancerCellEvolutionSimulator:
    return CancerCellEvolutionSimulator(config, seed=seed)


def _execute_isolated_simulation_stage(
    config: Mapping[str, Any],
    seed: int,
    timeout_seconds: int,
    rss_limit_bytes: int,
):
    return measured_stage(
        lambda: _simulate(_default_simulator_factory, config, seed),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def _execute_isolated_distance_stage(
    cells: Sequence[Genotype],
    runtime_config: CtbsRuntimeConfig,
    timeout_seconds: int,
    rss_limit_bytes: int,
):
    provider = Cnp2CnpFileDistanceProvider(runtime_config)
    return measured_stage(
        lambda: provider.compute(cells),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--simulation-timeout-seconds",
        type=int,
        default=DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--distance-timeout-seconds",
        type=int,
        default=DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cnp2cnp-process-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = generate_bank(
        output_root=arguments.output_root,
        base_config_path=arguments.base_config,
        base_seed=arguments.base_seed,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        progress=arguments.progress,
    )
    print(
        f"complete: {manifest['block_count']} truth blocks, "
        f"{manifest['condition_count']} H14/H24/H34 cases"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
