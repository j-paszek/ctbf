"""Fresh-process execution for resource-audited experimental tasks.

The resource contract measures absolute process-tree RSS.  A worker must
therefore execute only one declared task; otherwise allocator high-water marks
and reachable caches from an earlier task contaminate every later record.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing.pool import Pool
from typing import Any, Callable, Mapping


EXECUTION_ISOLATION_SCHEMA_VERSION = "ctbf-v5-fresh-process-execution-v1"
FRESH_SPAWN_START_METHOD = "spawn"
CASE_ARM_WORKER_UNIT = "case_arm_reconstruction_and_evaluation"
TRUTH_BLOCK_SIMULATION_WORKER_UNIT = "truth_block_simulation"
CONDITION_DISTANCE_WORKER_UNIT = "condition_distance_computation"


class FreshProcessTimeoutError(TimeoutError):
    """Raised when a fresh worker does not return within its outer deadline."""


def fresh_process_contract(worker_unit: str) -> dict[str, Any]:
    if not worker_unit or not worker_unit.strip():
        raise ValueError("Fresh-process worker_unit must be nonempty.")
    return {
        "schema_version": EXECUTION_ISOLATION_SCHEMA_VERSION,
        "isolation": "fresh_spawn_process_per_task",
        "start_method": FRESH_SPAWN_START_METHOD,
        "worker_unit": worker_unit,
        "max_tasks_per_child": 1,
        "absolute_worker_process_tree_rss": True,
        "parent_process_rss_excluded": True,
        "previous_task_state_reclaimed_by_process_exit": True,
    }


def validate_fresh_process_contract(
    value: Mapping[str, Any] | None,
    *,
    worker_unit: str,
) -> None:
    expected = fresh_process_contract(worker_unit)
    if value != expected:
        raise ValueError(
            f"Execution resources lack the required fresh-process {worker_unit} "
            "contract."
        )


class FreshSpawnPerTaskExecutor:
    """Sequential task executor backed by a new spawned process per task."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context(FRESH_SPAWN_START_METHOD)
        self._pool: Pool | None = None

    def _start_pool(self) -> Pool:
        if self._pool is None:
            self._pool = self._context.Pool(processes=1, maxtasksperchild=1)
        return self._pool

    def _terminate_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.terminate()
            pool.join()

    def run(
        self,
        function: Callable[..., Any],
        *args: Any,
        timeout_seconds: float | int | None = None,
    ) -> Any:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Fresh-process timeout_seconds must be positive.")
        pool = self._start_pool()
        pending = pool.apply_async(function, args)
        try:
            return pending.get(timeout=timeout_seconds)
        except multiprocessing.TimeoutError as error:
            self._terminate_pool()
            raise FreshProcessTimeoutError(
                "Fresh spawned task exceeded its outer "
                f"{timeout_seconds}-second deadline."
            ) from error
        except BaseException as error:
            if not isinstance(error, Exception):
                self._terminate_pool()
            raise

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
            pool.join()

    def __enter__(self) -> "FreshSpawnPerTaskExecutor":
        self._start_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._terminate_pool()


__all__ = [
    "CASE_ARM_WORKER_UNIT",
    "CONDITION_DISTANCE_WORKER_UNIT",
    "EXECUTION_ISOLATION_SCHEMA_VERSION",
    "FRESH_SPAWN_START_METHOD",
    "FreshProcessTimeoutError",
    "FreshSpawnPerTaskExecutor",
    "TRUTH_BLOCK_SIMULATION_WORKER_UNIT",
    "fresh_process_contract",
    "validate_fresh_process_contract",
]
