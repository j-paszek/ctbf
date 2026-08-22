"""Fresh-process execution for resource-audited experimental tasks.

The resource contract measures absolute process-tree RSS.  A worker must
therefore execute only one declared task; otherwise allocator high-water marks
and reachable caches from an earlier task contaminate every later record.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing.pool import Pool
import queue
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


EXECUTION_ISOLATION_SCHEMA_VERSION = "ctbf-v5-fresh-process-execution-v1"
FRESH_SPAWN_START_METHOD = "spawn"
CASE_ARM_WORKER_UNIT = "case_arm_reconstruction_and_evaluation"
TRUTH_BLOCK_SIMULATION_WORKER_UNIT = "truth_block_simulation"
CONDITION_DISTANCE_WORKER_UNIT = "condition_distance_computation"


class FreshProcessTimeoutError(TimeoutError):
    """Raised when a fresh worker does not return within its outer deadline."""


_POOL_TASK_START_QUEUE: Any = None


def _initialize_pool_task_start_queue(start_queue: Any) -> None:
    global _POOL_TASK_START_QUEUE
    _POOL_TASK_START_QUEUE = start_queue


def _execute_pool_task(
    task_index: int,
    function: Callable[..., Any],
    arguments: Sequence[Any],
) -> Any:
    if _POOL_TASK_START_QUEUE is None:
        raise RuntimeError("Fresh task-pool start queue was not initialized.")
    _POOL_TASK_START_QUEUE.put(task_index)
    return function(*arguments)


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


class FreshSpawnTaskPool:
    """Bounded parallel tasks with one fresh spawned process per task.

    Results are yielded in submission order. A small out-of-order buffer keeps
    workers occupied without allowing completed records to accumulate without
    bound behind one slow earlier task.
    """

    def __init__(self, worker_count: int, *, buffer_factor: int = 16) -> None:
        if (
            isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count <= 0
        ):
            raise ValueError("worker_count must be a positive integer.")
        if (
            isinstance(buffer_factor, bool)
            or not isinstance(buffer_factor, int)
            or buffer_factor <= 0
        ):
            raise ValueError("buffer_factor must be a positive integer.")
        self.worker_count = worker_count
        self.max_buffered_tasks = worker_count * buffer_factor
        self._context = multiprocessing.get_context(FRESH_SPAWN_START_METHOD)
        self._pool: Pool | None = None
        self._start_queue: Any = None

    def _start_pool(self) -> Pool:
        if self._pool is None:
            start_queue = self._context.Queue()
            try:
                self._pool = self._context.Pool(
                    processes=self.worker_count,
                    maxtasksperchild=1,
                    initializer=_initialize_pool_task_start_queue,
                    initargs=(start_queue,),
                )
            except BaseException:
                start_queue.close()
                raise
            self._start_queue = start_queue
        return self._pool

    def _close_start_queue(self) -> None:
        start_queue, self._start_queue = self._start_queue, None
        if start_queue is not None:
            start_queue.close()
            start_queue.join_thread()

    def _terminate_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.terminate()
            pool.join()
        self._close_start_queue()

    def map_ordered(
        self,
        function: Callable[..., Any],
        argument_rows: Iterable[Sequence[Any]],
        *,
        timeout_seconds: float | int | None = None,
    ) -> Iterator[Any]:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Fresh-process timeout_seconds must be positive.")
        pool = self._start_pool()
        arguments = iter(argument_rows)
        pending: dict[int, tuple[Any, float | None]] = {}
        ready: dict[int, Any] = {}
        next_submit_index = 0
        next_yield_index = 0
        exhausted = False

        def submit_to_capacity() -> None:
            nonlocal exhausted, next_submit_index
            while (
                not exhausted
                and len(pending) + len(ready) < self.max_buffered_tasks
            ):
                try:
                    row = next(arguments)
                except StopIteration:
                    exhausted = True
                    return
                pending[next_submit_index] = (
                    pool.apply_async(
                        _execute_pool_task,
                        (next_submit_index, function, tuple(row)),
                    ),
                    None,
                )
                next_submit_index += 1

        def register_started_tasks() -> None:
            if self._start_queue is None:
                return
            while True:
                try:
                    task_index = self._start_queue.get_nowait()
                except queue.Empty:
                    return
                if task_index not in pending:
                    continue
                outcome, started_deadline = pending[task_index]
                if started_deadline is not None:
                    raise RuntimeError("A fresh task reported multiple starts.")
                pending[task_index] = (
                    outcome,
                    None
                    if timeout_seconds is None
                    else time.monotonic() + float(timeout_seconds),
                )

        submit_to_capacity()
        while pending or ready or not exhausted:
            made_progress = False
            register_started_tasks()
            for task_index, (outcome, _deadline) in tuple(pending.items()):
                if not outcome.ready():
                    continue
                ready[task_index] = outcome.get()
                del pending[task_index]
                made_progress = True

            while next_yield_index in ready:
                value = ready.pop(next_yield_index)
                next_yield_index += 1
                made_progress = True
                yield value

            submit_to_capacity()
            if not pending and not ready and exhausted:
                return

            now = time.monotonic()
            expired = [
                task_index
                for task_index, (_outcome, deadline) in pending.items()
                if deadline is not None and deadline <= now
            ]
            if expired:
                self._terminate_pool()
                raise FreshProcessTimeoutError(
                    "Fresh spawned task exceeded its outer "
                    f"{timeout_seconds}-second deadline."
                )
            if not made_progress:
                deadlines = [
                    deadline
                    for _outcome, deadline in pending.values()
                    if deadline is not None
                ]
                wait_seconds = 0.01
                if deadlines:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.0, min(deadlines) - now),
                    )
                time.sleep(wait_seconds)

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
            pool.join()
        self._close_start_queue()

    def __enter__(self) -> "FreshSpawnTaskPool":
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
    "FreshSpawnTaskPool",
    "TRUTH_BLOCK_SIMULATION_WORKER_UNIT",
    "fresh_process_contract",
    "validate_fresh_process_contract",
]
