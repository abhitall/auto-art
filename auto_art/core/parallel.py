"""
Parallel attack execution engine.

Executes independent attacks concurrently using ProcessPool for CPU-bound
numpy attacks and ThreadPool for GPU-bound framework-specific attacks.
Supports configurable max workers and timeout per attack.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

GPU_ATTACKS = frozenset({
    "shadow_attack", "composite", "overload", "adversarial_patch",
})


@dataclass
class AttackTask:
    """A single attack task to be executed."""
    name: str
    callable: Callable[..., Dict[str, Any]]
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    use_gpu: bool = False
    timeout: float = 300.0


@dataclass
class AttackResult:
    """Result from a single attack execution."""
    name: str
    success: bool
    data: Dict[str, Any]
    duration: float
    error: Optional[str] = None


class ParallelAttackRunner:
    """Execute independent attacks concurrently.

    Uses ThreadPoolExecutor for GPU-bound attacks (Python GIL released
    inside native CUDA/framework code) and ProcessPoolExecutor for
    CPU-only numpy-based attacks.
    """

    def __init__(
        self,
        max_cpu_workers: int = 4,
        max_gpu_workers: int = 2,
        default_timeout: float = 300.0,
    ):
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_workers = max_gpu_workers
        self.default_timeout = default_timeout

    def run(self, tasks: List[AttackTask]) -> List[AttackResult]:
        """Execute all tasks, returning results in completion order."""
        gpu_tasks = [t for t in tasks if t.use_gpu]
        cpu_tasks = [t for t in tasks if not t.use_gpu]
        results: List[AttackResult] = []

        gpu_futures: Dict[Future, AttackTask] = {}
        cpu_futures: Dict[Future, AttackTask] = {}

        gpu_executor = ThreadPoolExecutor(max_workers=self.max_gpu_workers) if gpu_tasks else None
        cpu_executor = ThreadPoolExecutor(max_workers=self.max_cpu_workers) if cpu_tasks else None

        try:
            if gpu_executor:
                for task in gpu_tasks:
                    future = gpu_executor.submit(
                        self._execute_task, task,
                    )
                    gpu_futures[future] = task

            if cpu_executor:
                for task in cpu_tasks:
                    future = cpu_executor.submit(
                        self._execute_task, task,
                    )
                    cpu_futures[future] = task

            all_futures = {**gpu_futures, **cpu_futures}
            for future in as_completed(all_futures):
                task = all_futures[future]
                timeout = task.timeout or self.default_timeout
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Attack {task.name} failed: {e}")
                    results.append(AttackResult(
                        name=task.name,
                        success=False,
                        data={},
                        duration=0.0,
                        error=str(e),
                    ))
        finally:
            if gpu_executor:
                gpu_executor.shutdown(wait=False)
            if cpu_executor:
                cpu_executor.shutdown(wait=False)

        return results

    @staticmethod
    def _execute_task(task: AttackTask) -> AttackResult:
        start = time.time()
        try:
            data = task.callable(*task.args, **task.kwargs)
            return AttackResult(
                name=task.name,
                success=True,
                data=data if isinstance(data, dict) else {"result": data},
                duration=time.time() - start,
            )
        except Exception as e:
            return AttackResult(
                name=task.name,
                success=False,
                data={},
                duration=time.time() - start,
                error=str(e),
            )
