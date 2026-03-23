# Parallel evasion execution

Set `execution.mode: parallel` to run **evasion** attacks concurrently via `ParallelAttackRunner` (`auto_art.core.parallel`).

## Settings

| Field | Role |
|--------|------|
| `max_workers` | CPU-side thread pool size |
| `gpu_workers` | Thread pool for attacks tagged GPU-heavy (`GPU_ATTACKS`) |
| `timeout_per_attack` | Per-task timeout (seconds) |

GPU-bound attack names include `shadow_attack`, `composite`, `overload`, `adversarial_patch` (see `parallel.GPU_ATTACKS`).

**Note:** Use either `adaptive` or `parallel` per run, not both (see `ExecutionConfig` docstring).
