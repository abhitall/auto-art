# Adaptive attack selection

YAML `execution.mode: adaptive` reorders **evasion** attacks using `AdaptiveAttackSelector` (`auto_art.core.adaptive`). Selection uses a lightweight in-memory model of past success rates and durations.

## Requirements

- Non-empty `attacks.evasion` list with `name` fields matching `AttackGenerator` types.
- After each run, memory is updated from completed evasion results.

See also `docs/automation/orchestrator.md` and `docs/configuration.md`.
