# Robustness monitoring

Production-oriented hooks in `auto_art.core.monitoring` (drift, retraining signals). Pair with RDI baselines (`docs/metrics/rdi.md`) and telemetry (`docs/telemetry.md`).

## Typical flow

1. Establish baseline metrics after a certified release.
2. On schedule, re-run orchestrator or lightweight metric jobs.
3. Compare RDI / accuracy / attack success against thresholds; alert on regression.
