# Gradient masking detection

FOSC-style gradient masking checks live in `auto_art.core.gradient_masking` (`GradientMaskingDetector`). Use when suspicious that robust accuracy is inflated by masking rather than true robustness.

## Integration

Call from custom evaluation scripts or extend the orchestrator evaluation block; see `docs/architecture.md` and module docstrings for constructor options.
