# RDI metric

**Robustness Diagnostic Index (RDI)** — fast, attack-independent screening implemented in `auto_art.core.evaluation.metrics.rdi` as `RDICalculator`.

## Usage

```python
from auto_art.core.evaluation.metrics import RDICalculator

calc = RDICalculator()
report = calc.compute(art_classifier, x, y)  # see class docstring for kwargs
```

Use RDI to compare runs over time (baseline vs current) for drift-style monitoring; pair with `docs/production/monitoring.md`.
