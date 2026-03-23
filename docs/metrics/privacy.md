# Privacy metrics

Membership-style risk signals (e.g. PDTP, SHAPr) via `PrivacyMetricsCalculator` in `auto_art.core.evaluation.metrics.privacy`.

## Usage

```python
from auto_art.core.evaluation.metrics import PrivacyMetricsCalculator

calc = PrivacyMetricsCalculator()
report = calc.compute_all(classifier, x_train, y_train, x_test, y_test)
# Or: calc.compute_pdtp(...) / calc.compute_shapr(...)
```

Use alongside inference-attack phases in the orchestrator YAML for end-to-end privacy posture.
