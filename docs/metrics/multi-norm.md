# Multi-norm evaluation

Evaluates robustness under multiple norms (e.g. L1, L2, Linf) in one pass. Implementation: `auto_art.core.evaluation.metrics.multi_norm` — `MultiNormEvaluator`, `MultiNormReport`.

## Usage

```python
from auto_art.core.evaluation.metrics import MultiNormEvaluator

ev = MultiNormEvaluator(norms=["linf", "l2"])
report = ev.evaluate(classifier=estimator, x=x, y=y, model=raw_model, metadata=meta)
```

Reports can include MultiRobustBench-style aggregates (CR, SC, worst-case) where configured.
