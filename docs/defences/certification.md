# Robustness Certification

Formal methods for provably bounding adversarial vulnerability.

## Randomized Smoothing

Provides certified L2 robustness guarantees by constructing a smoothed classifier that is provably robust within a certified radius.

**Reference:** Cohen et al., 2019 -- "Certified Adversarial Robustness via Randomized Smoothing"

```python
from auto_art.core.evaluation.metrics.certification import RandomizedSmoothingCertifier

certifier = RandomizedSmoothingCertifier(
    sigma=0.25,        # Noise standard deviation
    nb_samples=100,    # Monte Carlo samples for certification
    alpha=0.001,       # Confidence level
)

# Create a smoothed classifier from your PyTorch model
smoothed = certifier.create_smoothed_classifier_pytorch(
    model=my_pytorch_model,
    nb_classes=10,
    input_shape=(3, 32, 32),
)

# Certify robustness
results = certifier.certify(smoothed, x_test, y_test)
print(f"Mean certified radius: {results['mean_certified_radius']:.4f}")
print(f"Certified accuracy at eps=0.5: {results['certified_accuracy']['eps_0.5']:.4f}")
```

## GREAT Score

Global Robustness Evaluation of Adversarial Perturbation using Generative Models. Added in ART 1.20.0.

```python
from auto_art.core.evaluation.metrics.certification import compute_great_score

score = compute_great_score(
    classifier=art_estimator,
    x=x_test,
    y=y_test,
    nb_samples=100,
)
print(f"GREAT Score: {score}")
```

**Note:** Requires ART >= 1.20.0. Returns `None` if not available.

## MetricsCalculator Integration

Both metrics are integrated into the standard `MetricsCalculator`:

```python
from auto_art.core.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()

# GREAT Score
great = calc.calculate_great_score(classifier, data, labels)

# Standard metrics
robustness = calc.calculate_robustness_metrics(classifier, data, labels)
```
