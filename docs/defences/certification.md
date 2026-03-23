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

## DeRandomized Smoothing

DeRandomized smoothing uses a structured, deterministic noise pattern (rather than only i.i.d. Gaussian sampling) when defining the smoothed classifier, which can tighten certified robustness guarantees in some regimes. In upstream ART, PyTorch support is exposed as `PyTorchDeRandomizedSmoothing` under `art.estimators.certification`.

The Auto-ART `DefenceStrategy` wrapper for this flow is **documented as living alongside other certified trainers** in `auto_art/core/evaluation/defences/trainer_certified.py`. Use the block below as a **code placeholder** until that export is available from the package `__init__`:

```python
# DeRandomized smoothing DefenceStrategy (trainer_certified.py)
# from auto_art.core.evaluation.defences.trainer_certified import DeRandomizedSmoothingDefence
#
# ders = DeRandomizedSmoothingDefence(...)
# estimator = ders.apply(art_estimator, x_train=x_train, y_train=y_train)
```

For direct ART usage today, see [ART certification estimators](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/certification.html) and the `PyTorchDeRandomizedSmoothing` symbol in your installed version.

## DeepZ Certification

DeepZ performs layer-wise abstract interpretation with zonotope domains to bound network outputs and certify robustness. In current ART sources, the PyTorch entry point is spelled **`PytorchDeepZ`** (exported from `art.estimators.certification` when PyTorch is installed).

```python
from art.estimators.certification import PytorchDeepZ

# Construct per ART docs for your version: model, clip_values, input_shape, labels, device, etc.
# certifier = PytorchDeepZ(...)
# ... certifier.predict, bounds, or certify APIs as documented upstream
```

Constructor arguments and methods vary by ART release; always confirm against the [certification module](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/certification.html) for the version pinned in your environment.

## Interval Bound Propagation

Interval Bound Propagation (IBP) trains networks so that interval bounds propagated through each layer stay within a certified region around each input. In Auto-ART this is exposed as a trainer defence wrapping ART's certified PyTorch trainer with `loss_type="interval"`.

```python
from auto_art.core.evaluation.defences import IntervalBoundPropagationDefence

ibp = IntervalBoundPropagationDefence(nb_epochs=20, batch_size=128)
robust_classifier = ibp.apply(art_estimator, x_train=x_train, y_train=y_train)
```

For a more general certified trainer (configurable `bound` and `loss_type`), use `CertifiedAdversarialTrainingDefence` from the same module.

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
