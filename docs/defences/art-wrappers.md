# ART Defence Wrappers

Auto-ART wraps ART's built-in defences into the `DefenceStrategy` interface for use with `ARTEvaluator`.

## Preprocessor Defences

Applied to inputs before they reach the model.

```python
from auto_art.core.evaluation.defences import (
    SpatialSmoothingDefence,
    FeatureSqueezingDefence,
    JpegCompressionDefence,
    GaussianAugmentationDefence,
)

# Apply to an ART estimator
smooth = SpatialSmoothingDefence(window_size=3)
defended_estimator = smooth.apply(art_estimator)

# Or transform data directly
clean_data = smooth.transform(noisy_data)
```

| Defence | Parameters | Effect |
|---------|-----------|--------|
| `SpatialSmoothingDefence` | `window_size`, `channels_first` | Median/mean filter on spatial dimensions |
| `FeatureSqueezingDefence` | `bit_depth`, `clip_values` | Reduces color precision |
| `JpegCompressionDefence` | `quality`, `channels_first`, `clip_values` | JPEG compression/decompression cycle |
| `GaussianAugmentationDefence` | `sigma`, `augmentation`, `ratio`, `clip_values` | Additive Gaussian noise |

## Postprocessor Defences

Applied to model outputs to reduce information leakage.

```python
from auto_art.core.evaluation.defences import (
    ReverseSigmoidDefence,
    HighConfidenceDefence,
    GaussianNoiseDefence,
    ClassLabelsDefence,
)

reverse_sig = ReverseSigmoidDefence(beta=1.0, gamma=0.1)
defended_estimator = reverse_sig.apply(art_estimator)
```

| Defence | Parameters | Effect |
|---------|-----------|--------|
| `ReverseSigmoidDefence` | `beta`, `gamma` | Perturbs output probabilities via reverse sigmoid |
| `HighConfidenceDefence` | `cutoff` | Replaces low-confidence outputs with uniform distribution |
| `GaussianNoiseDefence` | `scale` | Adds Gaussian noise to output probabilities |
| `ClassLabelsDefence` | (none) | Returns only class label, strips all probability info |

## Trainer Defences

Modify the training process to improve robustness.

```python
from auto_art.core.evaluation.defences import AdversarialTrainingPGDDefence

adv_train = AdversarialTrainingPGDDefence(
    nb_epochs=20, eps=0.3, eps_step=0.1, max_iter=7,
)
robust_estimator = adv_train.apply(
    art_estimator, x_train=x_train, y_train=y_train,
)
```

## Detector Defences

Identify poisoned samples in training data.

```python
from auto_art.core.evaluation.defences import (
    ActivationDefenceWrapper,
    SpectralSignatureDefenceWrapper,
)

# Activation-based poison detection
act_defence = ActivationDefenceWrapper(nb_clusters=2, reduce="PCA")
report = act_defence.detect_poison(art_estimator, x_train, y_train)
print(f"Detected {report.detected_poison} poisoned samples")
print(f"Clean mask: {report.is_clean_mask}")

# Spectral signature detection
spec_defence = SpectralSignatureDefenceWrapper(expected_pp_poison=0.1)
report = spec_defence.detect_poison(art_estimator, x_train, y_train)
```
