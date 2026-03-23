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
|---------|------------|--------|
| `SpatialSmoothingDefence` | `window_size`, `channels_first` | Median/mean filter on spatial dimensions |
| `FeatureSqueezingDefence` | `bit_depth`, `clip_values` | Reduces color precision |
| `JpegCompressionDefence` | `quality`, `channels_first`, `clip_values` | JPEG compression/decompression cycle |
| `GaussianAugmentationDefence` | `sigma`, `augmentation`, `ratio`, `clip_values` | Additive Gaussian noise |

### Augmentation Preprocessors

Cutout, Mixup, and CutMix are defined in `preprocessor_augmentation` and exposed from the defences package.

```python
from auto_art.core.evaluation.defences import CutoutDefence, MixupDefence, CutMixDefence

cutout = CutoutDefence(length=16, channels_first=False)
estimator = cutout.apply(art_estimator)

mixup = MixupDefence(num_classes=10, alpha=1.0)
estimator = mixup.apply(estimator)

cutmix = CutMixDefence(num_classes=10, alpha=1.0)
estimator = cutmix.apply(estimator)
```

| Defence | Parameters | Effect |
|---------|------------|--------|
| `CutoutDefence` | `length`, `channels_first` | Random square erasing on images |
| `MixupDefence` | `num_classes`, `alpha`, `channels_first` | Mixup convex combinations |
| `CutMixDefence` | `num_classes`, `alpha`, `channels_first` | CutMix patch swap + label mix |

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
|---------|------------|--------|
| `ReverseSigmoidDefence` | `beta`, `gamma` | Perturbs output probabilities via reverse sigmoid |
| `HighConfidenceDefence` | `cutoff` | Replaces low-confidence outputs with uniform distribution |
| `GaussianNoiseDefence` | `scale` | Adds Gaussian noise to output probabilities |
| `ClassLabelsDefence` | (none) | Returns only class label, strips all probability info |

Rounding reduces precision of reported probabilities (model extraction / MIA hardening).

```python
from auto_art.core.evaluation.defences import RoundingDefence

round_def = RoundingDefence(decimals=4)
defended_estimator = round_def.apply(art_estimator)
```

| Defence | Parameters | Effect |
|---------|------------|--------|
| `RoundingDefence` | `decimals` | Rounds output probabilities to fixed decimal places |

## Trainer Defences

Modify the training process to improve robustness.

### Adversarial Training (PGD)

```python
from auto_art.core.evaluation.defences import AdversarialTrainingPGDDefence

adv_train = AdversarialTrainingPGDDefence(
    nb_epochs=20, eps=0.3, eps_step=0.1, max_iter=7,
)
robust_estimator = adv_train.apply(
    art_estimator, x_train=x_train, y_train=y_train,
)
```

### Fast Is Better Than Free (`trainer_fbf`)

```python
from auto_art.core.evaluation.defences import FastIsBetterThanFreeDefence

fbf = FastIsBetterThanFreeDefence(nb_epochs=20, eps=0.3, batch_size=128)
robust_estimator = fbf.apply(art_estimator, x_train=x_train, y_train=y_train)
```

### Certified Adversarial Training and IBP (`trainer_certified`)

`CertifiedAdversarialTrainingDefence` wraps ART's certified trainer with configurable `bound` and `loss_type`. `IntervalBoundPropagationDefence` fixes interval (IBP) training.

```python
from auto_art.core.evaluation.defences import (
    CertifiedAdversarialTrainingDefence,
    IntervalBoundPropagationDefence,
)

cert = CertifiedAdversarialTrainingDefence(
    nb_epochs=20, batch_size=128, bound=0.1, loss_type="interval",
)
robust_estimator = cert.apply(art_estimator, x_train=x_train, y_train=y_train)

ibp = IntervalBoundPropagationDefence(nb_epochs=20, batch_size=128)
robust_estimator = ibp.apply(art_estimator, x_train=x_train, y_train=y_train)
```

### OAAT (`trainer_oaat`)

```python
from auto_art.core.evaluation.defences import OAATDefence

oaat = OAATDefence(
    nb_epochs=20, eps=0.3, eps_step=0.1, max_iter=10, batch_size=128,
)
robust_estimator = oaat.apply(art_estimator, x_train=x_train, y_train=y_train)
```

## Transformer Defences

Poisoning and evasion transformers return a new or mitigated estimator after `apply` (often requires `x_train` / `y_train`).

### Neural Cleanse and STRIP (`transformer_cleanse`)

```python
from auto_art.core.evaluation.defences import NeuralCleanseDefence, STRIPDefence

nc = NeuralCleanseDefence(steps=1000, learning_rate=0.1)
mitigated = nc.apply(art_estimator, x_train=x_train, y_train=y_train)

strip = STRIPDefence(num_samples=20)
mitigated = strip.apply(art_estimator, x_train=x_train, y_train=y_train)
```

### Defensive Distillation (`transformer_distillation`)

```python
from auto_art.core.evaluation.defences import DefensiveDistillationDefence

distill = DefensiveDistillationDefence(
    batch_size=128, nb_epochs=10, temperature=10.0,
)
student = distill.apply(art_estimator, x_train=x_train, y_train=y_train)
```

## Detector Defences (Poisoning)

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

## Evasion Detectors

Runtime detectors for adversarial inputs (`detector_evasion`, `detector_beyond`). Most require extra kwargs (e.g. a trained binary `detector_classifier`, or background data). After `apply`, use `detect(x)` where implemented.

### Basic Input, Activation, Subset Scan (`detector_evasion`)

```python
import numpy as np
from auto_art.core.evaluation.defences import (
    BasicInputDetectorWrapper,
    ActivationDetectorWrapper,
    SubsetScanDetectorWrapper,
)

# Trained binary ART classifier: clean vs adversarial
basic = BasicInputDetectorWrapper()
basic.apply(art_estimator, detector_classifier=my_binary_detector)
flags = basic.detect(x_batch)

act = ActivationDetectorWrapper(hidden_layer_index=-1)
act.apply(
    art_estimator,
    detector_classifier=my_activation_detector,
    x_train=x_calib,
    y_train=y_calib,
)
flags = act.detect(x_batch)

subset = SubsetScanDetectorWrapper(bgd_data=np.asarray(clean_background))
subset.apply(art_estimator, bgd_data=np.asarray(clean_background))
is_adversarial = subset.detect(x_batch)
```

### BEYOND (`detector_beyond`, ART 1.19+)

```python
from auto_art.core.evaluation.defences import BEYONDDetectorWrapper

beyond = BEYONDDetectorWrapper(nb_classes=10, nb_neighbors=50, batch_size=128)
beyond.apply(art_estimator, x_train=x_train, y_train=y_train)
is_adversarial = beyond.detect(x_batch)
```

## Additional Poison Detectors

Provenance and RONI wrap ART poison detectors that need full training arrays (`detector_provenance`).

```python
from auto_art.core.evaluation.defences import (
    DataProvenanceDefenceWrapper,
    RONIDefenceWrapper,
)

# p_train: provenance / trust label per training point
prov = DataProvenanceDefenceWrapper()
prov.apply(art_estimator, x_train=x_train, y_train=y_train, p_train=p_train)
out = prov.detect_poison()

roni = RONIDefenceWrapper()
roni.apply(art_estimator, x_train=x_train, y_train=y_train)
out = roni.detect_poison()
```
