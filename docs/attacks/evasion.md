# Evasion Attacks

Evasion attacks craft inputs that are misclassified at inference time without changing model weights. Below, `AttackConfig` examples match `AttackGenerator` in `auto_art/core/attacks/attack_generator.py` (only parameters that the generator forwards are shown; for full ART signatures see the [ART evasion module](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html)).

```python
from auto_art.core.interfaces import AttackConfig
# attack = attack_generator.create_attack(model, metadata, config)
# x_adv = attack_generator.apply_attack(attack, x_test, y_test)
```

## White-Box Attacks (gradient-based)

These require loss gradients through the classifier.

### FGSM (Fast Gradient Sign Method)

Single-step attack using the sign of the loss gradient. Fast but weaker than multi-step methods. (Goodfellow et al., 2015)

```python
config = AttackConfig(attack_type="fgsm", epsilon=0.3)
```

### PGD (Projected Gradient Descent)

Iterative L∞ (or other norm) attack inside an ε-ball; standard for adversarial training and robustness benchmarks. (Madry et al., 2018)

```python
config = AttackConfig(
    attack_type="pgd",
    epsilon=0.3,
    eps_step=0.01,
    max_iter=40,
    norm="inf",
)
```

### C&W L2 (Carlini & Wagner)

Strong optimisation-based attack minimising L2 perturbation with a margin objective. (Carlini & Wagner, 2016)

```python
config = AttackConfig(
    attack_type="carlini_wagner_l2",
    confidence=0.0,
    max_iter=100,
    learning_rate=0.01,
)
```

### DeepFool

Iteratively pushes samples toward the decision boundary with small cumulative perturbation. (Moosavi-Dezfooli et al., 2016)

```python
config = AttackConfig(attack_type="deepfool", epsilon=1e-6, max_iter=100)
```

### AutoAttack

Ensemble of strong parameter-free attacks (e.g. APGD variants, FAB, Square); common RobustBench-style evaluation. (Croce & Hein, 2020)

```python
config = AttackConfig(attack_type="autoattack", epsilon=0.3, norm="inf")
```

### BIM (Basic Iterative Method)

Multi-step FGSM with projection to the ε-ball; strictly stronger than one-step FGSM. (Kurakin et al., “Adversarial Examples in the Physical World”, 2017)

```python
config = AttackConfig(
    attack_type="bim",
    epsilon=0.3,
    eps_step=0.01,
    max_iter=40,
)
```

### Auto-PGD

Adaptive-step PGD with optional random restarts and loss choices (e.g. cross-entropy or difference-in-logits ratio). (Croce & Hein, 2020 — [arXiv:2003.01690](https://arxiv.org/abs/2003.01690))

```python
config = AttackConfig(
    attack_type="auto_pgd",
    epsilon=0.3,
    eps_step=0.1,
    max_iter=100,
    additional_params={"nb_random_start": 5, "loss_type": None},
)
```

`AttackGenerator` does not forward `AttackConfig.norm` into Auto-PGD; the wrapper defaults to an L∞-style threat model (see `AutoPGDWrapper`).

### Elastic Net (EAD)

Elastic-net regularised attack related to Carlini–Wagner with L1-style sparsity encouragement. (Chen et al., 2018 — [arXiv:1709.04114](https://arxiv.org/abs/1709.04114))

```python
config = AttackConfig(
    attack_type="elastic_net",
    confidence=0.0,
    learning_rate=0.01,
    max_iter=100,
    binary_search_steps=9,
    initial_const=0.01,
    additional_params={"beta": 0.001, "decision_rule": "EN"},
)
```

### JSMA (Jacobian Saliency Map Attack)

Greedy saliency-guided change of a bounded fraction of input features (typical for low-dimensional or visual pixel settings). (Papernot et al., 2016 — [arXiv:1511.07528](https://arxiv.org/abs/1511.07528))

```python
config = AttackConfig(
    attack_type="jsma",
    additional_params={"theta": 0.1, "gamma": 1.0},
)
```

### NewtonFool

Uses second-order style information to push inputs off the model’s local prediction. (Jang et al., 2017 — [ACM DL](https://doi.org/10.1145/3134600.3134635))

```python
config = AttackConfig(
    attack_type="newtonfool",
    max_iter=100,
    additional_params={"eta": 0.01},
)
```

### Universal Perturbation

Finds a quasi-universal perturbation that fools many inputs under a norm budget. (Moosavi-Dezfooli et al., 2017 — [arXiv:1610.08401](https://arxiv.org/abs/1610.08401))

```python
config = AttackConfig(
    attack_type="universal_perturbation",
    epsilon=0.3,
    max_iter=50,
    additional_params={"delta": 0.2},
)
```

### Shadow Attack

Crafts perturbations constrained to stay visually close in colour/statistics to the original (stealthy L∞-style threat model). (Sitawarin et al., 2020 — [arXiv:2003.08937](https://arxiv.org/abs/2003.08937))

```python
config = AttackConfig(attack_type="shadow_attack", batch_size=32)
```

### Wasserstein

Adversarial examples under a Wasserstein threat model via projected Sinkhorn iterations. (Wong et al., 2019 — [arXiv:1902.07906](https://arxiv.org/abs/1902.07906))

```python
config = AttackConfig(
    attack_type="wasserstein",
    epsilon=0.3,
    eps_step=0.1,
    max_iter=400,
    targeted=False,
)
```

Extra ART knobs (e.g. `kernel_size`, Sinkhorn iteration counts) stay at the wrapper defaults unless you extend `AttackGenerator`.

### Feature Adversaries

Matches internal layer representations to a target, inducing misclassification via feature space. (Sabour et al., 2016 — [arXiv:1511.05122](https://arxiv.org/abs/1511.05122))

```python
config = AttackConfig(
    attack_type="feature_adversaries",
    additional_params={"delta": 0.2, "layer": -1},
)
```

## Black-Box Attacks (query / decision-based)

These do not assume gradient access to the victim (some still use gradients inside a substitute or rely on scores from the victim).

### Boundary Attack

Starts from a large adversarial perturbation and reduces it while staying adversarial; needs only hard labels. (Brendel et al., 2018 — [arXiv:1712.04248](https://arxiv.org/abs/1712.04248))

```python
config = AttackConfig(
    attack_type="boundary_attack",
    delta=0.01,
    step_adapt=0.667,
    max_iter=5000,
    additional_params={"boundary_epsilon": 0.01, "num_trial": 25},
)
```

### Square Attack

Score-based random search on square regions; very query-efficient. (Andriushchenko et al., 2020 — [arXiv:1912.00049](https://arxiv.org/abs/1912.00049))

```python
config = AttackConfig(
    attack_type="square_attack",
    epsilon=0.3,
    max_iter=100,
    norm="inf",
    additional_params={"nb_restarts": 1},
)
```

### HopSkipJump

Decision-based boundary attack using only final predicted labels. (Chen et al., 2019 — [arXiv:1904.02144](https://arxiv.org/abs/1904.02144))

```python
config = AttackConfig(
    attack_type="hopskipjump",
    max_iter=50,
    additional_params={
        "hopskipjump_norm": "2",
        "max_eval": 10000,
        "init_eval": 100,
        "init_size": 100,
    },
)
```

### SimBA (Simple Black-box Adversarial)

Adds/subtracts basis directions (e.g. DCT) one at a time using model scores. (Guo et al., 2019 — [arXiv:1911.06502](https://arxiv.org/abs/1911.06502))

```python
config = AttackConfig(
    attack_type="simba",
    epsilon=0.1,
    max_iter=3000,
    additional_params={"simba_attack_type": "dct"},
)
```

### ZOO (Zeroth-Order Optimisation)

Black-box variant of C&W-style optimisation using coordinate finite differences. (Chen et al., 2017 — [arXiv:1708.03999](https://arxiv.org/abs/1708.03999))

```python
config = AttackConfig(
    attack_type="zoo",
    confidence=0.0,
    learning_rate=0.01,
    max_iter=10,
    binary_search_steps=1,
    initial_const=0.001,
)
```

### Brendel-Bethge

High-quality minimum-norm style attack along the decision boundary as implemented in ART (Brendel–Bethge line of attacks; see ART `BrendelBethgeAttack` and [Bethge lab resources](https://bethgelab.org/)).

```python
config = AttackConfig(attack_type="brendel_bethge", targeted=False, batch_size=1)
```

### GeoDA (Geometric Decision-based Attack)

Exploits geometric structure of the boundary with only label feedback. (Rahmati et al., 2020 — [arXiv:2003.06468](https://arxiv.org/abs/2003.06468))

```python
config = AttackConfig(attack_type="geoda", max_iter=4000, batch_size=64)
```

GeoDA’s `norm` and `sub_dim` use the wrapper defaults unless you extend `AttackGenerator` to pass them.

### Pixel Attack

Evolutionary search over sparse pixel changes (related to one-pixel and few-pixel attacks). (Su et al., 2019; Vargas et al., 2019 — [arXiv:1906.06026](https://arxiv.org/abs/1906.06026))

```python
config = AttackConfig(attack_type="pixel_attack", targeted=False)
```

### Threshold Attack

Evolutionary attack operating on pixels above a threshold (ART `ThresholdAttack`). (Vargas et al., 2019 — [arXiv:1906.06026](https://arxiv.org/abs/1906.06026))

```python
config = AttackConfig(attack_type="threshold_attack", targeted=False)
```

### Spatial Transformation Attack

Adversarially chosen spatial deformations of the input. (Xiao et al., 2018 — [arXiv:1712.02779](https://arxiv.org/abs/1712.02779))

```python
config = AttackConfig(attack_type="spatial_transformation")
```

### Adversarial Patch

Optimises a localized patch (scale/rotation-augmented) to cause misclassification when placed on images. (Brown et al., 2017; ART patch variants — e.g. [arXiv:1806.02299](https://arxiv.org/abs/1806.02299))

```python
config = AttackConfig(
    attack_type="adversarial_patch",
    max_iter=1000,
    learning_rate=0.01,
    additional_params={"rotation_max": 22.5, "scale_min": 0.3, "scale_max": 1.0},
)
```

### Decision Tree Attack

Transfer-style attack tailored to tree ensembles / decision paths. (Papernot et al., 2016 — [arXiv:1605.07277](https://arxiv.org/abs/1605.07277))

```python
config = AttackConfig(attack_type="decision_tree_attack")
```

## Advanced Attacks (ART 1.17+)

### Composite (Composite Adversarial Attack, PyTorch)

Schedules semantic perturbations (hue, saturation, rotation, brightness, contrast) plus an L∞ PGD sub-attack for realistic threat models. Requires PyTorch RGB inputs in [0, 1]. (Yu et al., 2022 — [arXiv:2202.04235](https://arxiv.org/abs/2202.04235); ART 1.17+)

```python
config = AttackConfig(attack_type="composite")
```

### Overload

Targets adaptive inference networks by increasing compute along expensive paths. (Hong et al., 2021; ART `OverloadPyTorch`, 1.18+)

```python
config = AttackConfig(attack_type="overload", epsilon=0.3, max_iter=100, batch_size=32)
```

## Evaluation-Layer Strategies

For direct use with `ARTEvaluator.evaluate_model()`:

```python
from auto_art.core.evaluation.attacks.evasion import FGMAttack, PGDAttack, CarliniL2Attack

fgm = FGMAttack(eps=0.3)
pgd = PGDAttack(eps=0.3, max_iter=40)
cw = CarliniL2Attack(confidence=0.0, max_iter=100)

adv_examples, success_rate = fgm.execute(classifier, x_test, y_test)
```
