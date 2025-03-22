# Evasion Attacks

Evasion attacks craft adversarial examples that cause misclassification at inference time without modifying the model.

## White-Box Attacks

Require gradient access to the model.

### FGSM (Fast Gradient Sign Method)

Single-step attack using the sign of the loss gradient. Fast but less effective than iterative methods.

```python
# Via AttackGenerator
from auto_art.core.interfaces import AttackConfig
config = AttackConfig(attack_type="fgsm", epsilon=0.3)
attack = attack_generator.create_attack(model, metadata, config)
adv_examples = attack_generator.apply_attack(attack, x_test, y_test)
```

### PGD (Projected Gradient Descent)

Iterative version of FGSM. The gold standard for adversarial training and evaluation.

```python
config = AttackConfig(attack_type="pgd", epsilon=0.3, eps_step=0.01, max_iter=40)
```

### C&W L2 (Carlini & Wagner)

Optimization-based attack minimizing L2 perturbation. One of the strongest white-box attacks.

```python
config = AttackConfig(attack_type="carlini_wagner_l2", confidence=0.0, max_iter=100)
```

### AutoAttack

Ensemble of 4 parameter-free attacks (APGD-CE, APGD-DLR, FAB, Square). The RobustBench standard for reliable evaluation.

```python
config = AttackConfig(attack_type="autoattack", epsilon=0.3)
```

## Black-Box Attacks

Require only query access (predicted labels or probabilities).

### Square Attack

Score-based random search. State-of-the-art query efficiency among black-box attacks.

```python
config = AttackConfig(attack_type="square_attack", epsilon=0.3, max_iter=100)
```

### HopSkipJump

Decision-based attack requiring only final predicted labels. Uses binary search along the decision boundary.

```python
config = AttackConfig(
    attack_type="hopskipjump", max_iter=50,
    additional_params={"hopskipjump_norm": "2", "max_eval": 10000},
)
```

### SimBA (Simple Black-box Adversarial)

Query-efficient attack perturbing one dimension at a time.

```python
config = AttackConfig(
    attack_type="simba", epsilon=0.1, max_iter=3000,
    additional_params={"simba_attack_type": "dct"},
)
```

## Evaluation-Layer Strategies

For use with `ARTEvaluator.evaluate_model()`:

```python
from auto_art.core.evaluation.attacks.evasion import FGMAttack, PGDAttack, CarliniL2Attack

fgm = FGMAttack(eps=0.3)
pgd = PGDAttack(eps=0.3, max_iter=40)
cw = CarliniL2Attack(confidence=0.0, max_iter=100)

adv_examples, success_rate = fgm.execute(classifier, x_test, y_test)
```
