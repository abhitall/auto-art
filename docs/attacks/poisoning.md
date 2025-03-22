# Poisoning Attacks

Poisoning attacks corrupt training data to implant backdoors or degrade model performance.

## Available Attacks

| Attack | Mechanism | Requires |
|--------|-----------|----------|
| **Backdoor** | Stamps a trigger pattern on training samples and relabels them | Training data access |
| **Clean Label** | Perturbs training samples of the target class so a trigger activates at test time, without changing labels | Model gradients + training data |
| **Feature Collision** | Crafts poison samples whose features collide with a target sample in the model's representation space | Model with feature extraction |
| **Gradient Matching** | Optimizes poison samples so their gradient influence mimics that of the target | Model gradients |

## Usage

```python
from auto_art.core.interfaces import AttackConfig

# Backdoor attack
config = AttackConfig(
    attack_type="backdoor",
    additional_params={
        "backdoor_trigger_fn": lambda x: x + 0.1,
        "target_class_idx": 0,
        "poisoning_rate": 0.1,
    },
)
attack = attack_generator.create_attack(model, metadata, config)
# Use attack.generate() to create poisoned training data

# Clean label attack
config = AttackConfig(
    attack_type="clean_label",
    additional_params={
        "target_class_idx": 5,
        "poisoning_rate": 0.05,
    },
)
```

## Detecting Poisoned Data

See [Agentic Defences](../defences/agentic.md) for RAG poisoning detection, and [ART Defence Wrappers](../defences/art-wrappers.md) for Activation Defence and Spectral Signatures.
