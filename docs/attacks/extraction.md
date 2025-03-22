# Extraction Attacks

Model extraction attacks attempt to steal or replicate a model through query access.

## Available Attacks

| Attack | Mechanism | Query Type |
|--------|-----------|-----------|
| **Copycat CNN** | Trains a substitute model on labeled queries to the victim | Prediction API |
| **Knockoff Nets** | Similar to Copycat with adaptive sampling strategies | Prediction API |
| **Functionally Equivalent** | Extracts a model that is functionally identical to the original | Prediction API |

## Usage

These attacks require an ART estimator instance as the victim model:

```python
config = AttackConfig(
    attack_type="copycat_cnn",
    additional_params={
        "nb_epochs_copycat": 10,
        "nb_stolen_samples": 1000,
    },
)
attack_wrapper = attack_generator.create_attack(art_estimator, metadata, config)
# Use attack_wrapper.extract() to perform extraction
```

## Defending Against Extraction

- **Reverse Sigmoid** -- Perturbs output probabilities
- **High Confidence** -- Only returns high-confidence predictions
- **Class Labels** -- Returns only the predicted class, no probabilities
- **Gaussian Noise** -- Adds noise to output distributions

See [ART Defence Wrappers](../defences/art-wrappers.md) for details.
