# Inference Attacks

Inference attacks deduce sensitive information about training data or the model itself.

## Available Attacks

| Attack | Goal | Access Required |
|--------|------|----------------|
| **Membership Inference (Black-Box)** | Determine if a specific sample was in the training set | Prediction API |
| **Attribute Inference (Black-Box)** | Infer sensitive attributes of training data | Prediction API |
| **Model Inversion (MIFace)** | Reconstruct representative training samples from model outputs | Model access |

## Usage

```python
# Membership inference
config = AttackConfig(
    attack_type="membership_inference_bb",
    additional_params={"attack_model_type": "rf", "input_type": "prediction"},
)
attack_wrapper = attack_generator.create_attack(art_estimator, metadata, config)
# Use attack_wrapper.fit() and attack_wrapper.infer()

# Model inversion
config = AttackConfig(
    attack_type="model_inversion_miface",
    max_iter=100,
    additional_params={"learning_rate_miface": 0.01},
)
```
