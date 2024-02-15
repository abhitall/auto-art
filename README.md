# Auto-ART: Automated Adversarial Robustness Testing Framework

A comprehensive framework for testing and evaluating machine learning models against various inference attacks and adversarial threats. This framework supports testing models for vulnerabilities to membership inference, attribute inference, and model inversion attacks, with a focus on automated testing and analysis.

## Features

- **Universal Model Loading**: Automatically detects and loads models from different frameworks (PyTorch, TensorFlow, HuggingFace)
- **Comprehensive Inference Attacks**: Supports multiple types of inference attacks:
  - Membership Inference (Black-Box)
  - Attribute Inference (Black-Box and White-Box)
  - Model Inversion
- **Flexible Attack Configuration**: Customizable parameters for each attack type
- **Detailed Metrics**: Generates success rates and attack results for each attack type

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/auto-art.git
cd auto-art
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from auto_art.core.evaluation.attacks.inference import (
    MembershipInferenceBlackBoxAttack,
    AttributeInferenceBlackBoxAttack,
    ModelInversionAttack
)

# Initialize attacks
membership_attack = MembershipInferenceBlackBoxAttack(
    confidence_threshold=0.5,
    num_shadow_models=5
)

attribute_attack = AttributeInferenceBlackBoxAttack(
    attack_feature=0,
    sensitive_features=None,
    confidence_threshold=0.5
)

model_inversion = ModelInversionAttack(
    target_class=0,
    learning_rate=0.01,
    max_iter=100
)

# Execute attacks
membership_preds, membership_success = membership_attack.execute(
    classifier=your_model,
    x=test_data,
    y=test_labels
)

attribute_preds, attribute_success = attribute_attack.execute(
    classifier=your_model,
    x=test_data,
    y=test_labels
)

reconstructed_data, inversion_success = model_inversion.execute(
    classifier=your_model,
    x=test_data
)
```

## Supported Attacks

### Membership Inference
- Black-Box attack using neural network-based attack model
- Configurable confidence threshold and number of shadow models

### Attribute Inference
- Black-Box attack for inferring sensitive attributes
- White-Box attack with direct model access
- Support for multiple sensitive features

### Model Inversion
- Reconstruction of training data samples
- Configurable learning rate and maximum iterations
- Success rate based on target class prediction

## Project Structure

```
auto_art/
├── core/
│   └── evaluation/
│       └── attacks/
│           ├── base.py
│           ├── inference.py
│           └── ...
├── examples/
└── tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- Focused on addressing privacy concerns in machine learning systems

## Detailed Documentation
For a comprehensive explanation of the codebase, refer to the [Detailed Overview](DETAILED_OVERVIEW.md) file.
