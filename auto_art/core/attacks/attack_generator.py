"""
Adversarial attack generator module for creating and applying different types of attacks.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.inference import ModelInversion
from art.estimators.classification import PyTorchClassifier, TensorFlowClassifier
from art.estimators.generation import PyTorchGenerator
from art.estimators.regression import PyTorchRegressor
from .model_analyzer import ModelMetadata
from .test_generator import TestData

@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    attack_type: str
    epsilon: float = 0.3
    eps_step: float = 0.01
    max_iter: int = 100
    targeted: bool = False
    num_random_init: int = 0
    batch_size: int = 32

class AttackGenerator:
    """Generates and applies appropriate adversarial attacks based on model type."""
    
    def __init__(self):
        self.supported_attacks = {
            'classification': ['fgsm', 'pgd', 'deepfool'],
            'regression': ['fgsm', 'pgd'],
            'generator': ['inversion'],
            'llm': ['textfool', 'hotflip']
        }
    
    def create_attack(self, model: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an appropriate adversarial attack based on model type and configuration."""
        if metadata.model_type == 'classification':
            return self._create_classification_attack(model, metadata, config)
        elif metadata.model_type == 'regression':
            return self._create_regression_attack(model, metadata, config)
        elif metadata.model_type == 'generator':
            return self._create_generator_attack(model, metadata, config)
        elif metadata.model_type == 'llm':
            return self._create_llm_attack(model, metadata, config)
        else:
            raise ValueError(f"Unsupported model type: {metadata.model_type}")
    
    def _create_classification_attack(self, model: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an attack appropriate for classification models."""
        if metadata.framework == 'pytorch':
            classifier = PyTorchClassifier(
                model=model,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=metadata.input_shape,
                nb_classes=metadata.output_shape[-1]
            )
        elif metadata.framework == 'tensorflow':
            classifier = TensorFlowClassifier(
                model=model,
                nb_classes=metadata.output_shape[-1],
                input_shape=metadata.input_shape
            )
        else:
            raise ValueError(f"Unsupported framework for classification: {metadata.framework}")
        
        if config.attack_type == 'fgsm':
            return FastGradientMethod(
                estimator=classifier,
                eps=config.epsilon,
                targeted=config.targeted
            )
        elif config.attack_type == 'pgd':
            return ProjectedGradientDescent(
                estimator=classifier,
                eps=config.epsilon,
                eps_step=config.eps_step,
                max_iter=config.max_iter,
                targeted=config.targeted,
                num_random_init=config.num_random_init,
                batch_size=config.batch_size
            )
        elif config.attack_type == 'deepfool':
            return DeepFool(
                classifier=classifier,
                epsilon=config.epsilon,
                max_iter=config.max_iter,
                batch_size=config.batch_size
            )
        else:
            raise ValueError(f"Unsupported attack type for classification: {config.attack_type}")
    
    def _create_regression_attack(self, model: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an attack appropriate for regression models."""
        if metadata.framework == 'pytorch':
            regressor = PyTorchRegressor(
                model=model,
                loss=torch.nn.MSELoss(),
                input_shape=metadata.input_shape
            )
        else:
            raise ValueError(f"Unsupported framework for regression: {metadata.framework}")
        
        if config.attack_type == 'fgsm':
            return FastGradientMethod(
                estimator=regressor,
                eps=config.epsilon,
                targeted=config.targeted
            )
        elif config.attack_type == 'pgd':
            return ProjectedGradientDescent(
                estimator=regressor,
                eps=config.epsilon,
                eps_step=config.eps_step,
                max_iter=config.max_iter,
                targeted=config.targeted,
                num_random_init=config.num_random_init,
                batch_size=config.batch_size
            )
        else:
            raise ValueError(f"Unsupported attack type for regression: {config.attack_type}")
    
    def _create_generator_attack(self, model: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an attack appropriate for generative models."""
        if metadata.framework == 'pytorch':
            generator = PyTorchGenerator(
                model=model,
                input_shape=metadata.input_shape
            )
        else:
            raise ValueError(f"Unsupported framework for generators: {metadata.framework}")
        
        if config.attack_type == 'inversion':
            return ModelInversion(
                estimator=generator,
                max_iter=config.max_iter,
                batch_size=config.batch_size
            )
        else:
            raise ValueError(f"Unsupported attack type for generators: {config.attack_type}")
    
    def _create_llm_attack(self, model: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an attack appropriate for language models."""
        # This is a placeholder for LLM-specific attacks
        # In practice, you'd implement specific text-based attacks
        raise NotImplementedError("LLM attacks not yet implemented")
    
    def apply_attack(self, attack: Any, test_data: TestData) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the adversarial attack to the test data."""
        if isinstance(test_data.inputs, dict):  # Multimodal input
            # Handle multimodal attacks (would need special implementation)
            raise NotImplementedError("Multimodal attacks not yet implemented")
        
        # Generate adversarial examples
        adversarial_examples = attack.generate(
            x=test_data.inputs,
            y=test_data.expected_outputs if test_data.expected_outputs is not None else None
        )
        
        return adversarial_examples, test_data.inputs 