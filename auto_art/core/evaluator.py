"""
Main evaluator module for automated adversarial robustness testing.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .model_analyzer import ModelAnalyzer, ModelMetadata
from .test_generator import TestDataGenerator, TestData
from .attack_generator import AttackGenerator, AttackConfig

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    perturbation_size: float
    attack_time: float
    model_type: str
    attack_type: str

class RobustnessEvaluator:
    """Main class for automated adversarial robustness evaluation."""
    
    def __init__(self):
        self.model_analyzer = ModelAnalyzer()
        self.test_generator = TestDataGenerator()
        self.attack_generator = AttackGenerator()
    
    def evaluate_model(self, model_path: str, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluates the robustness of a model against adversarial attacks."""
        # Load and analyze model
        model, framework = self.model_analyzer.load_model(model_path)
        metadata = self.model_analyzer.analyze_architecture(model, framework)
        
        # Generate test data
        test_data = self.test_generator.generate_test_data(metadata, num_samples)
        test_data.expected_outputs = self.test_generator.generate_expected_outputs(model, test_data)
        
        # Evaluate for each supported attack type
        results = {}
        for attack_type in self.attack_generator.supported_attacks[metadata.model_type]:
            metrics = self._evaluate_attack(
                model=model,
                metadata=metadata,
                test_data=test_data,
                attack_type=attack_type
            )
            results[attack_type] = metrics
        
        return {
            'model_metadata': metadata,
            'attack_results': results,
            'summary': self._generate_summary(results)
        }
    
    def _evaluate_attack(self, model: Any, metadata: ModelMetadata, 
                        test_data: TestData, attack_type: str) -> EvaluationMetrics:
        """Evaluates the model against a specific attack type."""
        # Create attack configuration
        config = AttackConfig(attack_type=attack_type)
        
        # Create and apply attack
        attack = self.attack_generator.create_attack(model, metadata, config)
        adversarial_examples, clean_examples = self.attack_generator.apply_attack(attack, test_data)
        
        # Calculate metrics
        clean_accuracy = self._calculate_accuracy(model, clean_examples, test_data.expected_outputs)
        adversarial_accuracy = self._calculate_accuracy(model, adversarial_examples, test_data.expected_outputs)
        attack_success_rate = 1 - adversarial_accuracy
        perturbation_size = self._calculate_perturbation_size(clean_examples, adversarial_examples)
        
        return EvaluationMetrics(
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            attack_success_rate=attack_success_rate,
            perturbation_size=perturbation_size,
            attack_time=0.0,  # Would need to measure actual time
            model_type=metadata.model_type,
            attack_type=attack_type
        )
    
    def _calculate_accuracy(self, model: Any, inputs: np.ndarray, 
                          expected_outputs: np.ndarray) -> float:
        """Calculates the accuracy of model predictions."""
        if isinstance(model, tuple) and len(model) == 2:  # Transformers model with tokenizer
            model, tokenizer = model
        else:
            tokenizer = None
        
        with torch.no_grad():
            if tokenizer is not None:
                outputs = model(**tokenizer(inputs, return_tensors="pt"))
            else:
                outputs = model(torch.from_numpy(inputs))
            
            if isinstance(outputs, torch.Tensor):
                predictions = outputs.argmax(dim=-1).numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).numpy()
            
            return np.mean(predictions == expected_outputs)
    
    def _calculate_perturbation_size(self, clean_examples: np.ndarray, 
                                   adversarial_examples: np.ndarray) -> float:
        """Calculates the average perturbation size."""
        return np.mean(np.abs(adversarial_examples - clean_examples))
    
    def _generate_summary(self, results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """Generates a summary of the evaluation results."""
        summary = {
            'average_clean_accuracy': np.mean([m.clean_accuracy for m in results.values()]),
            'average_adversarial_accuracy': np.mean([m.adversarial_accuracy for m in results.values()]),
            'average_attack_success_rate': np.mean([m.attack_success_rate for m in results.values()]),
            'average_perturbation_size': np.mean([m.perturbation_size for m in results.values()]),
            'best_attack': max(results.items(), key=lambda x: x[1].attack_success_rate)[0],
            'worst_attack': min(results.items(), key=lambda x: x[1].attack_success_rate)[0]
        }
        return summary 