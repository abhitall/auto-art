"""
ART-based model evaluator for adversarial robustness using advanced design patterns.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from functools import lru_cache
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from art.estimators.classification import (
    PyTorchClassifier,
    TensorFlowV2Classifier,
    KerasClassifier,
)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.object_detection import PyTorchObjectDetector
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
)
from art.attacks.poisoning import (
    PoisoningAttackBackdoor,
    PoisoningAttackSVM,
)
from art.defences.preprocessor import (
    FeatureSqueezing,
    SpatialSmoothing,
    GaussianAugmentation,
)
from art.defences.postprocessor import (
    HighConfidence,
    Rounded,
)
from art.utils import compute_success
from art.metrics import (
    empirical_robustness,
    loss_sensitivity,
    clever_u,
    RobustnessVerificationTreeModelsCliqueMethod,
)

from ...core.base import BaseEvaluator
from ...core.interfaces import EvaluatorInterface
from ...utils.logging import LogManager
from ...utils.validation import validate_model, validate_data

from .config.evaluation_config import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationBuilder,
    ModelType,
    Framework
)
from .factories.classifier_factory import ClassifierFactory
from .metrics.calculator import MetricsCalculator
from .attacks.base import AttackStrategy
from .observers import EvaluationObserver

# Type variables for generics
T = TypeVar('T')
ModelType = TypeVar('ModelType')
DataType = TypeVar('DataType')

# Constants
MAX_WORKERS = 4
CACHE_SIZE = 128

class Observer(Protocol):
    """Observer protocol for evaluation events."""
    def update(self, event_type: str, data: Any) -> None: ...

class EvaluationObserver:
    """Concrete observer for evaluation events."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def update(self, event_type: str, data: Any) -> None:
        self.logger.info(f"Evaluation event: {event_type} - {data}")

class AttackStrategy(ABC):
    """Abstract base class for attack strategies."""
    
    @abstractmethod
    def execute(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Execute attack and return adversarial examples and success rate."""
        pass

class EvasionAttackStrategy(AttackStrategy):
    """Strategy for evasion attacks."""
    
    def __init__(self, attack_class: Type[Any], params: Dict[str, Any]):
        self.attack_class = attack_class
        self.params = params

    def execute(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        attack = self.attack_class(classifier, **self.params)
        adversarial_examples = attack.generate(x=data)
        success_rate = compute_success(
            classifier,
            data,
            labels,
            adversarial_examples
        )
        return adversarial_examples, float(success_rate)

class DefenceStrategy(ABC):
    """Abstract base class for defence strategies."""
    
    @abstractmethod
    def apply(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Apply defence and return processed data and accuracy."""
        pass

class PreprocessingDefenceStrategy(DefenceStrategy):
    """Strategy for preprocessing defences."""
    
    def __init__(self, defence_class: Type[Any], params: Dict[str, Any]):
        self.defence_class = defence_class
        self.params = params

    def apply(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        defence = self.defence_class(**self.params)
        processed_data, _ = defence(data)
        predictions = classifier.predict(processed_data)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
        return processed_data, float(accuracy)

class MetricsCalculator:
    """Calculator for various robustness metrics."""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def calculate_empirical_robustness(
        classifier: Any,
        data: np.ndarray,
        eps: float = 0.3
    ) -> float:
        """Calculate empirical robustness with caching."""
        return float(empirical_robustness(
            classifier=classifier,
            x=data,
            attack_name="FastGradientMethod",
            attack_params={"eps": eps}
        ))

    @staticmethod
    def calculate_loss_sensitivity(
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate loss sensitivity."""
        return float(loss_sensitivity(classifier, data, labels))

    @staticmethod
    def calculate_clever_score(
        classifier: Any,
        sample: np.ndarray,
        nb_batches: int = 10,
        batch_size: int = 100,
        radius: float = 0.3,
        norm: int = 2
    ) -> float:
        """Calculate CLEVER score for a sample."""
        return float(clever_u(
            classifier=classifier,
            x=sample,
            nb_batches=nb_batches,
            batch_size=batch_size,
            radius=radius,
            norm=norm
        ))

class ARTEvaluator(BaseEvaluator, EvaluatorInterface):
    """Advanced ART-based evaluator for adversarial robustness."""
    
    def __init__(
        self,
        model: Any,
        config: Optional[EvaluationConfig] = None,
        observers: Optional[List[EvaluationObserver]] = None
    ):
        """Initialize evaluator with model and optional configuration."""
        self.model = model
        self.config = config or EvaluationBuilder().build()
        self.observers = observers or []
        self.logger = LogManager().logger
        self._classifier = None
        self._metrics_calculator = MetricsCalculator()
        
        # Add default observer for logging
        self.observers.append(EvaluationObserver(self.logger))

    def notify_observers(self, event_type: str, data: Any) -> None:
        """Notify all observers of an event."""
        for observer in self.observers:
            observer.update(event_type, data)

    @property
    def classifier(self) -> Any:
        """Lazy initialization of classifier."""
        if self._classifier is None:
            self._classifier = ClassifierFactory.create_classifier(
                model=self.model,
                model_type=self.config.model_type,
                framework=self.config.framework
            )
        return self._classifier

    def evaluate_model(
        self,
        test_data: Any,
        test_labels: Any,
        attacks: Optional[List[AttackStrategy]] = None,
        defences: Optional[List[Any]] = None
    ) -> EvaluationResult:
        """Evaluate model robustness using configured attacks and defences."""
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(test_data, test_labels)
            
            # Initialize results
            results = {
                'attacks': {},
                'defences': {},
                'metrics': {}
            }
            
            # Calculate basic metrics
            results['metrics'].update(
                self._metrics_calculator.calculate_basic_metrics(
                    self.classifier,
                    test_data,
                    test_labels
                )
            )
            
            # Calculate robustness metrics if configured
            if "robustness" in self.config.metrics:
                results['metrics'].update(
                    self._metrics_calculator.calculate_robustness_metrics(
                        self.classifier,
                        test_data,
                        test_labels
                    )
                )
            
            # Evaluate attacks in parallel
            if attacks:
                self._evaluate_attacks(attacks, test_data, test_labels, results)
            
            # Evaluate defences
            if defences:
                self._evaluate_defences(defences, test_data, test_labels, results)
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                success=True,
                metrics=results,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return EvaluationResult(
                success=False,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )

    def _validate_inputs(self, test_data: Any, test_labels: Any) -> None:
        """Validate input data and labels."""
        if test_data is None or test_labels is None:
            raise ValueError("Data and labels cannot be None")
        if not isinstance(test_data, (np.ndarray, torch.Tensor)):
            raise ValueError("Data must be a numpy array or torch tensor")
        if not isinstance(test_labels, (np.ndarray, torch.Tensor)):
            raise ValueError("Labels must be a numpy array or torch tensor")
        if test_data.shape[0] != test_labels.shape[0]:
            raise ValueError("Data and labels must have the same number of samples")

    def _evaluate_attacks(
        self,
        attacks: List[AttackStrategy],
        test_data: Any,
        test_labels: Any,
        results: Dict[str, Any]
    ) -> None:
        """Evaluate attacks in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_attack = {
                executor.submit(
                    self._evaluate_single_attack,
                    attack,
                    test_data,
                    test_labels
                ): attack.__class__.__name__
                for attack in attacks
            }
            
            for future in future_to_attack:
                attack_name = future_to_attack[future]
                try:
                    results['attacks'][attack_name] = future.result(
                        timeout=self.config.timeout
                    )
                except Exception as e:
                    self.logger.error(f"Attack {attack_name} failed: {str(e)}")
                    results['attacks'][attack_name] = None

    def _evaluate_single_attack(
        self,
        attack: AttackStrategy,
        test_data: Any,
        test_labels: Any
    ) -> Dict[str, Any]:
        """Evaluate a single attack strategy."""
        try:
            adversarial_examples, success_rate = attack.execute(
                self.classifier,
                test_data,
                test_labels
            )
            
            return {
                'success_rate': success_rate,
                'adversarial_examples_shape': adversarial_examples.shape,
                'perturbation_size': float(np.mean(np.abs(adversarial_examples - test_data)))
            }
            
        except Exception as e:
            self.logger.error(f"Attack evaluation failed: {str(e)}")
            raise

    def _evaluate_defences(
        self,
        defences: List[Any],
        test_data: Any,
        test_labels: Any,
        results: Dict[str, Any]
    ) -> None:
        """Evaluate defences."""
        for defence in defences:
            defence_name = defence.__class__.__name__
            try:
                processed_data, accuracy = defence.apply(
                    self.classifier,
                    test_data,
                    test_labels
                )
                results['defences'][defence_name] = {
                    'accuracy': accuracy,
                    'processed_data_shape': processed_data.shape
                }
            except Exception as e:
                self.logger.error(f"Defence {defence_name} failed: {str(e)}")
                results['defences'][defence_name] = None

    def generate_report(self, result: EvaluationResult) -> str:
        """Generate a detailed evaluation report."""
        if not result.success:
            return f"Evaluation failed with errors: {', '.join(result.errors)}"
        
        report = []
        report.append("Adversarial Robustness Evaluation Report")
        report.append("=====================================")
        
        # Add execution time
        report.append(f"\nExecution Time: {result.execution_time:.2f} seconds")
        
        # Add attack results
        if 'attacks' in result.metrics:
            report.append("\nAttack Results:")
            for attack_name, attack_results in result.metrics['attacks'].items():
                if attack_results:
                    report.append(f"  {attack_name}:")
                    for metric, value in attack_results.items():
                        if isinstance(value, float):
                            report.append(f"    - {metric}: {value:.4f}")
                        else:
                            report.append(f"    - {metric}: {value}")
                else:
                    report.append(f"  {attack_name}: Failed")
        
        # Add defence results
        if 'defences' in result.metrics:
            report.append("\nDefence Results:")
            for defence_name, defence_results in result.metrics['defences'].items():
                if defence_results:
                    report.append(f"  {defence_name}:")
                    for metric, value in defence_results.items():
                        if isinstance(value, float):
                            report.append(f"    - {metric}: {value:.4f}")
                        else:
                            report.append(f"    - {metric}: {value}")
                else:
                    report.append(f"  {defence_name}: Failed")
        
        # Add metrics
        if 'metrics' in result.metrics:
            report.append("\nModel Metrics:")
            for metric_name, value in result.metrics['metrics'].items():
                if isinstance(value, float):
                    report.append(f"  - {metric_name}: {value:.4f}")
                else:
                    report.append(f"  - {metric_name}: {value}")
        
        return "\n".join(report) 