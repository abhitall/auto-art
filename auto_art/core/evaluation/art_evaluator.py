"""
ART-based model evaluator for adversarial robustness using advanced design patterns.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, Type, TypeVar, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import torch
# import torch.nn as nn # Not directly used in ARTEvaluator class snippet
from functools import lru_cache
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from ...core.analysis.model_analyzer import analyze_model_architecture as analyze_model_architecture_utility
from ...core.testing.test_generator import TestDataGenerator, TestData
from ...core.attacks.attack_generator import AttackGenerator
from ...core.interfaces import AttackConfig as AutoARTAttackConfig # Renamed to avoid clash if ARTEvaluator has its own AttackConfig
from ...core.base import ModelMetadata # Already imported by ARTEvaluator via other means, but explicit is fine.
from ...implementations.models.factory import ModelFactory


# Assuming these ART components are used by ClassifierFactory or Attack Strategies
# from art.estimators.classification import (
#     PyTorchClassifier,
#     TensorFlowV2Classifier,
#     KerasClassifier,
# )
# from art.utils import compute_success # Used in conceptual EvasionAttackStrategy

from ...core.base import BaseEvaluator
from ...core.interfaces import EvaluatorInterface
from ...utils.logging import LogManager
# from ...utils.validation import validate_model, validate_data

from .config.evaluation_config import (
    EvaluationConfig,
    EvaluationResult,
    ModelType as ModelTypeEnum,
    Framework as FrameworkEnum
)
from .factories.classifier_factory import ClassifierFactory
from .metrics.calculator import MetricsCalculator
# EvasionAttackStrategy is defined in this file.
# No longer aliasing from .attacks.base. If .attacks.base.AttackStrategy is needed, import it directly.
from .defences.base import DefenceStrategy # New import

# Type variables
T = TypeVar('T')
AnyModelType = TypeVar('AnyModelType')
DataType = TypeVar('DataType')

# Constants
MAX_WORKERS = 4


# Moved EvasionAttackStrategy to be defined before ARTEvaluator uses it in type hints.
# Also, aliasing AttackStrategy from .attacks.base to EvasionAttackStrategy was confusing.
# Let's rename the class here to ConcreteEvasionAttackStrategy if it's meant to be a specific implementation
# or ensure it correctly implements the AttackStrategy protocol from .attacks.base.
# For now, keeping its original name here as it's a concrete class.

class EvasionAttackStrategy:
    """Represents a strategy for executing a specific evasion attack.

    This class wraps an ART attack class and its parameters, providing a standardized
    `execute` method to generate adversarial examples.
    """
    def __init__(self, attack_class: Type[Any], params: Dict[str, Any], attack_name: str = "UnnamedEvasionAttack"):
        """Initializes the EvasionAttackStrategy.

        Args:
            attack_class: The ART attack class (e.g., `art.attacks.evasion.FastGradientMethod`).
            params: A dictionary of parameters to initialize the `attack_class`.
                    This may include `y_target` for targeted attacks.
            attack_name: A descriptive name for this attack strategy instance.
        """
        self.attack_class = attack_class
        # params may include 'y_target' for targeted attacks
        self.params = params
        self.attack_name = attack_name
        self.logger = LogManager().logger # Added logger for warnings

    def execute(
        self,
        classifier: Any, # This is an ART Estimator
        data: np.ndarray,
        labels: np.ndarray # True labels
    ) -> Tuple[str, np.ndarray, float]:
        """Executes the configured evasion attack.

        Args:
            classifier: An ART estimator instance (e.g., PyTorchClassifier) against which
                        the attack is performed.
            data: A numpy array of original input samples.
            labels: A numpy array of true labels for the input samples.

        Returns:
            A tuple containing:
                - attack_name (str): Name of the attack.
                - adversarial_examples (np.ndarray): The generated adversarial examples.
                - success_rate (float): The success rate of the attack.

        Raises:
            ValueError: If `y_target` is provided for a targeted attack but its length
                        does not match the length of `data`.
        """
        from art.utils import compute_success # Delayed import

        # Simplified instantiation: ART attacks typically take 'estimator' as the primary arg.
        # User-provided params should not conflict with this.
        attack_params_for_creation = self.params.copy()
        y_target = attack_params_for_creation.pop('y_target', None) # Extract y_target if present

        attack_instance = self.attack_class(estimator=classifier, **attack_params_for_creation)

        is_targeted = getattr(attack_instance, 'targeted', False) # Check instance's targeted status

        y_for_generate = labels # Default for untargeted (ART uses true labels to ensure misclassification)
        if is_targeted:
            if y_target is not None:
                if len(y_target) != len(data):
                    raise ValueError(f"Target labels y_target (len {len(y_target)}) must have the same length as data (len {len(data)}) for targeted attack {self.attack_name}.")
                y_for_generate = y_target
            else:
                # This case means the attack is marked as targeted, but no target labels are provided.
                # Some ART attacks might handle this (e.g. "least likely class" if y is None),
                # but it's safer to warn or raise. For now, proceed with true labels (effectively untargeted against that class).
                self.logger.warning(f"Attack {self.attack_name} is targeted, but no 'y_target' was provided in params. "
                                     "Proceeding with original labels for generation, which might not be standard targeted behavior.")

        adversarial_examples = attack_instance.generate(x=data, y=y_for_generate)

        # Compute success rate, ensuring labels used for compute_success are the original true labels
        success_rate = compute_success(
            classifier, data, labels, adversarial_examples, targeted=is_targeted
        )
        return self.attack_name, adversarial_examples, float(success_rate)


@dataclass
class EvaluationMetrics: # From auto_art/core/evaluator.py
    """Container for evaluation metrics."""
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    perturbation_size: float
    attack_time: float
    model_type: str
    attack_type: str


class Observer(Protocol):
    """Defines the interface for observers of evaluation events."""
    def update(self, event_type: str, data: Any) -> None:
        """Receives updates from the ARTEvaluator about evaluation progress.

        Args:
            event_type: A string identifying the type of event (e.g., "evaluation_started").
            data: Data associated with the event.
        """
        ...

class EvaluationObserver:
    """A concrete observer that logs evaluation events.

    This observer prints information about evaluation stages and results to a logger.
    """
    def __init__(self, logger: logging.Logger):
        """Initializes the EvaluationObserver.

        Args:
            logger: The logger instance to use for recording events.
        """
        self.logger = logger

    def update(self, event_type: str, data: Any) -> None:
        """Logs an evaluation event.

        Args:
            event_type: The type of evaluation event.
            data: Data associated with the event, logged as a string.
        """
        log_data_str = str(data)
        if len(log_data_str) > 200:
            log_data_str = log_data_str[:197] + "..."
        self.logger.info(f"Evaluation event: {event_type} - Data: {log_data_str}")

# Removed the duplicate, older EvasionAttackStrategy definition that was here.
# The correct one is defined above ARTEvaluator.

class ARTEvaluator(BaseEvaluator, EvaluatorInterface): # type: ignore
    """Orchestrates adversarial robustness evaluations using the Adversarial Robustness Toolbox (ART).

    This evaluator can assess models against various adversarial attacks and defenses.
    It supports two main evaluation flows:
    1. `evaluate_model`: Evaluates a pre-loaded model object using provided test data,
       a list of `EvasionAttackStrategy` objects, and optionally `DefenceStrategy` objects.
    2. `evaluate_robustness_from_path`: Loads a model from a specified path and framework,
       generates test data, and runs a suite of default attacks based on model type,
       mimicking the behavior of the former `RobustnessEvaluator`.

    The evaluator uses an `EvaluationConfig` to control various aspects of the
    evaluation, such as batch sizes, metrics to compute, and device preferences.
    It notifies registered observers of key events during the evaluation process.

    Attributes:
        model_obj: The machine learning model instance to be evaluated. Can be None initially
                   if `evaluate_robustness_from_path` is used.
        config: An `EvaluationConfig` object storing evaluation parameters.
        observers: A list of `EvaluationObserver` instances to notify of events.
        logger: A logger instance for recording messages.
        model_metadata: Optional `ModelMetadata` for the `model_obj`. Can be set by
                        `evaluate_robustness_from_path` or passed during initialization.
    """
    def __init__(
        self,
        model_obj: Any, # Can be None if model_path is used later by evaluate_robustness_from_path
        config: EvaluationConfig,
        observers: Optional[List[EvaluationObserver]] = None,
        model_metadata: Optional[ModelMetadata] = None # Allow passing metadata directly
    ):
        """Initializes the ARTEvaluator.

        Args:
            model_obj: The raw model object (e.g., PyTorch Module, Keras Model).
                       Can be None if `evaluate_robustness_from_path` will be used to load it.
            config: The `EvaluationConfig` for this evaluation session.
            observers: An optional list of `EvaluationObserver` instances.
            model_metadata: Optional `ModelMetadata` if the model has already been analyzed.
        """
        self.model_obj = model_obj
        self.config = config
        self.observers = observers if observers is not None else []
        self.logger = LogManager().logger
        self._art_estimator_instance: Optional[Any] = None
        self._metrics_calculator = MetricsCalculator()
        self.model_metadata: Optional[ModelMetadata] = model_metadata # Store provided metadata
        
        if not self.observers:
            self.observers.append(EvaluationObserver(self.logger))

    def notify_observers(self, event_type: str, data: Any) -> None:
        """Notifies all registered observers of an evaluation event.

        Args:
            event_type: A string identifying the type of event.
            data: Data associated with the event.
        """
        for observer in self.observers:
            observer.update(event_type, data)

    @property
    def art_estimator(self) -> Any:
        """Provides an ART-compatible estimator for the model.

        Lazily initializes the ART estimator using `ClassifierFactory` based on
        `self.model_obj`, `self.model_metadata` (if available), and `self.config`.

        Returns:
            An ART estimator instance (e.g., PyTorchClassifier).

        Raises:
            ValueError: If `self.model_obj` is None or if essential parameters for
                        creating the ART estimator cannot be resolved.
        """
        if self._art_estimator_instance is None:
            if self.model_obj is None:
                raise ValueError("Cannot create ART estimator: model_obj is None. Load a model first or provide it during initialization.")

            # Prioritize self.model_metadata if available (e.g., from evaluate_robustness_from_path)
            meta_input_shape = None
            meta_nb_classes = None
            meta_model_type = self.config.model_type # Fallback to config
            meta_framework = self.config.framework # Fallback to config

            if self.model_metadata is not None:
                self.logger.info("Using self.model_metadata for ART estimator creation.")
                meta_input_shape = self.model_metadata.input_shape
                if self.model_metadata.output_shape and isinstance(self.model_metadata.output_shape[-1], int) and self.model_metadata.output_shape[-1] > 0:
                    meta_nb_classes = self.model_metadata.output_shape[-1]

                # Convert string model_type/framework from ModelMetadata to Enum expected by ClassifierFactory
                try:
                    meta_model_type = ModelTypeEnum(self.model_metadata.model_type.lower())
                except ValueError:
                    self.logger.warning(f"Invalid model_type '{self.model_metadata.model_type}' in self.model_metadata. Falling back to config: {self.config.model_type}")
                try:
                    meta_framework = FrameworkEnum(self.model_metadata.framework.lower())
                except ValueError:
                     self.logger.warning(f"Invalid framework '{self.model_metadata.framework}' in self.model_metadata. Falling back to config: {self.config.framework}")

            # If not available from self.model_metadata, try from self.config
            input_shape = meta_input_shape if meta_input_shape else getattr(self.config, 'input_shape', None)
            nb_classes = meta_nb_classes if meta_nb_classes else getattr(self.config, 'nb_classes', None)

            # If still None, try to infer from self.model_obj (existing logic)
            if input_shape is None and hasattr(self.model_obj, 'input_shape') and isinstance(self.model_obj.input_shape, tuple) and len(self.model_obj.input_shape) > 1:
                raw_shape = self.model_obj.input_shape
                # ART classifiers typically expect input_shape without the batch dimension
                input_shape = raw_shape[1:] if raw_shape[0] is None or not isinstance(raw_shape[0], int) else raw_shape
                self.logger.info(f"Inferred input_shape from model_obj: {input_shape}")

            if nb_classes is None and hasattr(self.model_obj, 'output_shape') and isinstance(self.model_obj.output_shape, tuple) and self.model_obj.output_shape:
                raw_out_shape = self.model_obj.output_shape
                if raw_out_shape and isinstance(raw_out_shape[-1], int) and raw_out_shape[-1] > 0 :
                    nb_classes = raw_out_shape[-1]
                    self.logger.info(f"Inferred nb_classes from model_obj: {nb_classes}")

            # Ensure input_shape for ClassifierFactory does not have a batch dimension if it's symbolic (None)
            # Some model_metadata might already provide it correctly (e.g. (C,H,W) not (None,C,H,W))
            if isinstance(input_shape, tuple) and len(input_shape) > 0 and (input_shape[0] is None or not isinstance(input_shape[0], int)):
                if all(isinstance(dim, int) and dim > 0 for dim in input_shape[1:]): # Check if rest of shape is valid
                    input_shape = input_shape[1:]
                    self.logger.info(f"Adjusted input_shape for ClassifierFactory (removed batch dim): {input_shape}")


            loss_function = getattr(self.config, 'loss_function', None)
            device_pref = getattr(self.config, 'device_preference', None)

            if input_shape is None or nb_classes is None:
                self.logger.warning("input_shape or nb_classes not fully resolved for ClassifierFactory. ART estimator might be incorrect or fail.")

            self._art_estimator_instance = ClassifierFactory.create_classifier(
                model=self.model_obj,
                model_type=meta_model_type, # Use resolved model_type (Enum)
                framework=meta_framework,   # Use resolved framework (Enum)
                input_shape=input_shape,
                nb_classes=nb_classes,
                loss_fn=loss_function,
                device_type=device_pref
            )
        return self._art_estimator_instance

    def evaluate_model(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        attacks: Optional[List[EvasionAttackStrategy]] = None,
        defences: Optional[List[DefenceStrategy]] = None # Use new DefenceStrategy
    ) -> EvaluationResult:
        """Evaluates a pre-loaded model against specified attacks and defenses.

        This method calculates basic model metrics, applies any specified defenses,
        evaluates robustness against the provided list of evasion attack strategies,
        and calculates an overall security score.

        Args:
            test_data: Numpy array of test input samples.
            test_labels: Numpy array of true labels for the test data.
            attacks: An optional list of `EvasionAttackStrategy` objects to apply.
            defences: An optional list of `DefenceStrategy` objects to apply and evaluate.

        Returns:
            An `EvaluationResult` object containing all metrics, attack/defence results,
            and any errors encountered.

        Raises:
            ValueError: If `test_data` and `test_labels` are not numpy arrays or have
                        mismatched sample counts.
        """
        start_time = time.time()
        results_aggregator: Dict[str, Any] = {'attacks': {}, 'defences': {}, 'metrics': {}}
            
        try:
            if not isinstance(test_data, np.ndarray) or not isinstance(test_labels, np.ndarray):
                raise ValueError("test_data and test_labels must be numpy arrays.")
            if test_data.shape[0] != test_labels.shape[0]:
                raise ValueError("test_data and test_labels must have the same number of samples.")

            self.notify_observers("evaluation_started", {"num_samples": len(test_data)})
            
            basic_metrics_dict = self._metrics_calculator.calculate_basic_metrics(
                self.art_estimator, test_data, test_labels)
            results_aggregator['metrics'].update(basic_metrics_dict)
            self.notify_observers("basic_metrics_calculated", basic_metrics_dict)
            
            metrics_to_calc = getattr(self.config, 'metrics_to_calculate', ["empirical_robustness"])
            if any(m in metrics_to_calc for m in ["empirical_robustness", "loss_sensitivity", "average_clever_score", "tree_verification"]):
                robustness_metrics_dict = self._metrics_calculator.calculate_robustness_metrics(
                    self.art_estimator, test_data, test_labels,
                    num_samples=getattr(self.config, 'num_samples_for_adv_metrics', 5))
                results_aggregator['metrics'].update(robustness_metrics_dict)
                self.notify_observers("robustness_metrics_calculated", robustness_metrics_dict)
            
            if attacks:
                self._evaluate_attacks_in_parallel(attacks, test_data, test_labels, results_aggregator)
                self.notify_observers("attacks_evaluated", results_aggregator['attacks'])
            
            if defences:
                self.logger.info(f"Evaluating {len(defences)} defence strategies.")
                original_estimator = self.art_estimator # Keep a reference to the original estimator

                for defence_strategy in defences:
                    defence_name = defence_strategy.name
                    self.notify_observers("defence_evaluation_started", {"defence_name": defence_name})
                    try:
                        self.logger.info(f"Applying defence: {defence_name}")
                        # Pass any relevant params from self.config if DefenceStrategy.apply expects them
                        # For now, assuming apply takes estimator and optional generic kwargs
                        # Example: defence_params = self.config.defence_params.get(defence_name, {})
                        defended_estimator = defence_strategy.apply(original_estimator) # Apply to original

                        # Evaluate defended estimator's accuracy on clean data
                        defence_metrics = self._metrics_calculator.calculate_basic_metrics(
                            defended_estimator, test_data, test_labels
                        )
                        # We are primarily interested in the accuracy after defence
                        results_aggregator['defences'][defence_name] = {
                            'accuracy_after_defence': defence_metrics.get('accuracy', np.nan),
                            'params': defence_strategy.get_params()
                        }
                        self.logger.info(f"Defence {defence_name} applied. Accuracy: {defence_metrics.get('accuracy', np.nan):.4f}")
                        self.notify_observers("defence_evaluation_completed", {"defence_name": defence_name, "metrics": results_aggregator['defences'][defence_name]})
                    except Exception as e:
                        self.logger.error(f"Failed to apply or evaluate defence {defence_name}: {str(e)}", exc_info=True)
                        results_aggregator['defences'][defence_name] = {'error': str(e), 'accuracy_after_defence': np.nan}
                        self.notify_observers("defence_evaluation_failed", {"defence_name": defence_name, "error": str(e)})
                self.notify_observers("all_defences_evaluated", results_aggregator['defences'])

            current_metrics_for_score = results_aggregator.get('metrics', {})
            base_acc = current_metrics_for_score.get('accuracy', 0.0)
            
            specific_robustness_metrics_for_score = {
                k: v for k, v in current_metrics_for_score.items()
                if k in ['empirical_robustness', 'average_clever_score', 'tree_verification'] and v is not None
            }

            security_score = self._metrics_calculator.calculate_security_score(
                base_accuracy=base_acc,
                attack_results=results_aggregator.get('attacks', {}),
                robustness_metrics=specific_robustness_metrics_for_score)
            results_aggregator['metrics']['security_score'] = security_score
            self.notify_observers("security_score_calculated", security_score)

            execution_time = time.time() - start_time
            self.notify_observers("evaluation_completed", {"execution_time": execution_time})
            
            return EvaluationResult(success=True, metrics_data=results_aggregator, execution_time=execution_time, errors=[])
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            execution_time = time.time() - start_time
            self.notify_observers("evaluation_failed", {"error": str(e), "execution_time": execution_time})
            return EvaluationResult(success=False, metrics_data={}, errors=[str(e)], execution_time=execution_time)

    def _evaluate_attacks_in_parallel(
        self, attacks: List[EvasionAttackStrategy], test_data: np.ndarray, test_labels: np.ndarray, results_aggregator: Dict[str, Any]
    ) -> None:
        """Evaluates multiple evasion attacks in parallel using a ThreadPoolExecutor.

        Args:
            attacks: A list of `EvasionAttackStrategy` instances to execute.
            test_data: Input data (numpy array) for the attacks.
            test_labels: True labels (numpy array) for the test data.
            results_aggregator: The dictionary to store attack results.
        """
        num_workers = getattr(self.config, 'num_workers', MAX_WORKERS)
        timeout_seconds = getattr(self.config, 'timeout_per_attack', 300.0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_attack_name = {
                executor.submit(self._evaluate_single_attack, attack_strategy, test_data, test_labels): attack_strategy.attack_name
                for attack_strategy in attacks
            }
            for future in future_to_attack_name:
                attack_name_key = future_to_attack_name[future]
                try:
                    attack_specific_results = future.result(timeout=timeout_seconds)
                    results_aggregator['attacks'][attack_name_key] = attack_specific_results
                except Exception as e:
                    self.logger.error(f"Attack {attack_name_key} failed or timed out: {str(e)}")
                    results_aggregator['attacks'][attack_name_key] = {'error': str(e), 'success_rate': np.nan}


    def _evaluate_single_attack(
        self, attack_strategy: EvasionAttackStrategy, test_data: np.ndarray, test_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Executes a single evasion attack strategy and collects its results.

        This method is typically called by `_evaluate_attacks_in_parallel`.

        Args:
            attack_strategy: The `EvasionAttackStrategy` instance to execute.
            test_data: Input data (numpy array) for the attack.
            test_labels: True labels (numpy array) for the test data.

        Returns:
            A dictionary containing results for the attack, including success rate,
            perturbation size, and potentially other metrics like Wasserstein distance.
            If an error occurs, it returns a dict with an 'error' key.
        """
        try:
            _attack_name, adversarial_examples, success_rate = attack_strategy.execute(
                self.art_estimator, test_data, test_labels
            )
            
            attack_eval_results = {
                'success_rate': success_rate,
                'adversarial_examples_shape': adversarial_examples.shape,
                'perturbation_size': float(np.mean(np.abs(adversarial_examples - test_data)))
            }

            if adversarial_examples is not None and test_data is not None:
                current_test_data_np = test_data
                if hasattr(torch, 'Tensor') and isinstance(current_test_data_np, torch.Tensor):
                    current_test_data_np = current_test_data_np.cpu().numpy()

                current_adv_examples_np = adversarial_examples
                if hasattr(torch, 'Tensor') and isinstance(current_adv_examples_np, torch.Tensor):
                    current_adv_examples_np = current_adv_examples_np.cpu().numpy()

                if isinstance(current_test_data_np, np.ndarray) and isinstance(current_adv_examples_np, np.ndarray):
                    wasserstein_dist = self._metrics_calculator.calculate_wasserstein_distance(
                        current_test_data_np,
                        current_adv_examples_np
                    )
                    if wasserstein_dist is not None:
                        attack_eval_results['wasserstein_dist_adv'] = wasserstein_dist
                else:
                    self.logger.debug("Skipping Wasserstein distance: original or adversarial examples not numpy arrays after conversion.")

            return attack_eval_results
            
        except Exception as e:
            self.logger.error(f"Attack {attack_strategy.attack_name} execution failed: {str(e)}", exc_info=True)
            return {'error': str(e), 'success_rate': np.nan, 'perturbation_size': np.nan}

    def generate_report(self, result: EvaluationResult) -> str:
        """Generates a human-readable string report from an EvaluationResult.

        Args:
            result: The `EvaluationResult` object to report on.

        Returns:
            A string containing the formatted evaluation report.
        """
        if not result.success:
            error_messages = ', '.join(result.errors) if result.errors else 'Unknown error'
            return f"Evaluation failed with errors: {error_messages}"
        
        report_lines: List[str] = []
        report_lines.append("Adversarial Robustness Evaluation Report")
        report_lines.append("=====================================")
        report_lines.append(f"\nExecution Time: {result.execution_time:.2f} seconds")

        metrics_data_content = result.metrics_data if hasattr(result, 'metrics_data') else {}

        report_lines.append("\n--- Overall Assessment ---")
        metrics_summary = metrics_data_content.get('metrics', {})
        
        base_accuracy = metrics_summary.get('accuracy')
        if base_accuracy is not None:
            report_lines.append(f"  Base Model Accuracy: {base_accuracy:.4f}")

        security_score = metrics_summary.get('security_score')
        if security_score is not None:
            report_lines.append(f"  Overall Security Score: {security_score:.2f} / 100.0")
        
        emp_robustness = metrics_summary.get('empirical_robustness')
        if emp_robustness is not None:
            report_lines.append(f"  Empirical Robustness (vs FGSM default): {emp_robustness:.4f}")
        report_lines.append("--------------------------")

        attacks_data = metrics_data_content.get('attacks', {})
        if attacks_data:
            report_lines.append("\n--- Attack Results ---")
            for attack_name, attack_data_dict in attacks_data.items():
                if attack_data_dict:
                    report_lines.append(f"  Attack: {attack_name}")
                    for metric_key, metric_value in attack_data_dict.items():
                        formatted_key = metric_key.replace('_', ' ').title()
                        if isinstance(metric_value, float):
                            report_lines.append(f"    - {formatted_key}: {metric_value:.4f}")
                        else:
                            report_lines.append(f"    - {formatted_key}: {metric_value}")
                else:
                    report_lines.append(f"  Attack: {attack_name}: Failed or No Results")
            report_lines.append("----------------------")

        defences_data = metrics_data_content.get('defences', {})
        if defences_data:
            report_lines.append("\n--- Defence Results ---")
        if not defences_data:
            report_lines.append("  No defences evaluated.")
        else:
            for defence_name, defence_data_dict in defences_data.items():
                    report_lines.append(f"  Defence: {defence_name}")
                if 'error' in defence_data_dict:
                    report_lines.append(f"    - Status: Failed ({defence_data_dict['error']})")
                else:
                    for metric_key, metric_value in defence_data_dict.items():
                        formatted_key = metric_key.replace('_', ' ').title()
                        if isinstance(metric_value, float):
                            report_lines.append(f"    - {formatted_key}: {metric_value:.4f}")
                        elif isinstance(metric_value, dict) and metric_key == 'Params': # Special handling for params
                            report_lines.append(f"    - {formatted_key}:")
                            for p_name, p_val in metric_value.items():
                                report_lines.append(f"      - {str(p_name).replace('_', ' ').title()}: {p_val}")
                        else:
                            report_lines.append(f"    - {formatted_key}: {metric_value}")
            report_lines.append("-----------------------")

        report_lines.append("\n--- Detailed Model & Robustness Metrics ---")
        summary_keys = {'accuracy', 'security_score', 'empirical_robustness'}
        detailed_metrics_to_print = {
            k: v for k, v in metrics_summary.items() if k not in summary_keys
        }

        if detailed_metrics_to_print:
            for metric_name, value in detailed_metrics_to_print.items():
                formatted_name = metric_name.replace('_', ' ').title()
                if isinstance(value, float):
                    report_lines.append(f"  - {formatted_name}: {value:.4f}")
                else:
                    report_lines.append(f"  - {formatted_name}: {value}")
        else:
            report_lines.append("  (No other detailed metrics to display or already summarized)")
        report_lines.append("-------------------------------------------")
        
        return "\n".join(report_lines)

    def add_observer(self, observer: EvaluationObserver) -> None:
        """Adds an observer to the list of subscribers for evaluation events.

        Args:
            observer: An instance of a class implementing the `Observer` protocol.
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer: EvaluationObserver) -> None:
        """Removes an observer from the list of subscribers.

        Args:
            observer: The observer instance to remove.
        """
        try:
            self.observers.remove(observer)
        except ValueError:
            self.logger.warning("Observer not found for removal.")

    def evaluate_robustness_from_path(self, model_path: str, framework: str, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluates model robustness by loading a model from path and running default attacks.

        This method mirrors the functionality of the deprecated `RobustnessEvaluator`.
        It performs model loading, analysis, test data generation, and then evaluates
        against a set of standard attacks appropriate for the model type. Attack parameters
        are sourced from `self.config.attack_params`.

        Args:
            model_path: Path to the model file.
            framework: The machine learning framework of the model (e.g., "pytorch", "tensorflow").
            num_samples: The number of test samples to generate for evaluation.

        Returns:
            A dictionary containing evaluation results, including model metadata,
            per-attack metrics, and a summary.

        Raises:
            NotImplementedError: If dictionary-based multimodal inputs are encountered, as
                                 the underlying attack generation flow does not yet fully support them.
        """
        self.notify_observers("robustness_evaluation_from_path_started", {"model_path": model_path, "framework": framework, "num_samples": num_samples})

        # 1. Load Model
        model_loader = ModelFactory.create_model(framework)
        if model_loader is None:
            self.logger.error(f"Unsupported framework: {framework} for model loading.")
            self.notify_observers("robustness_evaluation_from_path_failed", {"error": f"Unsupported framework: {framework}"})
            return {"error": f"Unsupported framework: {framework}"}

        try:
            # Assign to self.model_obj so art_estimator property can use it if needed
            self.model_obj, loaded_framework = model_loader.load_model(model_path)
            self.logger.info(f"Model loaded successfully from {model_path} using framework {loaded_framework}.")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
            self.notify_observers("robustness_evaluation_from_path_failed", {"error": f"Model loading failed: {str(e)}"})
            return {"error": f"Model loading failed: {str(e)}"}

        # 2. Analyze Architecture (get metadata)
        try:
            # Assuming analyze_model_architecture_utility is imported correctly
            # Assign to self.model_metadata so art_estimator property can use it
            self.model_metadata = analyze_model_architecture_utility(self.model_obj, loaded_framework)
            self.logger.info(f"Model architecture analyzed. Type: {self.model_metadata.model_type}, Input: {self.model_metadata.input_shape}")
        except Exception as e:
            self.logger.error(f"Failed to analyze model architecture: {str(e)}", exc_info=True)
            self.notify_observers("robustness_evaluation_from_path_failed", {"error": f"Model analysis failed: {str(e)}"})
            return {"error": f"Model analysis failed: {str(e)}"}

        # 3. Generate Test Data
        try:
            # Ensure TestDataGenerator and TestData are imported
            tdg = TestDataGenerator()
            # TestData class is from ..testing.test_generator
            # Use self.model_metadata and self.model_obj which are now set
            if self.model_metadata is None or self.model_obj is None:
                # This should not happen if model loading and analysis succeeded
                err_msg = "Model metadata or model object is None before test data generation."
                self.logger.error(err_msg)
                self.notify_observers("robustness_evaluation_from_path_failed", {"error": err_msg})
                return {"error": err_msg}
            test_data_obj: TestData = tdg.generate_test_data(self.model_metadata, num_samples)
            # The generate_expected_outputs method in TestDataGenerator can take the raw model_obj
            test_data_obj.expected_outputs = tdg.generate_expected_outputs(self.model_obj, test_data_obj)
            if test_data_obj.expected_outputs is None:
                # This can happen if the model type isn't fully supported for auto-prediction by TestDataGenerator
                self.logger.warning("Could not generate expected outputs for the test data. Accuracy calculations might be affected.")
                # Depending on strictness, one might raise an error here.
                # For now, we'll allow it and metrics calculation will likely fail or yield NaNs for accuracy.
            self.logger.info(f"Test data generated. Input shape: {test_data_obj.inputs.shape if isinstance(test_data_obj.inputs, np.ndarray) else 'dict'}, Expected outputs generated: {test_data_obj.expected_outputs is not None}")
        except Exception as e:
            self.logger.error(f"Failed to generate test data: {str(e)}", exc_info=True)
            self.notify_observers("robustness_evaluation_from_path_failed", {"error": f"Test data generation failed: {str(e)}"})
            return {"error": f"Test data generation failed: {str(e)}"}

        # 4. Initialize AttackGenerator
        attack_generator = AttackGenerator() # Ensure AttackGenerator and AutoARTAttackConfig are imported
        attack_results: Dict[str, EvaluationMetrics] = {}

        # The attack loop and metric calculation are implemented below.
        # Removed the TODO and associated logging/observer calls that stated it was pending.

        # Loop through supported attacks
        if self.model_metadata and self.model_metadata.model_type in attack_generator.supported_attacks: # Use self.model_metadata
            for attack_type_str in attack_generator.supported_attacks[self.model_metadata.model_type]:
                self.logger.info(f"Evaluating attack: {attack_type_str}")
                try:
                    # Ensure TestData has expected_outputs for targeted attacks or accuracy calc
                    if test_data_obj.expected_outputs is None and any(kw in attack_type_str.lower() for kw in ["targeted", "accuracy", "success"]): # Heuristic
                         self.logger.warning(f"Attack {attack_type_str} might need expected_outputs, but they are None. Skipping or results may be NaN.")
                         # continue # Or allow to proceed and let metrics be NaN

                    metrics = self._evaluate_single_attack_for_robustness(
                        self.model_obj, self.model_metadata, test_data_obj, attack_type_str, attack_generator
                    )
                    attack_results[attack_type_str] = metrics
                    self.notify_observers("robustness_attack_evaluated", {"attack": attack_type_str, "metrics": metrics})
                except Exception as e:
                    self.logger.error(f"Failed to evaluate attack {attack_type_str}: {str(e)}", exc_info=True)
                    # Optionally store error information in results
                    attack_results[attack_type_str] = EvaluationMetrics( # type: ignore
                        clean_accuracy=np.nan, adversarial_accuracy=np.nan, attack_success_rate=np.nan,
                        perturbation_size=np.nan, attack_time=0.0, model_type=self.model_metadata.model_type, # Use self.model_metadata
                        attack_type=attack_type_str + " (Failed)"
                    )
        else:
            self.logger.warning(f"No supported attacks found for model type: {self.model_metadata.model_type}") # Use self.model_metadata

        # Generate summary
        summary = self._generate_summary_for_robustness(attack_results)
        self.logger.info(f"Robustness evaluation from path completed. Summary: {summary}")
        self.notify_observers("robustness_evaluation_from_path_completed", {"summary": summary})

        # Convert ModelMetadata to dict for JSON serializability if needed, or select fields
        # Ensure self.model_metadata is not None before accessing its attributes
        serializable_metadata = {}
        if self.model_metadata:
            serializable_metadata = {
                "model_type": self.model_metadata.model_type,
                "framework": self.model_metadata.framework,
                "input_shape": self.model_metadata.input_shape,
                "output_shape": self.model_metadata.output_shape,
                "input_type": self.model_metadata.input_type,
                "output_type": self.model_metadata.output_type,
                # "layer_info": self.model_metadata.layer_info, # Can be very verbose
                "additional_info": self.model_metadata.additional_info
            }

        return {
            "model_metadata": serializable_metadata, # Use self.model_metadata
            "attack_results": {k: v.__dict__ for k, v in attack_results.items()}, # Convert EvaluationMetrics to dict
            "summary": summary
        }

    def _calculate_accuracy_for_robustness(self, model_obj: Any, inputs: np.ndarray, expected_outputs: Optional[np.ndarray]) -> float:
        """Calculates model accuracy on given inputs against expected outputs.

        This helper is used by `evaluate_robustness_from_path`. It directly uses the
        raw model object for predictions, attempting to handle different model types
        (PyTorch, scikit-learn, etc.) through duck typing.

        Args:
            model_obj: The raw model instance.
            inputs: Numpy array of input samples.
            expected_outputs: Numpy array of true labels or expected values.

        Returns:
            The calculated accuracy (float), or np.nan if calculation fails or
            expected_outputs are not provided.
        """
        if expected_outputs is None:
            self.logger.warning("Cannot calculate accuracy without expected_outputs.")
            return np.nan
        if inputs.shape[0] == 0:
            self.logger.warning("Cannot calculate accuracy with empty inputs.")
            return 0.0 # Or np.nan, 0.0 if we consider it "0% correct on 0 items"

        predictions_raw = None
        try:
            if hasattr(torch, 'Tensor') and isinstance(model_obj, torch.nn.Module):
                model_device = next(model_obj.parameters()).device
                # Ensure model is in eval mode
                model_obj.eval()
                with torch.no_grad():
                    # Batch processing for potentially large inputs to avoid OOM
                    batch_size = getattr(self.config, 'batch_size_for_prediction', 32)
                    all_outputs = []
                    for i in range(0, inputs.shape[0], batch_size):
                        batch_inputs = inputs[i:i+batch_size]
                        torch_inputs = torch.from_numpy(batch_inputs.astype(np.float32)).to(model_device)

                        # Handle dict inputs if model expects them (e.g. HuggingFace)
                        if isinstance(torch_inputs, torch.Tensor) and hasattr(model_obj, 'dummy_inputs'): # A common pattern for HF model input structure
                            # This part is heuristic and might need model-specific adaptation
                            # For now, assumes 'input_ids' or direct tensor if not a dict model
                             input_dict_keys = getattr(model_obj, 'input_names', None)
                             if input_dict_keys and isinstance(input_dict_keys, list) and len(input_dict_keys) == 1:
                                 torch_inputs = {input_dict_keys[0]: torch_inputs}
                                 outputs = model_obj(**torch_inputs)
                             else: # Default to direct tensor input
                                 outputs = model_obj(torch_inputs)
                        else: # Standard tensor input
                            outputs = model_obj(torch_inputs)

                        if isinstance(outputs, torch.Tensor):
                            all_outputs.append(outputs.detach().cpu().numpy())
                        elif hasattr(outputs, 'logits') and isinstance(outputs.logits, torch.Tensor): # HuggingFace style
                            all_outputs.append(outputs.logits.detach().cpu().numpy())
                        else: # Fallback for other output types if they can be converted
                            try: all_outputs.append(np.array(outputs))
                            except:  #NOSONAR
                                self.logger.error(f"Unsupported output type from PyTorch model: {type(outputs)}")
                                return np.nan
                    if not all_outputs: return np.nan
                    predictions_raw = np.concatenate(all_outputs, axis=0)

            elif hasattr(model_obj, 'predict_proba'): # sklearn style for probabilities
                predictions_raw = model_obj.predict_proba(inputs)
            elif hasattr(model_obj, 'predict'): # sklearn style or Keras/TF
                predictions_raw = model_obj.predict(inputs)
            else:
                self.logger.error(f"Model type {type(model_obj)} not directly supported by _calculate_accuracy_for_robustness for prediction.")
                return np.nan

            if predictions_raw is None: return np.nan

            # Determine how to get final class predictions from raw output
            if predictions_raw.ndim > 1 and predictions_raw.shape[-1] > 1 : # Logits or probabilities for multi-class
                predictions_classes = np.argmax(predictions_raw, axis=-1)
            elif predictions_raw.ndim > 1 and predictions_raw.shape[-1] == 1: # Output for binary classification or regression
                # For binary classification, often > 0.5 is positive class.
                # For this generic accuracy, assuming labels are 0/1 if shape is (N,1)
                predictions_classes = (predictions_raw > 0.5).astype(int).squeeze()
            else: # Single dimension array already (e.g. direct class predictions)
                predictions_classes = predictions_raw.squeeze()

            # Ensure expected_outputs is also squeezed for comparison
            squeezed_expected_outputs = expected_outputs.squeeze()
            if predictions_classes.shape != squeezed_expected_outputs.shape:
                self.logger.warning(f"Shape mismatch between predictions ({predictions_classes.shape}) and labels ({squeezed_expected_outputs.shape}). Accuracy may be incorrect.")
                # Attempt to reconcile if one is (N,) and other is (N,1) or vice-versa, very common.
                if predictions_classes.ndim == 1 and squeezed_expected_outputs.ndim == 2 and squeezed_expected_outputs.shape[1] == 1:
                    squeezed_expected_outputs = squeezed_expected_outputs.ravel()
                elif squeezed_expected_outputs.ndim == 1 and predictions_classes.ndim == 2 and predictions_classes.shape[1] == 1:
                    predictions_classes = predictions_classes.ravel()
                else: # Can't easily reconcile
                    return np.nan


            return np.mean(predictions_classes == squeezed_expected_outputs)

        except Exception as e:
            self.logger.error(f"Error during accuracy calculation: {e}", exc_info=True)
            return np.nan

    def _calculate_perturbation_size_for_robustness(self, clean_examples: np.ndarray, adversarial_examples: Optional[np.ndarray]) -> float:
        """Calculates the mean absolute perturbation between clean and adversarial examples.

        Args:
            clean_examples: Numpy array of original input samples.
            adversarial_examples: Numpy array of adversarial samples. Can be None if attack failed.

        Returns:
            The mean perturbation size (float), or np.nan if inputs are invalid or
            adversarial_examples is None.
        """
        if adversarial_examples is None:
            self.logger.warning("Adversarial examples are None, cannot calculate perturbation size.")
            return np.nan
        if clean_examples.shape != adversarial_examples.shape:
            self.logger.warning(f"Shapes of clean ({clean_examples.shape}) and adversarial examples ({adversarial_examples.shape}) do not match for perturbation calculation.")
            return np.nan
        return float(np.mean(np.abs(adversarial_examples - clean_examples)))

    def _generate_summary_for_robustness(self, results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        if not results: return {}

        # Filter out potential NaN values before calculating mean, max, min to avoid warnings/errors
        clean_accuracies = [m.clean_accuracy for m in results.values() if not np.isnan(m.clean_accuracy)]
        adv_accuracies = [m.adversarial_accuracy for m in results.values() if not np.isnan(m.adversarial_accuracy)]
        success_rates = [m.attack_success_rate for m in results.values() if not np.isnan(m.attack_success_rate)]
        perturb_sizes = [m.perturbation_size for m in results.values() if not np.isnan(m.perturbation_size)]

        summary = {
            'average_clean_accuracy': np.mean(clean_accuracies) if clean_accuracies else np.nan,
            'average_adversarial_accuracy': np.mean(adv_accuracies) if adv_accuracies else np.nan,
            'average_attack_success_rate': np.mean(success_rates) if success_rates else np.nan,
            'average_perturbation_size': np.mean(perturb_sizes) if perturb_sizes else np.nan,
        }

        valid_results_for_ranking = {k: v for k, v in results.items() if not np.isnan(v.attack_success_rate)}
        if valid_results_for_ranking:
            summary['best_attack'] = max(valid_results_for_ranking.items(), key=lambda x: x[1].attack_success_rate)[0]
            summary['worst_attack'] = min(valid_results_for_ranking.items(), key=lambda x: x[1].attack_success_rate)[0]
        else:
            summary['best_attack'] = "N/A"
            summary['worst_attack'] = "N/A"
        return summary

    def _evaluate_single_attack_for_robustness(self, model_obj: Any, model_metadata: ModelMetadata,
                                               test_data_obj: TestData, attack_type: str,
                                               attack_generator: AttackGenerator) -> EvaluationMetrics:
        """Evaluates the model against a single specified attack type using AttackGenerator.

        This helper is used by `evaluate_robustness_from_path`. It creates an attack
        configuration from `self.config.attack_params`, generates adversarial examples,
        and calculates relevant `EvaluationMetrics`.

        Args:
            model_obj: The raw model instance.
            model_metadata: ModelMetadata for the `model_obj`.
            test_data_obj: `TestData` object containing inputs and expected_outputs.
            attack_type: String identifier for the attack to be performed.
            attack_generator: An instance of `AttackGenerator`.

        Returns:
            An `EvaluationMetrics` object for the performed attack.

        Raises:
            NotImplementedError: If `test_data_obj.inputs` is a dictionary (multimodal)
                                 as this flow doesn't fully support it yet.
        """
        attack_start_time = time.time()

        # Use AutoARTAttackConfig (imported from ..interfaces)
        # Populate from self.config.attack_params, providing defaults from AttackConfig if not found
        cfg_attack_params = self.config.attack_params or {}

        # Ensure batch_size from EvaluationConfig is used if available and not overridden by attack_params
        # AttackConfig has its own batch_size, so we prioritize self.config.batch_size for the evaluation run.
        # However, AttackGenerator's create_attack will use the batch_size from AttackConfig.
        # This is a bit tricky. Let's assume attack_params in EvaluationConfig can override general batch_size.
        final_batch_size = cfg_attack_params.get('batch_size', self.config.batch_size)

        attack_config_params = {
            'attack_type': attack_type,
            'epsilon': cfg_attack_params.get('epsilon', AutoARTAttackConfig.epsilon),
            'eps_step': cfg_attack_params.get('eps_step', AutoARTAttackConfig.eps_step),
            'max_iter': cfg_attack_params.get('max_iter', AutoARTAttackConfig.max_iter),
            'targeted': cfg_attack_params.get('targeted', AutoARTAttackConfig.targeted),
            'num_random_init': cfg_attack_params.get('num_random_init', AutoARTAttackConfig.num_random_init),
            'batch_size': final_batch_size,
            'norm': cfg_attack_params.get('norm', AutoARTAttackConfig.norm),
            'confidence': cfg_attack_params.get('confidence', AutoARTAttackConfig.confidence),
            'learning_rate': cfg_attack_params.get('learning_rate', AutoARTAttackConfig.learning_rate),
            'binary_search_steps': cfg_attack_params.get('binary_search_steps', AutoARTAttackConfig.binary_search_steps),
            'initial_const': cfg_attack_params.get('initial_const', AutoARTAttackConfig.initial_const),
            'delta': cfg_attack_params.get('delta', AutoARTAttackConfig.delta),
            'step_adapt': cfg_attack_params.get('step_adapt', AutoARTAttackConfig.step_adapt),
            'additional_params': cfg_attack_params.get('additional_params', {}) # Ensure it's a dict
        }
        # Filter out any keys that are not actual fields of AutoARTAttackConfig to prevent errors
        # This is important if cfg_attack_params contains extra general keys.
        # However, AttackConfig's fields are fixed. We are mapping known keys.
        # Any other params from cfg_attack_params should go into 'additional_params' if not directly mapped.

        # Collect all other parameters from cfg_attack_params into additional_params
        current_additional_params = attack_config_params['additional_params']
        for k, v in cfg_attack_params.items():
            if k not in attack_config_params and k not in ['attack_type']: # don't overwrite already set params or attack_type
                current_additional_params[k] = v
        attack_config_params['additional_params'] = current_additional_params

        attack_config = AutoARTAttackConfig(**attack_config_params)
        attack_instance = attack_generator.create_attack(model_obj, model_metadata, attack_config)

        if not isinstance(test_data_obj.inputs, np.ndarray):
            self.logger.error("Attack application currently requires np.ndarray inputs. "
                              "Received dict, likely for multimodal data. This path is not yet fully supported by the underlying AttackGenerator.")
            raise NotImplementedError("Attack application for dictionary-based multimodal inputs is not supported in this automated flow. "
                                      "Consider adapting attacks for your specific multimodal architecture or applying attacks to individual modalities.")

        adversarial_examples = None
        try:
            adversarial_examples = attack_generator.apply_attack(
                attack_instance,
                test_data_obj.inputs,
                test_data_obj.expected_outputs # Pass labels if available, for targeted attacks or some ART attacks that use them
            )
        except Exception as e:
            self.logger.error(f"Error applying attack {attack_type}: {e}", exc_info=True)
            # Fall through, adversarial_examples will be None

        clean_examples_for_eval = test_data_obj.inputs

        # Metrics calculation
        clean_accuracy = self._calculate_accuracy_for_robustness(model_obj, clean_examples_for_eval, test_data_obj.expected_outputs)

        if adversarial_examples is not None:
            adversarial_accuracy = self._calculate_accuracy_for_robustness(model_obj, adversarial_examples, test_data_obj.expected_outputs)
            perturbation_size = self._calculate_perturbation_size_for_robustness(clean_examples_for_eval, adversarial_examples)
        else: # Attack failed to generate examples
            self.logger.warning(f"Adversarial examples for attack {attack_type} are None. Metrics will be NaN.")
            adversarial_accuracy = np.nan
            perturbation_size = np.nan

        if np.isnan(clean_accuracy) or np.isnan(adversarial_accuracy):
            attack_success_rate = np.nan
        else:
            # Standard definition: (Acc on clean - Acc on adv) / Acc on clean
            # Or simply 1 - adversarial_accuracy if attack is untargeted and aims to misclassify
            # RobustnessEvaluator used: 1 - adversarial_accuracy
            attack_success_rate = 1.0 - adversarial_accuracy
            # A common alternative: (Number of successful attacks) / (Number of samples correctly classified originally)
            # For now, stick to RobustnessEvaluator's simpler definition.

        actual_attack_time = time.time() - attack_start_time

        return EvaluationMetrics(
            clean_accuracy=float(clean_accuracy),
            adversarial_accuracy=float(adversarial_accuracy),
            attack_success_rate=float(attack_success_rate),
            perturbation_size=float(perturbation_size),
            attack_time=0.0,  # Per RobustnessEvaluator, though actual_attack_time is available
            model_type=model_metadata.model_type,
            attack_type=attack_type
        )
