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
from .attacks.base import AttackStrategy # Conceptual

# Type variables
T = TypeVar('T')
AnyModelType = TypeVar('AnyModelType')
DataType = TypeVar('DataType')

# Constants
MAX_WORKERS = 4

class Observer(Protocol):
    def update(self, event_type: str, data: Any) -> None: ...

class EvaluationObserver:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def update(self, event_type: str, data: Any) -> None:
        log_data_str = str(data)
        if len(log_data_str) > 200:
            log_data_str = log_data_str[:197] + "..."
        self.logger.info(f"Evaluation event: {event_type} - Data: {log_data_str}")

class EvasionAttackStrategy(AttackStrategy): # Made concrete for the example
    def __init__(self, attack_class: Type[Any], params: Dict[str, Any], attack_name: str = "UnnamedEvasionAttack"):
        self.attack_class = attack_class
        self.params = params
        self.attack_name = attack_name

    def execute(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[str, np.ndarray, float]:
        from art.utils import compute_success

        current_params = self.params.copy()
        if 'estimator' not in current_params and 'classifier' not in current_params:
            current_params['estimator'] = classifier
        elif 'classifier' in current_params and 'estimator' not in current_params :
            pass
        elif 'estimator' in current_params:
            current_params['estimator'] = classifier

        attack_instance = self.attack_class(**current_params)
        is_targeted = getattr(attack_instance, 'targeted', self.params.get('targeted', False))
        y_for_generate = labels
        if is_targeted:
            pass

        adversarial_examples = attack_instance.generate(x=data, y=y_for_generate)
        success_rate = compute_success(
            classifier, data, labels, adversarial_examples, targeted=is_targeted
        )
        return self.attack_name, adversarial_examples, float(success_rate)


class ARTEvaluator(BaseEvaluator, EvaluatorInterface): # type: ignore
    def __init__(
        self,
        model_obj: Any,
        config: EvaluationConfig,
        observers: Optional[List[EvaluationObserver]] = None
    ):
        self.model_obj = model_obj
        self.config = config
        self.observers = observers if observers is not None else []
        self.logger = LogManager().logger
        self._art_estimator_instance: Optional[Any] = None
        self._metrics_calculator = MetricsCalculator()
        
        if not self.observers:
            self.observers.append(EvaluationObserver(self.logger))

    def notify_observers(self, event_type: str, data: Any) -> None:
        for observer in self.observers:
            observer.update(event_type, data)

    @property
    def art_estimator(self) -> Any:
        if self._art_estimator_instance is None:
            input_shape = getattr(self.config, 'input_shape', None)
            nb_classes = getattr(self.config, 'nb_classes', None)
            loss_function = getattr(self.config, 'loss_function', None)
            # Get device_preference from EvaluationConfig
            device_pref = getattr(self.config, 'device_preference', None)

            if input_shape is None and hasattr(self.model_obj, 'input_shape') and isinstance(self.model_obj.input_shape, tuple) and len(self.model_obj.input_shape)>1:
                raw_shape = self.model_obj.input_shape
                input_shape = raw_shape[1:] if raw_shape[0] is None else raw_shape
                self.logger.info(f"Inferred input_shape from model_obj: {input_shape}")

            if nb_classes is None and hasattr(self.model_obj, 'output_shape') and isinstance(self.model_obj.output_shape, tuple) and self.model_obj.output_shape:
                raw_out_shape = self.model_obj.output_shape
                nb_classes = raw_out_shape[-1]
                self.logger.info(f"Inferred nb_classes from model_obj: {nb_classes}")

            if input_shape is None or nb_classes is None :
                self.logger.warning("input_shape or nb_classes not fully resolved for ClassifierFactory. ART estimator might be incorrect.")

            self._art_estimator_instance = ClassifierFactory.create_classifier(
                model=self.model_obj,
                model_type=self.config.model_type,
                framework=self.config.framework,
                input_shape=input_shape,
                nb_classes=nb_classes,
                loss_fn=loss_function,
                device_type=device_pref # Pass device_preference to the factory
            )
        return self._art_estimator_instance

    def evaluate_model(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        attacks: Optional[List[EvasionAttackStrategy]] = None,
        defences: Optional[List[Any]] = None
    ) -> EvaluationResult:
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
                self.logger.info(f"Conceptual: {len(defences)} defence strategies provided. Actual execution not implemented here.")
                self.notify_observers("defences_evaluated_conceptually", {"num_defences": len(defences)})

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
            for defence_name, defence_data_dict in defences_data.items():
                if defence_data_dict:
                    report_lines.append(f"  Defence: {defence_name}")
                    for metric_key, metric_value in defence_data_dict.items():
                        formatted_key = metric_key.replace('_', ' ').title()
                        if isinstance(metric_value, float):
                            report_lines.append(f"    - {formatted_key}: {metric_value:.4f}")
                        else:
                            report_lines.append(f"    - {formatted_key}: {metric_value}")
                else:
                    report_lines.append(f"  Defence: {defence_name}: Failed or No Results")
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
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer: EvaluationObserver) -> None:
        try:
            self.observers.remove(observer)
        except ValueError:
            self.logger.warning("Observer not found for removal.")
