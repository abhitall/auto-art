"""
Adversarial attack generator module for creating and applying different types of attacks.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.attacks.inference import ModelInversion as ARTModelInversion
from art.estimators.classification import PyTorchClassifier, TensorFlowClassifier # Keep for type checks, but factory will create
from art.estimators.generation import PyTorchGenerator
from art.estimators.regression import PyTorchRegressor
# ClassifierNeuralNetwork will be imported conditionally where needed

from ...core.base import ModelMetadata
from ..interfaces import AttackConfig
# Use ART Enums for Framework and ModelType for type safety with ClassifierFactory
from ...core.evaluation.config.evaluation_config import ModelType as ARTModelTypeEnum, Framework as ARTFrameworkEnum
from ...core.evaluation.factories.classifier_factory import ClassifierFactory

# Evasion attack wrappers
from .evasion.auto_attack import AutoAttackWrapper
from .evasion.carlini_wagner import CarliniWagnerL2Wrapper
from .evasion.boundary_attack import BoundaryAttackWrapper

# Poisoning attack wrappers
from .poisoning.backdoor_attack import BackdoorAttackWrapper
from .poisoning.clean_label_attack import CleanLabelAttackWrapper
from .poisoning.feature_collision_attack import FeatureCollisionAttackWrapper
from .poisoning.gradient_matching_attack import GradientMatchingAttackWrapper

# Extraction attack wrappers
from .extraction.copycat_cnn import CopycatCNNWrapper
from .extraction.knockoff_nets import KnockoffNetsWrapper
from .extraction.functionally_equivalent_extraction import FunctionallyEquivalentExtractionWrapper

# Inference attack wrappers
from .inference.membership_inference import MembershipInferenceBlackBoxWrapper
from .inference.attribute_inference import AttributeInferenceBlackBoxWrapper
from .inference.model_inversion import MIFaceWrapper

# LLM attack wrappers
from .llm.hotflip import HotFlipWrapper

import torch # For default PyTorch loss in helper

class AttackGenerator:
    """Generates and applies appropriate adversarial attacks based on model type."""

    def __init__(self):
        self.supported_attacks = {
            'classification': ['fgsm', 'pgd', 'deepfool', 'autoattack', 'carlini_wagner_l2', 'boundary_attack'],
            'poisoning': ['backdoor', 'clean_label', 'feature_collision', 'gradient_matching'],
            'extraction': ['copycat_cnn', 'knockoff_nets', 'functionally_equivalent_extraction'],
            'inference': ['membership_inference_bb', 'attribute_inference_bb', 'model_inversion_miface'],
            'regression': ['fgsm', 'pgd'],
            'generator': ['inversion'], # ART's ModelInversion (targets a classifier usually)
            'llm': ['textfool', 'hotflip']
        }

    def _get_art_classifier_for_crafting(
        self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig, default_num_classes: int
    ) -> Any:
        """Helper to create ART classifier, used by poisoning and LLM attack creation."""
        art_input_shape = metadata.input_shape
        if isinstance(art_input_shape, tuple) and len(art_input_shape) > 0 and art_input_shape[0] is None:
            art_input_shape = art_input_shape[1:]

        try:
            framework_enum = ARTFrameworkEnum(metadata.framework.lower())
        except ValueError:
            raise ValueError(f"Invalid framework string '{metadata.framework}' for ART FrameworkEnum.")

        factory_kwargs = {
            'input_shape': art_input_shape,
            'nb_classes': default_num_classes, # Use the passed default_num_classes
            'loss_fn': config.additional_params.get('loss_fn') if metadata.framework == 'pytorch' else None,
            'device_type': config.additional_params.get('device_type_override')
        }
        factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}

        return ClassifierFactory.create_classifier(
            model=model_obj,
            model_type=ARTModelTypeEnum.CLASSIFICATION, # Assume these need a classifier interface
            framework=framework_enum,
            **factory_kwargs
        )

    def create_attack(self, model: Optional[Any], metadata: Optional[ModelMetadata], config: AttackConfig) -> Any:
        num_classes = 0
        if metadata: # Calculate once if metadata is available
            if metadata.output_shape and isinstance(metadata.output_shape[-1], int) and metadata.output_shape[-1] > 0:
                num_classes = metadata.output_shape[-1]
            elif metadata.model_type.lower() in ['classification', 'llm']: # LLMs for classification tasks
                num_classes = 2 # Default for classification/LLM if output_shape not conclusive

        attack_category = None
        attack_type_lower = config.attack_type.lower()
        for category, attacks in self.supported_attacks.items():
            if attack_type_lower in attacks:
                attack_category = category
                break

        if attack_category == 'classification':
            if model is None or metadata is None:
                 raise ValueError("Model and metadata are required for classification evasion attacks.")
            return self._create_classification_attack(model, metadata, config, num_classes)
        elif attack_category == 'poisoning':
            return self._create_poisoning_attack(model, metadata, config, num_classes) # Pass num_classes
        elif attack_category == 'extraction':
            if model is None:
                 raise ValueError("Victim model (ART estimator instance) is required for extraction attacks.")
            return self._create_extraction_attack(model, config)
        elif attack_category == 'inference':
            if model is None:
                 raise ValueError("Target model (ART estimator instance) is required for inference attacks.")
            return self._create_inference_attack(model, config)
        elif attack_category == 'regression':
            if model is None or metadata is None:
                 raise ValueError("Model and metadata are required for regression evasion attacks.")
            return self._create_regression_attack(model, metadata, config)
        elif attack_category == 'generator':
            # 'inversion' here refers to ART's ModelInversion which targets a classifier.
            if model is None: # Model is the target classifier (ART estimator)
                 raise ValueError("Target classifier (as ART estimator) is required for ModelInversion attack.")
            return self._create_generator_attack(model, metadata, config)
        elif attack_category == 'llm':
            if model is None or metadata is None:
                 raise ValueError(f"LLM attack type '{attack_type_lower}' requires a model and its metadata.")
            return self._create_llm_attack(model, metadata, config, num_classes)
        else:
            raise ValueError(f"Unsupported attack type: {config.attack_type} or category not determined.")

    def _create_classification_attack(self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig, num_classes: int) -> Any:
        art_input_shape = metadata.input_shape
        if isinstance(art_input_shape, tuple) and len(art_input_shape) > 0 and art_input_shape[0] is None:
            art_input_shape = art_input_shape[1:]
        try:
            framework_enum = ARTFrameworkEnum(metadata.framework.lower())
        except ValueError:
            raise ValueError(f"Invalid framework string '{metadata.framework}' for ART FrameworkEnum.")

        factory_kwargs = {
            'input_shape': art_input_shape, 'nb_classes': num_classes,
            'loss_fn': config.additional_params.get('loss_fn') if framework_enum == ARTFrameworkEnum.PYTORCH else None,
            'device_type': config.additional_params.get('device_type_override')
        }
        factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
        classifier = ClassifierFactory.create_classifier(model=model_obj, model_type=ARTModelTypeEnum.CLASSIFICATION,
                                                       framework=framework_enum, **factory_kwargs)

        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}
        if attack_type_lower == 'fgsm':
            return FastGradientMethod(estimator=classifier, eps=config.epsilon, targeted=config.targeted, batch_size=config.batch_size)
        elif attack_type_lower == 'pgd':
            norm_val = config.norm
            if isinstance(norm_val, str): norm_val = (np.inf if norm_val == 'inf' else int(norm_val.replace('l',''))) # type: ignore
            return ProjectedGradientDescent(estimator=classifier, eps=config.epsilon, eps_step=config.eps_step, max_iter=config.max_iter,
                                            targeted=config.targeted, num_random_init=config.num_random_init, batch_size=config.batch_size, norm=norm_val) # type: ignore
        elif attack_type_lower == 'deepfool':
            return DeepFool(classifier=classifier, epsilon=config.epsilon, max_iter=config.max_iter, batch_size=config.batch_size)
        elif attack_type_lower == 'autoattack':
            return AutoAttackWrapper(estimator=classifier, norm=config.norm, eps=config.epsilon, eps_step=config.eps_step,
                                     batch_size=config.batch_size, targeted=config.targeted, verbose=params.get('verbose', True), **params).attack
        elif attack_type_lower == 'carlini_wagner_l2':
            return CarliniWagnerL2Wrapper(estimator=classifier, confidence=config.confidence, targeted=config.targeted, learning_rate=config.learning_rate,
                                          binary_search_steps=config.binary_search_steps, max_iter=config.max_iter, initial_const=config.initial_const,
                                          max_halvings=params.get('max_halvings', 5), max_doublings=params.get('max_doublings', 5),
                                          batch_size=config.batch_size, verbose=params.get('verbose', True), **params.get('wrapper_specific_kwargs', {})).attack
        elif attack_type_lower == 'boundary_attack':
            return BoundaryAttackWrapper(estimator=classifier, targeted=config.targeted, delta=config.delta, epsilon=params.get('boundary_epsilon', 0.01),
                                         step_adapt=config.step_adapt, max_iter=config.max_iter, num_trial=params.get('num_trial', 25),
                                         sample_size=params.get('sample_size', 20), init_size=params.get('init_size', 100),
                                         verbose=params.get('verbose', True)).attack
        else:
            raise ValueError(f"Unsupported classification attack type: {config.attack_type}")

    def _create_poisoning_attack(self, model_for_crafting: Optional[Any],
                                 metadata_for_crafting_model: Optional[ModelMetadata],
                                 config: AttackConfig, num_classes: int) -> Any: # Added num_classes
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}

        classifier_for_crafting = None
        # These attacks require an ART classifier representing the model to be targeted for crafting poisons
        if attack_type_lower in ['clean_label', 'feature_collision', 'gradient_matching']:
            if model_for_crafting is None or metadata_for_crafting_model is None:
                raise ValueError(f"Poisoning attack {attack_type_lower} requires a model_for_crafting and its metadata.")
            classifier_for_crafting = self._get_art_classifier_for_crafting( # Renamed helper
                model_for_crafting, metadata_for_crafting_model, config, num_classes
            )

        # For type checking ART ClassifierNeuralNetwork
        _ClassifierNeuralNetwork = None
        try:
            from art.estimators.classification import ClassifierNeuralNetwork as ARTClassifierNeuralNetwork
            _ClassifierNeuralNetwork = ARTClassifierNeuralNetwork
        except ImportError:
            class DummyClassifierNeuralNetwork: pass # type: ignore
            _ClassifierNeuralNetwork = DummyClassifierNeuralNetwork # type: ignore

        if attack_type_lower == 'backdoor':
            return BackdoorAttackWrapper(backdoor_trigger_fn=params.get('backdoor_trigger_fn', lambda x: x + 0.1),
                                         target_class_idx=params.get('target_class_idx', 0), poisoning_rate=params.get('poisoning_rate', 0.1),
                                         **(params.get('wrapper_specific_kwargs', {})))
        elif attack_type_lower == 'clean_label':
            return CleanLabelAttackWrapper(target_classifier=classifier_for_crafting, backdoor_trigger_fn=params.get('backdoor_trigger_fn', lambda x: x),
                                           target_class_idx=params.get('target_class_idx', 0), poisoning_rate=params.get('poisoning_rate', 0.1),
                                           max_iter_attack=params.get('max_iter_attack', 100), max_iter_perturb=params.get('max_iter_perturb', 10),
                                           perturb_eps=params.get('perturb_eps', config.epsilon), batch_size=config.batch_size,
                                           verbose=params.get('verbose', True), **(params.get('wrapper_specific_kwargs', {})))
        elif attack_type_lower == 'feature_collision':
            if not isinstance(classifier_for_crafting, _ClassifierNeuralNetwork):
                raise TypeError(f"FeatureCollisionAttack requires a ClassifierNeuralNetwork, got {type(classifier_for_crafting)}.")
            return FeatureCollisionAttackWrapper(target_classifier=classifier_for_crafting, target_feature_layer=params.get('target_feature_layer', ''),
                                                 target_image=params.get('target_image', np.array([])), target_label_for_collision=params.get('target_label_for_collision', 0),
                                                 max_iter=config.max_iter, learning_rate=config.learning_rate, batch_size=config.batch_size,
                                                 verbose=params.get('verbose', True), poisoning_rate=params.get('poisoning_rate', 0.1),
                                                 **(params.get('wrapper_specific_kwargs', {})))
        elif attack_type_lower == 'gradient_matching':
            if not isinstance(classifier_for_crafting, _ClassifierNeuralNetwork):
                raise TypeError(f"GradientMatchingAttack requires a ClassifierNeuralNetwork, got {type(classifier_for_crafting)}.")
            return GradientMatchingAttackWrapper(target_classifier=classifier_for_crafting, learning_rate=config.learning_rate, max_iter=config.max_iter,
                                                 lambda_hyper=params.get('lambda_hyper', 0.1), batch_size=config.batch_size,
                                                 verbose=params.get('verbose', True), poisoning_rate=params.get('poisoning_rate', 0.1),
                                                 **(params.get('wrapper_specific_kwargs', {})))
        else:
            raise ValueError(f"Unsupported poisoning attack type: {config.attack_type}")

    def _create_extraction_attack(self, victim_classifier_obj: Any, config: AttackConfig) -> Any:
        # victim_classifier_obj is already an ART estimator
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}
        if attack_type_lower == 'copycat_cnn':
            return CopycatCNNWrapper(victim_classifier=victim_classifier_obj, batch_size_query=params.get('batch_size_query', config.batch_size),
                                     nb_epochs_copycat=params.get('nb_epochs_copycat', 10), nb_stolen_samples=params.get('nb_stolen_samples', 1000),
                                     use_probabilities=params.get('use_probabilities', True), verbose=params.get('verbose', True),
                                     **params.get('wrapper_specific_kwargs', {}))
        elif attack_type_lower == 'knockoff_nets':
            return KnockoffNetsWrapper(victim_classifier=victim_classifier_obj, batch_size_query=params.get('batch_size_query', config.batch_size),
                                       nb_epochs_thief=params.get('nb_epochs_thief', 10), nb_stolen_samples=params.get('nb_stolen_samples', 1000),
                                       use_probabilities=params.get('use_probabilities', True), verbose=params.get('verbose', True),
                                       **params.get('wrapper_specific_kwargs', {}))
        elif attack_type_lower == 'functionally_equivalent_extraction':
            return FunctionallyEquivalentExtractionWrapper(victim_classifier=victim_classifier_obj, num_neurons=params.get('num_neurons_thief', [128, 64]),
                                                           activation=params.get('activation_thief', 'relu'), verbose=params.get('verbose', True),
                                                           **params.get('wrapper_specific_kwargs', {}))
        else:
            raise ValueError(f"Unsupported extraction attack type: {config.attack_type}")

    def _create_inference_attack(self, target_model_obj: Any, config: AttackConfig) -> Any:
        # target_model_obj is already an ART estimator
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}
        if attack_type_lower == 'membership_inference_bb':
            return MembershipInferenceBlackBoxWrapper(target_model_estimator=target_model_obj, attack_model_type=params.get('attack_model_type', 'rf'),
                                                      attack_model_instance=params.get('attack_model_instance'), input_type=params.get('input_type', 'prediction'),
                                                      **params.get('wrapper_specific_kwargs', {}))
        elif attack_type_lower == 'attribute_inference_bb':
            if 'attack_feature_index' not in params:
                raise ValueError("AttributeInferenceBlackBox requires 'attack_feature_index' in additional_params.")
            return AttributeInferenceBlackBoxWrapper(target_model_estimator=target_model_obj, attack_feature_index=params['attack_feature_index'],
                                                     attack_model_type=params.get('attack_model_type', 'rf'), attack_model_instance=params.get('attack_model_instance'),
                                                     non_attack_feature_indices=params.get('non_attack_feature_indices'),
                                                     **params.get('wrapper_specific_kwargs', {}))
        elif attack_type_lower == 'model_inversion_miface':
            return MIFaceWrapper(target_classifier=target_model_obj, max_iter=params.get('max_iter', config.max_iter),
                                 batch_size=params.get('batch_size_miface', config.batch_size), learning_rate=params.get('learning_rate_miface', config.learning_rate),
                                 lambda_tv=params.get('lambda_tv', 0.1), lambda_l2=params.get('lambda_l2', 0.001),
                                 verbose=params.get('verbose', True), **params.get('wrapper_specific_kwargs', {}))
        else:
            raise ValueError(f"Unsupported inference attack type: {config.attack_type}")

    def _create_regression_attack(self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        art_input_shape = metadata.input_shape
        if isinstance(art_input_shape, tuple) and len(art_input_shape) > 0 and art_input_shape[0] is None:
            art_input_shape = art_input_shape[1:]

        device_override = config.additional_params.get('device_type_override')
        # Use ClassifierFactory's protected method for now.
        resolved_device_for_regressor = ClassifierFactory._determine_actual_device(device_override)

        if metadata.framework == 'pytorch':
            regressor = PyTorchRegressor(model=model_obj, loss=config.additional_params.get('loss_fn', torch.nn.MSELoss()),
                                         input_shape=art_input_shape, device_type=resolved_device_for_regressor)
        # TODO: Add TensorFlowRegressor and other framework regressors if ART supports them and they are needed.
        # elif metadata.framework in ['tensorflow', 'keras']:
        #     regressor = TensorFlowRegressor(...) # TensorFlowRegressor does not take device_type
        else:
            raise ValueError(f"Unsupported framework for regression: {metadata.framework}")

        attack_type_lower = config.attack_type.lower()
        if attack_type_lower == 'fgsm':
            return FastGradientMethod(estimator=regressor, eps=config.epsilon, targeted=config.targeted, batch_size=config.batch_size)
        elif attack_type_lower == 'pgd':
            norm_val = config.norm
            if isinstance(norm_val, str): norm_val = (np.inf if norm_val == 'inf' else int(norm_val.replace('l',''))) # type: ignore
            return ProjectedGradientDescent(estimator=regressor, eps=config.epsilon, eps_step=config.eps_step, max_iter=config.max_iter,
                                            targeted=config.targeted, num_random_init=config.num_random_init, batch_size=config.batch_size, norm=norm_val) # type: ignore
        else:
            raise ValueError(f"Unsupported attack type for regression: {config.attack_type}")

    def _create_generator_attack(self, model_estimator_or_raw_model: Any, metadata: Optional[ModelMetadata], config: AttackConfig) -> Any:
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}

        if attack_type_lower == 'inversion': # ART's ModelInversion targets a classifier to reconstruct inputs
            if not (hasattr(model_estimator_or_raw_model, 'input_shape') and hasattr(model_estimator_or_raw_model, 'predict')):
                raise ValueError("ART ModelInversion attack ('inversion' type in 'generator' category) requires a pre-configured ART *classifier* estimator as the 'model' argument.")
            # model_estimator_or_raw_model is already an ART estimator of the target classifier
            return ARTModelInversion(estimator=model_estimator_or_raw_model, max_iter=config.max_iter,
                                     batch_size=config.batch_size, **params)
        # If other attack types are added for "generator" models (e.g., to attack a GAN's generator directly)
        # then this part would need to handle wrapping the raw generator network.
        # Example:
        # elif metadata and metadata.model_type.lower() == 'generator' and metadata.framework == 'pytorch':
        #     if metadata is None: raise ValueError("Metadata required for PyTorchGenerator creation if raw model is passed.")
        #     art_input_shape = metadata.input_shape
        #     if isinstance(art_input_shape, tuple) and len(art_input_shape) > 0 and art_input_shape[0] is None:
        #         art_input_shape = art_input_shape[1:]
        #     device_override = config.additional_params.get('device_type_override')
        #     resolved_device = ClassifierFactory._determine_actual_device(device_override)
        #     return PyTorchGenerator(model=model_estimator_or_raw_model, input_shape=art_input_shape, device_type=resolved_device)
        else:
            raise ValueError(f"Unsupported attack type '{attack_type_lower}' or model setup for generator category.")

    def _create_llm_attack(self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig, num_classes: int) -> Any:
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}

        text_classifier_estimator = self._get_art_classifier_for_crafting( # Use the renamed helper
            model_obj, metadata, config, num_classes
        )

        if attack_type_lower == 'hotflip':
            vocab_size = metadata.additional_info.get('vocab_size', params.get('vocab_size'))
            if vocab_size is None:
                raise ValueError("HotFlip attack requires 'vocab_size'. Provide it in model metadata's additional_info or attack's additional_params.")
            return HotFlipWrapper(text_classifier_estimator=text_classifier_estimator, max_iter=config.max_iter, batch_size=config.batch_size,
                                  verbose=params.get('verbose', True), vocab_size=vocab_size, **params.get('wrapper_specific_kwargs', {})).attack
        # elif attack_type_lower == 'textfooler': # Example for future
            # return TextFoolerWrapper(...).attack
        else:
            raise ValueError(f"Unsupported LLM attack type: {config.attack_type}")

    def apply_attack(self, attack_instance: Any, test_inputs: np.ndarray,
                     test_labels: Optional[np.ndarray] = None,
                     batch_size_override: Optional[int] = None) -> np.ndarray:
        instance_class_name = attack_instance.__class__.__name__
        # Check if it's one of our specific wrappers that have their own generate/extract/infer methods
        if any(wrapper_name in instance_class_name for wrapper_name in
               ["BackdoorAttackWrapper", "CleanLabelAttackWrapper", "FeatureCollisionAttackWrapper", "GradientMatchingAttackWrapper",
                "CopycatCNNWrapper", "KnockoffNetsWrapper", "FunctionallyEquivalentExtractionWrapper",
                "MembershipInferenceBlackBoxWrapper", "AttributeInferenceBlackBoxWrapper", "MIFaceWrapper"]):

            expected_method = "'extract'" # Default for extraction
            if any(name_part in instance_class_name for name_part in ["Poisoning", "Backdoor", "CleanLabel", "FeatureCollision", "GradientMatching"]):
                expected_method = "'generate' (for data poisoning)"
            elif any(name_part in instance_class_name for name_part in ["Inference", "MIFace"]):
                expected_method = "'fit' and/or 'infer'"

            raise TypeError(f"Attack wrapper {instance_class_name} should not be used with apply_attack. "
                            f"Call its {expected_method} method directly with appropriate parameters.")

        # If not a special wrapper, assume it's a direct ART evasion attack object or evasion wrapper that returned .attack
        actual_art_attack_obj = getattr(attack_instance, 'attack', attack_instance)

        generate_kwargs = {}
        # ART's BoundaryAttack class name is 'BoundaryAttack'
        is_boundary = actual_art_attack_obj.__class__.__name__ == 'BoundaryAttack'

        if is_boundary:
            if batch_size_override is not None:
                 generate_kwargs['batch_size'] = batch_size_override

        is_targeted_attack = getattr(actual_art_attack_obj, 'targeted', False)
        # ART's AutoAttack class name is 'AutoAttack'
        is_autoattack = actual_art_attack_obj.__class__.__name__ == 'AutoAttack'

        if is_targeted_attack and test_labels is None and not is_autoattack:
            if not (hasattr(actual_art_attack_obj, '_allow_no_y_for_targeted') and actual_art_attack_obj._allow_no_y_for_targeted):
                 raise ValueError(f"Attack {actual_art_attack_obj.__class__.__name__} is targeted, but no target labels (y) provided.")

        y_for_generate = test_labels
        if is_autoattack and not is_targeted_attack:
            y_for_generate = None

        return actual_art_attack_obj.generate(x=test_inputs, y=y_for_generate, **generate_kwargs)
