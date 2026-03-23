"""
Adversarial attack generator module for creating and applying different types of attacks.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.attacks.inference.model_inversion import MIFace as ARTModelInversion
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier as TensorFlowClassifier # Keep for type checks
try:
    from art.estimators.generation import PyTorchGenerator
except ImportError:
    PyTorchGenerator = None  # Not available in this ART version
from art.estimators.regression import PyTorchRegressor, KerasRegressor
try:
    from art.estimators.regression import TensorFlowRegressor
except ImportError:
    TensorFlowRegressor = None  # Not available in this ART version
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
from .evasion.blackbox import SquareAttackWrapper, HopSkipJumpWrapper, SimBAWrapper

# Poisoning attack wrappers
from .poisoning.backdoor_attack import BackdoorAttackWrapper
from .poisoning.clean_label_attack import CleanLabelAttackWrapper
from .poisoning.feature_collision_attack import FeatureCollisionAttackWrapper
from .poisoning.gradient_matching_attack import GradientMatchingAttackWrapper
from .poisoning.baddet import BadDetOGAWrapper, BadDetRMAWrapper, BadDetGMAWrapper, BadDetODAWrapper
from .poisoning.dgm import DGMReDWrapper, DGMTrailWrapper

# Extraction attack wrappers
from .extraction.copycat_cnn import CopycatCNNWrapper
from .extraction.knockoff_nets import KnockoffNetsWrapper
from .extraction.functionally_equivalent_extraction import FunctionallyEquivalentExtractionWrapper

# Inference attack wrappers
from .inference.membership_inference import MembershipInferenceBlackBoxWrapper
from .inference.attribute_inference import AttributeInferenceBlackBoxWrapper
from .inference.model_inversion import MIFaceWrapper
from .inference.label_only import LabelOnlyBoundaryDistanceWrapper, LabelOnlyGapAttackWrapper
from .inference.attribute_inference_wb import AttributeInferenceWhiteBoxDTWrapper, AttributeInferenceWhiteBoxLifestyleDTWrapper
from .inference.db_reconstruction import DatabaseReconstructionWrapper

# Audio attack wrappers
from .audio.carlini_wagner_audio import CarliniWagnerAudioWrapper
from .audio.imperceptible_asr import ImperceptibleASRWrapper

# LLM attack wrappers
from .llm.hotflip import HotFlipWrapper

import torch # For default PyTorch loss in helper

class AttackGenerator:
    """Generates and applies appropriate adversarial attacks based on model type.

    The generator can create instances of ART attacks or custom wrappers defined
    within this module. It uses `AttackConfig` to determine the type and parameters
    of the attack to be created.
    """

    def __init__(self):
        """Initializes AttackGenerator and its list of supported attack types per category."""
        self.supported_attacks = {
            'classification': [
                'fgsm', 'pgd', 'deepfool', 'autoattack',
                'carlini_wagner_l2', 'boundary_attack',
                'square_attack', 'hopskipjump', 'simba',
                'bim', 'auto_pgd', 'adversarial_patch',
                'elastic_net', 'jsma', 'newtonfool',
                'universal_perturbation', 'zoo',
                'shadow_attack', 'wasserstein',
                'decision_tree_attack', 'brendel_bethge',
                'pixel_attack', 'threshold_attack',
                'spatial_transformation', 'feature_adversaries',
                'composite', 'geoda', 'overload',
                'sign_opt', 'lowprofool', 'virtual_adversarial',
                'auto_conjugate_gradient', 'laser_attack',
                'high_confidence_low_uncertainty',
                'graphite_blackbox', 'graphite_whitebox',
                'adversarial_texture', 'dpatch', 'robust_dpatch',
                'frame_saliency', 'shapeshifter',
            ],
            'poisoning': [
                'backdoor', 'clean_label', 'feature_collision', 'gradient_matching',
                'baddet_oga', 'baddet_rma', 'baddet_gma', 'baddet_oda',
                'dgm_red', 'dgm_trail',
                'sleeper_agent', 'hidden_trigger', 'bullseye_polytope',
            ],
            'extraction': ['copycat_cnn', 'knockoff_nets', 'functionally_equivalent_extraction'],
            'inference': [
                'membership_inference_bb', 'attribute_inference_bb', 'model_inversion_miface',
                'label_only_boundary', 'label_only_gap',
                'attribute_inference_wb_dt', 'attribute_inference_wb_lifestyle',
                'db_reconstruction',
            ],
            'regression': ['fgsm', 'pgd', 'bim'],
            'generator': ['inversion'],
            'audio': ['carlini_wagner_audio', 'imperceptible_asr'],
            'llm': ['hotflip']
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
        """Creates an attack instance based on the provided model, metadata, and configuration.

        The method determines the attack category (e.g., 'classification', 'poisoning') from the
        `config.attack_type`. It then dispatches to a category-specific helper method
        (e.g., `_create_classification_attack`, `_create_poisoning_attack`) to instantiate
        the attack.

        Args:
            model: The raw model object (e.g., PyTorch nn.Module, TensorFlow Keras model) or
                   an ART estimator instance. The type required depends on the attack category:
                   - Evasion (classification, regression), Poisoning (for crafting), LLM: Raw model object.
                   - Extraction, Inference, Generator (for ART's ModelInversion): ART estimator instance.
            metadata: ModelMetadata object containing information about the model. Required for
                      attacks that need to know model specifics like input/output shapes, framework, etc.,
                      especially when `model` is a raw model object.
            config: AttackConfig object specifying the `attack_type` and its parameters.
                    Direct fields from AttackConfig (e.g., `epsilon`, `max_iter`) are used where
                    applicable. Less common or attack-specific parameters should be provided via
                    `config.additional_params`.

        Returns:
            An instance of an ART attack class (e.g., art.attacks.evasion.FastGradientMethod) or
            an instance of a custom attack wrapper from this module (e.g., BackdoorAttackWrapper),
            depending on the attack type and category. For evasion attacks and some others,
            this often returns the direct ART attack object (e.g., wrapper_instance.attack).
            For wrappers managing their own execution (poisoning, extraction, some inference),
            it returns the wrapper instance itself.

        Raises:
            ValueError: If model/metadata requirements are not met for the attack type,
                        or if the attack type is unsupported.
            TypeError: If model types are incorrect for specific attacks (e.g., needing ClassifierNeuralNetwork).
        """
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
        elif attack_category == 'audio':
            if model is None:
                raise ValueError("Model (ART estimator) is required for audio attacks.")
            return self._create_audio_attack(model, config)
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
        elif attack_type_lower == 'square_attack':
            return SquareAttackWrapper(
                estimator=classifier, norm=str(config.norm), eps=config.epsilon,
                max_iter=config.max_iter, nb_restarts=params.get('nb_restarts', 1),
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'hopskipjump':
            return HopSkipJumpWrapper(
                estimator=classifier, targeted=config.targeted,
                norm=params.get('hopskipjump_norm', '2'),
                max_iter=config.max_iter, max_eval=params.get('max_eval', 10000),
                init_eval=params.get('init_eval', 100),
                init_size=params.get('init_size', 100),
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'simba':
            return SimBAWrapper(
                estimator=classifier, attack=params.get('simba_attack_type', 'dct'),
                max_iter=config.max_iter, epsilon=config.epsilon,
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'bim':
            from .evasion.bim import BasicIterativeMethodWrapper
            return BasicIterativeMethodWrapper(
                estimator=classifier, eps=config.epsilon, eps_step=config.eps_step,
                max_iter=config.max_iter, targeted=config.targeted,
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'auto_pgd':
            from .evasion.auto_pgd import AutoPGDWrapper
            return AutoPGDWrapper(
                estimator=classifier, eps=config.epsilon, eps_step=config.eps_step,
                max_iter=config.max_iter, targeted=config.targeted,
                batch_size=config.batch_size, verbose=params.get('verbose', True),
                nb_random_start=params.get('nb_random_start', 5),
                loss_type=params.get('loss_type', None),
            ).attack
        elif attack_type_lower == 'adversarial_patch':
            from .evasion.adversarial_patch import AdversarialPatchWrapper
            return AdversarialPatchWrapper(
                estimator=classifier, max_iter=config.max_iter,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size, targeted=config.targeted,
                rotation_max=params.get('rotation_max', 22.5),
                scale_min=params.get('scale_min', 0.3),
                scale_max=params.get('scale_max', 1.0),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'elastic_net':
            from .evasion.elastic_net import ElasticNetWrapper
            return ElasticNetWrapper(
                estimator=classifier, confidence=config.confidence,
                targeted=config.targeted, learning_rate=config.learning_rate,
                max_iter=config.max_iter, binary_search_steps=config.binary_search_steps,
                initial_const=config.initial_const, batch_size=config.batch_size,
                beta=params.get('beta', 0.001),
                decision_rule=params.get('decision_rule', 'EN'),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'jsma':
            from .evasion.jsma import JSMAWrapper
            return JSMAWrapper(
                estimator=classifier,
                theta=params.get('theta', 0.1),
                gamma=params.get('gamma', 1.0),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'newtonfool':
            from .evasion.newtonfool import NewtonFoolWrapper
            return NewtonFoolWrapper(
                estimator=classifier, max_iter=config.max_iter,
                eta=params.get('eta', 0.01),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'universal_perturbation':
            from .evasion.universal_perturbation import UniversalPerturbationWrapper
            return UniversalPerturbationWrapper(
                estimator=classifier, max_iter=config.max_iter,
                eps=config.epsilon, delta=params.get('delta', 0.2),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'zoo':
            from .evasion.zoo import ZOOWrapper
            return ZOOWrapper(
                estimator=classifier, confidence=config.confidence,
                targeted=config.targeted, learning_rate=config.learning_rate,
                max_iter=config.max_iter,
                binary_search_steps=config.binary_search_steps,
                initial_const=config.initial_const,
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'shadow_attack':
            from .evasion.shadow_attack import ShadowAttackWrapper
            return ShadowAttackWrapper(
                estimator=classifier, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'wasserstein':
            from .evasion.wasserstein import WassersteinAttackWrapper
            return WassersteinAttackWrapper(
                estimator=classifier, targeted=config.targeted,
                eps=config.epsilon, eps_step=config.eps_step,
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'decision_tree_attack':
            from .evasion.decision_tree_attack import DecisionTreeAttackWrapper
            return DecisionTreeAttackWrapper(
                classifier=classifier,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'brendel_bethge':
            from .evasion.brendel_bethge import BrendelBethgeWrapper
            return BrendelBethgeWrapper(
                estimator=classifier, targeted=config.targeted,
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'pixel_attack':
            from .evasion.pixel_attack import PixelAttackWrapper
            return PixelAttackWrapper(
                estimator=classifier, targeted=config.targeted,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'threshold_attack':
            from .evasion.threshold_attack import ThresholdAttackWrapper
            return ThresholdAttackWrapper(
                estimator=classifier, targeted=config.targeted,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'spatial_transformation':
            from .evasion.spatial_transformation import SpatialTransformationWrapper
            return SpatialTransformationWrapper(
                estimator=classifier,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'feature_adversaries':
            from .evasion.feature_adversaries import FeatureAdversariesWrapper
            return FeatureAdversariesWrapper(
                estimator=classifier, delta=params.get('delta', 0.2),
                layer=params.get('layer', -1),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'composite':
            from .evasion.composite import CompositeAdversarialAttackWrapper
            return CompositeAdversarialAttackWrapper(
                estimator=classifier,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'geoda':
            from .evasion.geometric_decision import GeoDAWrapper
            return GeoDAWrapper(
                estimator=classifier, max_iter=config.max_iter,
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'overload':
            from .evasion.overload import OverloadAttackWrapper
            return OverloadAttackWrapper(
                estimator=classifier, eps=config.epsilon,
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'sign_opt':
            from .evasion.sign_opt import SignOPTWrapper
            return SignOPTWrapper(
                estimator=classifier, targeted=config.targeted,
                epsilon=config.epsilon, max_iter=config.max_iter,
                num_trial=params.get('num_trial', 100), k=params.get('k', 200),
                alpha=params.get('alpha', 0.2), beta=params.get('beta', 0.001),
                batch_size=params.get('batch_size', config.batch_size),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'lowprofool':
            from .evasion.lowprofool import LowProFoolWrapper
            return LowProFoolWrapper(
                estimator=classifier, n_steps=params.get('n_steps', config.max_iter),
                threshold=params.get('threshold'), lambd=params.get('lambd', 1.5),
                eta=params.get('eta', 0.02), eta_decay=params.get('eta_decay', 0.98),
                eta_min=params.get('eta_min', 1e-7),
                norm=params.get('norm', 2),
                importance=params.get('importance'),
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'virtual_adversarial':
            from .evasion.virtual_adversarial import VirtualAdversarialWrapper
            return VirtualAdversarialWrapper(
                estimator=classifier, eps=params.get('eps', config.epsilon),
                finite_diff=params.get('finite_diff', 1e-6),
                max_iter=params.get('max_iter', 1),
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower in ('auto_conjugate_gradient', 'auto_cg'):
            from .evasion.auto_conjugate import AutoConjugateGradientWrapper
            return AutoConjugateGradientWrapper(
                estimator=classifier, eps=config.epsilon,
                eps_step=config.eps_step, max_iter=config.max_iter,
                targeted=config.targeted,
                nb_random_start=params.get('nb_random_start', 5),
                batch_size=config.batch_size,
                loss_type=params.get('loss_type'),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'laser_attack':
            from .evasion.laser_attack import LaserAttackWrapper
            return LaserAttackWrapper(
                estimator=classifier,
                iterations=params.get('iterations', 10),
                laser_generator=params.get('laser_generator'),
                image_generator=params.get('image_generator'),
                random_initializations=params.get('random_initializations', 1),
                optimisation_algorithm=params.get('optimisation_algorithm'),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower in ('high_confidence_low_uncertainty', 'hclu'):
            from .evasion.high_confidence import HighConfidenceLowUncertaintyWrapper
            return HighConfidenceLowUncertaintyWrapper(
                estimator=classifier,
                conf_threshold=params.get('conf_threshold', 0.75),
                unc_increase=params.get('unc_increase', 100.0),
                min_val=params.get('min_val', 0.0), max_val=params.get('max_val', 1.0),
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'graphite_blackbox':
            from .evasion.graphite import GRAPHITEBlackboxWrapper
            return GRAPHITEBlackboxWrapper(
                estimator=classifier,
                noise_budget=params.get('noise_budget', 0.1),
                num_xforms=params.get('num_xforms', 100),
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'graphite_whitebox':
            from .evasion.graphite import GRAPHITEWhiteboxWrapper
            return GRAPHITEWhiteboxWrapper(
                estimator=classifier,
                noise_budget=params.get('noise_budget', 0.1),
                num_xforms=params.get('num_xforms', 100),
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'adversarial_texture':
            from .evasion.adversarial_texture import AdversarialTextureWrapper
            return AdversarialTextureWrapper(
                estimator=classifier,
                patch_height=params.get('patch_height', 16),
                patch_width=params.get('patch_width', 16),
                x_min=params.get('x_min', 0.0), x_max=params.get('x_max', 1.0),
                step_size=params.get('step_size', 1.0 / 255),
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'dpatch':
            from .evasion.dpatch import DPatchWrapper
            return DPatchWrapper(
                estimator=classifier,
                patch_shape=tuple(params.get('patch_shape', (40, 40, 3))),
                learning_rate=params.get('learning_rate', 5.0),
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'robust_dpatch':
            from .evasion.dpatch import RobustDPatchWrapper
            return RobustDPatchWrapper(
                estimator=classifier,
                patch_shape=tuple(params.get('patch_shape', (40, 40, 3))),
                patch_location=params.get('patch_location'),
                crop_range=tuple(params.get('crop_range', (0, 0))),
                brightness_range=tuple(params.get('brightness_range', (1.0, 1.0))),
                rotation_weights=tuple(params.get('rotation_weights', (1, 0, 0, 0))),
                sample_size=params.get('sample_size', 1),
                learning_rate=params.get('learning_rate', 5.0),
                max_iter=config.max_iter, batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'frame_saliency':
            from .evasion.frame_saliency import FrameSaliencyWrapper
            inner_eps = params.get('inner_eps', 0.1)
            attacker = params.get('inner_attacker')
            if attacker is None:
                attacker = FastGradientMethod(
                    estimator=classifier, eps=inner_eps,
                    eps_step=params.get('inner_eps_step', inner_eps / 4.0),
                    batch_size=config.batch_size,
                )
            return FrameSaliencyWrapper(
                estimator=classifier, attacker=attacker,
                method=params.get('method', 'iterative_saliency'),
                frame_index=params.get('frame_index', 1),
                batch_size=config.batch_size, verbose=params.get('verbose', True),
            ).attack
        elif attack_type_lower == 'shapeshifter':
            from .evasion.shapeshifter import ShapeShifterWrapper
            return ShapeShifterWrapper(
                estimator=classifier,
                random_transform=params.get('random_transform'),
                batch_size=config.batch_size, max_iter=config.max_iter,
                texture_as_input=params.get('texture_as_input', False),
                verbose=params.get('verbose', True),
            ).attack
        else:
            raise ValueError(f"Unsupported classification attack type: {config.attack_type}")

    def _create_poisoning_attack(self, model_for_crafting: Optional[Any],
                                 metadata_for_crafting_model: Optional[ModelMetadata],
                                 config: AttackConfig, num_classes: int) -> Any:
        """Creates a poisoning attack instance.

        Args:
            model_for_crafting: The raw model object to be used for crafting some types of poisoning attacks
                                (e.g., clean_label, feature_collision, gradient_matching). This model's
                                gradients or internal states might be used by the attack.
            metadata_for_crafting_model: ModelMetadata for `model_for_crafting`.
            config: AttackConfig object.
            num_classes: Number of classes, used for creating ART estimator for crafting.

        Returns:
            An instance of a poisoning attack wrapper.
        """
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}

        classifier_for_crafting = None
        # These attacks require an ART classifier representing the model to be targeted for crafting poisons
        if attack_type_lower in [
            'clean_label', 'feature_collision', 'gradient_matching',
            'sleeper_agent', 'hidden_trigger', 'bullseye_polytope',
        ]:
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
        elif attack_type_lower == 'baddet_oga':
            return BadDetOGAWrapper(
                estimator=model_for_crafting, target_class=params.get('target_class', 0),
                poisoning_rate=params.get('poisoning_rate', 0.1),
                trigger_size=params.get('trigger_size', 30),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'baddet_rma':
            return BadDetRMAWrapper(
                estimator=model_for_crafting, target_class=params.get('target_class', 0),
                poisoning_rate=params.get('poisoning_rate', 0.1),
                trigger_size=params.get('trigger_size', 30),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'baddet_gma':
            return BadDetGMAWrapper(
                estimator=model_for_crafting, target_class=params.get('target_class', 0),
                poisoning_rate=params.get('poisoning_rate', 0.1),
                trigger_size=params.get('trigger_size', 30),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'baddet_oda':
            return BadDetODAWrapper(
                estimator=model_for_crafting, target_class=params.get('target_class', 0),
                poisoning_rate=params.get('poisoning_rate', 0.1),
                trigger_size=params.get('trigger_size', 30),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'dgm_red':
            if 'z_trigger' not in params or 'x_target' not in params:
                raise ValueError("DGMReD requires 'z_trigger' and 'x_target' in additional_params.")
            return DGMReDWrapper(
                generator=model_for_crafting, z_trigger=params['z_trigger'],
                x_target=params['x_target'],
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'dgm_trail':
            if 'z_trigger' not in params or 'x_target' not in params:
                raise ValueError("DGMTrail requires 'z_trigger' and 'x_target' in additional_params.")
            return DGMTrailWrapper(
                generator=model_for_crafting, z_trigger=params['z_trigger'],
                x_target=params['x_target'],
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'sleeper_agent':
            from .poisoning.sleeper_agent import SleeperAgentWrapper
            return SleeperAgentWrapper(
                classifier=classifier_for_crafting,
                percent_poison=params.get('percent_poison', 0.1),
                epsilon=params.get('epsilon', config.epsilon),
                max_trials=params.get('max_trials', 8),
                max_epochs=params.get('max_epochs', 250),
                learning_rate_schedule=params.get('learning_rate_schedule'),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            )
        elif attack_type_lower == 'hidden_trigger':
            from .poisoning.hidden_trigger import HiddenTriggerBackdoorWrapper
            return HiddenTriggerBackdoorWrapper(
                classifier=classifier_for_crafting,
                target=params.get('target', 0),
                source=params.get('source', 1),
                feature_layer=params.get('feature_layer', ''),
                eps=params.get('eps', config.epsilon),
                learning_rate=params.get('learning_rate', config.learning_rate),
                decay_coeff=params.get('decay_coeff', 0.5),
                decay_iter=params.get('decay_iter', 2000),
                max_iter=params.get('max_iter', config.max_iter),
                batch_size=config.batch_size,
                poison_percent=params.get('poison_percent', 0.1),
                verbose=params.get('verbose', True),
            )
        elif attack_type_lower == 'bullseye_polytope':
            if 'target' not in params:
                raise ValueError("bullseye_polytope requires 'target' (feature vector) in additional_params.")
            from .poisoning.bullseye_polytope import BullseyePolytopeWrapper
            return BullseyePolytopeWrapper(
                classifier=classifier_for_crafting,
                target=np.asarray(params['target']),
                feature_layer=params.get('feature_layer', ''),
                opt=params.get('opt', 'adam'),
                max_iter=params.get('max_iter', config.max_iter),
                learning_rate=params.get('learning_rate', config.learning_rate),
                momentum=params.get('momentum', 0.9),
                decay_iter=params.get('decay_iter', 10000),
                decay_coeff=params.get('decay_coeff', 0.5),
                epsilon=params.get('epsilon', config.epsilon),
                dropout=params.get('dropout', 0.3),
                net_repeat=params.get('net_repeat', 1),
                endtoend=params.get('endtoend', True),
                batch_size=config.batch_size,
                verbose=params.get('verbose', True),
            )
        else:
            raise ValueError(f"Unsupported poisoning attack type: {config.attack_type}")

    def _create_extraction_attack(self, victim_classifier_obj: Any, config: AttackConfig) -> Any:
        """Creates a model extraction attack instance.

        Args:
            victim_classifier_obj: An ART estimator instance representing the victim model.
            config: AttackConfig object.

        Returns:
            An instance of an extraction attack wrapper.
        """
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
        """Creates a model inference attack instance.

        Args:
            target_model_obj: An ART estimator instance representing the target model.
            config: AttackConfig object.

        Returns:
            An instance of an inference attack wrapper.
        """
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
        elif attack_type_lower == 'label_only_boundary':
            return LabelOnlyBoundaryDistanceWrapper(
                estimator=target_model_obj,
                distance_threshold_tau=params.get('distance_threshold_tau', 0.5),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'label_only_gap':
            return LabelOnlyGapAttackWrapper(
                estimator=target_model_obj,
                distance_threshold_tau=params.get('distance_threshold_tau', 0.5),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'attribute_inference_wb_dt':
            if 'attack_feature_index' not in params:
                raise ValueError("AttributeInferenceWhiteBoxDT requires 'attack_feature_index' in additional_params.")
            return AttributeInferenceWhiteBoxDTWrapper(
                estimator=target_model_obj,
                attack_feature_index=params['attack_feature_index'],
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'attribute_inference_wb_lifestyle':
            if 'attack_feature_index' not in params:
                raise ValueError("AttributeInferenceWhiteBoxLifestyleDT requires 'attack_feature_index' in additional_params.")
            return AttributeInferenceWhiteBoxLifestyleDTWrapper(
                estimator=target_model_obj,
                attack_feature_index=params['attack_feature_index'],
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'db_reconstruction':
            return DatabaseReconstructionWrapper(
                estimator=target_model_obj,
                **(params.get('wrapper_specific_kwargs', {})),
            )
        else:
            raise ValueError(f"Unsupported inference attack type: {config.attack_type}")

    def _create_regression_attack(self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig) -> Any:
        """Creates an evasion attack instance for regression models.

        Args:
            model_obj: The raw regression model object.
            metadata: ModelMetadata for `model_obj`.
            config: AttackConfig object.

        Returns:
            An instance of an ART evasion attack suitable for regression.
        """
        art_input_shape = metadata.input_shape
        if isinstance(art_input_shape, tuple) and len(art_input_shape) > 0 and art_input_shape[0] is None:
            art_input_shape = art_input_shape[1:]

        device_override = config.additional_params.get('device_type_override')
        # Use ClassifierFactory's protected method for now.
        resolved_device_for_regressor = ClassifierFactory._determine_actual_device(device_override)

        if metadata.framework == 'pytorch':
            regressor = PyTorchRegressor(model=model_obj, loss=config.additional_params.get('loss_fn', torch.nn.MSELoss()),
                                         input_shape=art_input_shape, device_type=resolved_device_for_regressor)
        elif metadata.framework == 'tensorflow':
            # TensorFlowRegressor does not take device_type. Loss function needs to be a TF one.
            # User should provide a suitable TF loss function in additional_params if not default.
            # For now, cannot assume tf is imported to provide a default tf.keras.losses.MeanSquaredError.
            # Let's make loss_fn mandatory in additional_params for TF/Keras if not providing a smart default here.
            tf_loss_fn = config.additional_params.get('loss_fn')
            if tf_loss_fn is None:
                # Consider raising an error or logging a warning and using a placeholder that ART might handle or ignore.
                # For now, let ART try to handle it or error out if loss is critical and not part of model.
                # print("Warning: No 'loss_fn' provided in additional_params for TensorFlowRegressor. Model must be compiled with loss or ART default used.", file=sys.stderr)
                pass # ART will use a default if model is not compiled with loss.
            regressor = TensorFlowRegressor(model=model_obj, input_shape=art_input_shape, loss=tf_loss_fn,
                                            preprocessing=getattr(model_obj, 'preprocess_input', None)) # Basic attempt to get preprocessing
        elif metadata.framework == 'keras':
            # KerasRegressor also does not take device_type.
            keras_loss_fn = config.additional_params.get('loss_fn')
            if keras_loss_fn is None:
                # print("Warning: No 'loss_fn' provided in additional_params for KerasRegressor. Model must be compiled with loss or ART default used.", file=sys.stderr)
                pass
            regressor = KerasRegressor(model=model_obj, input_shape=art_input_shape, loss=keras_loss_fn,
                                       preprocessing=getattr(model_obj, 'preprocess_input', None)) # Basic attempt
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
        """Creates an attack instance relevant to generative models or model inversion.

        Currently primarily supports ART's ModelInversion attack which targets a classifier.

        Args:
            model_estimator_or_raw_model: For 'inversion', this must be an ART classifier estimator.
                                           For future generator-specific attacks, this could be a raw generator model.
            metadata: ModelMetadata, potentially used if `model_estimator_or_raw_model` is a raw model.
            config: AttackConfig object.

        Returns:
            An instance of an ART attack (e.g., ModelInversion).
        """
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

    def _create_audio_attack(self, estimator_obj: Any, config: AttackConfig) -> Any:
        """Creates an audio adversarial attack instance.

        Args:
            estimator_obj: An ART estimator instance for the target ASR model.
            config: AttackConfig object.

        Returns:
            An instance of an audio attack wrapper.
        """
        attack_type_lower = config.attack_type.lower()
        params = config.additional_params or {}

        if attack_type_lower == 'carlini_wagner_audio':
            return CarliniWagnerAudioWrapper(
                estimator=estimator_obj, eps=config.epsilon,
                max_iter=config.max_iter, learning_rate=config.learning_rate,
                batch_size=config.batch_size, verbose=params.get('verbose', True),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        elif attack_type_lower == 'imperceptible_asr':
            return ImperceptibleASRWrapper(
                estimator=estimator_obj, eps=config.epsilon,
                max_iter=config.max_iter, learning_rate=config.learning_rate,
                batch_size=config.batch_size, verbose=params.get('verbose', True),
                **(params.get('wrapper_specific_kwargs', {})),
            )
        else:
            raise ValueError(f"Unsupported audio attack type: {config.attack_type}")

    def _create_llm_attack(self, model_obj: Any, metadata: ModelMetadata, config: AttackConfig, num_classes: int) -> Any:
        """Creates an attack instance for Large Language Models (LLMs).

        Args:
            model_obj: The raw LLM model object.
            metadata: ModelMetadata for `model_obj`.
            config: AttackConfig object.
            num_classes: Number of classes (used for wrapping LLM as a classifier).

        Returns:
            An instance of an ART LLM attack (via a wrapper, e.g., HotFlipWrapper's .attack property).
        """
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
        """Applies a generated evasion-style attack instance to input data.

        This method is intended for ART attack objects that have a `generate` method
        (e.g., evasion attacks like FGSM, PGD, or wrappers that expose such an ART object).
        It should NOT be used for attack wrappers that have their own distinct execution
        methods like `extract` (for extraction attacks) or `poison` (for some poisoning attacks).

        Args:
            attack_instance: The ART attack object or a wrapper instance whose `.attack`
                             property holds the ART attack object.
            test_inputs: Input data (numpy array) to generate adversarial examples for.
            test_labels: True labels for `test_inputs`. Required for targeted attacks
                         or some untargeted attacks that use true labels for guidance.
            batch_size_override: Optional batch size to use for this specific `generate` call,
                                 potentially overriding the batch size set at attack creation.
                                 Mainly relevant for attacks like BoundaryAttack.

        Returns:
            A numpy array containing the adversarial examples.

        Raises:
            TypeError: If `attack_instance` is a wrapper type that should use its own
                       specific execution method (e.g., `extract`, `infer`, `generate_poisons`)
                       instead of this generic `apply_attack`.
            ValueError: If a targeted attack requires `test_labels` but they are not provided.
        """
        instance_class_name = attack_instance.__class__.__name__
        # Check if it's one of our specific wrappers that have their own generate/extract/infer methods
        if any(wrapper_name in instance_class_name for wrapper_name in
               ["BackdoorAttackWrapper", "CleanLabelAttackWrapper", "FeatureCollisionAttackWrapper", "GradientMatchingAttackWrapper",
                "SleeperAgentWrapper", "HiddenTriggerBackdoorWrapper", "BullseyePolytopeWrapper",
                "CopycatCNNWrapper", "KnockoffNetsWrapper", "FunctionallyEquivalentExtractionWrapper",
                "MembershipInferenceBlackBoxWrapper", "AttributeInferenceBlackBoxWrapper", "MIFaceWrapper"]):

            expected_method = "'extract'" # Default for extraction
            if any(name_part in instance_class_name for name_part in ["Poisoning", "Backdoor", "CleanLabel", "FeatureCollision", "GradientMatching",
                                                                       "SleeperAgent", "HiddenTrigger", "BullseyePolytope"]):
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
