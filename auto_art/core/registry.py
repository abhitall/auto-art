"""
Attack and Defense registry system with metadata, search, filtering, and deferred loading.

Design patterns:
- Registry pattern for centralized discovery and lookup
- Deferred/lazy loading to reduce startup time (inspired by Claude Code ToolRegistry)
- Builder pattern for structured metadata
- Strategy pattern for attack/defense interchangeability

References:
- Claude Code ToolRegistry: 67% token reduction via deferred loading
- Foolbox Attack ABC with __call__ + distance measures
- TextAttack AttackRecipe registry with build() factory
- Berkeley FCL: reducing options improves selection accuracy by 67%
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


class AttackCategory(Enum):
    """Category of adversarial attack."""
    EVASION = "evasion"
    POISONING = "poisoning"
    EXTRACTION = "extraction"
    INFERENCE = "inference"
    LLM = "llm"
    NLP = "nlp"
    AUDIO = "audio"
    AGENTIC = "agentic"


class ThreatModel(Enum):
    """Threat model for attack classification."""
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GREY_BOX = "grey_box"
    TRANSFER = "transfer"


class NormType(Enum):
    """Perturbation norm type."""
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    LINF = "Linf"
    WASSERSTEIN = "wasserstein"
    PERCEPTUAL = "perceptual"
    SEMANTIC = "semantic"
    NONE = "none"


class DefenseType(Enum):
    """Category of defense strategy."""
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    TRAINER = "trainer"
    DETECTOR = "detector"
    CERTIFIED = "certified"
    INPUT_SANITIZER = "input_sanitizer"
    GUARDRAIL = "guardrail"


class CostLevel(Enum):
    """Computational cost estimate."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass(frozen=True)
class AttackMetadata:
    """Structured metadata for an attack, inspired by Claude Code buildTool() pattern.

    Each attack is a first-class object with rich metadata enabling
    smart selection, filtering, and documentation generation.
    """
    name: str
    display_name: str
    category: AttackCategory
    threat_model: ThreatModel
    norm_type: NormType
    cost_estimate: CostLevel
    description: str
    is_gradient_based: bool = False
    requires_gpu: bool = False
    timeout_estimate_seconds: int = 60
    min_samples: int = 1
    supported_frameworks: Tuple[str, ...] = ("pytorch", "tensorflow", "sklearn")
    references: Tuple[str, ...] = ()
    owasp_mapping: Tuple[str, ...] = ()
    mitre_atlas_ids: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DefenseMetadata:
    """Structured metadata for a defense strategy."""
    name: str
    display_name: str
    defense_type: DefenseType
    cost_estimate: CostLevel
    description: str
    compatible_attacks: Tuple[str, ...] = ()
    requires_training: bool = False
    requires_gpu: bool = False
    certification_level: Optional[str] = None
    supported_frameworks: Tuple[str, ...] = ("pytorch", "tensorflow", "sklearn")
    references: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()


@dataclass
class _RegistryEntry:
    """Internal registry entry supporting deferred loading."""
    metadata: Union[AttackMetadata, DefenseMetadata]
    module_path: str
    class_name: str
    _resolved_class: Optional[Type] = field(default=None, repr=False)
    is_plugin: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._resolved_class is not None

    def resolve(self) -> Type:
        """Lazily import and return the implementation class."""
        if self._resolved_class is None:
            try:
                module = importlib.import_module(self.module_path)
                self._resolved_class = getattr(module, self.class_name)
                logger.debug(f"Loaded {self.class_name} from {self.module_path}")
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load {self.class_name} from {self.module_path}: {e}"
                ) from e
        return self._resolved_class


class BaseRegistry(ABC):
    """Abstract base registry with search, filter, and deferred loading.

    Inspired by Claude Code's ToolRegistry which achieves 67% token reduction
    through deferred loading — only importing modules when actually needed.
    """

    def __init__(self):
        self._entries: Dict[str, _RegistryEntry] = {}
        self._presets: Dict[str, List[str]] = {}

    def register(
        self,
        metadata: Union[AttackMetadata, DefenseMetadata],
        module_path: str,
        class_name: str,
        is_plugin: bool = False,
    ) -> None:
        """Register an attack/defense with deferred loading."""
        name = metadata.name
        if name in self._entries and not is_plugin:
            logger.warning(f"Overwriting existing registry entry: {name}")
        self._entries[name] = _RegistryEntry(
            metadata=metadata,
            module_path=module_path,
            class_name=class_name,
            is_plugin=is_plugin,
        )
        logger.debug(f"Registered {'plugin ' if is_plugin else ''}{name}")

    def unregister(self, name: str) -> bool:
        """Remove an entry from the registry."""
        if name in self._entries:
            del self._entries[name]
            return True
        return False

    def get(self, name: str) -> Type:
        """Get and resolve (lazily load) an implementation class by name."""
        if name not in self._entries:
            raise KeyError(
                f"'{name}' not found in registry. "
                f"Available: {sorted(self._entries.keys())}"
            )
        return self._entries[name].resolve()

    def get_metadata(self, name: str) -> Union[AttackMetadata, DefenseMetadata]:
        """Get metadata without loading the implementation."""
        if name not in self._entries:
            raise KeyError(f"'{name}' not found in registry.")
        return self._entries[name].metadata

    def list_all(self) -> List[str]:
        """List all registered names (no loading required)."""
        return sorted(self._entries.keys())

    def list_loaded(self) -> List[str]:
        """List only entries that have been resolved/loaded."""
        return sorted(
            name for name, entry in self._entries.items() if entry.is_loaded
        )

    def search(self, query: str) -> List[str]:
        """Search entries by name, display_name, description, or tags."""
        query_lower = query.lower()
        results = []
        for name, entry in self._entries.items():
            meta = entry.metadata
            searchable = (
                f"{meta.name} {meta.display_name} {meta.description} "
                f"{' '.join(getattr(meta, 'tags', ()))}"
            ).lower()
            if query_lower in searchable:
                results.append(name)
        return sorted(results)

    def add_preset(self, preset_name: str, entry_names: List[str]) -> None:
        """Define a named preset (curated subset of entries)."""
        for name in entry_names:
            if name not in self._entries:
                raise KeyError(f"'{name}' not in registry, cannot add to preset.")
        self._presets[preset_name] = entry_names

    def get_preset(self, preset_name: str) -> List[str]:
        """Get entry names in a preset."""
        if preset_name not in self._presets:
            raise KeyError(
                f"Preset '{preset_name}' not found. "
                f"Available: {sorted(self._presets.keys())}"
            )
        return self._presets[preset_name]

    def list_presets(self) -> List[str]:
        """List all available presets."""
        return sorted(self._presets.keys())

    @abstractmethod
    def _register_builtins(self) -> None:
        """Register all built-in entries. Called during initialization."""
        ...

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


class AttackRegistry(BaseRegistry):
    """Registry for adversarial attacks with rich metadata and filtering.

    Supports:
    - Deferred loading (import only when attack is instantiated)
    - Filtering by category, threat model, norm, GPU requirement
    - Presets for common evaluation scenarios
    - Plugin registration for third-party attacks
    """

    def filter_by_category(self, category: AttackCategory) -> List[str]:
        """Filter attacks by category (evasion, poisoning, etc.)."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, AttackMetadata)
            and entry.metadata.category == category
        )

    def filter_by_threat_model(self, threat_model: ThreatModel) -> List[str]:
        """Filter attacks by threat model."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, AttackMetadata)
            and entry.metadata.threat_model == threat_model
        )

    def filter_by_norm(self, norm_type: NormType) -> List[str]:
        """Filter attacks by norm type."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, AttackMetadata)
            and entry.metadata.norm_type == norm_type
        )

    def filter_by_cost(self, max_cost: CostLevel) -> List[str]:
        """Filter attacks by maximum cost level."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, AttackMetadata)
            and entry.metadata.cost_estimate.value <= max_cost.value
        )

    def filter_gpu_required(self, requires_gpu: bool = True) -> List[str]:
        """Filter attacks by GPU requirement."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, AttackMetadata)
            and entry.metadata.requires_gpu == requires_gpu
        )

    def filter(
        self,
        category: Optional[AttackCategory] = None,
        threat_model: Optional[ThreatModel] = None,
        norm_type: Optional[NormType] = None,
        max_cost: Optional[CostLevel] = None,
        requires_gpu: Optional[bool] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[str]:
        """Multi-criteria filter for attacks."""
        results = set(self._entries.keys())

        if category is not None:
            results &= set(self.filter_by_category(category))
        if threat_model is not None:
            results &= set(self.filter_by_threat_model(threat_model))
        if norm_type is not None:
            results &= set(self.filter_by_norm(norm_type))
        if max_cost is not None:
            results &= set(self.filter_by_cost(max_cost))
        if requires_gpu is not None:
            results &= set(self.filter_gpu_required(requires_gpu))
        if tags is not None:
            tag_matches = set()
            for name, entry in self._entries.items():
                meta = entry.metadata
                if isinstance(meta, AttackMetadata) and tags.issubset(set(meta.tags)):
                    tag_matches.add(name)
            results &= tag_matches

        return sorted(results)

    def _register_builtins(self) -> None:
        """Register all 60+ built-in attacks with full metadata."""
        _register_builtin_attacks(self)


class DefenseRegistry(BaseRegistry):
    """Registry for defense strategies with metadata and filtering."""

    def filter_by_type(self, defense_type: DefenseType) -> List[str]:
        """Filter defenses by type."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, DefenseMetadata)
            and entry.metadata.defense_type == defense_type
        )

    def filter_by_compatible_attack(self, attack_name: str) -> List[str]:
        """Find defenses effective against a specific attack."""
        return sorted(
            name for name, entry in self._entries.items()
            if isinstance(entry.metadata, DefenseMetadata)
            and attack_name in entry.metadata.compatible_attacks
        )

    def filter(
        self,
        defense_type: Optional[DefenseType] = None,
        max_cost: Optional[CostLevel] = None,
        requires_training: Optional[bool] = None,
        requires_gpu: Optional[bool] = None,
        compatible_with: Optional[str] = None,
    ) -> List[str]:
        """Multi-criteria filter for defenses."""
        results = set(self._entries.keys())

        if defense_type is not None:
            results &= set(self.filter_by_type(defense_type))
        if max_cost is not None:
            cost_match = {
                name for name, entry in self._entries.items()
                if isinstance(entry.metadata, DefenseMetadata)
                and entry.metadata.cost_estimate.value <= max_cost.value
            }
            results &= cost_match
        if requires_training is not None:
            train_match = {
                name for name, entry in self._entries.items()
                if isinstance(entry.metadata, DefenseMetadata)
                and entry.metadata.requires_training == requires_training
            }
            results &= train_match
        if requires_gpu is not None:
            gpu_match = {
                name for name, entry in self._entries.items()
                if isinstance(entry.metadata, DefenseMetadata)
                and entry.metadata.requires_gpu == requires_gpu
            }
            results &= gpu_match
        if compatible_with is not None:
            results &= set(self.filter_by_compatible_attack(compatible_with))

        return sorted(results)

    def _register_builtins(self) -> None:
        """Register all built-in defenses with full metadata."""
        _register_builtin_defenses(self)


# ---------------------------------------------------------------------------
# Compatibility matrix: attack <-> defense effectiveness
# ---------------------------------------------------------------------------

class CompatibilityMatrix:
    """Maps which defenses are effective against which attacks.

    Based on empirical results from RobustBench, AutoAttack benchmarks,
    and academic literature (Croce & Hein 2020, Rebuffi et al. 2021).
    """

    def __init__(
        self,
        attack_registry: AttackRegistry,
        defense_registry: DefenseRegistry,
    ):
        self._attack_reg = attack_registry
        self._defense_reg = defense_registry
        self._matrix: Dict[str, Dict[str, float]] = {}
        self._build_default_matrix()

    def _build_default_matrix(self) -> None:
        """Build default compatibility scores from defense metadata."""
        for def_name in self._defense_reg.list_all():
            meta = self._defense_reg.get_metadata(def_name)
            if isinstance(meta, DefenseMetadata):
                for atk_name in meta.compatible_attacks:
                    if atk_name not in self._matrix:
                        self._matrix[atk_name] = {}
                    self._matrix[atk_name][def_name] = 0.7  # default effectiveness

    def set_effectiveness(
        self, attack_name: str, defense_name: str, score: float
    ) -> None:
        """Set effectiveness score (0.0 = ineffective, 1.0 = fully mitigates)."""
        if attack_name not in self._matrix:
            self._matrix[attack_name] = {}
        self._matrix[attack_name][defense_name] = max(0.0, min(1.0, score))

    def get_effectiveness(self, attack_name: str, defense_name: str) -> float:
        """Get effectiveness of a defense against an attack."""
        return self._matrix.get(attack_name, {}).get(defense_name, 0.0)

    def recommend_defenses(
        self, attack_name: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Recommend top-k defenses for a given attack."""
        scores = self._matrix.get(attack_name, {})
        sorted_defenses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_defenses[:top_k]

    def recommend_attacks(
        self, defense_name: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find attacks most likely to bypass a given defense (lowest effectiveness)."""
        attack_scores = []
        for atk_name, defenses in self._matrix.items():
            if defense_name in defenses:
                attack_scores.append((atk_name, defenses[defense_name]))
        sorted_attacks = sorted(attack_scores, key=lambda x: x[1])
        return sorted_attacks[:top_k]


# ---------------------------------------------------------------------------
# Built-in attack registrations
# ---------------------------------------------------------------------------

def _register_builtin_attacks(registry: AttackRegistry) -> None:
    """Register all 60+ built-in attacks with metadata and deferred loading."""
    _base = "auto_art.core.attacks"

    # === EVASION ATTACKS ===
    evasion_attacks = [
        AttackMetadata(
            name="fgsm", display_name="FGSM",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.VERY_LOW,
            description="Fast Gradient Sign Method — single-step gradient attack",
            is_gradient_based=True, timeout_estimate_seconds=10,
            references=("Goodfellow et al. 2014",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("fast", "gradient", "baseline"),
        ),
        AttackMetadata(
            name="bim", display_name="BIM",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.LOW,
            description="Basic Iterative Method — iterative FGSM",
            is_gradient_based=True, timeout_estimate_seconds=30,
            references=("Kurakin et al. 2016",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("iterative", "gradient"),
        ),
        AttackMetadata(
            name="pgd", display_name="PGD",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Projected Gradient Descent — strongest first-order attack",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Madry et al. 2018",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("iterative", "gradient", "strong"),
        ),
        AttackMetadata(
            name="auto_pgd", display_name="Auto-PGD",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Auto-PGD with adaptive step size (part of AutoAttack)",
            is_gradient_based=True, timeout_estimate_seconds=180,
            references=("Croce & Hein 2020",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("iterative", "gradient", "adaptive"),
        ),
        AttackMetadata(
            name="autoattack", display_name="AutoAttack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.HIGH,
            description="AutoAttack ensemble — parameter-free, reliable evaluation standard",
            is_gradient_based=True, timeout_estimate_seconds=600,
            references=("Croce & Hein 2020",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("ensemble", "reliable", "benchmark"),
        ),
        AttackMetadata(
            name="carlini_wagner_l2", display_name="C&W L2",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="Carlini & Wagner L2 — optimization-based, defeats distillation",
            is_gradient_based=True, timeout_estimate_seconds=300,
            references=("Carlini & Wagner 2017",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("optimization", "strong", "L2"),
        ),
        AttackMetadata(
            name="deepfool", display_name="DeepFool",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.MEDIUM,
            description="DeepFool — minimal perturbation to decision boundary",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Moosavi-Dezfooli et al. 2016",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("minimal", "boundary"),
        ),
        AttackMetadata(
            name="elastic_net", display_name="Elastic Net (EAD)",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L1, cost_estimate=CostLevel.HIGH,
            description="Elastic Net Attack — L1+L2 regularized optimization",
            is_gradient_based=True, timeout_estimate_seconds=300,
            references=("Chen et al. 2018",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("optimization", "L1", "sparse"),
        ),
        AttackMetadata(
            name="jsma", display_name="JSMA",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L0, cost_estimate=CostLevel.HIGH,
            description="Jacobian-based Saliency Map Attack — sparse perturbation",
            is_gradient_based=True, timeout_estimate_seconds=300,
            references=("Papernot et al. 2016",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("sparse", "saliency"),
        ),
        AttackMetadata(
            name="newtonfool", display_name="NewtonFool",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.MEDIUM,
            description="Newton's method-based fooling attack",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Jang et al. 2017",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("optimization",),
        ),
        AttackMetadata(
            name="virtual_adversarial", display_name="Virtual Adversarial",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.MEDIUM,
            description="Virtual Adversarial Training perturbation",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Miyato et al. 2018",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("regularization",),
        ),
        AttackMetadata(
            name="shadow_attack", display_name="Shadow Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.PERCEPTUAL, cost_estimate=CostLevel.VERY_HIGH,
            description="Shadow Attack — large but imperceptible perturbation",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            references=("Ghiasi et al. 2020",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("perceptual", "stealth"),
        ),
        AttackMetadata(
            name="wasserstein", display_name="Wasserstein Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.WASSERSTEIN, cost_estimate=CostLevel.HIGH,
            description="Wasserstein distance-based adversarial attack",
            is_gradient_based=True, timeout_estimate_seconds=300,
            references=("Wong et al. 2019",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("wasserstein", "distribution"),
        ),
        AttackMetadata(
            name="universal_perturbation", display_name="Universal Perturbation",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.VERY_HIGH,
            description="Universal adversarial perturbation — single perturbation fools all inputs",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=1800,
            min_samples=100,
            references=("Moosavi-Dezfooli et al. 2017",),
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("universal", "image-agnostic"),
        ),
        AttackMetadata(
            name="adversarial_patch", display_name="Adversarial Patch",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Adversarial Patch — physical-world patch attack",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            references=("Brown et al. 2017",),
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "patch"),
        ),
        # Black-box evasion attacks
        AttackMetadata(
            name="boundary_attack", display_name="Boundary Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="Decision-based boundary attack — no gradient access needed",
            timeout_estimate_seconds=600,
            references=("Brendel et al. 2018",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("decision-based", "query"),
        ),
        AttackMetadata(
            name="square_attack", display_name="Square Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Score-based square perturbation — query-efficient",
            timeout_estimate_seconds=300,
            references=("Andriushchenko et al. 2020",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("query-efficient", "score-based"),
        ),
        AttackMetadata(
            name="hopskipjump", display_name="HopSkipJump",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="HopSkipJump — decision-based boundary estimation",
            timeout_estimate_seconds=600,
            references=("Chen et al. 2020",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("decision-based", "query"),
        ),
        AttackMetadata(
            name="simba", display_name="SimBA",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Simple Black-box Attack — random direction search",
            timeout_estimate_seconds=300,
            references=("Guo et al. 2019",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("simple", "query"),
        ),
        AttackMetadata(
            name="zoo", display_name="ZOO",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.VERY_HIGH,
            description="Zeroth Order Optimization — gradient estimation via finite differences",
            timeout_estimate_seconds=900,
            references=("Chen et al. 2017",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("zeroth-order", "query-heavy"),
        ),
        AttackMetadata(
            name="geoda", display_name="GeoDA",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="Geometric Decision-based Attack",
            timeout_estimate_seconds=600,
            references=("Rahmati et al. 2020",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("geometric", "decision-based"),
        ),
        AttackMetadata(
            name="sign_opt", display_name="Sign-OPT",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.HIGH,
            description="Sign-OPT — sign gradient estimation for hard-label attacks",
            timeout_estimate_seconds=600,
            references=("Cheng et al. 2019",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("hard-label", "gradient-estimation"),
        ),
        AttackMetadata(
            name="pixel_attack", display_name="Pixel Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.L0, cost_estimate=CostLevel.MEDIUM,
            description="One/few pixel attack using differential evolution",
            timeout_estimate_seconds=300,
            references=("Su et al. 2019",),
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("sparse", "evolutionary"),
        ),
        AttackMetadata(
            name="threshold_attack", display_name="Threshold Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.LOW,
            description="Threshold-based binary perturbation attack",
            timeout_estimate_seconds=60,
            mitre_atlas_ids=("AML.T0043.002",),
            tags=("binary", "simple"),
        ),
        # Additional white-box evasion
        AttackMetadata(
            name="brendel_bethge", display_name="Brendel & Bethge",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.VERY_HIGH,
            description="Brendel & Bethge — gradient-based boundary refinement",
            is_gradient_based=True, timeout_estimate_seconds=900,
            references=("Brendel et al. 2019",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("boundary", "refined"),
        ),
        AttackMetadata(
            name="spatial_transformation", display_name="Spatial Transformation",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Spatial transformation attack — rotation, translation",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Engstrom et al. 2019",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("spatial", "geometric"),
        ),
        AttackMetadata(
            name="feature_adversaries", display_name="Feature Adversaries",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="Feature-space adversarial attack targeting internal representations",
            is_gradient_based=True, timeout_estimate_seconds=300,
            references=("Sabour et al. 2016",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("feature-space", "internal"),
        ),
        AttackMetadata(
            name="composite", display_name="Composite Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.VERY_HIGH,
            description="Composite adversarial attack combining multiple perturbation types",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("composite", "multi-perturbation"),
        ),
        AttackMetadata(
            name="auto_conjugate_gradient", display_name="Auto Conjugate Gradient",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Auto Conjugate Gradient method for adversarial attacks",
            is_gradient_based=True, timeout_estimate_seconds=180,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("conjugate", "gradient"),
        ),
        AttackMetadata(
            name="lowprofool", display_name="LowProFool",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.MEDIUM,
            description="Low-Profile Fooling — tabular data adversarial attack",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Ballet et al. 2019",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("tabular", "low-profile"),
        ),
        AttackMetadata(
            name="overload", display_name="Overload Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.MEDIUM,
            description="Overload attack targeting model computation",
            is_gradient_based=True, timeout_estimate_seconds=180,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("computational",),
        ),
        AttackMetadata(
            name="high_confidence_low_uncertainty",
            display_name="High Confidence Low Uncertainty",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.HIGH,
            description="Attack producing high-confidence adversarial examples",
            is_gradient_based=True, timeout_estimate_seconds=300,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("high-confidence",),
        ),
        AttackMetadata(
            name="laser_attack", display_name="Laser Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Laser beam physical adversarial attack",
            is_gradient_based=True, timeout_estimate_seconds=120,
            references=("Duan et al. 2021",),
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "laser"),
        ),
        AttackMetadata(
            name="frame_saliency", display_name="Frame Saliency",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.HIGH,
            description="Video frame saliency-based adversarial attack",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("video", "saliency"),
        ),
        AttackMetadata(
            name="decision_tree_attack", display_name="Decision Tree Attack",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=CostLevel.LOW,
            description="Adversarial attack on decision tree classifiers",
            timeout_estimate_seconds=30,
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("tree", "tabular"),
        ),
        # Object detection attacks
        AttackMetadata(
            name="dpatch", display_name="DPatch",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="DPatch — adversarial patch for object detectors",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            references=("Liu et al. 2019",),
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("patch", "object-detection"),
        ),
        AttackMetadata(
            name="robust_dpatch", display_name="Robust DPatch",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Robust DPatch — improved adversarial patch for detectors",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=600,
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("patch", "object-detection", "robust"),
        ),
        AttackMetadata(
            name="adversarial_texture", display_name="Adversarial Texture",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
            description="Adversarial texture attack for 3D physical-world scenarios",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=1800,
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "3D", "texture"),
        ),
        AttackMetadata(
            name="shapeshifter", display_name="ShapeShifter",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
            description="ShapeShifter — physical adversarial for object detectors",
            is_gradient_based=True, requires_gpu=True, timeout_estimate_seconds=1800,
            references=("Chen et al. 2019",),
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "object-detection"),
        ),
        AttackMetadata(
            name="graphite_blackbox", display_name="GRAPHITE (Black-box)",
            category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="GRAPHITE black-box physical-world attack",
            timeout_estimate_seconds=600,
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "graphite"),
        ),
        AttackMetadata(
            name="graphite_whitebox", display_name="GRAPHITE (White-box)",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="GRAPHITE white-box physical-world attack",
            is_gradient_based=True, timeout_estimate_seconds=600,
            mitre_atlas_ids=("AML.T0043.001",),
            tags=("physical", "graphite"),
        ),
    ]

    evasion_module_map = {
        "fgsm": (f"{_base}.evasion.fast_gradient_method", "FGSMWrapper"),
        "bim": (f"{_base}.evasion.bim", "BIMWrapper"),
        "pgd": (f"{_base}.evasion.fast_gradient_method", "PGDWrapper"),
        "auto_pgd": (f"{_base}.evasion.auto_pgd", "AutoPGDWrapper"),
        "autoattack": (f"{_base}.evasion.auto_attack", "AutoAttackWrapper"),
        "carlini_wagner_l2": (f"{_base}.evasion.carlini_wagner", "CarliniWagnerL2Wrapper"),
        "deepfool": (f"{_base}.evasion.fast_gradient_method", "DeepFoolWrapper"),
        "elastic_net": (f"{_base}.evasion.elastic_net", "ElasticNetWrapper"),
        "jsma": (f"{_base}.evasion.jsma", "JSMAWrapper"),
        "newtonfool": (f"{_base}.evasion.newtonfool", "NewtonFoolWrapper"),
        "virtual_adversarial": (f"{_base}.evasion.virtual_adversarial", "VirtualAdversarialWrapper"),
        "shadow_attack": (f"{_base}.evasion.shadow_attack", "ShadowAttackWrapper"),
        "wasserstein": (f"{_base}.evasion.wasserstein", "WassersteinAttackWrapper"),
        "universal_perturbation": (f"{_base}.evasion.universal_perturbation", "UniversalPerturbationWrapper"),
        "adversarial_patch": (f"{_base}.evasion.adversarial_patch", "AdversarialPatchWrapper"),
        "boundary_attack": (f"{_base}.evasion.boundary_attack", "BoundaryAttackWrapper"),
        "square_attack": (f"{_base}.evasion.blackbox", "SquareAttackWrapper"),
        "hopskipjump": (f"{_base}.evasion.blackbox", "HopSkipJumpWrapper"),
        "simba": (f"{_base}.evasion.blackbox", "SimBAWrapper"),
        "zoo": (f"{_base}.evasion.zoo", "ZOOWrapper"),
        "geoda": (f"{_base}.evasion.geometric_decision", "GeoDAWrapper"),
        "sign_opt": (f"{_base}.evasion.sign_opt", "SignOPTWrapper"),
        "pixel_attack": (f"{_base}.evasion.pixel_attack", "PixelAttackWrapper"),
        "threshold_attack": (f"{_base}.evasion.threshold_attack", "ThresholdAttackWrapper"),
        "brendel_bethge": (f"{_base}.evasion.brendel_bethge", "BrendelBethgeWrapper"),
        "spatial_transformation": (f"{_base}.evasion.spatial_transformation", "SpatialTransformationWrapper"),
        "feature_adversaries": (f"{_base}.evasion.feature_adversaries", "FeatureAdversariesWrapper"),
        "composite": (f"{_base}.evasion.composite", "CompositeAttackWrapper"),
        "auto_conjugate_gradient": (f"{_base}.evasion.auto_conjugate", "AutoConjugateGradientWrapper"),
        "lowprofool": (f"{_base}.evasion.lowprofool", "LowProFoolWrapper"),
        "overload": (f"{_base}.evasion.overload", "OverloadWrapper"),
        "high_confidence_low_uncertainty": (f"{_base}.evasion.high_confidence", "HighConfidenceLowUncertaintyWrapper"),
        "laser_attack": (f"{_base}.evasion.laser_attack", "LaserAttackWrapper"),
        "frame_saliency": (f"{_base}.evasion.frame_saliency", "FrameSaliencyWrapper"),
        "decision_tree_attack": (f"{_base}.evasion.decision_tree_attack", "DecisionTreeAttackWrapper"),
        "dpatch": (f"{_base}.evasion.dpatch", "DPatchWrapper"),
        "robust_dpatch": (f"{_base}.evasion.dpatch", "RobustDPatchWrapper"),
        "adversarial_texture": (f"{_base}.evasion.adversarial_texture", "AdversarialTextureWrapper"),
        "shapeshifter": (f"{_base}.evasion.shapeshifter", "ShapeShifterWrapper"),
        "graphite_blackbox": (f"{_base}.evasion.graphite", "GraphiteBlackboxWrapper"),
        "graphite_whitebox": (f"{_base}.evasion.graphite", "GraphiteWhiteboxWrapper"),
    }

    for meta in evasion_attacks:
        if meta.name in evasion_module_map:
            mod_path, cls_name = evasion_module_map[meta.name]
            registry.register(meta, mod_path, cls_name)

    # === POISONING ATTACKS ===
    poisoning_attacks = [
        AttackMetadata(
            name="backdoor", display_name="Backdoor Attack",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Backdoor poisoning — inject trigger pattern into training data",
            references=("Gu et al. 2017",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("backdoor", "training"),
        ),
        AttackMetadata(
            name="clean_label", display_name="Clean Label Attack",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Clean label poisoning — poison without modifying labels",
            references=("Shafahi et al. 2018",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("clean-label", "stealthy"),
        ),
        AttackMetadata(
            name="feature_collision", display_name="Feature Collision",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Feature collision poisoning attack",
            is_gradient_based=True,
            references=("Shafahi et al. 2018",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("feature-space",),
        ),
        AttackMetadata(
            name="gradient_matching", display_name="Gradient Matching",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
            description="Gradient matching poisoning — Witches' Brew",
            is_gradient_based=True, requires_gpu=True,
            references=("Geiping et al. 2021",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("gradient-matching", "witches-brew"),
        ),
        AttackMetadata(
            name="sleeper_agent", display_name="Sleeper Agent",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
            description="Sleeper Agent — dormant backdoor activated by trigger",
            requires_gpu=True,
            references=("Goldblum et al. 2022",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("sleeper", "dormant", "advanced"),
        ),
        AttackMetadata(
            name="hidden_trigger", display_name="Hidden Trigger Backdoor",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Hidden Trigger Backdoor — visually imperceptible trigger",
            references=("Saha et al. 2020",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("hidden-trigger", "stealthy"),
        ),
        AttackMetadata(
            name="bullseye_polytope", display_name="Bullseye Polytope",
            category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Bullseye Polytope clean-label poisoning",
            is_gradient_based=True,
            references=("Aghakhani et al. 2021",),
            mitre_atlas_ids=("AML.T0020",),
            owasp_mapping=("LLM04",),
            tags=("clean-label", "polytope"),
        ),
    ]

    poisoning_module_map = {
        "backdoor": (f"{_base}.poisoning.backdoor_attack", "BackdoorAttackWrapper"),
        "clean_label": (f"{_base}.poisoning.clean_label_attack", "CleanLabelAttackWrapper"),
        "feature_collision": (f"{_base}.poisoning.feature_collision_attack", "FeatureCollisionAttackWrapper"),
        "gradient_matching": (f"{_base}.poisoning.gradient_matching_attack", "GradientMatchingAttackWrapper"),
        "sleeper_agent": (f"{_base}.poisoning.sleeper_agent", "SleeperAgentWrapper"),
        "hidden_trigger": (f"{_base}.poisoning.hidden_trigger", "HiddenTriggerWrapper"),
        "bullseye_polytope": (f"{_base}.poisoning.bullseye_polytope", "BullseyePolytopeWrapper"),
    }

    for meta in poisoning_attacks:
        if meta.name in poisoning_module_map:
            mod_path, cls_name = poisoning_module_map[meta.name]
            registry.register(meta, mod_path, cls_name)

    # BadDet variants
    baddet_variants = [
        ("baddet_oga", "BadDet OGA", "BadDetOGAWrapper"),
        ("baddet_rma", "BadDet RMA", "BadDetRMAWrapper"),
        ("baddet_gma", "BadDet GMA", "BadDetGMAWrapper"),
        ("baddet_oda", "BadDet ODA", "BadDetODAWrapper"),
    ]
    for name, display, cls in baddet_variants:
        registry.register(
            AttackMetadata(
                name=name, display_name=display,
                category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
                norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
                description=f"{display} — object detection backdoor attack",
                mitre_atlas_ids=("AML.T0020",),
                owasp_mapping=("LLM04",),
                tags=("baddet", "object-detection", "backdoor"),
            ),
            f"{_base}.poisoning.baddet", cls,
        )

    # DGM variants
    for name, display, cls in [("dgm_red", "DGM ReD", "DGMReDWrapper"), ("dgm_trail", "DGM Trail", "DGMTrailWrapper")]:
        registry.register(
            AttackMetadata(
                name=name, display_name=display,
                category=AttackCategory.POISONING, threat_model=ThreatModel.WHITE_BOX,
                norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
                description=f"{display} — deep generative model poisoning",
                requires_gpu=True,
                mitre_atlas_ids=("AML.T0020",),
                tags=("generative", "dgm"),
            ),
            f"{_base}.poisoning.dgm", cls,
        )

    # === EXTRACTION ATTACKS ===
    extraction_attacks = [
        (AttackMetadata(
            name="copycat_cnn", display_name="CopyCat CNN",
            category=AttackCategory.EXTRACTION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="CopyCat CNN — model extraction via query distillation",
            requires_gpu=True,
            references=("Correia-Silva et al. 2018",),
            mitre_atlas_ids=("AML.T0024",),
            tags=("extraction", "distillation"),
        ), f"{_base}.extraction.copycat_cnn", "CopycatCNNWrapper"),
        (AttackMetadata(
            name="knockoff_nets", display_name="KnockoffNets",
            category=AttackCategory.EXTRACTION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="KnockoffNets — model stealing via active learning",
            requires_gpu=True,
            references=("Orekondy et al. 2019",),
            mitre_atlas_ids=("AML.T0024",),
            tags=("extraction", "active-learning"),
        ), f"{_base}.extraction.knockoff_nets", "KnockoffNetsWrapper"),
        (AttackMetadata(
            name="functionally_equivalent_extraction",
            display_name="Functionally Equivalent Extraction",
            category=AttackCategory.EXTRACTION, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.VERY_HIGH,
            description="Exact model extraction for simple architectures",
            references=("Tramer et al. 2016",),
            mitre_atlas_ids=("AML.T0024",),
            tags=("extraction", "exact"),
        ), f"{_base}.extraction.functionally_equivalent_extraction", "FunctionallyEquivalentExtractionWrapper"),
    ]

    for meta, mod_path, cls_name in extraction_attacks:
        registry.register(meta, mod_path, cls_name)

    # === INFERENCE ATTACKS ===
    inference_attacks = [
        (AttackMetadata(
            name="membership_inference_bb", display_name="Membership Inference (Black-box)",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Membership inference — determine if data was in training set",
            references=("Shokri et al. 2017",),
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "membership"),
        ), f"{_base}.inference.membership_inference", "MembershipInferenceBlackBoxWrapper"),
        (AttackMetadata(
            name="attribute_inference_bb", display_name="Attribute Inference (Black-box)",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Attribute inference — infer sensitive attributes",
            references=("Fredrikson et al. 2014",),
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "attribute"),
        ), f"{_base}.inference.attribute_inference", "AttributeInferenceBlackBoxWrapper"),
        (AttackMetadata(
            name="model_inversion", display_name="Model Inversion (MIFace)",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Model Inversion — reconstruct training data representations",
            is_gradient_based=True,
            references=("Fredrikson et al. 2015",),
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "reconstruction"),
        ), f"{_base}.inference.model_inversion", "MIFaceWrapper"),
        (AttackMetadata(
            name="label_only_boundary", display_name="Label-Only Boundary Distance",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Label-only membership inference via boundary distance",
            references=("Choquette-Choo et al. 2021",),
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "label-only"),
        ), f"{_base}.inference.label_only", "LabelOnlyBoundaryDistanceWrapper"),
        (AttackMetadata(
            name="label_only_gap", display_name="Label-Only Gap Attack",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="Label-only gap attack for membership inference",
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "gap"),
        ), f"{_base}.inference.label_only", "LabelOnlyGapAttackWrapper"),
        (AttackMetadata(
            name="attribute_inference_wb_dt", display_name="Attribute Inference WB (DT)",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.LOW,
            description="White-box attribute inference via decision tree",
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "white-box", "decision-tree"),
        ), f"{_base}.inference.attribute_inference_wb", "AttributeInferenceWhiteBoxDTWrapper"),
        (AttackMetadata(
            name="db_reconstruction", display_name="Database Reconstruction",
            category=AttackCategory.INFERENCE, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.HIGH,
            description="Database reconstruction from model parameters",
            mitre_atlas_ids=("AML.T0025",),
            owasp_mapping=("LLM02",),
            tags=("privacy", "reconstruction", "database"),
        ), f"{_base}.inference.db_reconstruction", "DatabaseReconstructionWrapper"),
    ]

    for meta, mod_path, cls_name in inference_attacks:
        registry.register(meta, mod_path, cls_name)

    # === AUDIO ATTACKS ===
    audio_attacks = [
        (AttackMetadata(
            name="carlini_wagner_audio", display_name="C&W Audio",
            category=AttackCategory.AUDIO, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.L2, cost_estimate=CostLevel.VERY_HIGH,
            description="Carlini & Wagner audio adversarial attack on speech models",
            is_gradient_based=True, requires_gpu=True,
            references=("Carlini & Wagner 2018",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("audio", "speech"),
        ), f"{_base}.audio.carlini_wagner_audio", "CarliniWagnerAudioWrapper"),
        (AttackMetadata(
            name="imperceptible_asr", display_name="Imperceptible ASR",
            category=AttackCategory.AUDIO, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.PERCEPTUAL, cost_estimate=CostLevel.VERY_HIGH,
            description="Imperceptible adversarial attack on ASR systems",
            is_gradient_based=True, requires_gpu=True,
            references=("Qin et al. 2019",),
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("audio", "asr", "imperceptible"),
        ), f"{_base}.audio.imperceptible_asr", "ImperceptibleASRWrapper"),
    ]

    for meta, mod_path, cls_name in audio_attacks:
        registry.register(meta, mod_path, cls_name)

    # === LLM ATTACKS ===
    registry.register(
        AttackMetadata(
            name="hotflip", display_name="HotFlip",
            category=AttackCategory.LLM, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.NONE, cost_estimate=CostLevel.MEDIUM,
            description="HotFlip — token-level gradient-based text attack",
            is_gradient_based=True,
            references=("Ebrahimi et al. 2018",),
            mitre_atlas_ids=("AML.T0043.000",),
            owasp_mapping=("LLM01",),
            tags=("token", "gradient", "text"),
        ),
        f"{_base}.llm.hotflip", "HotFlipWrapper",
    )

    # === NLP ATTACKS ===
    registry.register(
        AttackMetadata(
            name="semantic_attack", display_name="Semantic Attack",
            category=AttackCategory.NLP, threat_model=ThreatModel.BLACK_BOX,
            norm_type=NormType.SEMANTIC, cost_estimate=CostLevel.MEDIUM,
            description="Semantic perturbation attack for NLP models",
            mitre_atlas_ids=("AML.T0043.000",),
            tags=("nlp", "semantic", "text"),
        ),
        f"{_base}.nlp.semantic_attack", "SemanticAttackWrapper",
    )

    # === PRESETS ===
    registry.add_preset("quick_scan", ["fgsm", "bim", "square_attack"])
    registry.add_preset("standard", [
        "fgsm", "pgd", "autoattack", "carlini_wagner_l2",
        "square_attack", "hopskipjump",
    ])
    registry.add_preset("comprehensive", [
        "fgsm", "pgd", "auto_pgd", "autoattack", "carlini_wagner_l2",
        "deepfool", "boundary_attack", "square_attack", "hopskipjump",
        "elastic_net", "jsma", "brendel_bethge",
    ])
    registry.add_preset("black_box_only", [
        "boundary_attack", "square_attack", "hopskipjump", "simba",
        "zoo", "geoda", "sign_opt", "pixel_attack",
    ])
    registry.add_preset("poisoning_suite", [
        "backdoor", "clean_label", "feature_collision",
        "gradient_matching", "sleeper_agent", "hidden_trigger",
    ])
    registry.add_preset("privacy_audit", [
        "membership_inference_bb", "attribute_inference_bb",
        "model_inversion", "label_only_boundary", "db_reconstruction",
    ])


def _register_builtin_defenses(registry: DefenseRegistry) -> None:
    """Register all built-in defenses with metadata."""
    _base = "auto_art.core.evaluation.defences"

    defenses = [
        # Preprocessors
        (DefenseMetadata(
            name="feature_squeezing", display_name="Feature Squeezing",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="Reduce color bit depth to remove adversarial perturbations",
            compatible_attacks=("fgsm", "bim", "pgd", "carlini_wagner_l2"),
            references=("Xu et al. 2018",),
            tags=("input", "squeezing"),
        ), f"{_base}.preprocessor", "FeatureSqueezingDefence"),
        (DefenseMetadata(
            name="spatial_smoothing", display_name="Spatial Smoothing",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="Apply spatial smoothing filter to remove small perturbations",
            compatible_attacks=("fgsm", "bim", "pgd"),
            references=("Xu et al. 2018",),
            tags=("input", "smoothing"),
        ), f"{_base}.preprocessor", "SpatialSmoothingDefence"),
        (DefenseMetadata(
            name="jpeg_compression", display_name="JPEG Compression",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="JPEG compression as adversarial defense",
            compatible_attacks=("fgsm", "bim", "pgd", "adversarial_patch"),
            references=("Dziugaite et al. 2016",),
            tags=("input", "compression"),
        ), f"{_base}.preprocessor", "JPEGCompressionDefence"),
        (DefenseMetadata(
            name="gaussian_augmentation", display_name="Gaussian Augmentation",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.LOW,
            description="Add Gaussian noise for robustness via randomization",
            compatible_attacks=("fgsm", "bim", "pgd", "carlini_wagner_l2"),
            tags=("input", "noise", "randomization"),
        ), f"{_base}.preprocessor_augmentation", "GaussianAugmentationDefence"),
        (DefenseMetadata(
            name="thermometer_encoding", display_name="Thermometer Encoding",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.LOW,
            description="Thermometer encoding for input transformation defense",
            compatible_attacks=("fgsm", "bim"),
            references=("Buckman et al. 2018",),
            tags=("input", "encoding"),
        ), f"{_base}.preprocessor_advanced", "ThermometerEncodingDefence"),
        (DefenseMetadata(
            name="total_variance_minimization",
            display_name="Total Variance Minimization",
            defense_type=DefenseType.PREPROCESSOR, cost_estimate=CostLevel.MEDIUM,
            description="Total variance minimization for denoising",
            compatible_attacks=("fgsm", "pgd", "carlini_wagner_l2"),
            tags=("input", "denoising"),
        ), f"{_base}.preprocessor_advanced", "TotalVarianceMinimizationDefence"),
        # Postprocessors
        (DefenseMetadata(
            name="high_confidence", display_name="High Confidence Filter",
            defense_type=DefenseType.POSTPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="Reject low-confidence predictions as potential adversarial",
            compatible_attacks=("fgsm", "bim", "pgd", "deepfool"),
            tags=("output", "confidence"),
        ), f"{_base}.postprocessor", "HighConfidenceDefence"),
        (DefenseMetadata(
            name="gaussian_noise_output", display_name="Gaussian Noise (Output)",
            defense_type=DefenseType.POSTPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="Add Gaussian noise to output logits for randomized smoothing",
            compatible_attacks=("carlini_wagner_l2",),
            tags=("output", "noise"),
        ), f"{_base}.postprocessor", "GaussianNoiseDefence"),
        (DefenseMetadata(
            name="rounded_output", display_name="Rounded Output",
            defense_type=DefenseType.POSTPROCESSOR, cost_estimate=CostLevel.VERY_LOW,
            description="Round output probabilities to reduce precision attacks",
            compatible_attacks=("carlini_wagner_l2",),
            tags=("output", "rounding"),
        ), f"{_base}.postprocessor_rounding", "RoundedOutputDefence"),
        # Trainers (adversarial training)
        (DefenseMetadata(
            name="adversarial_training_pgd", display_name="Adversarial Training (PGD)",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.HIGH,
            description="Standard adversarial training with PGD inner maximization",
            compatible_attacks=("fgsm", "bim", "pgd", "auto_pgd", "autoattack"),
            requires_training=True, requires_gpu=True,
            references=("Madry et al. 2018",),
            tags=("training", "pgd"),
        ), f"{_base}.trainer", "AdversarialTrainingDefence"),
        (DefenseMetadata(
            name="trades", display_name="TRADES",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.HIGH,
            description="TRADES — accuracy-robustness tradeoff via KL regularization",
            compatible_attacks=("fgsm", "pgd", "autoattack"),
            requires_training=True, requires_gpu=True,
            references=("Zhang et al. 2019",),
            tags=("training", "tradeoff"),
        ), f"{_base}.trainer_trades", "TRADESDefence"),
        (DefenseMetadata(
            name="awp", display_name="Adversarial Weight Perturbation",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.VERY_HIGH,
            description="AWP — perturb model weights during adversarial training",
            compatible_attacks=("pgd", "autoattack", "carlini_wagner_l2"),
            requires_training=True, requires_gpu=True,
            references=("Wu et al. 2020",),
            tags=("training", "weight-perturbation"),
        ), f"{_base}.trainer_awp", "AWPDefence"),
        (DefenseMetadata(
            name="certified_adversarial_training",
            display_name="Certified Adversarial Training",
            defense_type=DefenseType.CERTIFIED, cost_estimate=CostLevel.VERY_HIGH,
            description="Training with certified robustness guarantees",
            compatible_attacks=("fgsm", "pgd", "autoattack"),
            requires_training=True, requires_gpu=True,
            certification_level="IBP",
            tags=("training", "certified"),
        ), f"{_base}.trainer_certified", "CertifiedAdversarialTrainingDefence"),
        (DefenseMetadata(
            name="fbf", display_name="Fast is Better than Free (FBF)",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.MEDIUM,
            description="FBF — efficient adversarial training with free updates",
            compatible_attacks=("fgsm", "pgd"),
            requires_training=True, requires_gpu=True,
            references=("Shafahi et al. 2019",),
            tags=("training", "efficient"),
        ), f"{_base}.trainer_fbf", "FBFDefence"),
        (DefenseMetadata(
            name="oaat", display_name="OAAT",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.HIGH,
            description="Once-for-All Adversarial Training",
            compatible_attacks=("pgd", "autoattack"),
            requires_training=True, requires_gpu=True,
            tags=("training", "universal"),
        ), f"{_base}.trainer_oaat", "OAATDefence"),
        # Detectors
        (DefenseMetadata(
            name="activation_detector", display_name="Activation Detector",
            defense_type=DefenseType.DETECTOR, cost_estimate=CostLevel.MEDIUM,
            description="Detect adversarial examples via activation analysis",
            compatible_attacks=("fgsm", "pgd", "carlini_wagner_l2"),
            tags=("detection", "activation"),
        ), f"{_base}.detector", "ActivationDetectorDefence"),
        (DefenseMetadata(
            name="binary_input_detector", display_name="Binary Input Detector",
            defense_type=DefenseType.DETECTOR, cost_estimate=CostLevel.MEDIUM,
            description="Binary classifier detecting adversarial inputs",
            compatible_attacks=("fgsm", "pgd", "carlini_wagner_l2", "deepfool"),
            requires_training=True,
            tags=("detection", "binary"),
        ), f"{_base}.detector", "BinaryInputDetectorDefence"),
        # Transformer-based
        (DefenseMetadata(
            name="neural_cleanse", display_name="Neural Cleanse",
            defense_type=DefenseType.DETECTOR, cost_estimate=CostLevel.HIGH,
            description="Neural Cleanse — detect and reverse-engineer backdoor triggers",
            compatible_attacks=("backdoor", "hidden_trigger", "sleeper_agent"),
            requires_gpu=True,
            references=("Wang et al. 2019",),
            tags=("backdoor-detection", "reverse-engineer"),
        ), f"{_base}.transformer_cleanse", "NeuralCleanseDefence"),
        (DefenseMetadata(
            name="defensive_distillation", display_name="Defensive Distillation",
            defense_type=DefenseType.TRAINER, cost_estimate=CostLevel.HIGH,
            description="Defensive distillation — smooth model outputs to resist attacks",
            compatible_attacks=("fgsm", "jsma"),
            requires_training=True, requires_gpu=True,
            references=("Papernot et al. 2016",),
            tags=("distillation", "smoothing"),
        ), f"{_base}.transformer_distillation", "DefensiveDistillationDefence"),
        # Specialized
        (DefenseMetadata(
            name="rag_poisoning_detector", display_name="RAG Poisoning Detector",
            defense_type=DefenseType.DETECTOR, cost_estimate=CostLevel.MEDIUM,
            description="Detect poisoned documents in RAG retrieval pipeline",
            compatible_attacks=("backdoor",),
            owasp_mapping=("LLM08",),
            tags=("rag", "poisoning", "detection"),
        ), f"{_base}.rag_poisoning_detector", "RAGPoisoningDetector"),
        (DefenseMetadata(
            name="in_context_defence", display_name="In-Context Defence",
            defense_type=DefenseType.INPUT_SANITIZER, cost_estimate=CostLevel.LOW,
            description="Detect and neutralize in-context prompt injection",
            compatible_attacks=("hotflip",),
            owasp_mapping=("LLM01",),
            tags=("llm", "prompt-injection", "sanitizer"),
        ), f"{_base}.in_context_defence", "InContextDefence"),
        (DefenseMetadata(
            name="input_sanitizer", display_name="Input Sanitizer",
            defense_type=DefenseType.INPUT_SANITIZER, cost_estimate=CostLevel.LOW,
            description="Multi-layer input sanitization (DOM, visual, semantic)",
            compatible_attacks=("hotflip",),
            owasp_mapping=("LLM01", "LLM05"),
            tags=("sanitizer", "multi-layer"),
        ), f"{_base}.input_sanitizer", "InputSanitizer"),
    ]

    for meta, mod_path, cls_name in defenses:
        registry.register(meta, mod_path, cls_name)

    # Presets
    registry.add_preset("basic_defense", [
        "feature_squeezing", "spatial_smoothing", "high_confidence",
    ])
    registry.add_preset("adversarial_training", [
        "adversarial_training_pgd", "trades", "awp", "fbf",
    ])
    registry.add_preset("detection_suite", [
        "activation_detector", "binary_input_detector", "neural_cleanse",
    ])
    registry.add_preset("llm_defense", [
        "in_context_defence", "input_sanitizer", "rag_poisoning_detector",
    ])


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_attack_registry: Optional[AttackRegistry] = None
_defense_registry: Optional[DefenseRegistry] = None


def get_attack_registry() -> AttackRegistry:
    """Get the global attack registry singleton (lazy-initialized)."""
    global _attack_registry
    if _attack_registry is None:
        _attack_registry = AttackRegistry()
        _attack_registry._register_builtins()
    return _attack_registry


def get_defense_registry() -> DefenseRegistry:
    """Get the global defense registry singleton (lazy-initialized)."""
    global _defense_registry
    if _defense_registry is None:
        _defense_registry = DefenseRegistry()
        _defense_registry._register_builtins()
    return _defense_registry
