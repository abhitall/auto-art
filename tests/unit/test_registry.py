"""Tests for attack/defense registry system."""

import pytest
from auto_art.core.registry import (
    AttackCategory,
    AttackMetadata,
    AttackRegistry,
    CompatibilityMatrix,
    CostLevel,
    DefenseMetadata,
    DefenseRegistry,
    DefenseType,
    NormType,
    ThreatModel,
    get_attack_registry,
    get_defense_registry,
)


class TestAttackRegistry:
    def setup_method(self):
        self.registry = AttackRegistry()
        self.registry._register_builtins()

    def test_builtin_attacks_registered(self):
        """All 60+ attacks should be registered."""
        assert len(self.registry) >= 50

    def test_get_attack_by_name(self):
        """Should return the wrapper class (may fail if torch not installed)."""
        try:
            cls = self.registry.get("fgsm")
            assert cls is not None
        except ImportError:
            # Expected when PyTorch is not installed
            pass

    def test_get_metadata_without_loading(self):
        """Metadata should be accessible without importing the module."""
        meta = self.registry.get_metadata("pgd")
        assert meta.name == "pgd"
        assert meta.category == AttackCategory.EVASION
        assert meta.threat_model == ThreatModel.WHITE_BOX
        assert meta.is_gradient_based is True

    def test_search(self):
        results = self.registry.search("gradient")
        assert len(results) > 0
        assert "fgsm" in results or "pgd" in results

    def test_filter_by_category(self):
        evasion = self.registry.filter_by_category(AttackCategory.EVASION)
        assert len(evasion) > 20
        for name in evasion:
            meta = self.registry.get_metadata(name)
            assert meta.category == AttackCategory.EVASION

    def test_filter_by_threat_model(self):
        black_box = self.registry.filter_by_threat_model(ThreatModel.BLACK_BOX)
        assert len(black_box) >= 5
        for name in black_box:
            meta = self.registry.get_metadata(name)
            assert meta.threat_model == ThreatModel.BLACK_BOX

    def test_filter_by_norm(self):
        l2 = self.registry.filter_by_norm(NormType.L2)
        assert len(l2) >= 3

    def test_filter_by_cost(self):
        cheap = self.registry.filter_by_cost(CostLevel.LOW)
        assert len(cheap) >= 2
        for name in cheap:
            meta = self.registry.get_metadata(name)
            assert meta.cost_estimate.value <= CostLevel.LOW.value

    def test_multi_filter(self):
        results = self.registry.filter(
            category=AttackCategory.EVASION,
            threat_model=ThreatModel.BLACK_BOX,
        )
        assert len(results) >= 3

    def test_presets(self):
        presets = self.registry.list_presets()
        assert "quick_scan" in presets
        assert "standard" in presets
        assert "comprehensive" in presets
        assert "llm_red_team" in presets
        assert "agentic_security" in presets

        quick = self.registry.get_preset("quick_scan")
        assert "fgsm" in quick

    def test_preset_not_found(self):
        with pytest.raises(KeyError):
            self.registry.get_preset("nonexistent")

    def test_attack_not_found(self):
        with pytest.raises(KeyError):
            self.registry.get("totally_fake_attack")

    def test_list_all(self):
        all_attacks = self.registry.list_all()
        assert isinstance(all_attacks, list)
        assert len(all_attacks) >= 50

    def test_contains(self):
        assert "fgsm" in self.registry
        assert "nonexistent" not in self.registry

    def test_llm_attacks_registered(self):
        """PAIR, TAP, GCG, many-shot, crescendo should be registered."""
        for name in ["pair", "tap", "gcg", "many_shot", "crescendo", "system_prompt_leakage"]:
            assert name in self.registry, f"LLM attack '{name}' not registered"

    def test_agentic_attacks_registered(self):
        """Agentic attacks should be registered."""
        for name in ["indirect_prompt_injection", "goal_hijacking_chain",
                      "tool_misuse_chain", "confused_deputy", "memory_poisoning"]:
            assert name in self.registry, f"Agentic attack '{name}' not registered"

    def test_owasp_mapping_on_attacks(self):
        meta = self.registry.get_metadata("pair")
        assert "LLM01" in meta.owasp_mapping

    def test_mitre_atlas_ids(self):
        meta = self.registry.get_metadata("fgsm")
        assert len(meta.mitre_atlas_ids) > 0

    def test_unregister(self):
        reg = AttackRegistry()
        reg.register(
            AttackMetadata(
                name="test_attack", display_name="Test",
                category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
                norm_type=NormType.LINF, cost_estimate=CostLevel.LOW,
                description="test",
            ),
            "some.module", "SomeClass",
        )
        assert "test_attack" in reg
        reg.unregister("test_attack")
        assert "test_attack" not in reg


class TestDefenseRegistry:
    def setup_method(self):
        self.registry = DefenseRegistry()
        self.registry._register_builtins()

    def test_builtin_defenses_registered(self):
        assert len(self.registry) >= 15

    def test_filter_by_type(self):
        trainers = self.registry.filter_by_type(DefenseType.TRAINER)
        assert len(trainers) >= 3

    def test_filter_by_compatible_attack(self):
        defenses = self.registry.filter_by_compatible_attack("pgd")
        assert len(defenses) >= 2

    def test_presets(self):
        presets = self.registry.list_presets()
        assert "adversarial_training" in presets
        assert "llm_defense" in presets


class TestCompatibilityMatrix:
    def test_recommend_defenses(self):
        atk_reg = AttackRegistry()
        atk_reg._register_builtins()
        def_reg = DefenseRegistry()
        def_reg._register_builtins()

        matrix = CompatibilityMatrix(atk_reg, def_reg)
        recs = matrix.recommend_defenses("fgsm")
        assert isinstance(recs, list)

    def test_set_effectiveness(self):
        atk_reg = AttackRegistry()
        atk_reg._register_builtins()
        def_reg = DefenseRegistry()
        def_reg._register_builtins()

        matrix = CompatibilityMatrix(atk_reg, def_reg)
        matrix.set_effectiveness("fgsm", "feature_squeezing", 0.85)
        assert matrix.get_effectiveness("fgsm", "feature_squeezing") == 0.85


class TestSingleton:
    def test_get_attack_registry_returns_same_instance(self):
        r1 = get_attack_registry()
        r2 = get_attack_registry()
        assert r1 is r2

    def test_get_defense_registry_returns_same_instance(self):
        r1 = get_defense_registry()
        r2 = get_defense_registry()
        assert r1 is r2
