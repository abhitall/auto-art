"""Tests for plugin architecture."""

import pytest
from auto_art.core.plugins import PluginManager, get_plugin_manager
from auto_art.core.registry import AttackMetadata, AttackCategory, ThreatModel, NormType, CostLevel


class TestPluginManager:
    def test_discover_plugins(self):
        pm = PluginManager()
        discovered = pm.discover_plugins()
        assert isinstance(discovered, dict)
        assert "auto_art.attacks" in discovered

    def test_hook_registration(self):
        pm = PluginManager()
        called = []

        def my_hook(**kwargs):
            called.append(kwargs)

        pm.register_hook("pre_attack", my_hook)
        pm.fire_hook("pre_attack", attack_name="fgsm")

        assert len(called) == 1
        assert called[0]["attack_name"] == "fgsm"

    def test_invalid_hook_event(self):
        pm = PluginManager()
        with pytest.raises(ValueError):
            pm.register_hook("nonexistent_event", lambda: None)

    def test_hook_events_list(self):
        pm = PluginManager()
        events = pm.hook_events
        assert "pre_attack" in events
        assert "post_attack" in events
        assert "on_error" in events

    def test_register_attack_plugin(self):
        pm = PluginManager()
        pm.register_attack_plugin(
            metadata=AttackMetadata(
                name="custom_attack", display_name="Custom Attack",
                category=AttackCategory.EVASION, threat_model=ThreatModel.BLACK_BOX,
                norm_type=NormType.LINF, cost_estimate=CostLevel.LOW,
                description="A custom plugin attack",
            ),
            module_path="some.module",
            class_name="CustomAttack",
        )

        from auto_art.core.registry import get_attack_registry
        reg = get_attack_registry()
        assert "custom_attack" in reg


class TestPluginManagerSingleton:
    def test_singleton(self):
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        assert pm1 is pm2
