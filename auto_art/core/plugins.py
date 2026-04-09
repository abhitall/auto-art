"""
Plugin architecture for Auto-ART with entry-point-based discovery.

Enables third-party extensions for attacks, defenses, and metrics without
modifying source code. Follows pytest plugin architecture and PEP 621
entry_points conventions.

Entry points group: "auto_art.plugins"
Sub-groups:
  - auto_art.attacks: Custom attack plugins
  - auto_art.defenses: Custom defense plugins
  - auto_art.metrics: Custom metric plugins
  - auto_art.reporters: Custom report format plugins

Usage in third-party pyproject.toml:
    [project.entry-points."auto_art.attacks"]
    my_attack = "my_package.attacks:register"

The register function receives the appropriate registry and registers entries.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .registry import (
    AttackMetadata,
    AttackRegistry,
    DefenseMetadata,
    DefenseRegistry,
    get_attack_registry,
    get_defense_registry,
)

logger = logging.getLogger(__name__)

PLUGIN_GROUPS = (
    "auto_art.attacks",
    "auto_art.defenses",
    "auto_art.metrics",
    "auto_art.reporters",
)


@runtime_checkable
class AttackPlugin(Protocol):
    """Protocol for attack plugins."""
    def register(self, registry: AttackRegistry) -> None: ...


@runtime_checkable
class DefensePlugin(Protocol):
    """Protocol for defense plugins."""
    def register(self, registry: DefenseRegistry) -> None: ...


class PluginManager:
    """Discovers and loads plugins from Python entry points and explicit registration.

    Lifecycle hooks allow plugins to inject behavior at key pipeline stages:
    - pre_attack: Before each attack execution
    - post_attack: After each attack execution
    - pre_report: Before report generation
    - post_report: After report generation
    - on_error: When an error occurs during execution
    """

    def __init__(self):
        self._loaded_plugins: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_attack": [],
            "post_attack": [],
            "pre_report": [],
            "post_report": [],
            "pre_evaluation": [],
            "post_evaluation": [],
            "on_error": [],
        }
        self._discovered = False

    def discover_plugins(self) -> Dict[str, List[str]]:
        """Discover all plugins from entry points (PEP 621).

        Returns:
            Dict mapping group names to lists of discovered plugin names.
        """
        discovered: Dict[str, List[str]] = {}

        for group in PLUGIN_GROUPS:
            discovered[group] = []
            try:
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns SelectableGroups, 3.9+ returns dict
                if hasattr(eps, 'select'):
                    group_eps = eps.select(group=group)
                elif isinstance(eps, dict):
                    group_eps = eps.get(group, [])
                else:
                    group_eps = [ep for ep in eps if ep.group == group]

                for ep in group_eps:
                    discovered[group].append(ep.name)
                    logger.debug(f"Discovered plugin: {group}/{ep.name}")
            except Exception as e:
                logger.warning(f"Error discovering plugins in {group}: {e}")

        self._discovered = True
        return discovered

    def load_plugins(self) -> int:
        """Load and register all discovered plugins.

        Returns:
            Number of successfully loaded plugins.
        """
        if not self._discovered:
            self.discover_plugins()

        loaded_count = 0

        for group in PLUGIN_GROUPS:
            try:
                eps = importlib.metadata.entry_points()
                if hasattr(eps, 'select'):
                    group_eps = eps.select(group=group)
                elif isinstance(eps, dict):
                    group_eps = eps.get(group, [])
                else:
                    group_eps = [ep for ep in eps if ep.group == group]

                for ep in group_eps:
                    try:
                        plugin_fn = ep.load()
                        self._apply_plugin(group, ep.name, plugin_fn)
                        self._loaded_plugins[f"{group}/{ep.name}"] = plugin_fn
                        loaded_count += 1
                        logger.info(f"Loaded plugin: {group}/{ep.name}")
                    except Exception as e:
                        logger.error(f"Failed to load plugin {group}/{ep.name}: {e}")

            except Exception as e:
                logger.warning(f"Error loading plugins from {group}: {e}")

        return loaded_count

    def _apply_plugin(self, group: str, name: str, plugin_fn: Any) -> None:
        """Apply a plugin to the appropriate registry."""
        if group == "auto_art.attacks":
            if callable(plugin_fn):
                plugin_fn(get_attack_registry())
            elif isinstance(plugin_fn, AttackPlugin):
                plugin_fn.register(get_attack_registry())
        elif group == "auto_art.defenses":
            if callable(plugin_fn):
                plugin_fn(get_defense_registry())
            elif isinstance(plugin_fn, DefensePlugin):
                plugin_fn.register(get_defense_registry())

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Args:
            event: Hook event name (pre_attack, post_attack, etc.)
            callback: Callable to invoke when the event fires.
        """
        if event not in self._hooks:
            raise ValueError(
                f"Unknown hook event '{event}'. "
                f"Available: {sorted(self._hooks.keys())}"
            )
        self._hooks[event].append(callback)
        logger.debug(f"Registered hook for '{event}': {callback}")

    def fire_hook(self, event: str, **kwargs: Any) -> List[Any]:
        """Fire all callbacks for a hook event.

        Args:
            event: Hook event name.
            **kwargs: Arguments passed to each callback.

        Returns:
            List of return values from callbacks.
        """
        results = []
        for callback in self._hooks.get(event, []):
            try:
                result = callback(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback error for '{event}': {e}")
                if event != "on_error":
                    self.fire_hook("on_error", error=e, source_event=event)
        return results

    def register_attack_plugin(
        self,
        metadata: AttackMetadata,
        module_path: str,
        class_name: str,
    ) -> None:
        """Programmatically register a third-party attack plugin."""
        registry = get_attack_registry()
        registry.register(metadata, module_path, class_name, is_plugin=True)

    def register_defense_plugin(
        self,
        metadata: DefenseMetadata,
        module_path: str,
        class_name: str,
    ) -> None:
        """Programmatically register a third-party defense plugin."""
        registry = get_defense_registry()
        registry.register(metadata, module_path, class_name, is_plugin=True)

    @property
    def loaded_plugin_names(self) -> List[str]:
        return sorted(self._loaded_plugins.keys())

    @property
    def hook_events(self) -> List[str]:
        return sorted(self._hooks.keys())


# Singleton
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager singleton."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
