"""
Plugins Package

This package contains all plugins for the Pokemon Crystal RL platform:
- Official plugins maintained by the core team
- Community plugins for extended functionality
- Plugin base classes and utilities
"""

from ..core.plugin_system import (
    BasePlugin,
    BattleStrategyPlugin,
    ExplorationPatternPlugin,
    RewardCalculatorPlugin,
    PluginType,
    PluginStatus,
    PluginMetadata,
    get_plugin_registry,
    initialize_plugin_system
)

__all__ = [
    'BasePlugin',
    'BattleStrategyPlugin',
    'ExplorationPatternPlugin',
    'RewardCalculatorPlugin',
    'PluginType',
    'PluginStatus',
    'PluginMetadata',
    'get_plugin_registry',
    'initialize_plugin_system'
]