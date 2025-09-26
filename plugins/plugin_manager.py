"""
Plugin Manager - High-level interface for plugin system management

This module provides a convenient high-level interface for managing plugins,
integrating with the training system, and providing plugin orchestration.
"""

import logging
from typing import Dict, Any, List, Optional, Type
from core.plugin_system import (
    get_plugin_registry, PluginType, PluginStatus, BasePlugin,
    BattleStrategyPlugin, ExplorationPatternPlugin, RewardCalculatorPlugin
)
from core.event_system import EventType, Event, EventSubscriber, get_event_bus


class PluginManager(EventSubscriber):
    """High-level plugin management interface"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("PluginManager")
        self.registry = get_plugin_registry()
        self.event_bus = get_event_bus()

        # Plugin configurations
        self.plugin_configs = self.config.get('plugins', {})
        self.auto_discover = self.config.get('auto_discover', True)
        self.auto_load = self.config.get('auto_load', [])

        # Active plugin tracking
        self.active_battle_strategies = []
        self.active_exploration_patterns = []
        self.active_reward_calculators = []

        # Subscribe to plugin events
        self.event_bus.subscribe(self)

        self.logger.info("PluginManager initialized")

    def initialize(self) -> bool:
        """Initialize the plugin manager and load default plugins"""
        try:
            # Discover plugins if enabled
            if self.auto_discover:
                discovered = self.registry.discover_plugins()
                self.logger.info(f"Discovered {len(discovered)} plugins")

            # Auto-load specified plugins
            for plugin_id in self.auto_load:
                config = self.plugin_configs.get(plugin_id, {})
                if self.load_and_activate_plugin(plugin_id, config):
                    self.logger.info(f"Auto-loaded plugin: {plugin_id}")
                else:
                    self.logger.warning(f"Failed to auto-load plugin: {plugin_id}")

            # Load default plugins if none specified
            if not self.auto_load:
                self._load_default_plugins()

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize plugin manager: {e}")
            return False

    def load_and_activate_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """Load and activate a plugin in one step"""
        if self.registry.load_plugin(plugin_id, config):
            if self.registry.activate_plugin(plugin_id):
                self._update_active_plugin_tracking()
                return True
        return False

    def deactivate_and_unload_plugin(self, plugin_id: str) -> bool:
        """Deactivate and unload a plugin in one step"""
        if self.registry.deactivate_plugin(plugin_id):
            if self.registry.unload_plugin(plugin_id):
                self._update_active_plugin_tracking()
                return True
        return False

    def get_battle_strategy_recommendation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get battle strategy recommendation from active battle plugins"""
        active_strategies = self.registry.get_active_plugins(PluginType.BATTLE_STRATEGY)

        if not active_strategies:
            return {
                'action': 5,  # Default A button
                'confidence': 0.3,
                'reasoning': "No battle strategy plugins active",
                'source': 'plugin_manager_fallback'
            }

        # Get recommendations from all active strategies
        recommendations = []
        for strategy in active_strategies:
            try:
                recommendation = strategy.recommend_move(game_state, battle_context.get('available_moves', []))
                recommendation['plugin_name'] = strategy.get_metadata().name
                recommendation['plugin_priority'] = strategy.get_metadata().priority
                recommendations.append(recommendation)
            except Exception as e:
                self.logger.error(f"Error getting recommendation from {strategy.__class__.__name__}: {e}")

        if not recommendations:
            return {
                'action': 5,
                'confidence': 0.3,
                'reasoning': "All battle strategy plugins failed",
                'source': 'plugin_manager_fallback'
            }

        # Choose best recommendation (highest confidence * priority)
        best_recommendation = max(
            recommendations,
            key=lambda r: r.get('confidence', 0) * r.get('plugin_priority', 5)
        )

        best_recommendation['source'] = 'plugin_manager'
        best_recommendation['total_strategies_consulted'] = len(recommendations)

        return best_recommendation

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get exploration direction from active exploration plugins"""
        active_patterns = self.registry.get_active_plugins(PluginType.EXPLORATION_PATTERN)

        if not active_patterns:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "No exploration pattern plugins active",
                'source': 'plugin_manager_fallback'
            }

        # Use the highest priority pattern
        pattern = max(active_patterns, key=lambda p: p.get_metadata().priority)

        try:
            direction_result = pattern.get_exploration_direction(game_state, exploration_context)
            direction_result['source'] = 'plugin_manager'
            direction_result['plugin_name'] = pattern.get_metadata().name
            return direction_result
        except Exception as e:
            self.logger.error(f"Error getting direction from {pattern.__class__.__name__}: {e}")
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': f"Exploration plugin failed: {e}",
                'source': 'plugin_manager_fallback'
            }

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate reward using active reward calculator plugins"""
        active_calculators = self.registry.get_active_plugins(PluginType.REWARD_CALCULATOR)

        if not active_calculators:
            # Fallback basic reward calculation
            return self._calculate_basic_reward(old_state, new_state, action)

        # Combine rewards from all active calculators
        total_reward = 0.0
        calculator_count = 0

        for calculator in active_calculators:
            try:
                reward = calculator.calculate_reward(old_state, new_state, action)
                priority = calculator.get_metadata().priority
                weighted_reward = reward * (priority / 10.0)  # Normalize priority to weight
                total_reward += weighted_reward
                calculator_count += 1
            except Exception as e:
                self.logger.error(f"Error calculating reward from {calculator.__class__.__name__}: {e}")

        if calculator_count > 0:
            return total_reward / calculator_count  # Average weighted rewards
        else:
            return self._calculate_basic_reward(old_state, new_state, action)

    def hot_reload_plugin(self, plugin_id: str) -> bool:
        """Hot reload a plugin during runtime"""
        self.logger.info(f"Hot reloading plugin: {plugin_id}")

        if self.registry.reload_plugin(plugin_id):
            self._update_active_plugin_tracking()

            # Publish plugin reloaded event
            event = Event(
                event_type=EventType.COMPONENT_INITIALIZED,
                timestamp=__import__('time').time(),
                source="plugin_manager",
                data={
                    'event_type': 'plugin_reloaded',
                    'plugin_id': plugin_id
                },
                priority=6
            )
            self.event_bus.publish(event)

            return True
        return False

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get comprehensive plugin system statistics"""
        registry_stats = self.registry.get_registry_stats()

        # Add plugin manager specific stats
        plugin_manager_stats = {
            'active_battle_strategies': len(self.active_battle_strategies),
            'active_exploration_patterns': len(self.active_exploration_patterns),
            'active_reward_calculators': len(self.active_reward_calculators),
            'auto_discover_enabled': self.auto_discover,
            'auto_load_count': len(self.auto_load),
            'configured_plugins': len(self.plugin_configs)
        }

        return {
            'registry': registry_stats,
            'manager': plugin_manager_stats
        }

    def list_available_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """List all available plugins with their metadata"""
        plugin_list = []
        plugin_ids = self.registry.list_plugins(plugin_type=plugin_type)

        for plugin_id in plugin_ids:
            plugin_info = self.registry.get_plugin_info(plugin_id)
            if plugin_info:
                plugin_list.append({
                    'id': plugin_id,
                    'name': plugin_info.metadata.name,
                    'version': plugin_info.metadata.version,
                    'description': plugin_info.metadata.description,
                    'author': plugin_info.metadata.author,
                    'type': plugin_info.metadata.plugin_type.value,
                    'status': plugin_info.status.value,
                    'priority': plugin_info.metadata.priority,
                    'hot_swappable': plugin_info.metadata.hot_swappable,
                    'tags': plugin_info.metadata.tags
                })

        return plugin_list

    def update_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration at runtime"""
        plugin_info = self.registry.get_plugin_info(plugin_id)
        if plugin_info and plugin_info.instance:
            return plugin_info.instance.update_config(config)
        return False

    def get_subscribed_events(self) -> set:
        """Return set of event types this subscriber is interested in"""
        return {
            EventType.COMPONENT_INITIALIZED,
            EventType.COMPONENT_SHUTDOWN,
            EventType.SYSTEM_ERROR
        }

    def handle_event(self, event: Event) -> None:
        """Handle events related to plugin management"""
        try:
            if event.event_type == EventType.SYSTEM_ERROR:
                self._handle_system_error_event(event)
            elif event.event_type in [EventType.COMPONENT_INITIALIZED, EventType.COMPONENT_SHUTDOWN]:
                if event.source == "plugin_registry":
                    self._handle_plugin_lifecycle_event(event)
        except Exception as e:
            self.logger.error(f"Error handling event in plugin manager: {e}")

    def shutdown(self) -> bool:
        """Shutdown plugin manager and all plugins"""
        try:
            # Deactivate and unload all plugins
            for plugin_type in PluginType:
                active_plugins = self.registry.list_plugins(
                    plugin_type=plugin_type,
                    status=PluginStatus.ACTIVE
                )
                for plugin_id in active_plugins:
                    self.deactivate_and_unload_plugin(plugin_id)

            self.logger.info("Plugin manager shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during plugin manager shutdown: {e}")
            return False

    def _load_default_plugins(self) -> None:
        """Load default plugins if none are specified"""
        default_plugins = [
            ('battle_strategy.balanced_battle', {}),
            ('exploration_pattern.systematic_sweep', {}),
        ]

        for plugin_id, config in default_plugins:
            if self.load_and_activate_plugin(plugin_id, config):
                self.logger.info(f"Loaded default plugin: {plugin_id}")

    def _update_active_plugin_tracking(self) -> None:
        """Update internal tracking of active plugins"""
        self.active_battle_strategies = [
            p.get_metadata().name for p in self.registry.get_active_plugins(PluginType.BATTLE_STRATEGY)
        ]
        self.active_exploration_patterns = [
            p.get_metadata().name for p in self.registry.get_active_plugins(PluginType.EXPLORATION_PATTERN)
        ]
        self.active_reward_calculators = [
            p.get_metadata().name for p in self.registry.get_active_plugins(PluginType.REWARD_CALCULATOR)
        ]

    def _calculate_basic_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Basic fallback reward calculation"""
        reward = 0.0

        # Basic progression rewards
        old_level = old_state.get('player_level', 0)
        new_level = new_state.get('player_level', 0)
        if new_level > old_level:
            reward += (new_level - old_level) * 100

        old_badges = old_state.get('badges_total', 0)
        new_badges = new_state.get('badges_total', 0)
        if new_badges > old_badges:
            reward += (new_badges - old_badges) * 1000

        # Small negative reward for each action to encourage efficiency
        reward -= 0.1

        return reward

    def _handle_system_error_event(self, event: Event) -> None:
        """Handle system error events"""
        error_source = event.data.get('component', 'unknown')
        if 'plugin' in error_source.lower():
            self.logger.warning(f"Plugin system error detected: {event.data}")
            # Could implement automatic plugin recovery here

    def _handle_plugin_lifecycle_event(self, event: Event) -> None:
        """Handle plugin lifecycle events"""
        event_type = event.data.get('event_type', '')
        plugin_id = event.data.get('plugin_id', 'unknown')

        if event_type in ['plugin_loaded', 'plugin_activated']:
            self._update_active_plugin_tracking()
            self.logger.debug(f"Plugin lifecycle: {event_type} - {plugin_id}")
        elif event_type in ['plugin_unloaded', 'plugin_deactivated']:
            self._update_active_plugin_tracking()
            self.logger.debug(f"Plugin lifecycle: {event_type} - {plugin_id}")


# Singleton instance for global access
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def initialize_plugin_manager(config: Dict[str, Any] = None) -> PluginManager:
    """Initialize the global plugin manager"""
    global _plugin_manager
    _plugin_manager = PluginManager(config)
    _plugin_manager.initialize()
    return _plugin_manager