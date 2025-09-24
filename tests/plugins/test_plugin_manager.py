"""
Tests for Plugin Manager - High-level Plugin System Interface

This module contains comprehensive tests for the plugin manager which provides
the main interface for using the plugin system in training.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from plugins.plugin_manager import PluginManager, get_plugin_manager, initialize_plugin_manager
from core.plugin_system import PluginType, PluginStatus, PluginMetadata, BasePlugin, BattleStrategyPlugin
from core.event_system import EventType, Event


class TestBattlePlugin(BattleStrategyPlugin):
    """Test battle strategy plugin for testing"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.test_confidence = config.get('confidence', 0.8) if config else 0.8

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_battle",
            version="1.0.0",
            description="Test battle strategy",
            author="Test",
            plugin_type=PluginType.BATTLE_STRATEGY,
            priority=8
        )

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def analyze_battle_situation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'strategy_type': 'test',
            'confidence': self.test_confidence
        }

    def recommend_move(self, game_state: Dict[str, Any], available_moves: list) -> Dict[str, Any]:
        return {
            'action': 5,
            'confidence': self.test_confidence,
            'reasoning': 'Test battle recommendation',
            'move_type': 'attack'
        }

    def assess_switch_opportunity(self, game_state: Dict[str, Any], party: list) -> Dict[str, Any]:
        return {
            'should_switch': False,
            'confidence': self.test_confidence,
            'reasoning': 'Test switch assessment'
        }


class TestExplorationPlugin(BasePlugin):
    """Test exploration plugin for testing"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_exploration",
            version="1.0.0",
            description="Test exploration pattern",
            author="Test",
            plugin_type=PluginType.EXPLORATION_PATTERN,
            priority=7
        )

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'direction': 'right',
            'action': 4,
            'confidence': 0.9,
            'reasoning': 'Test exploration direction'
        }


class TestRewardPlugin(BasePlugin):
    """Test reward calculator plugin for testing"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_reward",
            version="1.0.0",
            description="Test reward calculator",
            author="Test",
            plugin_type=PluginType.REWARD_CALCULATOR,
            priority=6
        )

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        return 10.0


class TestPluginManager:
    """Test PluginManager functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.config = {
            'auto_discover': False,  # Disable auto-discovery for tests
            'auto_load': [],
            'plugins': {
                'battle_strategy.test': {'confidence': 0.9},
                'exploration_pattern.test': {'pattern': 'test'},
                'reward_calculator.test': {'weight': 1.5}
            }
        }
        self.plugin_manager = PluginManager(self.config)

    def test_plugin_manager_initialization(self):
        """Test plugin manager initialization"""
        assert self.plugin_manager.config == self.config
        assert self.plugin_manager.auto_discover is False
        assert len(self.plugin_manager.auto_load) == 0
        assert 'battle_strategy.test' in self.plugin_manager.plugin_configs

    def test_plugin_manager_initialize_without_auto_discovery(self):
        """Test initialization without auto-discovery"""
        with patch.object(self.plugin_manager.registry, 'discover_plugins') as mock_discover:
            result = self.plugin_manager.initialize()

            assert result is True
            mock_discover.assert_not_called()  # Should not be called when auto_discover=False

    def test_plugin_manager_initialize_with_auto_discovery(self):
        """Test initialization with auto-discovery"""
        self.plugin_manager.auto_discover = True

        with patch.object(self.plugin_manager.registry, 'discover_plugins', return_value=['plugin1', 'plugin2']) as mock_discover:
            result = self.plugin_manager.initialize()

            assert result is True
            mock_discover.assert_called_once()

    def test_plugin_manager_auto_load(self):
        """Test auto-loading specified plugins"""
        self.plugin_manager.auto_load = ['battle_strategy.test']

        with patch.object(self.plugin_manager, 'load_and_activate_plugin', return_value=True) as mock_load:
            result = self.plugin_manager.initialize()

            assert result is True
            mock_load.assert_called_once_with('battle_strategy.test', {'confidence': 0.9})

    def test_load_and_activate_plugin(self):
        """Test loading and activating a plugin"""
        plugin_id = 'battle_strategy.test'
        config = {'test_param': 'value'}

        with patch.object(self.plugin_manager.registry, 'load_plugin', return_value=True) as mock_load, \
             patch.object(self.plugin_manager.registry, 'activate_plugin', return_value=True) as mock_activate, \
             patch.object(self.plugin_manager, '_update_active_plugin_tracking') as mock_update:

            result = self.plugin_manager.load_and_activate_plugin(plugin_id, config)

            assert result is True
            mock_load.assert_called_once_with(plugin_id, config)
            mock_activate.assert_called_once_with(plugin_id)
            mock_update.assert_called_once()

    def test_load_and_activate_plugin_failure(self):
        """Test failure to load and activate plugin"""
        plugin_id = 'battle_strategy.test'

        with patch.object(self.plugin_manager.registry, 'load_plugin', return_value=False):
            result = self.plugin_manager.load_and_activate_plugin(plugin_id)

            assert result is False

    def test_deactivate_and_unload_plugin(self):
        """Test deactivating and unloading a plugin"""
        plugin_id = 'battle_strategy.test'

        with patch.object(self.plugin_manager.registry, 'deactivate_plugin', return_value=True) as mock_deactivate, \
             patch.object(self.plugin_manager.registry, 'unload_plugin', return_value=True) as mock_unload, \
             patch.object(self.plugin_manager, '_update_active_plugin_tracking') as mock_update:

            result = self.plugin_manager.deactivate_and_unload_plugin(plugin_id)

            assert result is True
            mock_deactivate.assert_called_once_with(plugin_id)
            mock_unload.assert_called_once_with(plugin_id)
            mock_update.assert_called_once()

    def test_get_battle_strategy_recommendation_with_plugins(self):
        """Test getting battle strategy recommendation with active plugins"""
        # Mock active battle strategy plugins
        test_plugin = TestBattlePlugin()
        mock_plugins = [test_plugin]

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=mock_plugins):
            game_state = {'player_hp': 100}
            battle_context = {'available_moves': [{'name': 'tackle', 'action': 5}]}

            recommendation = self.plugin_manager.get_battle_strategy_recommendation(game_state, battle_context)

            assert recommendation['action'] == 5
            assert recommendation['confidence'] == 0.8
            assert recommendation['source'] == 'plugin_manager'
            assert recommendation['plugin_name'] == 'test_battle'
            assert recommendation['total_strategies_consulted'] == 1

    def test_get_battle_strategy_recommendation_no_plugins(self):
        """Test getting battle strategy recommendation with no active plugins"""
        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=[]):
            recommendation = self.plugin_manager.get_battle_strategy_recommendation({}, {})

            assert recommendation['action'] == 5  # Default A button
            assert recommendation['confidence'] == 0.3
            assert recommendation['source'] == 'plugin_manager_fallback'
            assert 'No battle strategy plugins active' in recommendation['reasoning']

    def test_get_battle_strategy_recommendation_plugin_failure(self):
        """Test handling plugin failures in battle strategy recommendation"""
        # Create a plugin that raises an exception
        failing_plugin = Mock()
        failing_plugin.recommend_move.side_effect = Exception("Plugin failed")
        failing_plugin.get_metadata.return_value = PluginMetadata(
            name="failing_plugin", version="1.0.0", description="Failing",
            author="Test", plugin_type=PluginType.BATTLE_STRATEGY, priority=5
        )

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=[failing_plugin]):
            recommendation = self.plugin_manager.get_battle_strategy_recommendation({}, {})

            assert recommendation['action'] == 5
            assert recommendation['confidence'] == 0.3
            assert 'failed' in recommendation['reasoning']

    def test_get_battle_strategy_recommendation_multiple_plugins(self):
        """Test battle strategy recommendation with multiple plugins"""
        # Create multiple plugins with different priorities and confidences
        plugin1 = TestBattlePlugin({'confidence': 0.7})
        plugin1.get_metadata = lambda: PluginMetadata(
            "plugin1", "1.0.0", "Test", "Test", PluginType.BATTLE_STRATEGY, priority=5
        )

        plugin2 = TestBattlePlugin({'confidence': 0.9})
        plugin2.get_metadata = lambda: PluginMetadata(
            "plugin2", "1.0.0", "Test", "Test", PluginType.BATTLE_STRATEGY, priority=8
        )

        mock_plugins = [plugin1, plugin2]

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=mock_plugins):
            recommendation = self.plugin_manager.get_battle_strategy_recommendation({}, {})

            # Should choose plugin2 (higher confidence * priority = 0.9 * 8 = 7.2 vs 0.7 * 5 = 3.5)
            assert recommendation['plugin_name'] == 'plugin2'
            assert recommendation['confidence'] == 0.9
            assert recommendation['total_strategies_consulted'] == 2

    def test_get_exploration_direction_with_plugin(self):
        """Test getting exploration direction with active plugin"""
        test_plugin = TestExplorationPlugin()
        mock_plugins = [test_plugin]

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=mock_plugins):
            game_state = {'player_x': 10, 'player_y': 15}
            exploration_context = {'available_directions': ['right', 'up']}

            direction = self.plugin_manager.get_exploration_direction(game_state, exploration_context)

            assert direction['direction'] == 'right'
            assert direction['action'] == 4
            assert direction['confidence'] == 0.9
            assert direction['source'] == 'plugin_manager'
            assert direction['plugin_name'] == 'test_exploration'

    def test_get_exploration_direction_no_plugins(self):
        """Test getting exploration direction with no active plugins"""
        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=[]):
            direction = self.plugin_manager.get_exploration_direction({}, {})

            assert direction['direction'] == 'right'
            assert direction['action'] == 4
            assert direction['confidence'] == 0.3
            assert direction['source'] == 'plugin_manager_fallback'

    def test_get_exploration_direction_plugin_failure(self):
        """Test handling plugin failure in exploration direction"""
        failing_plugin = Mock()
        failing_plugin.get_exploration_direction.side_effect = Exception("Exploration failed")
        failing_plugin.get_metadata.return_value = PluginMetadata(
            "failing", "1.0.0", "Test", "Test", PluginType.EXPLORATION_PATTERN
        )

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=[failing_plugin]):
            direction = self.plugin_manager.get_exploration_direction({}, {})

            assert direction['direction'] == 'right'
            assert direction['action'] == 4
            assert direction['confidence'] == 0.3
            assert 'failed' in direction['reasoning']

    def test_calculate_reward_with_plugins(self):
        """Test reward calculation with active plugins"""
        test_plugin = TestRewardPlugin()
        mock_plugins = [test_plugin]

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=mock_plugins):
            old_state = {'player_level': 5}
            new_state = {'player_level': 6}

            reward = self.plugin_manager.calculate_reward(old_state, new_state, 1)

            assert reward == 10.0  # From test plugin

    def test_calculate_reward_multiple_plugins(self):
        """Test reward calculation with multiple plugins"""
        plugin1 = TestRewardPlugin()
        plugin1.calculate_reward = Mock(return_value=5.0)
        plugin1.get_metadata = lambda: PluginMetadata(
            "reward1", "1.0.0", "Test", "Test", PluginType.REWARD_CALCULATOR, priority=5
        )

        plugin2 = TestRewardPlugin()
        plugin2.calculate_reward = Mock(return_value=15.0)
        plugin2.get_metadata = lambda: PluginMetadata(
            "reward2", "1.0.0", "Test", "Test", PluginType.REWARD_CALCULATOR, priority=10
        )

        mock_plugins = [plugin1, plugin2]

        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=mock_plugins):
            reward = self.plugin_manager.calculate_reward({}, {}, 1)

            # Should be weighted average: (5.0 * 0.5 + 15.0 * 1.0) / 2 = 10.0
            assert reward == 10.0

    def test_calculate_reward_no_plugins(self):
        """Test reward calculation with no active plugins (fallback)"""
        with patch.object(self.plugin_manager.registry, 'get_active_plugins', return_value=[]), \
             patch.object(self.plugin_manager, '_calculate_basic_reward', return_value=2.5) as mock_basic:

            reward = self.plugin_manager.calculate_reward({}, {}, 1)

            assert reward == 2.5
            mock_basic.assert_called_once()

    def test_calculate_basic_reward_fallback(self):
        """Test basic reward calculation fallback"""
        old_state = {'player_level': 5, 'badges_total': 0}
        new_state = {'player_level': 6, 'badges_total': 1}

        reward = self.plugin_manager._calculate_basic_reward(old_state, new_state, 1)

        # Should be: level_gain(1) * 100 + badge_gain(1) * 1000 - action_penalty(0.1) = 1099.9
        expected_reward = 100 + 1000 - 0.1
        assert abs(reward - expected_reward) < 0.001

    def test_hot_reload_plugin(self):
        """Test hot reloading a plugin"""
        plugin_id = 'battle_strategy.test'

        with patch.object(self.plugin_manager.registry, 'reload_plugin', return_value=True) as mock_reload, \
             patch.object(self.plugin_manager, '_update_active_plugin_tracking') as mock_update:

            result = self.plugin_manager.hot_reload_plugin(plugin_id)

            assert result is True
            mock_reload.assert_called_once_with(plugin_id)
            mock_update.assert_called_once()

        # Should publish event
        with patch.object(self.plugin_manager.event_bus, 'publish') as mock_publish:
            self.plugin_manager.hot_reload_plugin(plugin_id)
            mock_publish.assert_called_once()

    def test_get_plugin_statistics(self):
        """Test getting plugin statistics"""
        mock_registry_stats = {
            'total_plugins': 5,
            'status_distribution': {'active': 3, 'loaded': 2},
            'type_distribution': {'battle_strategy': 2, 'exploration_pattern': 3}
        }

        with patch.object(self.plugin_manager.registry, 'get_registry_stats', return_value=mock_registry_stats):
            # Mock active plugin tracking
            self.plugin_manager.active_battle_strategies = ['battle1', 'battle2']
            self.plugin_manager.active_exploration_patterns = ['exploration1']
            self.plugin_manager.active_reward_calculators = []

            stats = self.plugin_manager.get_plugin_statistics()

            assert 'registry' in stats
            assert 'manager' in stats
            assert stats['registry'] == mock_registry_stats
            assert stats['manager']['active_battle_strategies'] == 2
            assert stats['manager']['active_exploration_patterns'] == 1
            assert stats['manager']['active_reward_calculators'] == 0

    def test_list_available_plugins(self):
        """Test listing available plugins"""
        mock_plugin_ids = ['battle_strategy.test1', 'exploration_pattern.test2']

        mock_plugin_info_1 = Mock()
        mock_plugin_info_1.metadata = PluginMetadata(
            "test1", "1.0.0", "Test battle", "Author", PluginType.BATTLE_STRATEGY,
            priority=8, hot_swappable=True, tags=["test", "battle"]
        )
        mock_plugin_info_1.status = PluginStatus.ACTIVE

        mock_plugin_info_2 = Mock()
        mock_plugin_info_2.metadata = PluginMetadata(
            "test2", "2.0.0", "Test exploration", "Author", PluginType.EXPLORATION_PATTERN,
            priority=6, hot_swappable=False, tags=["test", "exploration"]
        )
        mock_plugin_info_2.status = PluginStatus.LOADED

        with patch.object(self.plugin_manager.registry, 'list_plugins', return_value=mock_plugin_ids), \
             patch.object(self.plugin_manager.registry, 'get_plugin_info',
                         side_effect=[mock_plugin_info_1, mock_plugin_info_2]):

            plugins = self.plugin_manager.list_available_plugins()

            assert len(plugins) == 2

            plugin1 = plugins[0]
            assert plugin1['id'] == 'battle_strategy.test1'
            assert plugin1['name'] == 'test1'
            assert plugin1['type'] == 'battle_strategy'
            assert plugin1['status'] == 'active'
            assert plugin1['priority'] == 8
            assert plugin1['hot_swappable'] is True

            plugin2 = plugins[1]
            assert plugin2['id'] == 'exploration_pattern.test2'
            assert plugin2['name'] == 'test2'
            assert plugin2['type'] == 'exploration_pattern'
            assert plugin2['status'] == 'loaded'
            assert plugin2['hot_swappable'] is False

    def test_list_available_plugins_with_filter(self):
        """Test listing plugins with type filter"""
        with patch.object(self.plugin_manager.registry, 'list_plugins') as mock_list:
            self.plugin_manager.list_available_plugins(PluginType.BATTLE_STRATEGY)
            mock_list.assert_called_once_with(plugin_type=PluginType.BATTLE_STRATEGY)

    def test_update_plugin_config(self):
        """Test updating plugin configuration"""
        plugin_id = 'battle_strategy.test'
        new_config = {'new_param': 'new_value'}

        mock_plugin_info = Mock()
        mock_instance = Mock()
        mock_instance.update_config.return_value = True
        mock_plugin_info.instance = mock_instance

        with patch.object(self.plugin_manager.registry, 'get_plugin_info', return_value=mock_plugin_info):
            result = self.plugin_manager.update_plugin_config(plugin_id, new_config)

            assert result is True
            mock_instance.update_config.assert_called_once_with(new_config)

    def test_update_plugin_config_no_instance(self):
        """Test updating config when plugin has no instance"""
        plugin_id = 'battle_strategy.test'

        mock_plugin_info = Mock()
        mock_plugin_info.instance = None

        with patch.object(self.plugin_manager.registry, 'get_plugin_info', return_value=mock_plugin_info):
            result = self.plugin_manager.update_plugin_config(plugin_id, {})

            assert result is False

    def test_event_handling(self):
        """Test event handling functionality"""
        # Test system error event
        error_event = Event(
            EventType.SYSTEM_ERROR,
            1234567890.0,
            "test_component",
            data={'component': 'plugin_system', 'error': 'test error'}
        )

        # Should not raise exception
        self.plugin_manager.handle_event(error_event)

        # Test plugin lifecycle event
        lifecycle_event = Event(
            EventType.COMPONENT_INITIALIZED,
            1234567890.0,
            "plugin_registry",
            data={'event_type': 'plugin_loaded', 'plugin_id': 'test.plugin'}
        )

        with patch.object(self.plugin_manager, '_update_active_plugin_tracking') as mock_update:
            self.plugin_manager.handle_event(lifecycle_event)
            mock_update.assert_called_once()

    def test_shutdown(self):
        """Test plugin manager shutdown"""
        mock_active_plugins = {
            PluginType.BATTLE_STRATEGY: ['battle1', 'battle2'],
            PluginType.EXPLORATION_PATTERN: ['exploration1'],
            PluginType.REWARD_CALCULATOR: []
        }

        with patch.object(self.plugin_manager.registry, 'list_plugins') as mock_list, \
             patch.object(self.plugin_manager, 'deactivate_and_unload_plugin', return_value=True) as mock_deactivate:

            # Mock list_plugins to return active plugins for each type
            mock_list.side_effect = lambda plugin_type, status: mock_active_plugins.get(plugin_type, [])

            result = self.plugin_manager.shutdown()

            assert result is True
            # Should deactivate all active plugins
            assert mock_deactivate.call_count == 3  # battle1, battle2, exploration1

    def test_update_active_plugin_tracking(self):
        """Test active plugin tracking update"""
        mock_battle_plugin = Mock()
        mock_battle_plugin.get_metadata.return_value.name = 'battle_test'

        mock_exploration_plugin = Mock()
        mock_exploration_plugin.get_metadata.return_value.name = 'exploration_test'

        with patch.object(self.plugin_manager.registry, 'get_active_plugins') as mock_get_active:
            def side_effect(plugin_type):
                if plugin_type == PluginType.BATTLE_STRATEGY:
                    return [mock_battle_plugin]
                elif plugin_type == PluginType.EXPLORATION_PATTERN:
                    return [mock_exploration_plugin]
                else:
                    return []

            mock_get_active.side_effect = side_effect

            self.plugin_manager._update_active_plugin_tracking()

            assert self.plugin_manager.active_battle_strategies == ['battle_test']
            assert self.plugin_manager.active_exploration_patterns == ['exploration_test']
            assert self.plugin_manager.active_reward_calculators == []


class TestGlobalPluginManager:
    """Test global plugin manager functions"""

    def test_get_plugin_manager_singleton(self):
        """Test global plugin manager singleton"""
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is manager2  # Should be the same instance

    def test_initialize_plugin_manager(self):
        """Test plugin manager initialization with config"""
        config = {'auto_discover': True, 'auto_load': ['test.plugin']}

        with patch.object(PluginManager, 'initialize', return_value=True) as mock_init:
            manager = initialize_plugin_manager(config)

            assert manager is not None
            assert isinstance(manager, PluginManager)
            mock_init.assert_called_once()


@pytest.mark.integration
class TestPluginManagerIntegration:
    """Integration tests for plugin manager"""

    def setup_method(self):
        """Setup for integration tests"""
        self.plugin_manager = PluginManager({'auto_discover': False, 'auto_load': []})

    def test_full_plugin_workflow(self):
        """Test complete plugin workflow through manager"""
        # Mock the registry methods for a complete workflow
        plugin_id = 'battle_strategy.test'

        with patch.object(self.plugin_manager.registry, 'load_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'activate_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'get_active_plugins') as mock_get_active, \
             patch.object(self.plugin_manager.registry, 'deactivate_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'unload_plugin', return_value=True):

            # Load and activate plugin
            result = self.plugin_manager.load_and_activate_plugin(plugin_id, {'test': 'config'})
            assert result is True

            # Mock that plugin is now active
            test_plugin = TestBattlePlugin()
            mock_get_active.return_value = [test_plugin]

            # Get recommendation from plugin
            recommendation = self.plugin_manager.get_battle_strategy_recommendation({}, {})
            assert recommendation['plugin_name'] == 'test_battle'

            # Deactivate and unload plugin
            result = self.plugin_manager.deactivate_and_unload_plugin(plugin_id)
            assert result is True

    def test_plugin_error_recovery(self):
        """Test plugin manager handles plugin errors gracefully"""
        # Mock a scenario where plugins fail but manager continues working
        failing_plugin = Mock()
        failing_plugin.recommend_move.side_effect = Exception("Plugin crashed")
        failing_plugin.get_metadata.return_value = PluginMetadata(
            "failing", "1.0.0", "Failing plugin", "Test", PluginType.BATTLE_STRATEGY, priority=5
        )

        working_plugin = TestBattlePlugin()

        with patch.object(self.plugin_manager.registry, 'get_active_plugins',
                         return_value=[failing_plugin, working_plugin]):

            # Should still get recommendation from working plugin
            recommendation = self.plugin_manager.get_battle_strategy_recommendation({}, {})
            assert recommendation['plugin_name'] == 'test_battle'
            assert recommendation['confidence'] == 0.8

    def test_concurrent_plugin_operations(self):
        """Test plugin manager handles concurrent operations"""
        import threading
        import time

        results = []
        errors = []

        def plugin_operation():
            try:
                # Simulate concurrent plugin loading/unloading
                for i in range(5):
                    plugin_id = f'test_plugin_{i}'
                    result = self.plugin_manager.load_and_activate_plugin(plugin_id)
                    results.append(result)
                    time.sleep(0.01)  # Small delay to encourage race conditions
                    self.plugin_manager.deactivate_and_unload_plugin(plugin_id)
            except Exception as e:
                errors.append(e)

        # Mock registry methods to return success
        with patch.object(self.plugin_manager.registry, 'load_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'activate_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'deactivate_plugin', return_value=True), \
             patch.object(self.plugin_manager.registry, 'unload_plugin', return_value=True):

            # Run multiple threads
            threads = [threading.Thread(target=plugin_operation) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Should not have any errors
            assert len(errors) == 0
            assert len(results) == 15  # 3 threads * 5 operations each