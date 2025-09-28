#!/usr/bin/env python3
"""
Test Suite for Web Dashboard API Endpoints

Tests the UnifiedApiEndpoints class that handles all web dashboard API functionality,
including dashboard data, game state, training stats, memory debug, LLM decisions,
system status, and visualization data endpoints.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from web_dashboard.api.endpoints import UnifiedApiEndpoints
from web_dashboard.api.models import (
    GameStateModel, TrainingStatsModel, MemoryDebugModel, LLMDecisionModel,
    SystemStatusModel, VisualizationDataModel, UnifiedDashboardModel, ApiResponseModel
)


class TestUnifiedApiEndpoints:
    """Test UnifiedApiEndpoints functionality."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer with comprehensive attributes."""
        trainer = Mock()

        # Mock statistics tracker
        stats_tracker = Mock()
        stats_tracker.get_current_stats.return_value = {
            'current_map': 5,
            'player_position': {'x': 10, 'y': 20},
            'money': 1500,
            'badges': 3,
            'party_count': 4,
            'level': 25,
            'hp_current': 80,
            'hp_max': 100,
            'in_battle': False,
            'facing_direction': 2,
            'total_actions': 1000,
            'actions_per_second': 2.5,
            'llm_calls': 50,
            'total_reward': 125.5,
            'session_duration': 400.0,
            'success_rate': 0.75,
            'exploration_rate': 0.3,
            'recent_rewards': [1.5, -0.2, 3.0, 0.5, 2.1],
            'action_counts': {'up': 200, 'down': 150, 'left': 100, 'right': 180, 'a': 70},
            'position_history': {5: [(10, 20), (11, 20), (12, 20)]}
        }
        trainer.stats_tracker = stats_tracker

        # Mock LLM decisions
        trainer.llm_decisions = [
            {
                'action': 1,
                'action_name': 'up',
                'reasoning': 'Moving north to explore',
                'confidence': 0.8,
                'response_time_ms': 250.0,
                'game_state': {'hp': 80, 'level': 25},
                'timestamp': time.time() - 100
            },
            {
                'action': 'a',
                'action_name': 'button_a',
                'reasoning': 'Interacting with NPC',
                'confidence': 0.9,
                'response_time_ms': 180.0,
                'game_state': {'in_battle': False, 'map': 5},
                'timestamp': time.time() - 50
            }
        ]

        # Mock emulation manager
        emulation_manager = Mock()
        pyboy_instance = Mock()
        emulation_manager.get_instance.return_value = pyboy_instance
        trainer.emulation_manager = emulation_manager

        # Mock memory reader
        memory_reader = Mock()
        memory_reader.read_game_state.return_value = {
            'player_x': 10,
            'player_y': 20,
            'money_low': 220,
            'money_mid': 5,
            'badges': 3
        }
        memory_reader.get_debug_info.return_value = {
            'cache_hits': 150,
            'cache_misses': 10,
            'last_read_time': time.time()
        }
        trainer.memory_reader = memory_reader

        # Mock web monitor
        web_monitor = Mock()
        web_monitor.active_connections = 2
        trainer.web_monitor = web_monitor

        # Training status
        trainer.training_active = True

        return trainer

    @pytest.fixture
    def endpoints(self, mock_trainer):
        """Create UnifiedApiEndpoints instance with mock trainer."""
        return UnifiedApiEndpoints(trainer=mock_trainer)

    @pytest.fixture
    def endpoints_no_trainer(self):
        """Create UnifiedApiEndpoints instance without trainer."""
        return UnifiedApiEndpoints(trainer=None)

    def test_initialization(self, mock_trainer):
        """Test UnifiedApiEndpoints initialization."""
        endpoints = UnifiedApiEndpoints(trainer=mock_trainer)

        assert endpoints.trainer == mock_trainer
        assert endpoints.logger is not None

    def test_initialization_no_trainer(self):
        """Test UnifiedApiEndpoints initialization without trainer."""
        endpoints = UnifiedApiEndpoints(trainer=None)

        assert endpoints.trainer is None
        assert endpoints.logger is not None

    def test_get_dashboard_data_success(self, endpoints):
        """Test successful dashboard data retrieval."""
        result = endpoints.get_dashboard_data()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result
        assert 'timestamp' in result

        data = result['data']
        assert 'game_state' in data
        assert 'training_stats' in data
        assert 'memory_debug' in data
        assert 'recent_llm_decisions' in data
        assert 'system_status' in data

        # Verify game state data
        game_state = data['game_state']
        assert game_state['current_map'] == 5
        assert game_state['money'] == 1500
        assert game_state['badges_earned'] == 3

        # Verify training stats
        training_stats = data['training_stats']
        assert training_stats['total_actions'] == 1000
        assert training_stats['total_reward'] == 125.5

        # Verify LLM decisions
        llm_decisions = data['recent_llm_decisions']
        assert len(llm_decisions) == 2
        assert llm_decisions[0]['action'] == 1
        assert llm_decisions[1]['action'] == 'a'

    def test_get_dashboard_data_no_trainer(self, endpoints_no_trainer):
        """Test dashboard data retrieval without trainer."""
        result = endpoints_no_trainer.get_dashboard_data()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        # Should return default/empty data
        data = result['data']
        assert data['game_state']['current_map'] == 0
        assert data['training_stats']['total_actions'] == 0
        assert data['recent_llm_decisions'] == []

    def test_get_dashboard_data_error_handling(self, endpoints):
        """Test dashboard data error handling."""
        # Mock an exception in stats tracker
        endpoints.trainer.stats_tracker.get_current_stats.side_effect = Exception("Stats error")

        result = endpoints.get_dashboard_data()

        assert isinstance(result, dict)
        # The endpoints gracefully handle errors and return default data, not error responses
        assert result['success'] is True
        assert 'data' in result
        # Should still return dashboard structure with default values
        data = result['data']
        assert 'game_state' in data
        assert 'training_stats' in data

    def test_get_game_state_success(self, endpoints):
        """Test successful game state retrieval."""
        result = endpoints.get_game_state()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert data['current_map'] == 5
        assert data['player_position'] == {'x': 10, 'y': 20}
        assert data['money'] == 1500
        assert data['badges_earned'] == 3
        assert data['party_count'] == 4
        assert data['player_level'] == 25
        assert data['hp_current'] == 80
        assert data['hp_max'] == 100
        assert data['in_battle'] is False
        assert data['facing_direction'] == 2

    def test_get_game_state_statistics_tracker_fallback(self, endpoints):
        """Test game state retrieval with statistics_tracker fallback."""
        # Remove stats_tracker, add statistics_tracker
        del endpoints.trainer.stats_tracker
        endpoints.trainer.statistics_tracker = Mock()
        endpoints.trainer.statistics_tracker.get_current_stats.return_value = {
            'map_id': 7,
            'player_position': {'x': 15, 'y': 25},
            'money': 2000
        }

        result = endpoints.get_game_state()

        assert result['success'] is True
        data = result['data']
        assert data['current_map'] == 7
        assert data['player_position'] == {'x': 15, 'y': 25}
        assert data['money'] == 2000

    def test_get_game_state_position_formats(self, endpoints):
        """Test game state with different position formats."""
        # Test with non-dict position
        endpoints.trainer.stats_tracker.get_current_stats.return_value = {
            'player_position': 'invalid',
            'current_map': 3
        }

        result = endpoints.get_game_state()

        assert result['success'] is True
        data = result['data']
        assert data['player_position'] == {'x': 0, 'y': 0}  # Default fallback

    def test_get_game_state_map_name_fallbacks(self, endpoints):
        """Test game state with different map name keys."""
        # Test different possible map keys
        test_cases = [
            ({'current_map': 5}, 5),
            ({'map_id': 7}, 7),
            ({'map': 9}, 9),
            ({'location': 11}, 11),
            ({}, 0)  # Default fallback
        ]

        for stats, expected_map in test_cases:
            endpoints.trainer.stats_tracker.get_current_stats.return_value = stats
            result = endpoints.get_game_state()

            assert result['success'] is True
            assert result['data']['current_map'] == expected_map

    def test_get_training_stats_success(self, endpoints):
        """Test successful training stats retrieval."""
        result = endpoints.get_training_stats()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert data['total_actions'] == 1000
        assert data['actions_per_second'] == 2.5
        assert data['llm_decisions'] == 50
        assert data['total_reward'] == 125.5
        assert data['session_duration'] == 400.0
        assert data['success_rate'] == 0.75
        assert data['exploration_rate'] == 0.3
        assert data['recent_rewards'] == [1.5, -0.2, 3.0, 0.5, 2.1]

    def test_get_memory_debug_with_pyboy_and_reader(self, endpoints):
        """Test memory debug with PyBoy instance and memory reader."""
        result = endpoints.get_memory_debug()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert data['memory_read_success'] is True
        assert data['pyboy_available'] is True
        assert 'memory_addresses' in data
        assert 'cache_info' in data

    def test_get_memory_debug_pyboy_only(self, endpoints):
        """Test memory debug with PyBoy instance but no memory reader."""
        # Remove memory reader
        del endpoints.trainer.memory_reader

        result = endpoints.get_memory_debug()

        assert result['success'] is True
        data = result['data']
        # When memory reader is removed, it falls back to stats tracker
        assert data['memory_read_success'] is True
        assert data['cache_info']['source'] == 'statistics_tracker'

    def test_get_memory_debug_stats_fallback(self, endpoints):
        """Test memory debug fallback to stats tracker."""
        # Remove memory reader but keep stats tracker
        del endpoints.trainer.memory_reader
        del endpoints.trainer.emulation_manager

        result = endpoints.get_memory_debug()

        assert result['success'] is True
        data = result['data']
        assert data['memory_read_success'] is True
        assert data['cache_info']['source'] == 'statistics_tracker'

    def test_get_memory_debug_no_resources(self, endpoints_no_trainer):
        """Test memory debug with no PyBoy or memory reader."""
        result = endpoints_no_trainer.get_memory_debug()

        assert result['success'] is True
        data = result['data']
        assert data['memory_read_success'] is False
        assert data['pyboy_available'] is False
        # Cache info is initialized as empty dict by default
        assert isinstance(data['cache_info'], dict)

    def test_get_llm_decisions_success(self, endpoints):
        """Test successful LLM decisions retrieval."""
        result = endpoints.get_llm_decisions()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert len(data) == 2

        # Verify first decision
        decision1 = data[0]
        assert decision1['action'] == 1
        assert decision1['action_name'] == 'up'
        assert decision1['reasoning'] == 'Moving north to explore'
        assert decision1['confidence'] == 0.8

        # Verify second decision
        decision2 = data[1]
        assert decision2['action'] == 'a'
        assert decision2['action_name'] == 'button_a'
        assert decision2['reasoning'] == 'Interacting with NPC'
        assert decision2['confidence'] == 0.9

    def test_get_llm_decisions_large_list(self, endpoints):
        """Test LLM decisions with more than 10 items (should limit to last 10)."""
        # Create 15 mock decisions
        decisions = []
        for i in range(15):
            decisions.append({
                'action': i,
                'action_name': f'action_{i}',
                'reasoning': f'Reason {i}',
                'confidence': 0.5 + (i * 0.03),
                'response_time_ms': 200.0,
                'game_state': {'step': i},
                'timestamp': time.time() - (15 - i) * 10
            })

        endpoints.trainer.llm_decisions = decisions

        result = endpoints.get_llm_decisions()

        assert result['success'] is True
        data = result['data']
        assert len(data) == 10  # Should limit to last 10
        assert data[0]['action'] == 5  # Should start from index 5 (last 10 of 15)
        assert data[9]['action'] == 14  # Should end at index 14

    def test_get_llm_decisions_no_decisions(self, endpoints):
        """Test LLM decisions when no decisions exist."""
        endpoints.trainer.llm_decisions = []

        result = endpoints.get_llm_decisions()

        assert result['success'] is True
        data = result['data']
        assert data == []

    def test_get_system_status_active_training(self, endpoints):
        """Test system status with active training."""
        result = endpoints.get_system_status()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert data['training_active'] is True
        assert data['web_server_status'] == 'active'
        assert data['websocket_connections'] == 2

    def test_get_system_status_inactive_training(self, endpoints):
        """Test system status with inactive training."""
        endpoints.trainer.training_active = False
        endpoints.trainer.stats_tracker.get_current_stats.return_value = {'total_actions': 0}

        result = endpoints.get_system_status()

        assert result['success'] is True
        data = result['data']
        assert data['training_active'] is False
        assert data['web_server_status'] == 'stopped'

    def test_get_system_status_stats_tracker_inference(self, endpoints):
        """Test system status inference from stats tracker activity."""
        # Remove explicit training_active attribute
        del endpoints.trainer.training_active

        result = endpoints.get_system_status()

        assert result['success'] is True
        data = result['data']
        # Should infer from stats tracker having actions > 0
        assert data['training_active'] is True

    def test_get_system_status_websocket_handler_fallback(self, endpoints):
        """Test system status with websocket_handler fallback."""
        # Replace web_monitor with websocket_handler
        del endpoints.trainer.web_monitor
        websocket_handler = Mock()
        websocket_handler.connection_count = 3
        endpoints.trainer.websocket_handler = websocket_handler

        result = endpoints.get_system_status()

        assert result['success'] is True
        data = result['data']
        assert data['websocket_connections'] == 3

    def test_get_visualization_data_success(self, endpoints):
        """Test successful visualization data retrieval."""
        result = endpoints.get_visualization_data()

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'data' in result

        data = result['data']
        assert 'reward_history' in data
        assert 'action_performance' in data
        assert 'decision_patterns' in data
        assert 'performance_metrics' in data
        assert 'exploration_data' in data

        # Verify reward history
        reward_history = data['reward_history']
        assert len(reward_history) == 5  # Should match recent_rewards length
        assert all('timestamp' in item for item in reward_history)
        assert all('reward' in item for item in reward_history)
        assert all('cumulative' in item for item in reward_history)

        # Verify action performance
        action_performance = data['action_performance']
        assert 'up' in action_performance
        assert action_performance['up']['count'] == 200
        assert 'frequency' in action_performance['up']

        # Verify performance metrics
        performance_metrics = data['performance_metrics']
        assert len(performance_metrics) == 1
        assert performance_metrics[0]['actions_per_second'] == 2.5
        assert performance_metrics[0]['total_reward'] == 125.5

        # Verify exploration data
        exploration_data = data['exploration_data']
        assert exploration_data['current_map'] == 5
        assert exploration_data['exploration_coverage'] == 0.3

        # Verify decision patterns
        decision_patterns = data['decision_patterns']
        assert len(decision_patterns) == 2
        assert decision_patterns[0]['action'] == 1
        assert decision_patterns[1]['action'] == 'a'

    def test_get_visualization_data_no_tracker(self, endpoints):
        """Test visualization data with no stats tracker."""
        del endpoints.trainer.stats_tracker

        result = endpoints.get_visualization_data()

        assert result['success'] is True
        data = result['data']
        # Should return empty visualization data
        assert data['reward_history'] == []
        assert data['action_performance'] == {}
        assert data['decision_patterns'] == []

    def test_error_handling_consistency(self, endpoints):
        """Test that all endpoints handle errors consistently."""
        # Mock an exception in the trainer
        endpoints.trainer.stats_tracker.get_current_stats.side_effect = Exception("Test error")

        endpoints_to_test = [
            'get_game_state',
            'get_training_stats',
            'get_memory_debug',
            'get_llm_decisions',
            'get_system_status',
            'get_visualization_data'
        ]

        for endpoint_name in endpoints_to_test:
            endpoint = getattr(endpoints, endpoint_name)
            result = endpoint()

            # All should return successful responses with fallback data
            assert isinstance(result, dict)
            assert result['success'] is True  # Endpoints gracefully handle errors
            assert 'data' in result  # Should return fallback data instead of errors
            assert 'timestamp' in result

    def test_private_method_game_state_extraction(self, endpoints):
        """Test private _get_game_state method directly."""
        game_state = endpoints._get_game_state()

        assert isinstance(game_state, GameStateModel)
        assert game_state.current_map == 5
        assert game_state.money == 1500
        assert game_state.badges_earned == 3

    def test_private_method_training_stats_extraction(self, endpoints):
        """Test private _get_training_stats method directly."""
        training_stats = endpoints._get_training_stats()

        assert isinstance(training_stats, TrainingStatsModel)
        assert training_stats.total_actions == 1000
        assert training_stats.total_reward == 125.5

    def test_private_method_memory_debug_extraction(self, endpoints):
        """Test private _get_memory_debug method directly."""
        memory_debug = endpoints._get_memory_debug()

        assert isinstance(memory_debug, MemoryDebugModel)
        assert memory_debug.memory_read_success is True
        assert memory_debug.pyboy_available is True

    def test_private_method_system_status_extraction(self, endpoints):
        """Test private _get_system_status method directly."""
        system_status = endpoints._get_system_status()

        assert isinstance(system_status, SystemStatusModel)
        assert system_status.training_active is True
        assert system_status.websocket_connections == 2

    def test_private_method_llm_decisions_extraction(self, endpoints):
        """Test private _get_recent_llm_decisions method directly."""
        llm_decisions = endpoints._get_recent_llm_decisions()

        assert isinstance(llm_decisions, list)
        assert len(llm_decisions) == 2
        assert all(isinstance(decision, LLMDecisionModel) for decision in llm_decisions)

    def test_private_method_visualization_data_extraction(self, endpoints):
        """Test private _get_visualization_data method directly."""
        viz_data = endpoints._get_visualization_data()

        assert isinstance(viz_data, VisualizationDataModel)
        assert len(viz_data.reward_history) > 0
        assert len(viz_data.action_performance) > 0

    def test_logging_functionality(self, endpoints):
        """Test that logging works correctly."""
        # Mock the logger to capture calls
        with patch.object(endpoints.logger, 'error') as mock_error:
            with patch.object(endpoints.logger, 'warning') as mock_warning:
                # Trigger an error
                endpoints.trainer.stats_tracker.get_current_stats.side_effect = Exception("Test error")
                endpoints.get_game_state()

                # Verify warning logging (endpoints use warnings for graceful fallback)
                mock_warning.assert_called()
                warning_call = mock_warning.call_args[0][0]
                assert "Could not get game state from statistics" in warning_call

    def test_response_format_consistency(self, endpoints):
        """Test that all endpoints return consistent response formats."""
        endpoints_to_test = [
            'get_dashboard_data',
            'get_game_state',
            'get_training_stats',
            'get_memory_debug',
            'get_llm_decisions',
            'get_system_status',
            'get_visualization_data'
        ]

        for endpoint_name in endpoints_to_test:
            endpoint = getattr(endpoints, endpoint_name)
            result = endpoint()

            # All should have consistent response format
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'timestamp' in result
            assert isinstance(result['success'], bool)
            assert isinstance(result['timestamp'], float)

            if result['success']:
                assert 'data' in result
            else:
                assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])