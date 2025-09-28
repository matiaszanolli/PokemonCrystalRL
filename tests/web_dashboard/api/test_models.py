#!/usr/bin/env python3
"""
Test Suite for Web Dashboard API Data Models

Tests all data model classes that form the foundation of the web dashboard API,
including GameStateModel, TrainingStatsModel, MemoryDebugModel, LLMDecisionModel,
SystemStatusModel, VisualizationDataModel, UnifiedDashboardModel, and ApiResponseModel.
"""

import pytest
import time
from unittest.mock import Mock
from dataclasses import asdict

from web_dashboard.api.models import (
    GameStateModel, TrainingStatsModel, MemoryDebugModel, LLMDecisionModel,
    SystemStatusModel, VisualizationDataModel, UnifiedDashboardModel, ApiResponseModel
)


class TestGameStateModel:
    """Test GameStateModel functionality."""

    def test_game_state_model_defaults(self):
        """Test GameStateModel default initialization."""
        model = GameStateModel()

        assert model.current_map == 0
        assert model.player_position == {"x": 0, "y": 0}
        assert model.money == 0
        assert model.badges_earned == 0
        assert model.party_count == 0
        assert model.player_level == 0
        assert model.hp_current == 0
        assert model.hp_max == 0
        assert model.in_battle is False
        assert model.facing_direction == 0
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_game_state_model_custom_values(self):
        """Test GameStateModel with custom values."""
        custom_position = {"x": 15, "y": 25}
        custom_timestamp = time.time() - 100

        model = GameStateModel(
            current_map=5,
            player_position=custom_position,
            money=1500,
            badges_earned=3,
            party_count=4,
            player_level=25,
            hp_current=80,
            hp_max=100,
            in_battle=True,
            facing_direction=2,
            timestamp=custom_timestamp
        )

        assert model.current_map == 5
        assert model.player_position == custom_position
        assert model.money == 1500
        assert model.badges_earned == 3
        assert model.party_count == 4
        assert model.player_level == 25
        assert model.hp_current == 80
        assert model.hp_max == 100
        assert model.in_battle is True
        assert model.facing_direction == 2
        assert model.timestamp == custom_timestamp

    def test_game_state_model_post_init(self):
        """Test GameStateModel __post_init__ method."""
        # Test with None values
        model = GameStateModel(player_position=None, timestamp=None)

        assert model.player_position == {"x": 0, "y": 0}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_game_state_model_to_dict(self):
        """Test GameStateModel to_dict conversion."""
        model = GameStateModel(
            current_map=3,
            money=2500,
            badges_earned=2,
            hp_current=50,
            hp_max=75
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["current_map"] == 3
        assert result["money"] == 2500
        assert result["badges_earned"] == 2
        assert result["hp_current"] == 50
        assert result["hp_max"] == 75
        assert "timestamp" in result
        assert "player_position" in result

    def test_game_state_model_hp_ratio_calculation(self):
        """Test health ratio calculations."""
        model = GameStateModel(hp_current=75, hp_max=100)

        # While not built into the model, verify the data is correct for calculations
        hp_ratio = model.hp_current / max(model.hp_max, 1)
        assert hp_ratio == 0.75

        # Test zero max HP protection
        model_zero_max = GameStateModel(hp_current=50, hp_max=0)
        hp_ratio_protected = model_zero_max.hp_current / max(model_zero_max.hp_max, 1)
        assert hp_ratio_protected == 50.0


class TestTrainingStatsModel:
    """Test TrainingStatsModel functionality."""

    def test_training_stats_model_defaults(self):
        """Test TrainingStatsModel default initialization."""
        model = TrainingStatsModel()

        assert model.total_actions == 0
        assert model.actions_per_second == 0.0
        assert model.llm_decisions == 0
        assert model.total_reward == 0.0
        assert model.session_duration == 0.0
        assert model.success_rate == 0.0
        assert model.exploration_rate == 0.0
        assert model.recent_rewards == []
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_training_stats_model_custom_values(self):
        """Test TrainingStatsModel with custom values."""
        recent_rewards = [1.5, -0.2, 3.0, 0.5]
        custom_timestamp = time.time() - 50

        model = TrainingStatsModel(
            total_actions=1000,
            actions_per_second=2.5,
            llm_decisions=50,
            total_reward=125.5,
            session_duration=400.0,
            success_rate=0.75,
            exploration_rate=0.3,
            recent_rewards=recent_rewards,
            timestamp=custom_timestamp
        )

        assert model.total_actions == 1000
        assert model.actions_per_second == 2.5
        assert model.llm_decisions == 50
        assert model.total_reward == 125.5
        assert model.session_duration == 400.0
        assert model.success_rate == 0.75
        assert model.exploration_rate == 0.3
        assert model.recent_rewards == recent_rewards
        assert model.timestamp == custom_timestamp

    def test_training_stats_model_post_init(self):
        """Test TrainingStatsModel __post_init__ method."""
        model = TrainingStatsModel(recent_rewards=None, timestamp=None)

        assert model.recent_rewards == []
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_training_stats_model_to_dict(self):
        """Test TrainingStatsModel to_dict conversion."""
        model = TrainingStatsModel(
            total_actions=500,
            total_reward=45.5,
            success_rate=0.8,
            recent_rewards=[1.0, 2.0, -0.5]
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["total_actions"] == 500
        assert result["total_reward"] == 45.5
        assert result["success_rate"] == 0.8
        assert result["recent_rewards"] == [1.0, 2.0, -0.5]
        assert "timestamp" in result


class TestMemoryDebugModel:
    """Test MemoryDebugModel functionality."""

    def test_memory_debug_model_defaults(self):
        """Test MemoryDebugModel default initialization."""
        model = MemoryDebugModel()

        assert model.memory_addresses == {}
        assert model.memory_read_success is False
        assert model.pyboy_available is False
        assert model.cache_info == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_memory_debug_model_custom_values(self):
        """Test MemoryDebugModel with custom values."""
        memory_addresses = {"player_x": 0xDCB8, "player_y": 0xDCB9, "badges": 0xD857}
        cache_info = {"hits": 150, "misses": 10, "hit_rate": 0.94}
        custom_timestamp = time.time() - 25

        model = MemoryDebugModel(
            memory_addresses=memory_addresses,
            memory_read_success=True,
            pyboy_available=True,
            cache_info=cache_info,
            timestamp=custom_timestamp
        )

        assert model.memory_addresses == memory_addresses
        assert model.memory_read_success is True
        assert model.pyboy_available is True
        assert model.cache_info == cache_info
        assert model.timestamp == custom_timestamp

    def test_memory_debug_model_post_init(self):
        """Test MemoryDebugModel __post_init__ method."""
        model = MemoryDebugModel(
            memory_addresses=None,
            cache_info=None,
            timestamp=None
        )

        assert model.memory_addresses == {}
        assert model.cache_info == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_memory_debug_model_to_dict(self):
        """Test MemoryDebugModel to_dict conversion."""
        model = MemoryDebugModel(
            memory_read_success=True,
            pyboy_available=True,
            cache_info={"size": 100}
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["memory_read_success"] is True
        assert result["pyboy_available"] is True
        assert result["cache_info"] == {"size": 100}
        assert "timestamp" in result


class TestLLMDecisionModel:
    """Test LLMDecisionModel functionality."""

    def test_llm_decision_model_defaults(self):
        """Test LLMDecisionModel default initialization with required fields."""
        model = LLMDecisionModel(action=1, action_name="up")

        assert model.action == 1
        assert model.action_name == "up"
        assert model.reasoning == ""
        assert model.confidence == 0.0
        assert model.response_time_ms == 0.0
        assert model.game_state_snapshot == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_llm_decision_model_custom_values(self):
        """Test LLMDecisionModel with custom values."""
        game_state = {"hp": 80, "level": 15, "in_battle": False}
        custom_timestamp = time.time() - 5

        model = LLMDecisionModel(
            action="a",
            action_name="button_a",
            reasoning="Need to interact with NPC",
            confidence=0.85,
            response_time_ms=250.5,
            game_state_snapshot=game_state,
            timestamp=custom_timestamp
        )

        assert model.action == "a"
        assert model.action_name == "button_a"
        assert model.reasoning == "Need to interact with NPC"
        assert model.confidence == 0.85
        assert model.response_time_ms == 250.5
        assert model.game_state_snapshot == game_state
        assert model.timestamp == custom_timestamp

    def test_llm_decision_model_post_init(self):
        """Test LLMDecisionModel __post_init__ method."""
        model = LLMDecisionModel(
            action=2,
            action_name="down",
            game_state_snapshot=None,
            timestamp=None
        )

        assert model.game_state_snapshot == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_llm_decision_model_to_dict(self):
        """Test LLMDecisionModel to_dict conversion."""
        model = LLMDecisionModel(
            action=3,
            action_name="left",
            reasoning="Exploring westward",
            confidence=0.7
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["action"] == 3
        assert result["action_name"] == "left"
        assert result["reasoning"] == "Exploring westward"
        assert result["confidence"] == 0.7
        assert "timestamp" in result

    def test_llm_decision_model_action_types(self):
        """Test LLMDecisionModel with different action types."""
        # Integer action
        model_int = LLMDecisionModel(action=1, action_name="up")
        assert isinstance(model_int.action, int)

        # String action
        model_str = LLMDecisionModel(action="select", action_name="select_button")
        assert isinstance(model_str.action, str)


class TestSystemStatusModel:
    """Test SystemStatusModel functionality."""

    def test_system_status_model_defaults(self):
        """Test SystemStatusModel default initialization."""
        model = SystemStatusModel()

        assert model.training_active is False
        assert model.web_server_status == "stopped"
        assert model.websocket_connections == 0
        assert model.last_update is not None
        assert model.errors == []
        assert model.uptime_seconds == 0.0
        assert isinstance(model.last_update, float)

    def test_system_status_model_custom_values(self):
        """Test SystemStatusModel with custom values."""
        errors = ["Connection timeout", "Memory allocation failed"]
        last_update = time.time() - 10

        model = SystemStatusModel(
            training_active=True,
            web_server_status="running",
            websocket_connections=3,
            last_update=last_update,
            errors=errors,
            uptime_seconds=3600.5
        )

        assert model.training_active is True
        assert model.web_server_status == "running"
        assert model.websocket_connections == 3
        assert model.last_update == last_update
        assert model.errors == errors
        assert model.uptime_seconds == 3600.5

    def test_system_status_model_post_init(self):
        """Test SystemStatusModel __post_init__ method."""
        model = SystemStatusModel(errors=None, last_update=None)

        assert model.errors == []
        assert model.last_update is not None
        assert isinstance(model.last_update, float)

    def test_system_status_model_to_dict(self):
        """Test SystemStatusModel to_dict conversion."""
        model = SystemStatusModel(
            training_active=True,
            web_server_status="running",
            websocket_connections=2,
            uptime_seconds=1800.0
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["training_active"] is True
        assert result["web_server_status"] == "running"
        assert result["websocket_connections"] == 2
        assert result["uptime_seconds"] == 1800.0
        assert "last_update" in result


class TestVisualizationDataModel:
    """Test VisualizationDataModel functionality."""

    def test_visualization_data_model_defaults(self):
        """Test VisualizationDataModel default initialization."""
        model = VisualizationDataModel()

        assert model.reward_history == []
        assert model.action_performance == {}
        assert model.decision_patterns == []
        assert model.performance_metrics == []
        assert model.exploration_data == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_visualization_data_model_custom_values(self):
        """Test VisualizationDataModel with custom values."""
        reward_history = [
            {"timestamp": 1000, "reward": 1.5, "cumulative": 1.5},
            {"timestamp": 1001, "reward": -0.2, "cumulative": 1.3}
        ]
        action_performance = {
            "up": {"count": 50, "success_rate": 0.8},
            "down": {"count": 45, "success_rate": 0.75}
        }
        decision_patterns = [{"pattern_id": 1, "frequency": 0.3}]
        performance_metrics = [{"timestamp": 1000, "accuracy": 0.85}]
        exploration_data = {"visited_locations": 25, "coverage": 0.4}
        custom_timestamp = time.time() - 30

        model = VisualizationDataModel(
            reward_history=reward_history,
            action_performance=action_performance,
            decision_patterns=decision_patterns,
            performance_metrics=performance_metrics,
            exploration_data=exploration_data,
            timestamp=custom_timestamp
        )

        assert model.reward_history == reward_history
        assert model.action_performance == action_performance
        assert model.decision_patterns == decision_patterns
        assert model.performance_metrics == performance_metrics
        assert model.exploration_data == exploration_data
        assert model.timestamp == custom_timestamp

    def test_visualization_data_model_post_init(self):
        """Test VisualizationDataModel __post_init__ method."""
        model = VisualizationDataModel(
            reward_history=None,
            action_performance=None,
            decision_patterns=None,
            performance_metrics=None,
            exploration_data=None,
            timestamp=None
        )

        assert model.reward_history == []
        assert model.action_performance == {}
        assert model.decision_patterns == []
        assert model.performance_metrics == []
        assert model.exploration_data == {}
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_visualization_data_model_to_dict(self):
        """Test VisualizationDataModel to_dict conversion."""
        model = VisualizationDataModel(
            reward_history=[{"timestamp": 1000, "reward": 2.0}],
            action_performance={"a": {"count": 10}},
            exploration_data={"coverage": 0.5}
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["reward_history"] == [{"timestamp": 1000, "reward": 2.0}]
        assert result["action_performance"] == {"a": {"count": 10}}
        assert result["exploration_data"] == {"coverage": 0.5}
        assert "timestamp" in result


class TestUnifiedDashboardModel:
    """Test UnifiedDashboardModel functionality."""

    def test_unified_dashboard_model_initialization(self):
        """Test UnifiedDashboardModel initialization with required components."""
        game_state = GameStateModel(current_map=5, money=1000)
        training_stats = TrainingStatsModel(total_actions=500)
        memory_debug = MemoryDebugModel(memory_read_success=True)
        llm_decisions = [
            LLMDecisionModel(action=1, action_name="up", reasoning="Move north"),
            LLMDecisionModel(action="a", action_name="button_a", reasoning="Interact")
        ]
        system_status = SystemStatusModel(training_active=True)

        model = UnifiedDashboardModel(
            game_state=game_state,
            training_stats=training_stats,
            memory_debug=memory_debug,
            recent_llm_decisions=llm_decisions,
            system_status=system_status
        )

        assert model.game_state == game_state
        assert model.training_stats == training_stats
        assert model.memory_debug == memory_debug
        assert model.recent_llm_decisions == llm_decisions
        assert model.system_status == system_status
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_unified_dashboard_model_post_init(self):
        """Test UnifiedDashboardModel __post_init__ method."""
        game_state = GameStateModel()
        training_stats = TrainingStatsModel()
        memory_debug = MemoryDebugModel()
        system_status = SystemStatusModel()

        model = UnifiedDashboardModel(
            game_state=game_state,
            training_stats=training_stats,
            memory_debug=memory_debug,
            recent_llm_decisions=[],
            system_status=system_status,
            timestamp=None
        )

        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_unified_dashboard_model_to_dict(self):
        """Test UnifiedDashboardModel to_dict conversion."""
        game_state = GameStateModel(current_map=3)
        training_stats = TrainingStatsModel(total_actions=200)
        memory_debug = MemoryDebugModel(pyboy_available=True)
        llm_decision = LLMDecisionModel(action=2, action_name="down")
        system_status = SystemStatusModel(web_server_status="running")

        model = UnifiedDashboardModel(
            game_state=game_state,
            training_stats=training_stats,
            memory_debug=memory_debug,
            recent_llm_decisions=[llm_decision],
            system_status=system_status
        )

        result = model.to_dict()

        assert isinstance(result, dict)
        assert "game_state" in result
        assert "training_stats" in result
        assert "memory_debug" in result
        assert "recent_llm_decisions" in result
        assert "system_status" in result
        assert "timestamp" in result

        # Verify nested dictionaries
        assert isinstance(result["game_state"], dict)
        assert isinstance(result["training_stats"], dict)
        assert isinstance(result["memory_debug"], dict)
        assert isinstance(result["recent_llm_decisions"], list)
        assert isinstance(result["system_status"], dict)

        # Verify specific values
        assert result["game_state"]["current_map"] == 3
        assert result["training_stats"]["total_actions"] == 200
        assert result["memory_debug"]["pyboy_available"] is True
        assert len(result["recent_llm_decisions"]) == 1
        assert result["recent_llm_decisions"][0]["action"] == 2
        assert result["system_status"]["web_server_status"] == "running"


class TestApiResponseModel:
    """Test ApiResponseModel functionality."""

    def test_api_response_model_success(self):
        """Test ApiResponseModel for successful responses."""
        data = {"message": "Operation completed", "count": 42}

        model = ApiResponseModel(success=True, data=data)

        assert model.success is True
        assert model.data == data
        assert model.error is None
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_api_response_model_error(self):
        """Test ApiResponseModel for error responses."""
        model = ApiResponseModel(success=False, error="Invalid request")

        assert model.success is False
        assert model.data is None
        assert model.error == "Invalid request"
        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_api_response_model_post_init(self):
        """Test ApiResponseModel __post_init__ method."""
        model = ApiResponseModel(success=True, timestamp=None)

        assert model.timestamp is not None
        assert isinstance(model.timestamp, float)

    def test_api_response_model_to_dict_with_data(self):
        """Test ApiResponseModel to_dict with data object."""
        game_state = GameStateModel(current_map=7, money=2500)
        model = ApiResponseModel(success=True, data=game_state)

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "data" in result
        assert isinstance(result["data"], dict)
        assert result["data"]["current_map"] == 7
        assert result["data"]["money"] == 2500
        assert "timestamp" in result

    def test_api_response_model_to_dict_with_list(self):
        """Test ApiResponseModel to_dict with list of objects."""
        decisions = [
            LLMDecisionModel(action=1, action_name="up"),
            LLMDecisionModel(action=2, action_name="down")
        ]
        model = ApiResponseModel(success=True, data=decisions)

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "data" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2
        assert result["data"][0]["action"] == 1
        assert result["data"][1]["action"] == 2

    def test_api_response_model_to_dict_with_simple_data(self):
        """Test ApiResponseModel to_dict with simple data types."""
        model = ApiResponseModel(success=True, data={"simple": "value", "number": 123})

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["data"] == {"simple": "value", "number": 123}

    def test_api_response_model_to_dict_with_error(self):
        """Test ApiResponseModel to_dict with error."""
        model = ApiResponseModel(success=False, error="Database connection failed")

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["error"] == "Database connection failed"
        assert "data" not in result or result.get("data") is None

    def test_api_response_model_to_dict_mixed_list(self):
        """Test ApiResponseModel to_dict with mixed list items."""
        mixed_data = [
            LLMDecisionModel(action=1, action_name="up"),
            {"raw_data": "value"},
            "simple_string"
        ]
        model = ApiResponseModel(success=True, data=mixed_data)

        result = model.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 3
        assert isinstance(result["data"][0], dict)  # Converted from LLMDecisionModel
        assert result["data"][1] == {"raw_data": "value"}
        assert result["data"][2] == "simple_string"


class TestDataModelIntegration:
    """Test integration between different data models."""

    def test_model_timestamp_consistency(self):
        """Test that timestamps are consistently applied across models."""
        # Create models around the same time
        start_time = time.time()

        game_state = GameStateModel()
        training_stats = TrainingStatsModel()
        memory_debug = MemoryDebugModel()
        system_status = SystemStatusModel()

        end_time = time.time()

        # All timestamps should be between start and end time
        assert start_time <= game_state.timestamp <= end_time
        assert start_time <= training_stats.timestamp <= end_time
        assert start_time <= memory_debug.timestamp <= end_time
        assert start_time <= system_status.last_update <= end_time  # SystemStatusModel uses last_update

    def test_nested_model_serialization(self):
        """Test serialization of nested data models."""
        # Create a complex nested structure
        llm_decisions = [
            LLMDecisionModel(action=1, action_name="up", confidence=0.8),
            LLMDecisionModel(action="a", action_name="button_a", confidence=0.9)
        ]

        unified_model = UnifiedDashboardModel(
            game_state=GameStateModel(current_map=10, money=5000),
            training_stats=TrainingStatsModel(total_actions=1000, success_rate=0.75),
            memory_debug=MemoryDebugModel(memory_read_success=True),
            recent_llm_decisions=llm_decisions,
            system_status=SystemStatusModel(training_active=True)
        )

        api_response = ApiResponseModel(success=True, data=unified_model)

        # Test full serialization chain
        result = api_response.to_dict()

        assert isinstance(result, dict)
        assert result["success"] is True
        assert isinstance(result["data"], dict)

        # Verify deep nesting
        data = result["data"]
        assert data["game_state"]["current_map"] == 10
        assert data["training_stats"]["total_actions"] == 1000
        assert data["memory_debug"]["memory_read_success"] is True
        assert data["system_status"]["training_active"] is True
        assert len(data["recent_llm_decisions"]) == 2
        assert data["recent_llm_decisions"][0]["confidence"] == 0.8

    def test_model_field_validation(self):
        """Test that model fields maintain proper types and values."""
        # Test that boolean fields are properly handled
        game_state = GameStateModel(in_battle=True)
        assert isinstance(game_state.in_battle, bool)
        assert game_state.in_battle is True

        # Test that numeric fields are properly handled
        training_stats = TrainingStatsModel(success_rate=0.75, total_actions=500)
        assert isinstance(training_stats.success_rate, float)
        assert isinstance(training_stats.total_actions, int)

        # Test that list fields are properly handled
        visualization_data = VisualizationDataModel(
            reward_history=[{"timestamp": 1000, "reward": 1.5}]
        )
        assert isinstance(visualization_data.reward_history, list)
        assert len(visualization_data.reward_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])