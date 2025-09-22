"""
Comprehensive tests for HybridAgent class.

Tests the hybrid LLM+RL agent architecture including initialization, decision
arbitration, curriculum learning, agent coordination, mode switching, performance
tracking, and state persistence.
"""

import pytest
import json
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from collections import deque

from agents.hybrid_agent import (
    HybridAgent, AgentMode, DecisionConfidence, AgentDecision,
    BaseAgent, LLMAgent, RLAgent
)


class TestAgentDecision:
    """Test AgentDecision dataclass functionality."""

    def test_agent_decision_creation(self):
        """Test creating AgentDecision objects."""
        decision = AgentDecision(
            agent_type="llm",
            action=2,
            confidence=0.8,
            reasoning="Strategic exploration",
            expected_reward=1.5,
            risk_assessment="low"
        )

        assert decision.agent_type == "llm"
        assert decision.action == 2
        assert decision.confidence == 0.8
        assert decision.reasoning == "Strategic exploration"
        assert decision.expected_reward == 1.5
        assert decision.risk_assessment == "low"
        assert decision.alternative_actions == []

    def test_agent_decision_with_alternatives(self):
        """Test AgentDecision with alternative actions."""
        alternatives = [(1, 0.6), (3, 0.4)]
        decision = AgentDecision(
            agent_type="rl",
            action=0,
            confidence=0.9,
            reasoning="High Q-value action",
            alternative_actions=alternatives
        )

        assert decision.alternative_actions == alternatives

    def test_agent_decision_post_init(self):
        """Test AgentDecision __post_init__ method."""
        decision = AgentDecision(
            agent_type="hybrid",
            action=1,
            confidence=0.7,
            reasoning="Blended decision"
        )

        # Should have empty list for alternatives after post_init
        assert decision.alternative_actions == []


class TestAgentModeEnum:
    """Test AgentMode enumeration."""

    def test_agent_mode_values(self):
        """Test AgentMode enum values."""
        assert AgentMode.LLM_GUIDED.value == "llm_guided"
        assert AgentMode.RL_OPTIMIZED.value == "rl_optimized"
        assert AgentMode.COLLABORATIVE.value == "collaborative"
        assert AgentMode.ADAPTIVE.value == "adaptive"
        assert AgentMode.LEARNING.value == "learning"

    def test_agent_mode_membership(self):
        """Test AgentMode membership testing."""
        assert AgentMode.ADAPTIVE in AgentMode
        assert "invalid_mode" not in [mode.value for mode in AgentMode]


class TestHybridAgentInitialization:
    """Test HybridAgent initialization."""

    @patch('agents.hybrid_agent.LLMAgent')
    @patch('agents.hybrid_agent.RLAgent')
    def test_init_basic(self, mock_rl_agent, mock_llm_agent):
        """Test basic HybridAgent initialization."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)

        assert agent.mode == AgentMode.ADAPTIVE
        assert agent.confidence_threshold == 0.6
        assert agent.experience_counter == 0
        assert agent.llm_preference_bonus == 0.1
        assert agent.curriculum_stage == 0
        assert agent.stage_progress == 0

        # Check that component agents were created
        mock_llm_agent.assert_called_once_with(mock_llm_manager, mock_adaptive_strategy)
        mock_rl_agent.assert_called_once_with(9)  # Default action_space_size

    @patch('agents.hybrid_agent.LLMAgent')
    @patch('agents.hybrid_agent.RLAgent')
    def test_init_custom_params(self, mock_rl_agent, mock_llm_agent):
        """Test HybridAgent initialization with custom parameters."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()
        custom_curriculum = {"stages": [{"name": "test", "llm_weight": 0.5, "duration": 500}]}

        agent = HybridAgent(
            mock_llm_manager,
            mock_adaptive_strategy,
            action_space_size=12,
            curriculum_config=custom_curriculum
        )

        assert agent.curriculum_config == custom_curriculum
        mock_rl_agent.assert_called_once_with(12)

    @patch('agents.hybrid_agent.LLMAgent')
    @patch('agents.hybrid_agent.RLAgent')
    def test_init_default_curriculum(self, mock_rl_agent, mock_llm_agent):
        """Test default curriculum configuration."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)

        curriculum = agent.curriculum_config
        assert "stages" in curriculum
        assert len(curriculum["stages"]) == 3
        assert curriculum["stages"][0]["name"] == "llm_heavy"
        assert curriculum["stages"][1]["name"] == "balanced"
        assert curriculum["stages"][2]["name"] == "rl_heavy"

    @patch('agents.hybrid_agent.LLMAgent')
    @patch('agents.hybrid_agent.RLAgent')
    def test_init_tracking_structures(self, mock_rl_agent, mock_llm_agent):
        """Test initialization of tracking structures."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)

        assert isinstance(agent.agent_usage_stats, dict)
        assert "llm" in agent.agent_usage_stats
        assert "rl" in agent.agent_usage_stats
        assert "hybrid" in agent.agent_usage_stats

        assert isinstance(agent.performance_by_agent, dict)
        assert isinstance(agent.performance_by_agent["llm"], deque)
        assert isinstance(agent.performance_by_agent["rl"], deque)

        assert isinstance(agent.recent_rewards, deque)


class TestHybridAgentActionSelection:
    """Test HybridAgent action selection and arbitration."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent') as mock_llm_agent_class:
            with patch('agents.hybrid_agent.RLAgent') as mock_rl_agent_class:
                mock_llm_agent = Mock()
                mock_rl_agent = Mock()
                mock_llm_agent_class.return_value = mock_llm_agent
                mock_rl_agent_class.return_value = mock_rl_agent

                agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
                agent.llm_agent = mock_llm_agent
                agent.rl_agent = mock_rl_agent

                return agent, mock_llm_agent, mock_rl_agent

    def test_get_action_llm_guided_mode(self):
        """Test action selection in LLM_GUIDED mode."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.LLM_GUIDED

        llm_decision = AgentDecision("llm", 1, 0.8, "LLM reasoning")
        rl_decision = AgentDecision("rl", 2, 0.6, "RL reasoning")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {"test": "data"}
        info = {"additional": "info"}

        action, decision_info = agent.get_action(observation, info)

        assert action == 1  # LLM action
        assert decision_info['source'] == "llm"
        assert decision_info['confidence'] == 0.8

    def test_get_action_rl_optimized_mode(self):
        """Test action selection in RL_OPTIMIZED mode."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.RL_OPTIMIZED

        llm_decision = AgentDecision("llm", 1, 0.8, "LLM reasoning")
        rl_decision = AgentDecision("rl", 2, 0.6, "RL reasoning")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {"test": "data"}
        info = {"additional": "info"}

        action, decision_info = agent.get_action(observation, info)

        assert action == 2  # RL action
        assert decision_info['source'] == "rl"
        assert decision_info['confidence'] == 0.6

    def test_get_action_adaptive_mode_llm_wins(self):
        """Test adaptive mode when LLM has higher confidence."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.ADAPTIVE

        llm_decision = AgentDecision("llm", 1, 0.9, "High confidence LLM")
        rl_decision = AgentDecision("rl", 2, 0.4, "Low confidence RL")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {"criticality": 1, "game_phase": 1}
        info = {}

        action, decision_info = agent.get_action(observation, info)

        assert action == 1  # LLM action should win
        assert decision_info['source'] == "llm"

    def test_get_action_adaptive_mode_rl_wins(self):
        """Test adaptive mode when RL has higher effective confidence."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.ADAPTIVE
        agent.llm_preference_bonus = 0.0  # Remove LLM bias
        agent.curriculum_stage = 1  # Use balanced stage (llm_weight: 0.5)

        llm_decision = AgentDecision("llm", 1, 0.3, "Low confidence LLM")
        rl_decision = AgentDecision("rl", 2, 0.8, "High confidence RL")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {"criticality": 1, "game_phase": 3}  # Later game phase
        info = {}

        action, decision_info = agent.get_action(observation, info)

        assert action == 2  # RL action should win
        assert decision_info['source'] == "rl"

    def test_get_action_collaborative_mode(self):
        """Test collaborative mode decision making."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.COLLABORATIVE

        llm_decision = AgentDecision("llm", 1, 0.8, "High confidence LLM")
        rl_decision = AgentDecision("rl", 2, 0.9, "Higher confidence RL")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {}
        info = {}

        action, decision_info = agent.get_action(observation, info)

        # Should choose higher confidence (RL in this case)
        assert action == 2
        assert decision_info['source'] == "hybrid"

    def test_get_action_learning_mode(self):
        """Test learning mode favors LLM."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.LEARNING

        llm_decision = AgentDecision("llm", 1, 0.5, "Learning LLM")
        rl_decision = AgentDecision("rl", 2, 0.8, "RL reasoning")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        observation = {}
        info = {}

        action, decision_info = agent.get_action(observation, info)

        assert action == 1  # Should favor LLM for learning
        assert decision_info['source'] == "llm"

    def test_emergency_situation_handling(self):
        """Test that emergency situations favor LLM."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()
        agent.mode = AgentMode.ADAPTIVE
        agent.llm_preference_bonus = 0.0  # Remove initial bias

        llm_decision = AgentDecision("llm", 1, 0.5, "Emergency LLM")
        rl_decision = AgentDecision("rl", 2, 0.6, "RL reasoning")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        # Emergency situation
        observation = {"criticality": 4, "game_phase": 2}  # High criticality
        info = {}

        action, decision_info = agent.get_action(observation, info)

        # LLM should be preferred in emergency despite lower base confidence
        assert action == 1
        assert decision_info['source'] == "llm"

    def test_usage_stats_tracking(self):
        """Test that usage statistics are properly tracked."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        llm_decision = AgentDecision("llm", 1, 0.8, "LLM reasoning")
        rl_decision = AgentDecision("rl", 2, 0.6, "RL reasoning")

        mock_llm_agent.get_action.return_value = llm_decision
        mock_rl_agent.get_action.return_value = rl_decision

        initial_llm_count = agent.agent_usage_stats["llm"]

        # Force LLM choice
        agent.mode = AgentMode.LLM_GUIDED
        agent.get_action({}, {})

        assert agent.agent_usage_stats["llm"] == initial_llm_count + 1


class TestHybridAgentUpdate:
    """Test HybridAgent update and learning functionality."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent') as mock_llm_agent_class:
            with patch('agents.hybrid_agent.RLAgent') as mock_rl_agent_class:
                mock_llm_agent = Mock()
                mock_rl_agent = Mock()
                mock_llm_agent_class.return_value = mock_llm_agent
                mock_rl_agent_class.return_value = mock_rl_agent

                agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
                agent.llm_agent = mock_llm_agent
                agent.rl_agent = mock_rl_agent

                return agent, mock_llm_agent, mock_rl_agent

    def test_update_both_agents(self):
        """Test that update calls both component agents."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        observation = {"test": "obs"}
        action = 1
        reward = 0.5
        next_observation = {"test": "next_obs"}
        done = False

        agent.update(observation, action, reward, next_observation, done)

        mock_llm_agent.update.assert_called_once_with(
            observation, action, reward, next_observation, done
        )
        mock_rl_agent.update.assert_called_once_with(
            observation, action, reward, next_observation, done
        )

    def test_update_performance_tracking(self):
        """Test that update tracks performance."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        initial_llm_performance = len(agent.performance_by_agent["llm"])
        initial_rl_performance = len(agent.performance_by_agent["rl"])

        agent.update({}, 1, 0.5, {}, False)

        assert len(agent.performance_by_agent["llm"]) == initial_llm_performance + 1
        assert len(agent.performance_by_agent["rl"]) == initial_rl_performance + 1
        assert agent.performance_by_agent["llm"][-1] == 0.5
        assert agent.performance_by_agent["rl"][-1] == 0.5

    def test_update_experience_counter(self):
        """Test that experience counter is incremented."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        initial_counter = agent.experience_counter

        agent.update({}, 1, 0.5, {}, False)

        assert agent.experience_counter == initial_counter + 1

    def test_curriculum_progression(self):
        """Test curriculum learning progression."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        # Set short duration for testing
        agent.curriculum_config = {
            "stages": [
                {"name": "stage1", "llm_weight": 0.9, "duration": 3},
                {"name": "stage2", "llm_weight": 0.5, "duration": 3},
                {"name": "stage3", "llm_weight": 0.1, "duration": float('inf')}
            ]
        }

        initial_stage = agent.curriculum_stage

        # Update enough times to progress stage
        for _ in range(4):
            agent.update({}, 1, 0.1, {}, False)

        assert agent.curriculum_stage == initial_stage + 1
        assert agent.stage_progress > 0  # Should have some progress in new stage

    def test_agent_preference_adjustment(self):
        """Test agent preference adjustment based on performance."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        initial_preference = agent.llm_preference_bonus

        # Simulate good LLM performance
        for _ in range(20):
            agent.performance_by_agent["llm"].append(0.5)  # Good performance
            agent.performance_by_agent["rl"].append(-0.1)  # Poor performance

        agent._update_agent_preferences()

        # LLM preference should increase
        assert agent.llm_preference_bonus > initial_preference

    def test_preference_adjustment_favor_rl(self):
        """Test preference adjustment favoring RL."""
        agent, mock_llm_agent, mock_rl_agent = self.create_mock_agent()

        initial_preference = agent.llm_preference_bonus

        # Simulate good RL performance
        for _ in range(20):
            agent.performance_by_agent["llm"].append(-0.2)  # Poor performance
            agent.performance_by_agent["rl"].append(0.4)  # Good performance

        agent._update_agent_preferences()

        # LLM preference should decrease
        assert agent.llm_preference_bonus < initial_preference


class TestHybridAgentModeManagement:
    """Test HybridAgent mode switching and management."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent'):
            with patch('agents.hybrid_agent.RLAgent'):
                return HybridAgent(mock_llm_manager, mock_adaptive_strategy)

    def test_set_mode(self):
        """Test setting different modes."""
        agent = self.create_mock_agent()

        initial_mode = agent.mode

        agent.set_mode(AgentMode.LLM_GUIDED)
        assert agent.mode == AgentMode.LLM_GUIDED
        assert agent.mode != initial_mode

        agent.set_mode(AgentMode.RL_OPTIMIZED)
        assert agent.mode == AgentMode.RL_OPTIMIZED

        agent.set_mode(AgentMode.COLLABORATIVE)
        assert agent.mode == AgentMode.COLLABORATIVE


class TestHybridAgentStatistics:
    """Test HybridAgent statistics and monitoring."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent') as mock_llm_agent_class:
            with patch('agents.hybrid_agent.RLAgent') as mock_rl_agent_class:
                mock_llm_agent = Mock()
                mock_rl_agent = Mock()
                mock_llm_agent.get_confidence.return_value = 0.7
                mock_rl_agent.get_confidence.return_value = 0.5
                mock_llm_agent_class.return_value = mock_llm_agent
                mock_rl_agent_class.return_value = mock_rl_agent

                agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
                agent.llm_agent = mock_llm_agent
                agent.rl_agent = mock_rl_agent

                return agent

    def test_get_stats_basic(self):
        """Test getting basic statistics."""
        agent = self.create_mock_agent()

        # Add some usage
        agent.agent_usage_stats["llm"] = 10
        agent.agent_usage_stats["rl"] = 5
        agent.experience_counter = 100

        stats = agent.get_stats()

        assert stats["mode"] == agent.mode.value
        assert stats["experience_counter"] == 100
        assert stats["curriculum_stage"] == 0
        assert stats["llm_preference_bonus"] == 0.1

        assert "agent_usage" in stats
        assert stats["agent_usage"]["llm"] == 10/15  # 10 out of 15 total
        assert stats["agent_usage"]["rl"] == 5/15   # 5 out of 15 total

        assert "agent_confidence" in stats
        assert stats["agent_confidence"]["llm"] == 0.7
        assert stats["agent_confidence"]["rl"] == 0.5

    def test_get_stats_with_performance(self):
        """Test statistics with performance data."""
        agent = self.create_mock_agent()

        # Add performance data
        for i in range(15):
            agent.performance_by_agent["llm"].append(i * 0.1)
            agent.performance_by_agent["rl"].append(i * 0.05)

        stats = agent.get_stats()

        assert "recent_performance" in stats
        assert "llm" in stats["recent_performance"]
        assert "rl" in stats["recent_performance"]

    def test_get_stats_no_usage(self):
        """Test statistics with no agent usage."""
        agent = self.create_mock_agent()

        stats = agent.get_stats()

        # All usage should be 0 when no decisions made
        assert stats["agent_usage"]["llm"] == 0
        assert stats["agent_usage"]["rl"] == 0
        assert stats["agent_usage"]["hybrid"] == 0


class TestHybridAgentStatePersistence:
    """Test HybridAgent state saving and loading."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent'):
            with patch('agents.hybrid_agent.RLAgent') as mock_rl_agent_class:
                mock_rl_agent = Mock()
                # Create a proper dictionary for q_table that supports .items()
                q_table_dict = {"test_state": np.array([1, 2, 3])}
                mock_rl_agent.q_table = q_table_dict
                mock_rl_agent_class.return_value = mock_rl_agent

                agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
                agent.rl_agent = mock_rl_agent

                return agent

    def test_save_state(self):
        """Test saving agent state to file."""
        agent = self.create_mock_agent()

        # Modify some state
        agent.curriculum_stage = 2
        agent.stage_progress = 150
        agent.llm_preference_bonus = 0.3
        agent.agent_usage_stats = {"llm": 50, "rl": 30, "hybrid": 10}

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'hybrid_agent_state.json')

            agent.save_state(filepath)

            assert os.path.exists(filepath)

            # Verify file contents
            with open(filepath, 'r') as f:
                saved_state = json.load(f)

            assert saved_state["curriculum_stage"] == 2
            assert saved_state["stage_progress"] == 150
            assert saved_state["llm_preference_bonus"] == 0.3
            assert saved_state["agent_usage_stats"]["llm"] == 50

    def test_load_state_success(self):
        """Test loading agent state from file."""
        agent = self.create_mock_agent()

        state_data = {
            "curriculum_stage": 3,
            "stage_progress": 200,
            "llm_preference_bonus": 0.4,
            "agent_usage_stats": {"llm": 60, "rl": 40, "hybrid": 20},
            "rl_q_table": {"test_state": [4, 5, 6]}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_state.json')

            with open(filepath, 'w') as f:
                json.dump(state_data, f)

            agent.load_state(filepath)

            assert agent.curriculum_stage == 3
            assert agent.stage_progress == 200
            assert agent.llm_preference_bonus == 0.4
            assert agent.agent_usage_stats["llm"] == 60

    def test_load_state_file_not_found(self):
        """Test loading state when file doesn't exist."""
        agent = self.create_mock_agent()

        initial_stage = agent.curriculum_stage

        # Should not crash when file doesn't exist
        agent.load_state("nonexistent_file.json")

        # State should be unchanged
        assert agent.curriculum_stage == initial_stage

    def test_load_state_corrupted_file(self):
        """Test loading state with corrupted JSON file."""
        agent = self.create_mock_agent()

        initial_stage = agent.curriculum_stage

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_filepath = f.name

        try:
            agent.load_state(temp_filepath)
            # Should not crash
            assert agent.curriculum_stage == initial_stage
        finally:
            os.unlink(temp_filepath)

    def test_get_state_dict(self):
        """Test getting state dictionary."""
        agent = self.create_mock_agent()

        # Add some test data
        agent.agent_usage_stats = {"llm": 10, "rl": 5, "hybrid": 2}

        state_dict = agent.get_state_dict()

        assert "rl_model_state" in state_dict
        assert "agent_usage_stats" in state_dict
        assert state_dict["agent_usage_stats"]["llm"] == 10

    def test_load_state_dict(self):
        """Test loading from state dictionary."""
        agent = self.create_mock_agent()

        state_dict = {
            "rl_model_state": {"test": "data"},
            "agent_usage_stats": {"llm": 15, "rl": 8, "hybrid": 3}
        }

        agent.load_state_dict(state_dict)

        assert agent.agent_usage_stats["llm"] == 15
        assert agent.agent_usage_stats["rl"] == 8


@pytest.mark.integration
class TestHybridAgentIntegration:
    """Integration tests for HybridAgent."""

    def create_mock_agent(self):
        """Create a HybridAgent with mocked but functional components."""
        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        with patch('agents.hybrid_agent.LLMAgent') as mock_llm_agent_class:
            with patch('agents.hybrid_agent.RLAgent') as mock_rl_agent_class:
                # Create somewhat realistic mock agents
                mock_llm_agent = Mock()
                mock_rl_agent = Mock()

                def llm_get_action(obs, info):
                    return AgentDecision("llm", 1, 0.7, "LLM decision")

                def rl_get_action(obs, info):
                    return AgentDecision("rl", 2, 0.5, "RL decision")

                mock_llm_agent.get_action = llm_get_action
                mock_rl_agent.get_action = rl_get_action
                mock_llm_agent.get_confidence.return_value = 0.7
                mock_rl_agent.get_confidence.return_value = 0.5

                # Add q_table for state persistence tests
                mock_rl_agent.q_table = {"test_state": np.array([1, 2, 3])}

                mock_llm_agent_class.return_value = mock_llm_agent
                mock_rl_agent_class.return_value = mock_rl_agent

                agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
                agent.llm_agent = mock_llm_agent
                agent.rl_agent = mock_rl_agent

                return agent

    def test_full_decision_cycle(self):
        """Test complete decision making and update cycle."""
        agent = self.create_mock_agent()

        observation = {"game_phase": 1, "criticality": 2}
        info = {"test": "info"}

        # Get action
        action, decision_info = agent.get_action(observation, info)

        assert action in [1, 2]  # LLM or RL action
        assert decision_info['source'] in ['llm', 'rl']
        assert 'confidence' in decision_info

        # Update with experience
        reward = 0.5
        next_observation = {"game_phase": 1, "criticality": 1}
        done = False

        initial_counter = agent.experience_counter

        agent.update(observation, action, reward, next_observation, done)

        assert agent.experience_counter == initial_counter + 1

    def test_mode_switching_effects(self):
        """Test that mode switching affects decision making."""
        agent = self.create_mock_agent()

        observation = {}
        info = {}

        # Test LLM guided mode
        agent.set_mode(AgentMode.LLM_GUIDED)
        action1, info1 = agent.get_action(observation, info)
        assert info1['source'] == 'llm'

        # Test RL optimized mode
        agent.set_mode(AgentMode.RL_OPTIMIZED)
        action2, info2 = agent.get_action(observation, info)
        assert info2['source'] == 'rl'

        # Actions might be different
        # (though not guaranteed due to mocking)

    def test_curriculum_learning_progression(self):
        """Test curriculum learning over extended training."""
        agent = self.create_mock_agent()

        # Set short durations for testing
        agent.curriculum_config = {
            "stages": [
                {"name": "stage1", "llm_weight": 0.9, "duration": 5},
                {"name": "stage2", "llm_weight": 0.5, "duration": 5},
                {"name": "stage3", "llm_weight": 0.1, "duration": float('inf')}
            ]
        }

        initial_stage = agent.curriculum_stage

        # Simulate training
        for i in range(12):
            observation = {"step": i}
            action, _ = agent.get_action(observation, {})
            reward = 0.1
            next_obs = {"step": i + 1}
            agent.update(observation, action, reward, next_obs, False)

        # Should have progressed through curriculum stages
        assert agent.curriculum_stage > initial_stage

    def test_performance_based_adaptation(self):
        """Test that agent adapts based on performance."""
        agent = self.create_mock_agent()

        initial_preference = agent.llm_preference_bonus

        # Simulate many updates with varying performance
        for i in range(25):
            observation = {"step": i}
            action, _ = agent.get_action(observation, {})

            # Simulate LLM performing better
            if i < 20:
                agent.performance_by_agent["llm"].append(0.5)
                agent.performance_by_agent["rl"].append(-0.1)

            agent.update(observation, action, 0.1, {"step": i + 1}, False)

        # LLM preference should have increased
        assert agent.llm_preference_bonus >= initial_preference

    def test_state_persistence_cycle(self):
        """Test complete state save/load cycle."""
        agent = self.create_mock_agent()

        # Modify agent state
        agent.curriculum_stage = 2
        agent.agent_usage_stats["llm"] = 25
        agent.llm_preference_bonus = 0.35

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'agent_state.json')

            # Save state
            agent.save_state(filepath)

            # Create new agent and load state
            new_agent = self.create_mock_agent()
            new_agent.load_state(filepath)

            # States should match
            assert new_agent.curriculum_stage == 2
            assert new_agent.agent_usage_stats["llm"] == 25
            assert new_agent.llm_preference_bonus == 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])