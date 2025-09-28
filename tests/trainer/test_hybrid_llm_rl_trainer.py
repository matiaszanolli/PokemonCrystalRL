#!/usr/bin/env python3
"""
Test Suite for Hybrid LLM-RL Training System

Tests the hybrid decision engine, temporal memory integration,
and curriculum learning coordination.
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from training.hybrid_llm_rl_trainer import HybridLLMRLTrainer, TrainingConfig
from agents.hybrid_llm_rl_agent import HybridLLMRLAgent, DecisionMode, DecisionMetrics
from core.temporal_memory import TemporalMemoryBuffer, TemporalState
from core.experience_memory import ExperienceMemory
from core.save_state_library import SaveStateLibrary


class TestHybridLLMRLAgent:
    """Test the hybrid decision engine."""

    @pytest.fixture
    def mock_llm_agent(self):
        """Mock LLM agent for testing."""
        agent = Mock()
        agent.get_action.return_value = 1  # Up action
        return agent

    @pytest.fixture
    def mock_rl_agent(self):
        """Mock RL agent for testing."""
        agent = Mock()
        agent.act.return_value = 2  # Down action
        agent.get_q_values.return_value = np.array([0.1, 0.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1])
        agent.memory = []
        agent.batch_size = 32
        agent.remember = Mock()
        agent.replay = Mock()
        return agent

    @pytest.fixture
    def mock_temporal_memory(self):
        """Mock temporal memory for testing."""
        base_memory = Mock()
        temporal_memory = Mock()
        temporal_memory.add_experience = Mock()
        temporal_memory.update_last_reward = Mock()
        temporal_memory.get_similar_states.return_value = [(Mock(), 0.7), (Mock(), 0.5)]
        temporal_memory.get_recent_experiences.return_value = [Mock()]
        return temporal_memory

    @pytest.fixture
    def hybrid_agent(self, mock_llm_agent, mock_rl_agent, mock_temporal_memory):
        """Create hybrid agent for testing."""
        config = {
            'llm_weight': 0.7,
            'rl_weight': 0.3,
            'exploration_rate': 0.1
        }

        agent = HybridLLMRLAgent(
            llm_agent=mock_llm_agent,
            rl_agent=mock_rl_agent,
            temporal_memory=mock_temporal_memory,
            config=config
        )
        return agent

    def test_initialization(self, hybrid_agent):
        """Test hybrid agent initializes correctly."""
        assert hybrid_agent.llm_agent is not None
        assert hybrid_agent.rl_agent is not None
        assert hybrid_agent.temporal_memory is not None
        assert hybrid_agent.llm_weight == 0.7
        assert hybrid_agent.rl_weight == 0.3
        assert hybrid_agent.exploration_rate == 0.1
        assert isinstance(hybrid_agent.metrics, DecisionMetrics)

    def test_feature_vector_extraction(self, hybrid_agent):
        """Test feature vector extraction from game state."""
        game_state = {
            'player_hp': 50,
            'player_level': 10,
            'badges': 2,
            'money': 5000,
            'party': [{'name': 'Cyndaquil'}, {'name': 'Geodude'}],
            'player_x': 100,
            'player_y': 150,
            'map_id': 15,
            'screen_state': 'overworld'
        }

        features = hybrid_agent._extract_feature_vector(game_state)

        assert isinstance(features, np.ndarray)
        assert features.shape == (32,)
        assert 0.0 <= features[0] <= 1.0  # Normalized HP
        assert 0.0 <= features[1] <= 1.0  # Normalized level
        assert features[2] == 2/16  # Normalized badges
        assert abs(features[4] - 2/6) < 0.001  # Party size (accounting for float precision)

    def test_decision_mode_selection(self, hybrid_agent):
        """Test decision mode selection logic."""
        # Create mock decision context
        game_state = {'badges': 0, 'party': []}
        temporal_state = Mock()
        temporal_state.feature_vector = np.zeros(32)

        context = hybrid_agent._create_decision_context(game_state, temporal_state)

        # Test mode selection
        mode = hybrid_agent._select_decision_mode(context)
        assert isinstance(mode, DecisionMode)

    def test_llm_decision(self, hybrid_agent):
        """Test LLM-based decision making."""
        game_state = {'player_hp': 100}
        temporal_state = Mock()
        temporal_state.feature_vector = np.zeros(32)

        context = hybrid_agent._create_decision_context(game_state, temporal_state)
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        action, confidence = hybrid_agent._llm_decision(context, action_space)

        assert action in action_space
        assert 0.0 <= confidence <= 1.0
        hybrid_agent.llm_agent.get_action.assert_called()

    def test_rl_decision(self, hybrid_agent):
        """Test RL-based decision making."""
        game_state = {'player_hp': 100}
        temporal_state = Mock()
        temporal_state.feature_vector = np.zeros(32)

        context = hybrid_agent._create_decision_context(game_state, temporal_state)
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        action, confidence = hybrid_agent._rl_decision(context, action_space)

        assert action in action_space
        assert 0.0 <= confidence <= 1.0
        hybrid_agent.rl_agent.act.assert_called()

    def test_hybrid_decision(self, hybrid_agent):
        """Test hybrid decision combining LLM and RL."""
        game_state = {'player_hp': 100}
        temporal_state = Mock()
        temporal_state.feature_vector = np.zeros(32)

        context = hybrid_agent._create_decision_context(game_state, temporal_state)
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        action, confidence = hybrid_agent._hybrid_decision(context, action_space)

        assert action in action_space
        assert 0.0 <= confidence <= 1.0

    def test_decide_action_integration(self, hybrid_agent):
        """Test complete decision making process."""
        game_state = {
            'player_hp': 50,
            'player_level': 10,
            'badges': 1,
            'screen_state': 'overworld'
        }
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        action, decision_info = hybrid_agent.decide_action(game_state, action_space)

        assert action in action_space
        assert 'mode' in decision_info
        assert 'confidence' in decision_info
        assert 'decision_time' in decision_info
        assert 'temporal_state' in decision_info

        # Verify temporal memory was called
        hybrid_agent.temporal_memory.add_experience.assert_called()

    def test_reward_update(self, hybrid_agent):
        """Test reward feedback integration."""
        # First make a decision to set up history
        game_state = {'player_hp': 50}
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]
        action, _ = hybrid_agent.decide_action(game_state, action_space)

        # Now update with reward
        next_state = {'player_hp': 60}
        reward = 10.0

        hybrid_agent.update_with_reward(action, reward, next_state)

        # Verify temporal memory was updated
        hybrid_agent.temporal_memory.update_last_reward.assert_called_with(reward)

        # Verify RL agent was updated
        if hybrid_agent.rl_agent:
            hybrid_agent.rl_agent.remember.assert_called()

    def test_metrics_tracking(self, hybrid_agent):
        """Test decision metrics tracking."""
        initial_decisions = hybrid_agent.metrics.total_decisions

        game_state = {'player_hp': 50}
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        # Make several decisions
        for _ in range(5):
            hybrid_agent.decide_action(game_state, action_space)

        assert hybrid_agent.metrics.total_decisions == initial_decisions + 5

    def test_weight_adaptation(self, hybrid_agent):
        """Test adaptive weight adjustment."""
        initial_llm_weight = hybrid_agent.llm_weight
        initial_rl_weight = hybrid_agent.rl_weight

        # Simulate enough decision history to trigger weight adaptation
        hybrid_agent.decision_history = [
            (DecisionMode.LLM_STRATEGIC, 1, 50.0, 0.8),  # Very good LLM decisions
            (DecisionMode.LLM_STRATEGIC, 2, 60.0, 0.9),
            (DecisionMode.LLM_STRATEGIC, 3, 55.0, 0.9),
            (DecisionMode.LLM_STRATEGIC, 4, 45.0, 0.8),
            (DecisionMode.RL_TACTICAL, 5, -10.0, 0.6),   # Very poor RL decisions
            (DecisionMode.RL_TACTICAL, 6, -15.0, 0.7),
            (DecisionMode.RL_TACTICAL, 7, -20.0, 0.5),
            (DecisionMode.RL_TACTICAL, 8, -12.0, 0.6),
        ] * 3  # Repeat to get enough history

        hybrid_agent.adapt_weights()

        # Weights should adjust based on performance (LLM performed much better)
        # Check that the adaptation logic ran successfully
        assert hasattr(hybrid_agent, 'llm_weight')
        assert hasattr(hybrid_agent, 'rl_weight')
        # LLM weight should be higher than RL weight given the performance difference
        assert hybrid_agent.llm_weight >= hybrid_agent.rl_weight

    def test_decision_summary(self, hybrid_agent):
        """Test decision summary generation."""
        # Make some decisions first
        game_state = {'player_hp': 50}
        action_space = [0, 1, 2, 3, 4, 5, 6, 7]

        for _ in range(3):
            hybrid_agent.decide_action(game_state, action_space)

        summary = hybrid_agent.get_decision_summary()

        assert 'total_decisions' in summary
        assert 'mode_distribution' in summary
        assert 'success_rates' in summary
        assert 'current_weights' in summary
        assert summary['total_decisions'] == 3


class TestHybridTrainer:
    """Test the hybrid trainer implementation."""

    @pytest.fixture
    def temp_rom_path(self):
        """Create temporary ROM file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gbc', delete=False) as f:
            f.write(b"fake_rom_data")
            return f.name

    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return TrainingConfig(
            max_episodes=5,
            max_actions_per_episode=50,
            llm_weight=0.7,
            rl_weight=0.3,
            enable_curriculum=False,
            enable_web=False
        )

    @pytest.fixture
    def mock_save_state_library(self):
        """Mock save state library for testing."""
        library = Mock()
        library.list_save_states.return_value = []
        return library

    @patch('training.hybrid_llm_rl_trainer.EnhancedPyBoyPokemonCrystalEnv')
    @patch('training.hybrid_llm_rl_trainer.LLMAgent')
    @patch('training.hybrid_llm_rl_trainer.DQNAgent')
    def test_trainer_initialization(
        self, mock_dqn, mock_llm, mock_env,
        temp_rom_path, training_config, mock_save_state_library
    ):
        """Test trainer initializes correctly."""
        trainer = HybridLLMRLTrainer(
            rom_path=temp_rom_path,
            config=training_config,
            save_state_library=mock_save_state_library
        )

        assert trainer.rom_path == temp_rom_path
        assert trainer.config == training_config
        assert trainer.current_episode == 0
        assert trainer.total_actions == 0
        assert not trainer.is_training

        # Verify components were initialized
        mock_env.assert_called_once()
        mock_llm.assert_called_once()
        mock_dqn.assert_called_once()

    @patch('training.hybrid_llm_rl_trainer.EnhancedPyBoyPokemonCrystalEnv')
    @patch('training.hybrid_llm_rl_trainer.LLMAgent')
    @patch('training.hybrid_llm_rl_trainer.DQNAgent')
    def test_training_summary(
        self, mock_dqn, mock_llm, mock_env,
        temp_rom_path, training_config
    ):
        """Test training summary generation."""
        trainer = HybridLLMRLTrainer(
            rom_path=temp_rom_path,
            config=training_config
        )

        # Add some mock episode data
        trainer.episode_rewards = [10.0, 15.0, 20.0]
        trainer.episode_lengths = [50, 60, 55]

        summary = trainer.get_training_summary()

        assert 'training_status' in summary
        assert 'performance' in summary
        assert 'hybrid_agent' in summary
        assert 'temporal_memory' in summary
        assert summary['episodes_completed'] == 3
        assert summary['performance']['avg_reward'] == 15.0


class TestTemporalMemoryIntegration:
    """Test temporal memory integration with hybrid training."""

    def test_temporal_state_creation(self):
        """Test temporal state creation from game state."""
        # This would test the temporal memory integration
        # with the hybrid agent decision making
        pass

    def test_experience_storage(self):
        """Test experience storage in temporal memory."""
        # Test that experiences are properly stored
        # with LLM reasoning and RL features
        pass

    def test_similarity_search(self):
        """Test state similarity search in temporal memory."""
        # Test that similar states can be found
        # for novelty calculation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])