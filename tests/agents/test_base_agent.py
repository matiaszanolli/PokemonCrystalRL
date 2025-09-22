"""
Comprehensive tests for BaseAgent class.

Tests the basic agent functionality including initialization, state tracking,
statistics management, and interface contracts.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from agents.base_agent import BaseAgent


class ConcreteTestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        """Simple test implementation of get_action."""
        return "up"


class TestBaseAgent:
    """Test BaseAgent initialization and basic functionality."""

    def test_init_with_no_config(self):
        """Test initialization with no configuration."""
        agent = ConcreteTestAgent()

        assert agent.config == {}
        assert agent.total_steps == 0
        assert agent.total_reward == 0.0
        assert agent.current_episode == 0
        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0
        assert agent._is_training is True

    def test_init_with_empty_config(self):
        """Test initialization with empty configuration."""
        agent = ConcreteTestAgent({})

        assert agent.config == {}
        assert agent.total_steps == 0
        assert agent.total_reward == 0.0
        assert agent.current_episode == 0
        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0
        assert agent._is_training is True

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'learning_rate': 0.001,
            'epsilon': 0.1,
            'custom_param': 'test_value'
        }
        agent = ConcreteTestAgent(config)

        assert agent.config == config
        assert agent.total_steps == 0
        assert agent.total_reward == 0.0

    def test_init_with_none_config(self):
        """Test initialization with None configuration."""
        agent = ConcreteTestAgent(None)

        assert agent.config == {}


class TestBaseAgentUpdate:
    """Test BaseAgent reward update functionality."""

    def test_update_positive_reward(self):
        """Test updating with positive reward."""
        agent = ConcreteTestAgent()

        agent.update(5.0)

        assert agent.total_reward == 5.0
        assert agent.episode_reward == 5.0
        assert agent.total_steps == 1
        assert agent.episode_steps == 1

    def test_update_negative_reward(self):
        """Test updating with negative reward."""
        agent = ConcreteTestAgent()

        agent.update(-2.5)

        assert agent.total_reward == -2.5
        assert agent.episode_reward == -2.5
        assert agent.total_steps == 1
        assert agent.episode_steps == 1

    def test_update_zero_reward(self):
        """Test updating with zero reward."""
        agent = ConcreteTestAgent()

        agent.update(0.0)

        assert agent.total_reward == 0.0
        assert agent.episode_reward == 0.0
        assert agent.total_steps == 1
        assert agent.episode_steps == 1

    def test_multiple_updates(self):
        """Test multiple reward updates."""
        agent = ConcreteTestAgent()

        agent.update(1.0)
        agent.update(2.5)
        agent.update(-0.5)

        assert agent.total_reward == 3.0
        assert agent.episode_reward == 3.0
        assert agent.total_steps == 3
        assert agent.episode_steps == 3

    def test_update_accumulation(self):
        """Test that updates accumulate correctly over time."""
        agent = ConcreteTestAgent()

        rewards = [1.0, -0.5, 2.0, -0.25, 1.5]
        expected_total = sum(rewards)

        for reward in rewards:
            agent.update(reward)

        assert agent.total_reward == expected_total
        assert agent.episode_reward == expected_total
        assert agent.total_steps == len(rewards)
        assert agent.episode_steps == len(rewards)


class TestBaseAgentReset:
    """Test BaseAgent episode reset functionality."""

    def test_reset_fresh_agent(self):
        """Test resetting a fresh agent."""
        agent = ConcreteTestAgent()

        agent.reset()

        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0
        assert agent.current_episode == 1
        # Total stats should remain unchanged
        assert agent.total_steps == 0
        assert agent.total_reward == 0.0

    def test_reset_after_updates(self):
        """Test resetting after some updates."""
        agent = ConcreteTestAgent()

        # Do some updates
        agent.update(5.0)
        agent.update(-1.0)

        # Reset episode
        agent.reset()

        # Episode stats should reset
        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0
        assert agent.current_episode == 1

        # Total stats should remain
        assert agent.total_steps == 2
        assert agent.total_reward == 4.0

    def test_multiple_resets(self):
        """Test multiple episode resets."""
        agent = ConcreteTestAgent()

        # Episode 1
        agent.update(2.0)
        agent.reset()

        # Episode 2
        agent.update(3.0)
        agent.update(1.0)
        agent.reset()

        # Episode 3
        agent.update(-1.0)
        agent.reset()

        assert agent.current_episode == 3
        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0
        assert agent.total_reward == 5.0  # 2.0 + 3.0 + 1.0 + (-1.0)
        assert agent.total_steps == 4

    def test_reset_preserves_training_mode(self):
        """Test that reset preserves training mode."""
        agent = ConcreteTestAgent()

        agent.eval()
        agent.reset()
        assert agent.is_training is False

        agent.train()
        agent.reset()
        assert agent.is_training is True


class TestBaseAgentTrainingMode:
    """Test BaseAgent training/evaluation mode functionality."""

    def test_initial_training_mode(self):
        """Test agent starts in training mode."""
        agent = ConcreteTestAgent()
        assert agent.is_training is True

    def test_set_training_mode(self):
        """Test setting training mode."""
        agent = ConcreteTestAgent()

        agent.train()
        assert agent.is_training is True
        assert agent._is_training is True

    def test_set_evaluation_mode(self):
        """Test setting evaluation mode."""
        agent = ConcreteTestAgent()

        agent.eval()
        assert agent.is_training is False
        assert agent._is_training is False

    def test_mode_switching(self):
        """Test switching between training and evaluation modes."""
        agent = ConcreteTestAgent()

        # Start in training
        assert agent.is_training is True

        # Switch to eval
        agent.eval()
        assert agent.is_training is False

        # Switch back to training
        agent.train()
        assert agent.is_training is True

        # Switch to eval again
        agent.eval()
        assert agent.is_training is False

    def test_is_training_property(self):
        """Test is_training property access."""
        agent = ConcreteTestAgent()

        # Property should reflect internal state
        assert agent.is_training == agent._is_training

        agent.eval()
        assert agent.is_training == agent._is_training

        agent.train()
        assert agent.is_training == agent._is_training


class TestBaseAgentStatistics:
    """Test BaseAgent statistics functionality."""

    def test_get_stats_initial(self):
        """Test getting statistics from fresh agent."""
        agent = ConcreteTestAgent()

        stats = agent.get_stats()

        expected_stats = {
            'total_steps': 0,
            'total_reward': 0.0,
            'current_episode': 0,
            'episode_reward': 0.0,
            'episode_steps': 0,
            'is_training': True
        }

        assert stats == expected_stats

    def test_get_stats_after_updates(self):
        """Test getting statistics after updates."""
        agent = ConcreteTestAgent()

        agent.update(2.5)
        agent.update(-0.5)

        stats = agent.get_stats()

        expected_stats = {
            'total_steps': 2,
            'total_reward': 2.0,
            'current_episode': 0,
            'episode_reward': 2.0,
            'episode_steps': 2,
            'is_training': True
        }

        assert stats == expected_stats

    def test_get_stats_after_reset(self):
        """Test getting statistics after episode reset."""
        agent = ConcreteTestAgent()

        agent.update(5.0)
        agent.reset()
        agent.update(1.0)

        stats = agent.get_stats()

        expected_stats = {
            'total_steps': 2,
            'total_reward': 6.0,
            'current_episode': 1,
            'episode_reward': 1.0,
            'episode_steps': 1,
            'is_training': True
        }

        assert stats == expected_stats

    def test_get_stats_in_eval_mode(self):
        """Test getting statistics in evaluation mode."""
        agent = ConcreteTestAgent()

        agent.eval()
        agent.update(3.0)

        stats = agent.get_stats()

        assert stats['is_training'] is False
        assert stats['total_reward'] == 3.0

    def test_stats_consistency(self):
        """Test that statistics remain consistent over operations."""
        agent = ConcreteTestAgent()

        # Perform various operations
        agent.update(1.0)
        stats1 = agent.get_stats()

        agent.eval()
        stats2 = agent.get_stats()

        agent.train()
        agent.update(2.0)
        stats3 = agent.get_stats()

        agent.reset()
        stats4 = agent.get_stats()

        # Check consistency
        assert stats1['total_steps'] == 1
        assert stats2['total_steps'] == 1  # No change
        assert stats3['total_steps'] == 2  # Incremented
        assert stats4['total_steps'] == 2  # Preserved after reset

        assert stats1['total_reward'] == 1.0
        assert stats2['total_reward'] == 1.0  # No change
        assert stats3['total_reward'] == 3.0  # Incremented
        assert stats4['total_reward'] == 3.0  # Preserved after reset

        assert stats4['episode_reward'] == 0.0  # Reset
        assert stats4['current_episode'] == 1  # Incremented


class TestBaseAgentConfiguration:
    """Test BaseAgent configuration handling."""

    def test_config_immutability(self):
        """Test that config is stored properly."""
        original_config = {'param1': 'value1', 'param2': 42}
        agent = ConcreteTestAgent(original_config)

        # Modify original config
        original_config['param1'] = 'modified'

        # Agent config should be unaffected (if it was copied)
        # Note: BaseAgent currently doesn't copy, so this tests current behavior
        assert agent.config == original_config

    def test_config_access(self):
        """Test accessing configuration parameters."""
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'device': 'cpu'
        }
        agent = ConcreteTestAgent(config)

        assert agent.config['learning_rate'] == 0.001
        assert agent.config['batch_size'] == 32
        assert agent.config['device'] == 'cpu'

    def test_config_modification(self):
        """Test modifying configuration after initialization."""
        config = {'param': 'initial_value'}
        agent = ConcreteTestAgent(config)

        # Modify config
        agent.config['param'] = 'modified_value'
        agent.config['new_param'] = 'new_value'

        assert agent.config['param'] == 'modified_value'
        assert agent.config['new_param'] == 'new_value'


class TestBaseAgentEdgeCases:
    """Test BaseAgent edge cases and error conditions."""

    def test_large_reward_values(self):
        """Test handling large reward values."""
        agent = ConcreteTestAgent()

        large_positive = 1e6
        large_negative = -1e6

        agent.update(large_positive)
        assert agent.total_reward == large_positive

        agent.update(large_negative)
        assert agent.total_reward == 0.0

    def test_very_small_reward_values(self):
        """Test handling very small reward values."""
        agent = ConcreteTestAgent()

        tiny_reward = 1e-10
        agent.update(tiny_reward)

        assert agent.total_reward == tiny_reward
        assert agent.episode_reward == tiny_reward

    def test_many_episodes(self):
        """Test handling many episode resets."""
        agent = ConcreteTestAgent()

        for i in range(1000):
            agent.update(0.1)
            agent.reset()

        assert agent.current_episode == 1000
        assert agent.total_steps == 1000
        assert abs(agent.total_reward - 100.0) < 1e-10  # 1000 * 0.1 (with floating-point tolerance)
        assert agent.episode_reward == 0.0
        assert agent.episode_steps == 0

    def test_stats_return_type(self):
        """Test that get_stats returns proper types."""
        agent = ConcreteTestAgent()
        agent.update(1.5)

        stats = agent.get_stats()

        assert isinstance(stats, dict)
        assert isinstance(stats['total_steps'], int)
        assert isinstance(stats['total_reward'], float)
        assert isinstance(stats['current_episode'], int)
        assert isinstance(stats['episode_reward'], float)
        assert isinstance(stats['episode_steps'], int)
        assert isinstance(stats['is_training'], bool)

    def test_stats_keys_consistency(self):
        """Test that get_stats always returns same keys."""
        agent = ConcreteTestAgent()

        stats1 = agent.get_stats()
        agent.update(1.0)
        stats2 = agent.get_stats()
        agent.reset()
        stats3 = agent.get_stats()
        agent.eval()
        stats4 = agent.get_stats()

        expected_keys = {
            'total_steps', 'total_reward', 'current_episode',
            'episode_reward', 'episode_steps', 'is_training'
        }

        assert set(stats1.keys()) == expected_keys
        assert set(stats2.keys()) == expected_keys
        assert set(stats3.keys()) == expected_keys
        assert set(stats4.keys()) == expected_keys


@pytest.mark.integration
class TestBaseAgentIntegration:
    """Integration tests for BaseAgent functionality."""

    def test_training_simulation(self):
        """Test simulating a full training scenario."""
        agent = ConcreteTestAgent({'experiment': 'test'})

        # Simulate 3 episodes
        for episode in range(3):
            episode_rewards = []

            # Simulate steps in episode
            for step in range(10):
                # Simulate varying rewards
                reward = 1.0 if step % 3 == 0 else -0.1
                episode_rewards.append(reward)
                agent.update(reward)

            # Check episode stats
            assert agent.episode_steps == 10
            assert agent.episode_reward == sum(episode_rewards)
            assert agent.current_episode == episode

            # Reset for next episode
            agent.reset()

        # Final checks
        assert agent.current_episode == 3
        assert agent.total_steps == 30
        assert agent.episode_steps == 0  # Reset after last episode

    def test_eval_training_cycle(self):
        """Test cycling between training and evaluation."""
        agent = ConcreteTestAgent()

        # Training phase
        agent.train()
        for _ in range(5):
            agent.update(1.0)
        training_stats = agent.get_stats()

        # Evaluation phase
        agent.eval()
        for _ in range(3):
            agent.update(0.5)
        eval_stats = agent.get_stats()

        # Back to training
        agent.train()
        agent.update(2.0)
        final_stats = agent.get_stats()

        assert training_stats['is_training'] is True
        assert eval_stats['is_training'] is False
        assert final_stats['is_training'] is True

        assert final_stats['total_steps'] == 9  # 5 + 3 + 1
        assert final_stats['total_reward'] == 8.5  # 5*1.0 + 3*0.5 + 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])