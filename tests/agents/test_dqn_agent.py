"""
Comprehensive tests for DQNAgent class.

Tests the Deep Q-Network reinforcement learning agent including neural network
initialization, state processing, action selection, experience replay, training,
model saving/loading, and memory management.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from agents.dqn_agent import DQNAgent, DQNNetwork, ReplayBuffer, Experience


class TestDQNNetwork:
    """Test DQNNetwork neural network implementation."""

    def test_init_default_params(self):
        """Test DQNNetwork initialization with default parameters."""
        network = DQNNetwork()

        assert network.state_size == 32
        assert network.action_size == 8
        assert isinstance(network.network, torch.nn.Sequential)

    def test_init_custom_params(self):
        """Test DQNNetwork initialization with custom parameters."""
        network = DQNNetwork(state_size=64, action_size=16, hidden_sizes=[512, 256])

        assert network.state_size == 64
        assert network.action_size == 16

    def test_forward_pass(self):
        """Test forward pass through network."""
        network = DQNNetwork(state_size=32, action_size=8)

        # Create random input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 32)

        output = network(input_tensor)

        assert output.shape == (batch_size, 8)
        assert not torch.isnan(output).any()

    def test_forward_single_input(self):
        """Test forward pass with single input."""
        network = DQNNetwork()

        input_tensor = torch.randn(1, 32)
        output = network(input_tensor)

        assert output.shape == (1, 8)

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        network = DQNNetwork()

        # Check that weights are not zero
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestReplayBuffer:
    """Test ReplayBuffer experience storage and sampling."""

    def test_init_empty_buffer(self):
        """Test ReplayBuffer initialization."""
        buffer = ReplayBuffer(capacity=1000)

        assert len(buffer) == 0
        assert buffer.capacity == 1000

    def test_push_experience(self):
        """Test adding experience to buffer."""
        buffer = ReplayBuffer(capacity=10)

        state = torch.randn(32)
        action = 2
        reward = 1.5
        next_state = torch.randn(32)
        done = False

        buffer.push(state, action, reward, next_state, done)

        assert len(buffer) == 1

    def test_push_multiple_experiences(self):
        """Test adding multiple experiences."""
        buffer = ReplayBuffer(capacity=10)

        for i in range(5):
            state = torch.randn(32)
            buffer.push(state, i % 8, i * 0.1, torch.randn(32), i == 4)

        assert len(buffer) == 5

    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is exceeded."""
        buffer = ReplayBuffer(capacity=3)

        # Add more experiences than capacity
        for i in range(5):
            state = torch.randn(32)
            buffer.push(state, i, i, torch.randn(32), False)

        assert len(buffer) == 3  # Should not exceed capacity

    def test_sample_experiences(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=10)

        # Add experiences
        for i in range(5):
            state = torch.randn(32)
            buffer.push(state, i, i * 0.1, torch.randn(32), False)

        # Sample
        batch = buffer.sample(3)

        assert len(batch) == 3
        assert all(isinstance(exp, Experience) for exp in batch)

    def test_sample_insufficient_experiences(self):
        """Test sampling when buffer has insufficient experiences."""
        buffer = ReplayBuffer(capacity=10)

        # Add only 2 experiences
        for i in range(2):
            state = torch.randn(32)
            buffer.push(state, i, i, torch.randn(32), False)

        # Try to sample more than available
        with pytest.raises(ValueError):
            buffer.sample(5)


class TestDQNAgentInitialization:
    """Test DQNAgent initialization."""

    def test_init_default_params(self):
        """Test DQNAgent initialization with default parameters."""
        agent = DQNAgent()

        assert agent.state_size == 32
        assert agent.action_size == 8
        assert agent.learning_rate == 1e-4
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.995
        assert agent.batch_size == 32
        assert agent.target_update == 1000

        assert isinstance(agent.q_network, DQNNetwork)
        assert isinstance(agent.target_network, DQNNetwork)
        assert isinstance(agent.memory, ReplayBuffer)

    def test_init_custom_params(self):
        """Test DQNAgent initialization with custom parameters."""
        agent = DQNAgent(
            state_size=64,
            action_size=16,
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=0.8,
            memory_size=50000,
            batch_size=64
        )

        assert agent.state_size == 64
        assert agent.action_size == 16
        assert agent.learning_rate == 0.001
        assert agent.gamma == 0.95
        assert agent.epsilon == 0.8
        assert agent.batch_size == 64

    def test_device_selection(self):
        """Test device selection (CPU/CUDA)."""
        agent = DQNAgent()

        # Should use available device
        assert agent.device in [torch.device('cpu'), torch.device('cuda')]

    def test_target_network_copy(self):
        """Test that target network is properly copied from main network."""
        agent = DQNAgent()

        # Check that networks have same structure but different instances
        assert type(agent.q_network) == type(agent.target_network)
        assert agent.q_network is not agent.target_network

        # Check that initial weights are the same
        for param1, param2 in zip(agent.q_network.parameters(), agent.target_network.parameters()):
            assert torch.allclose(param1, param2)


class TestDQNAgentStateProcessing:
    """Test DQNAgent state processing functionality."""

    def test_state_to_tensor_basic(self):
        """Test basic state to tensor conversion."""
        agent = DQNAgent()

        game_state = {
            'party_count': 1,
            'player_level': 10,
            'player_hp': 50,
            'player_max_hp': 100,
            'badges_total': 2,
            'money': 1500
        }
        screen_analysis = {
            'state': 'overworld',
            'variance': 1000,
            'colors': 16,
            'brightness': 128
        }

        tensor = agent.state_to_tensor(game_state, screen_analysis)

        assert tensor.shape == (1, 32)  # Batch size 1, state size 32
        assert tensor.device.type == agent.device.type  # Same device type (cpu/cuda)
        assert not torch.isnan(tensor).any()

    def test_state_to_tensor_normalization(self):
        """Test that state values are properly normalized."""
        agent = DQNAgent()

        game_state = {
            'party_count': 6,  # Max party
            'player_level': 100,  # Max level
            'player_hp': 100,
            'player_max_hp': 100,  # Full health
            'badges_total': 16,  # All badges
            'money': 999999  # Rich
        }
        screen_analysis = {
            'variance': 50000,
            'colors': 256,
            'brightness': 255
        }

        tensor = agent.state_to_tensor(game_state, screen_analysis)

        # Check that normalized values are in reasonable ranges
        values = tensor.squeeze().cpu().numpy()
        assert all(0.0 <= val <= 1.0 for val in values[:5])  # First few should be normalized to [0,1]

    def test_state_to_tensor_missing_values(self):
        """Test state processing with missing values."""
        agent = DQNAgent()

        # Empty states should use defaults
        empty_game_state = {}
        empty_screen_analysis = {}

        tensor = agent.state_to_tensor(empty_game_state, empty_screen_analysis)

        assert tensor.shape == (1, 32)
        assert not torch.isnan(tensor).any()

    def test_state_to_tensor_padding(self):
        """Test that state tensor is properly padded to state_size."""
        agent = DQNAgent(state_size=50)  # Larger state size

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        tensor = agent.state_to_tensor(game_state, screen_analysis)

        assert tensor.shape == (1, 50)


class TestDQNAgentActionSelection:
    """Test DQNAgent action selection methods."""

    def test_action_to_index_valid(self):
        """Test converting valid action strings to indices."""
        agent = DQNAgent()

        test_cases = [
            ('up', 0),
            ('down', 1),
            ('left', 2),
            ('right', 3),
            ('a', 4),
            ('b', 5),
            ('start', 6),
            ('select', 7)
        ]

        for action, expected_index in test_cases:
            index = agent.action_to_index(action)
            assert index == expected_index

    def test_action_to_index_invalid(self):
        """Test converting invalid action strings."""
        agent = DQNAgent()

        invalid_actions = ['invalid', 'x', 'y', 'unknown']

        for action in invalid_actions:
            index = agent.action_to_index(action)
            assert index == 0  # Should default to 'up'

    def test_index_to_action_valid(self):
        """Test converting valid indices to action strings."""
        agent = DQNAgent()

        for i in range(8):
            action = agent.index_to_action(i)
            assert action in agent.actions

    def test_index_to_action_invalid(self):
        """Test converting invalid indices."""
        agent = DQNAgent()

        invalid_indices = [-1, 8, 100]

        for index in invalid_indices:
            action = agent.index_to_action(index)
            assert action == 'up'  # Should default to 'up'

    def test_get_action_exploration(self):
        """Test action selection during exploration (high epsilon)."""
        agent = DQNAgent(epsilon_start=1.0)  # Always explore

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        # Multiple calls should potentially return different actions
        actions = set()
        for _ in range(20):
            action, q_value = agent.get_action(game_state, screen_analysis, training=True)
            actions.add(action)
            assert action in agent.actions
            # Random actions should have low Q-values (due to untrained network) or be 0.0
            assert -1.0 <= q_value <= 1.0  # Q-values should be in reasonable range

        # Should have some variety in actions
        assert len(actions) > 1

    def test_get_action_exploitation(self):
        """Test action selection during exploitation (low epsilon)."""
        agent = DQNAgent(epsilon_start=0.0)  # Never explore

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        action1, q_value1 = agent.get_action(game_state, screen_analysis, training=True)
        action2, q_value2 = agent.get_action(game_state, screen_analysis, training=True)

        # Should be consistent when not exploring
        assert action1 == action2
        assert isinstance(q_value1, float)

    def test_get_action_eval_mode(self):
        """Test action selection in evaluation mode."""
        agent = DQNAgent(epsilon_start=1.0)  # High exploration in training

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        action, q_value = agent.get_action(game_state, screen_analysis, training=False)

        # In eval mode, should not explore
        assert action in agent.actions
        assert isinstance(q_value, float)

    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        agent = DQNAgent(epsilon_start=1.0, epsilon_decay=0.9)

        initial_epsilon = agent.epsilon

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        # Take action to trigger epsilon decay
        agent.get_action(game_state, screen_analysis, training=True)

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end

    def test_get_action_values(self):
        """Test getting Q-values for all actions."""
        agent = DQNAgent()

        game_state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}

        action_values = agent.get_action_values(game_state, screen_analysis)

        assert isinstance(action_values, dict)
        assert len(action_values) == len(agent.actions)

        for action in agent.actions:
            assert action in action_values
            assert isinstance(action_values[action], float)


class TestDQNAgentExperienceReplay:
    """Test DQNAgent experience replay functionality."""

    def test_store_experience(self):
        """Test storing experience in replay buffer."""
        agent = DQNAgent()

        state = {'party_count': 1}
        screen_analysis = {'state': 'overworld'}
        action = 'up'
        reward = 1.0
        next_state = {'party_count': 1, 'player_x': 11}
        next_screen_analysis = {'state': 'overworld'}
        done = False

        initial_memory_size = len(agent.memory)

        agent.store_experience(state, screen_analysis, action, reward,
                             next_state, next_screen_analysis, done)

        assert len(agent.memory) == initial_memory_size + 1

    def test_store_multiple_experiences(self):
        """Test storing multiple experiences."""
        agent = DQNAgent()

        for i in range(10):
            state = {'party_count': 1, 'step': i}
            screen_analysis = {'state': 'overworld'}
            action = 'up'
            reward = i * 0.1
            next_state = {'party_count': 1, 'step': i + 1}
            next_screen_analysis = {'state': 'overworld'}
            done = i == 9

            agent.store_experience(state, screen_analysis, action, reward,
                                 next_state, next_screen_analysis, done)

        assert len(agent.memory) == 10


class TestDQNAgentTraining:
    """Test DQNAgent training functionality."""

    def test_train_step_insufficient_memory(self):
        """Test training step with insufficient replay memory."""
        agent = DQNAgent(batch_size=32)

        # Add only a few experiences (less than batch_size)
        for i in range(5):
            state = {'step': i}
            agent.store_experience(state, {}, 'up', 0.0, {'step': i+1}, {}, False)

        loss = agent.train_step()

        assert loss == 0.0  # Should not train with insufficient memory

    def test_train_step_sufficient_memory(self):
        """Test training step with sufficient replay memory."""
        agent = DQNAgent(batch_size=4)

        # Add enough experiences
        for i in range(10):
            state = {'step': i}
            agent.store_experience(state, {}, 'up', i * 0.1, {'step': i+1}, {}, i == 9)

        loss = agent.train_step()

        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_target_network_update(self):
        """Test target network periodic update."""
        agent = DQNAgent(target_update=5, batch_size=2)

        # Add experiences
        for i in range(10):
            state = {'step': i}
            agent.store_experience(state, {}, 'up', 0.1, {'step': i+1}, {}, False)

        # Get initial target network weights
        initial_target_params = [param.clone() for param in agent.target_network.parameters()]

        # Train enough steps to trigger target update
        for _ in range(6):
            agent.train_step()

        # Check that target network was updated
        updated_target_params = list(agent.target_network.parameters())

        # At least one parameter should have changed
        params_changed = any(
            not torch.allclose(initial, updated)
            for initial, updated in zip(initial_target_params, updated_target_params)
        )
        assert params_changed

    def test_gradient_clipping(self):
        """Test that gradient clipping is applied during training."""
        agent = DQNAgent(batch_size=2)

        # Add experiences with extreme rewards to potentially cause large gradients
        for i in range(10):
            state = {'step': i}
            reward = 1000.0 if i % 2 == 0 else -1000.0
            agent.store_experience(state, {}, 'up', reward, {'step': i+1}, {}, False)

        # Training should complete without NaN values
        loss = agent.train_step()

        assert not np.isnan(loss)

        # Check that no parameters are NaN
        for param in agent.q_network.parameters():
            assert not torch.isnan(param).any()


class TestDQNAgentModelPersistence:
    """Test DQNAgent model saving and loading."""

    def test_save_model(self):
        """Test saving model to file."""
        agent = DQNAgent()

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pth')

            agent.save_model(filepath)

            assert os.path.exists(filepath)

    def test_load_model_success(self):
        """Test loading model from file."""
        agent1 = DQNAgent()

        # Modify agent state
        agent1.epsilon = 0.5
        agent1.steps_done = 100
        agent1.episode_rewards = [1.0, 2.0, 3.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pth')

            # Save model
            agent1.save_model(filepath)

            # Create new agent and load
            agent2 = DQNAgent()
            success = agent2.load_model(filepath)

            assert success is True
            assert agent2.epsilon == 0.5
            assert agent2.steps_done == 100
            assert agent2.episode_rewards == [1.0, 2.0, 3.0]

    def test_load_model_file_not_found(self):
        """Test loading model when file doesn't exist."""
        agent = DQNAgent()

        success = agent.load_model('nonexistent_file.pth')

        assert success is False

    @patch('torch.load')
    def test_load_model_corrupted_file(self, mock_torch_load):
        """Test loading model with corrupted file."""
        mock_torch_load.side_effect = Exception("Corrupted file")

        agent = DQNAgent()

        with tempfile.NamedTemporaryFile(suffix='.pth') as temp_file:
            success = agent.load_model(temp_file.name)

        assert success is False


class TestDQNAgentStatistics:
    """Test DQNAgent statistics and monitoring."""

    def test_get_training_stats(self):
        """Test getting training statistics."""
        agent = DQNAgent()

        # Add some training data
        agent.steps_done = 500
        agent.epsilon = 0.3
        agent.losses = [0.1, 0.2, 0.15, 0.08]
        agent.episode_rewards = [10.0, 15.0, 12.0]

        stats = agent.get_training_stats()

        assert stats['steps_trained'] == 500
        assert stats['epsilon'] == 0.3
        assert stats['memory_size'] == len(agent.memory)
        assert 'avg_recent_loss' in stats
        assert 'avg_recent_reward' in stats
        assert stats['total_episodes'] == 3
        assert 'device' in stats

    def test_get_memory_usage(self):
        """Test getting memory usage statistics."""
        agent = DQNAgent(memory_size=1000)

        # Add some experiences
        for i in range(100):
            state = {'step': i}
            agent.store_experience(state, {}, 'up', 0.1, {'step': i+1}, {}, False)

        memory_stats = agent.get_memory_usage()

        assert 'replay_buffer' in memory_stats
        assert memory_stats['replay_buffer']['size'] == 100
        assert memory_stats['replay_buffer']['capacity'] == 1000
        assert memory_stats['replay_buffer']['usage_percent'] == 10.0

        assert 'model' in memory_stats
        assert 'total_estimated_mb' in memory_stats
        assert 'device' in memory_stats

    def test_get_memory_usage_error_handling(self):
        """Test memory usage statistics error handling."""
        agent = DQNAgent()

        # Delete memory to cause error
        del agent.memory

        memory_stats = agent.get_memory_usage()

        assert 'error' in memory_stats
        assert 'replay_buffer' in memory_stats


class TestDQNAgentIntegration:
    """Integration tests for DQNAgent."""

    def test_full_training_episode(self):
        """Test a complete training episode."""
        agent = DQNAgent(batch_size=4, target_update=10)

        episode_states = []
        episode_actions = []
        episode_rewards = []

        # Simulate episode
        for step in range(20):
            state = {'step': step, 'party_count': 1}
            screen_analysis = {'state': 'overworld'}

            # Get action
            action, q_value = agent.get_action(state, screen_analysis, training=True)

            # Simulate reward
            reward = 1.0 if step % 5 == 0 else 0.1

            # Simulate next state
            next_state = {'step': step + 1, 'party_count': 1}
            next_screen_analysis = {'state': 'overworld'}
            done = step == 19

            # Store experience
            agent.store_experience(state, screen_analysis, action, reward,
                                 next_state, next_screen_analysis, done)

            # Train if enough experience
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                assert loss >= 0.0

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

        # Verify episode completion
        assert len(episode_states) == 20
        assert len(agent.memory) == 20
        assert agent.steps_done > 0

    def test_action_consistency(self):
        """Test that action selection is consistent in eval mode."""
        agent = DQNAgent(epsilon_start=0.0)  # No exploration

        state = {'party_count': 1, 'player_level': 10}
        screen_analysis = {'state': 'overworld'}

        # Get action multiple times
        actions = []
        q_values = []
        for _ in range(5):
            action, q_value = agent.get_action(state, screen_analysis, training=False)
            actions.append(action)
            q_values.append(q_value)

        # All actions should be the same
        assert all(action == actions[0] for action in actions)
        # All Q-values should be the same
        assert all(abs(q_val - q_values[0]) < 1e-6 for q_val in q_values)

    @pytest.mark.slow
    def test_learning_progression(self):
        """Test that agent shows learning progression over time."""
        agent = DQNAgent(batch_size=8, epsilon_decay=0.99)

        initial_epsilon = agent.epsilon
        initial_losses = []
        final_losses = []

        # Train for multiple episodes
        for episode in range(5):
            for step in range(20):
                state = {'episode': episode, 'step': step}
                screen_analysis = {'state': 'overworld'}

                action, _ = agent.get_action(state, screen_analysis, training=True)
                reward = np.random.uniform(-1, 1)
                next_state = {'episode': episode, 'step': step + 1}
                next_screen_analysis = {'state': 'overworld'}
                done = step == 19

                agent.store_experience(state, screen_analysis, action, reward,
                                     next_state, next_screen_analysis, done)

                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()

                    if episode == 0:
                        initial_losses.append(loss)
                    elif episode == 4:
                        final_losses.append(loss)

        # Check that epsilon has decayed
        assert agent.epsilon < initial_epsilon

        # Check that we have training data
        assert len(initial_losses) > 0
        assert len(final_losses) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])