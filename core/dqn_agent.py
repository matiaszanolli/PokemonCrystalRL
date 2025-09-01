#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Pokemon Crystal

This module implements a DQN agent that learns to play Pokemon Crystal
by learning Q-values (action values) for different game states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import json
import os
import time

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network for Pokemon Crystal state-action value estimation"""
    
    def __init__(self, state_size: int = 32, action_size: int = 8, hidden_sizes: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for training stability"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent that learns to play Pokemon Crystal"""
    
    def __init__(self, 
                 state_size: int = 32,
                 action_size: int = 8,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 100000,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 device: str = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧠 DQN Agent using device: {self.device}")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Training tracking
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
        self.q_values_history = []
        
        # Action mapping
        self.actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
    
    def state_to_tensor(self, game_state: Dict, screen_analysis: Dict) -> torch.Tensor:
        """Convert game state to neural network input tensor"""
        # Extract key features from game state
        features = [
            # Player stats (normalized)
            game_state.get('party_count', 0) / 6.0,
            game_state.get('player_level', 0) / 100.0,
            game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1),
            game_state.get('badges_total', 0) / 16.0,
            game_state.get('money', 0) / 1000000.0,  # Normalize to reasonable range
            
            # Location features
            game_state.get('player_map', 0) / 255.0,
            game_state.get('player_x', 0) / 255.0,
            game_state.get('player_y', 0) / 255.0,
            
            # Battle state
            float(game_state.get('in_battle', 0)),
            game_state.get('enemy_level', 0) / 100.0,
            game_state.get('enemy_species', 0) / 255.0,
            
            # Screen analysis features
            screen_analysis.get('variance', 0) / 50000.0,  # Normalize screen variance
            screen_analysis.get('colors', 0) / 256.0,
            screen_analysis.get('brightness', 0) / 255.0,
            
            # Screen state one-hot encoding (simplified)
            float(screen_analysis.get('state', '') == 'overworld'),
            float(screen_analysis.get('state', '') == 'battle'),
            float(screen_analysis.get('state', '') == 'menu'),
            float(screen_analysis.get('state', '') == 'dialogue'),
            float(screen_analysis.get('state', '') == 'loading'),
            
            # Game progression indicators
            float(game_state.get('party_count', 0) > 0),  # Has Pokemon
            float(game_state.get('badges_total', 0) > 0),  # Has badges
            float(game_state.get('player_level', 0) > 5),  # Reasonable level
            
            # Health status
            1.0 if game_state.get('party_count', 0) == 0 else game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1),
            float(game_state.get('player_hp', 0) < game_state.get('player_max_hp', 1) * 0.3),  # Low health
            
            # Additional context features
            game_state.get('wram_c2a7', 0) / 255.0,  # Additional memory state
            game_state.get('wram_d358', 0) / 255.0,  # Additional memory state
            game_state.get('wram_d359', 0) / 255.0,  # Additional memory state
            game_state.get('wram_c2a5', 0) / 255.0,  # Additional memory state
            
            # Derived features
            float(game_state.get('money', 0) > 1000),  # Has money
            float(game_state.get('player_map', 0) != 0),  # Not in default map
            
            # Padding to reach state_size (32)
            0.0, 0.0  # Can be used for additional features later
        ]
        
        # Ensure we have exactly state_size features
        features = features[:self.state_size]
        while len(features) < self.state_size:
            features.append(0.0)
        
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        try:
            return self.actions.index(action.lower())
        except ValueError:
            return 0  # Default to 'up'
    
    def index_to_action(self, index: int) -> str:
        """Convert action index to string"""
        return self.actions[index] if 0 <= index < len(self.actions) else 'up'
    
    def get_action(self, game_state: Dict, screen_analysis: Dict, training: bool = True) -> Tuple[str, float]:
        """Get action using epsilon-greedy policy"""
        state_tensor = self.state_to_tensor(game_state, screen_analysis)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_size)
            q_value = 0.0  # Random action has no meaningful Q-value
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.max(1)[1].item()
                q_value = q_values.max(1)[0].item()
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return self.index_to_action(action_idx), q_value
    
    def get_action_values(self, game_state: Dict, screen_analysis: Dict) -> Dict[str, float]:
        """Get Q-values for all actions"""
        state_tensor = self.state_to_tensor(game_state, screen_analysis)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
        
        return {action: q_value for action, q_value in zip(self.actions, q_values)}
    
    def store_experience(self, state: Dict, screen_analysis: Dict, action: str, 
                        reward: float, next_state: Dict, next_screen_analysis: Dict, done: bool):
        """Store experience in replay buffer"""
        state_tensor = self.state_to_tensor(state, screen_analysis).cpu().squeeze(0)
        next_state_tensor = self.state_to_tensor(next_state, next_screen_analysis).cpu().squeeze(0)
        action_idx = self.action_to_index(action)
        
        self.memory.push(state_tensor, action_idx, reward, next_state_tensor, done)
    
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at step {self.steps_done}")
        
        # Track loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"🔄 DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            print(f"⚠️ Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.losses = checkpoint.get('losses', [])
            
            print(f"🔄 DQN model loaded from {filepath}")
            print(f"   Steps trained: {self.steps_done}")
            print(f"   Current epsilon: {self.epsilon:.4f}")
            print(f"   Episodes recorded: {len(self.episode_rewards)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return False
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        recent_losses = self.losses[-100:] if self.losses else [0]
        recent_rewards = self.episode_rewards[-10:] if self.episode_rewards else [0]
        
        return {
            'steps_trained': self.steps_done,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_recent_loss': np.mean(recent_losses),
            'avg_recent_reward': np.mean(recent_rewards),
            'total_episodes': len(self.episode_rewards),
            'device': str(self.device)
        }

class HybridAgent:
    """Hybrid agent that combines LLM reasoning with DQN action values"""
    
    def __init__(self, dqn_agent: DQNAgent, llm_agent, 
                 dqn_weight: float = 0.3, exploration_bonus: float = 0.1):
        self.dqn_agent = dqn_agent
        self.llm_agent = llm_agent
        self.dqn_weight = dqn_weight  # How much to trust DQN vs LLM
        self.exploration_bonus = exploration_bonus
        
        # Adaptive weighting
        self.min_dqn_weight = 0.1
        self.max_dqn_weight = 0.8
        self.performance_history = deque(maxlen=100)
    
    def get_hybrid_action(self, game_state: Dict, screen_analysis: Dict, 
                         recent_actions: List[str]) -> Tuple[str, str]:
        """Get action using hybrid LLM + DQN approach"""
        
        # Get DQN action values
        dqn_action, dqn_q_value = self.dqn_agent.get_action(game_state, screen_analysis, training=True)
        action_values = self.dqn_agent.get_action_values(game_state, screen_analysis)
        
        # Get LLM suggestion
        llm_action, llm_reasoning = self.llm_agent.get_decision(game_state, screen_analysis, recent_actions)
        
        # Adaptive weighting based on training progress
        current_dqn_weight = self._get_adaptive_weight()
        
        # If DQN is confident and trained enough, consider its suggestion
        if (self.dqn_agent.steps_done > 1000 and 
            dqn_q_value > 0.5 and 
            random.random() < current_dqn_weight):
            chosen_action = dqn_action
            reasoning = f"DQN choice (Q={dqn_q_value:.3f}, weight={current_dqn_weight:.2f}): {dqn_action}"
        else:
            chosen_action = llm_action
            reasoning = f"LLM choice (DQN weight={current_dqn_weight:.2f}): {llm_reasoning[:50]}..."
        
        # Add DQN values to reasoning for debugging
        top_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)[:3]
        dqn_info = ", ".join([f"{act}:{val:.2f}" for act, val in top_actions])
        reasoning += f" | DQN top: {dqn_info}"
        
        return chosen_action, reasoning
    
    def _get_adaptive_weight(self) -> float:
        """Calculate adaptive weight based on DQN performance"""
        base_weight = self.dqn_weight
        
        # Increase DQN weight as training progresses
        training_progress = min(self.dqn_agent.steps_done / 10000, 1.0)
        progress_bonus = training_progress * 0.3
        
        # Adjust based on recent performance
        performance_bonus = 0.0
        if len(self.performance_history) > 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            if recent_performance > 0.1:  # If recent rewards are positive
                performance_bonus = 0.2
        
        final_weight = base_weight + progress_bonus + performance_bonus
        return np.clip(final_weight, self.min_dqn_weight, self.max_dqn_weight)
    
    def record_performance(self, reward: float):
        """Record performance for adaptive weighting"""
        self.performance_history.append(reward)
    
    def get_info(self) -> str:
        """Get hybrid agent status info"""
        dqn_stats = self.dqn_agent.get_training_stats()
        current_weight = self._get_adaptive_weight()
        
        return (f"DQN: {dqn_stats['steps_trained']} steps, ε={dqn_stats['epsilon']:.3f}, "
                f"weight={current_weight:.2f}, mem={dqn_stats['memory_size']}")
