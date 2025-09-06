#!/usr/bin/env python3
"""
Hybrid LLM-RL Trainer for Pokemon Crystal.

This module merges the functionality of the standalone and the integrated trainer,
providing a unified interface for both LLM-guided and DQN-based training.
"""

import json
import sys
import os
import time
import logging
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from pyboy import PyBoy
from http.server import HTTPServer

from core.enhanced_reward_system import RewardCalculator
from core.state.analyzer import GameStateAnalyzer
from core.decision_validator import DecisionValidator
from core.strategic_context_builder import StrategicContextBuilder
from llm_trainer import LLMAgent, WebMonitor, build_observation


@dataclass
class TrainingConfig:
    """Configuration for hybrid LLM-RL training."""
    rom_path: str
    headless: bool = True
    observation_type: str = "multi_modal"
    llm_model: str = "smollm2:1.7b"
    llm_base_url: str = "http://localhost:11434"
    max_context_length: int = 8000
    initial_strategy: str = "llm_heavy"
    decision_db_path: str = "decisions.db"
    save_dir: str = "training_checkpoints"
    enable_web: bool = True
    web_port: int = 8080
    web_host: str = "localhost"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Hybrid training parameters
    dqn_params: Dict = field(default_factory=lambda: {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'memory_size': 50000,
        'training_frequency': 4,
        'target_update': 1000,
        'gamma': 0.99,
        'epsilon_start': 0.9,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995
    })
    
    # LLM integration parameters
    llm_params: Dict = field(default_factory=lambda: {
        'interval': 20,
        'temperature': 0.7,
        'use_memory': True,
        'context_window': 10,
        'decision_cache_size': 1000
    })
    
    # Strategy adaptation parameters
    strategy_params: Dict = field(default_factory=lambda: {
        'success_threshold': 0.6,
        'failure_threshold': 0.3,
        'adaptation_period': 100,
        'min_llm_ratio': 0.2,
        'max_llm_ratio': 0.8
    })


class HybridTrainer:
    """Unified trainer combining LLM guidance with DQN learning."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the hybrid trainer."""
        self.config = config
        self.initialize_logging()
        
        # Initialize core components
        self.initialize_pyboy()
        self.initialize_agents()
        self.initialize_web_monitoring()
        self.initialize_metrics()
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        self.logger.info(
            f"Hybrid trainer initialized: LLM={config.llm_model}, "
            f"Strategy={config.initial_strategy}"
        )
    
    def initialize_logging(self):
        """Set up logging system."""
        self.logger = logging.getLogger('HybridTrainer')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Add file handler
        os.makedirs(self.config.save_dir, exist_ok=True)
        fh = logging.FileHandler(
            Path(self.config.save_dir) / 'training.log'
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
        
        # Add console handler if in debug mode
        if self.config.debug_mode:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(
                '%(levelname)s - %(message)s'
            ))
            self.logger.addHandler(ch)
    
    def initialize_pyboy(self):
        """Initialize PyBoy emulator."""
        self.logger.info(f"Initializing PyBoy with ROM: {self.config.rom_path}")
        try:
            self.pyboy = PyBoy(
                self.config.rom_path,
                window="null" if self.config.headless else "SDL2",
                debug=self.config.debug_mode
            )
            self.previous_state = build_observation(self.pyboy.memory)
        except Exception as e:
            self.logger.error(f"PyBoy initialization failed: {e}")
            raise
    
    def initialize_agents(self):
        """Initialize LLM and DQN agents."""
        # Create LLM agent
        self.llm_agent = LLMAgent(
            model_name=self.config.llm_model,
            base_url=self.config.llm_base_url
        )
        
        # Create DQN agent if needed
        if self.config.initial_strategy != "llm_only":
            from core.dqn_agent import DQNAgent
            self.dqn_agent = DQNAgent(**self.config.dqn_params)
        else:
            self.dqn_agent = None
        
        # Initialize hybrid strategy
        self.strategy = self.config.initial_strategy
        self.llm_ratio = 0.8 if self.strategy == "llm_heavy" else 0.2
    
    def initialize_web_monitoring(self):
        """Set up web monitoring if enabled."""
        if not self.config.enable_web:
            return
            
        try:
            self.web_server = HTTPServer(
                (self.config.web_host, self.config.web_port),
                WebMonitor
            )
            self.web_server.trainer = self
            self.web_thread = threading.Thread(
                target=self.web_server.serve_forever,
                daemon=True
            )
            self.web_thread.start()
            self.logger.info(
                f"Web monitoring active at http://{self.config.web_host}:"
                f"{self.config.web_port}"
            )
        except Exception as e:
            self.logger.warning(f"Web monitoring initialization failed: {e}")
            self.config.enable_web = False
    
    def initialize_metrics(self):
        """Initialize training metrics and statistics."""
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'llm_decisions': [],
            'dqn_losses': [],
            'strategy_switches': 0,
            'start_time': time.time(),
            'total_steps': 0,
            'current_episode': 0
        }
        
        # Performance tracking
        self.performance_tracking = {
            'reward_window': [],
            'llm_success_window': [],
            'action_counts': {},
            'state_transitions': {},
            'reward_stats': {
                'mean': 0.0,
                'std': 0.0,
                'best': float('-inf'),
                'recent': []
            },
            'strategy_stats': {
                'llm_success_rate': 0.0,
                'dqn_success_rate': 0.0,
                'adaptation_history': []
            }
        }
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        import signal
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        self.running = True
    
    def shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown."""
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Stop PyBoy
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()
        
        # Stop web server
        if hasattr(self, 'web_server'):
            self.web_server.shutdown()
            if hasattr(self, 'web_thread'):
                self.web_thread.join(timeout=1.0)
        
        # Save final metrics
        self.save_metrics()
        self.logger.info("Shutdown complete")
    
    def save_metrics(self):
        """Save training metrics and models."""
        save_path = Path(self.config.save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = save_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save DQN model if enabled
        if self.dqn_agent:
            model_path = save_path / "final_dqn_model.pt"
            self.dqn_agent.save_model(str(model_path))
        
        # Save performance stats
        stats_path = save_path / "performance_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.performance_tracking, f, indent=2)
        
        self.logger.info(f"Training data saved to {save_path}")
    
    def get_action(self) -> Tuple[int, str, float]:
        """Get next action from current strategy."""
        state = build_observation(self.pyboy.memory)
        
        # Always get LLM decision at regular intervals
        use_llm = (
            self.metrics['total_steps'] % 
            self.config.llm_params['interval'] == 0
        )
        
        if use_llm and self.llm_agent.available:
            action, reasoning = self.llm_agent.get_decision(
                state,
                self.analyze_screen(),
                self.get_recent_actions()
            )
            return action, reasoning, 1.0
            
        # Use DQN if available
        if self.dqn_agent:
            action = self.dqn_agent.get_action(state, training=True)
            return action, "DQN decision", 0.0
            
        # Fallback to rule-based
        return self.get_rule_based_action(state), "Rule-based fallback", 0.0
    
    def get_rule_based_action(self, state: Dict) -> int:
        """Fallback rule-based action selection."""
        # Map game state to simple actions
        if state.get('in_battle', False):
            return 0  # 'a' for battle
        elif state.get('menu_active', False):
            return 1  # 'b' to exit menus
        else:
            # Cycle through movement pattern
            step = self.metrics['total_steps'] % 4
            return 2 + step  # up, right, down, left
    
    def update_strategy(self, episode_reward: float):
        """Update strategy based on performance."""
        window = self.config.strategy_params['adaptation_period']
        if len(self.metrics['episode_rewards']) < window:
            return
            
        # Calculate success metrics
        recent_rewards = self.metrics['episode_rewards'][-window:]
        success_rate = sum(1 for r in recent_rewards if r > 0) / window
        
        # Get strategy parameters
        params = self.config.strategy_params
        
        if success_rate < params['failure_threshold']:
            # Poor performance - adjust strategy
            if self.strategy == "llm_heavy":
                self.strategy = "balanced"
                self.llm_ratio = 0.5
            elif self.strategy == "balanced":
                self.strategy = "dqn_heavy"
                self.llm_ratio = params['min_llm_ratio']
            
            self.metrics['strategy_switches'] += 1
            self.logger.info(
                f"Strategy adjusted to {self.strategy} "
                f"(success rate: {success_rate:.2f})"
            )
            
        elif success_rate > params['success_threshold']:
            # Good performance - consider increasing LLM usage
            if self.llm_ratio < params['max_llm_ratio']:
                self.llm_ratio = min(
                    self.llm_ratio + 0.1,
                    params['max_llm_ratio']
                )
                self.logger.info(
                    f"Increased LLM ratio to {self.llm_ratio:.2f} "
                    f"(success rate: {success_rate:.2f})"
                )
    
    def train(
        self,
        total_episodes: int = 50,
        max_steps_per_episode: int = 1000,
        save_interval: int = 10,
        eval_interval: int = 15,
        curriculum_patience: int = 10
    ) -> Dict:
        """Run complete training process."""
        self.logger.info("Starting training...")
        
        try:
            for episode in range(total_episodes):
                if not self.running:
                    break
                    
                self.logger.info(f"Starting episode {episode + 1}")
                episode_reward = self.run_episode(max_steps_per_episode)
                
                # Update metrics
                self.metrics['episode_rewards'].append(episode_reward)
                self.metrics['current_episode'] = episode + 1
                
                # Update strategy
                self.update_strategy(episode_reward)
                
                # Save checkpoints
                if (episode + 1) % save_interval == 0:
                    self.save_metrics()
                
                # Evaluate if needed
                if (episode + 1) % eval_interval == 0:
                    self.evaluate()
                
            return self.compile_training_summary()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.shutdown()
    
    def run_episode(self, max_steps: int) -> float:
        """Run a single training episode."""
        total_reward = 0.0
        steps = 0
        
        # Reset environment
        self.pyboy.reset()
        last_state = build_observation(self.pyboy.memory)
        
        while steps < max_steps and self.running:
            # Get action from current strategy
            action, reasoning, llm_weight = self.get_action()
            
            # Execute action
            self._execute_action(action)
            steps += 1
            
            # Get reward
            current_state = build_observation(self.pyboy.memory)
            reward = self.calculate_reward(current_state, last_state)
            total_reward += reward
            
            # Store experience for DQN if enabled
            if self.dqn_agent and llm_weight < 1.0:
                self.dqn_agent.store_experience(
                    last_state, action, reward,
                    current_state, False
                )
            
            # Update tracking
            self.performance_tracking['reward_window'].append(reward)
            self.metrics['total_steps'] += 1
            last_state = current_state
            
            # Train DQN if needed
            if (self.dqn_agent and 
                steps % self.config.dqn_params['training_frequency'] == 0):
                loss = self.dqn_agent.train_step()
                if loss is not None:
                    self.metrics['dqn_losses'].append(float(loss))
        
        self.metrics['episode_lengths'].append(steps)
        return total_reward
    
    def evaluate(self, num_episodes: int = 5) -> Dict:
        """Evaluate current performance."""
        self.logger.info("Running evaluation...")
        rewards = []
        lengths = []
        llm_usage = []
        
        for episode in range(num_episodes):
            if not self.running:
                break
                
            episode_reward = self.run_evaluation_episode()
            rewards.append(episode_reward['reward'])
            lengths.append(episode_reward['length'])
            llm_usage.append(episode_reward['llm_usage'])
        
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'avg_llm_usage': np.mean(llm_usage)
        }
        
        self.logger.info(
            f"Evaluation results: "
            f"Reward={results['avg_reward']:.2f}±{results['std_reward']:.2f}, "
            f"Length={results['avg_length']:.1f}, "
            f"LLM={results['avg_llm_usage']:.1%}"
        )
        
        return results
    
    def run_evaluation_episode(self) -> Dict:
        """Run a single evaluation episode."""
        total_reward = 0.0
        steps = 0
        llm_decisions = 0
        
        # Run episode with current strategy but no exploration
        self.pyboy.reset()
        last_state = build_observation(self.pyboy.memory)
        
        while steps < 1000 and self.running:  # Max 1000 steps
            action, _, llm_weight = self.get_action()
            if llm_weight > 0:
                llm_decisions += 1
            
            self._execute_action(action)
            current_state = build_observation(self.pyboy.memory)
            reward = self.calculate_reward(current_state, last_state)
            
            total_reward += reward
            steps += 1
            last_state = current_state
        
        return {
            'reward': total_reward,
            'length': steps,
            'llm_usage': llm_decisions / steps if steps > 0 else 0
        }
    
    def compile_training_summary(self) -> Dict:
        """Compile complete training summary."""
        # Calculate key metrics
        final_eval = self.evaluate()
        
        return {
            'total_episodes': self.metrics['current_episode'],
            'total_steps': self.metrics['total_steps'],
            'best_reward': max(self.metrics['episode_rewards']),
            'final_avg_reward': np.mean(self.metrics['episode_rewards'][-10:]),
            'strategy_switches': self.metrics['strategy_switches'],
            'avg_episode_length': np.mean(self.metrics['episode_lengths']),
            'avg_llm_usage': self.llm_ratio,
            'final_evaluation': final_eval,
            'training_time': time.time() - self.metrics['start_time']
        }
    
    def _execute_action(self, action: int):
        """Execute action in environment."""
        # Map action index to PyBoy button
        button_map = ['a', 'b', 'up', 'right', 'down', 'left', 'start', 'select']
        button = button_map[action]
        
        # Press button
        self.pyboy.send_input(button)
        for _ in range(4):  # Hold for 4 frames
            self.pyboy.tick()
        
        # Release button
        self.pyboy.send_input([])  # Clear inputs
        for _ in range(2):  # Wait 2 frames
            self.pyboy.tick()
    
    def calculate_reward(self, current: Dict, previous: Dict) -> float:
        """Calculate reward for state transition."""
        calculator = RewardCalculator()
        return calculator.calculate_reward(current, previous)
    
    def analyze_screen(self) -> Dict:
        """Analyze current screen state."""
        analyzer = GameStateAnalyzer()
        return analyzer.analyze_screen(self.pyboy.screen.ndarray)
    
    def get_recent_actions(self, window: int = 10) -> List[str]:
        """Get recent action history."""
        if not hasattr(self, '_action_history'):
            self._action_history = []
        return self._action_history[-window:]
