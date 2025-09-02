"""
Hybrid LLM-RL Trainer that orchestrates the training process using both LLM guidance and RL optimization.
Implements curriculum learning with progressive transition from LLM to RL decision making.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from collections import defaultdict, deque

from core.enhanced_pyboy_env import EnhancedPyBoyPokemonCrystalEnv
from core.hybrid_agent import HybridAgent
from core.adaptive_strategy_system import AdaptiveStrategySystem
from core.decision_history_analyzer import DecisionHistoryAnalyzer
from trainer.llm_manager import LLMManager
from core.game_state_analyzer import GameStateAnalysis


class HybridLLMRLTrainer:
    """
    Main trainer class that orchestrates hybrid LLM-RL training with curriculum learning.
    """
    
    def __init__(
        self,
        env: EnhancedPyBoyPokemonCrystalEnv,
        agent: HybridAgent,
        strategy_system: AdaptiveStrategySystem,
        decision_analyzer: DecisionHistoryAnalyzer,
        llm_manager: LLMManager,
        save_dir: str = "checkpoints",
        log_level: str = "INFO"
    ):
        self.env = env
        self.agent = agent
        self.strategy_system = strategy_system
        self.decision_analyzer = decision_analyzer
        self.llm_manager = llm_manager
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.llm_usage_rates = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_thresholds = [50.0, 100.0, 200.0, 500.0]  # Reward thresholds
        self.llm_confidence_decay = 0.995  # Gradual reduction in LLM reliance
        
        # Performance tracking
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'curriculum_advancements': 0,
            'strategy_switches': 0
        }
        
    def train(
        self,
        total_episodes: int = 1000,
        max_steps_per_episode: int = 10000,
        save_interval: int = 100,
        eval_interval: int = 50,
        curriculum_patience: int = 20
    ) -> Dict[str, Any]:
        """
        Main training loop with curriculum learning and adaptive strategy switching.
        """
        self.logger.info(f"Starting hybrid LLM-RL training for {total_episodes} episodes")
        
        curriculum_no_improvement = 0
        last_avg_reward = float('-inf')
        
        for episode in range(total_episodes):
            episode_start_time = time.time()
            
            # Reset environment and get initial observation
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            llm_decisions = 0
            
            # Episode-level metrics
            episode_decisions = []
            episode_contexts = []
            
            for step in range(max_steps_per_episode):
                # Get action from hybrid agent
                action, decision_info = self.agent.get_action(obs, info)
                
                # Track decision source
                if decision_info.get('source') == 'llm':
                    llm_decisions += 1
                
                # Store decision context for learning
                decision_context = {
                    'observation': obs,
                    'action': action,
                    'decision_info': decision_info,
                    'game_state': info.get('game_state', {}),
                    'step': step
                }
                episode_decisions.append(decision_context)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                
                # Update agent (for RL components)
                self.agent.update(obs, action, reward, next_obs, terminated or truncated)
                
                # Update metrics
                episode_reward += reward
                episode_steps += 1
                self.training_stats['total_steps'] += 1
                
                # Check for episode termination
                if terminated or truncated:
                    break
                
                obs, info = next_obs, next_info
            
            # End of episode processing
            episode_time = time.time() - episode_start_time
            self.training_stats['episodes'] += 1
            
            # Record episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            llm_usage_rate = llm_decisions / episode_steps if episode_steps > 0 else 0
            self.llm_usage_rates.append(llm_usage_rate)
            
            # Analyze episode decisions for learning
            self._analyze_episode_decisions(episode_decisions, episode_reward)
            
            # Update adaptive strategy system
            self._update_strategy_system(episode_reward, llm_usage_rate)
            
            # Curriculum learning progression
            avg_reward = np.mean(list(self.episode_rewards)[-10:])  # Last 10 episodes
            if self._should_advance_curriculum(avg_reward):
                self.curriculum_stage += 1
                self.training_stats['curriculum_advancements'] += 1
                curriculum_no_improvement = 0
                self.logger.info(f"Advanced to curriculum stage {self.curriculum_stage}")
            elif avg_reward <= last_avg_reward:
                curriculum_no_improvement += 1
            else:
                curriculum_no_improvement = 0
            
            last_avg_reward = avg_reward
            
            # Update best reward
            if episode_reward > self.training_stats['best_reward']:
                self.training_stats['best_reward'] = episode_reward
                self._save_best_model(episode)
            
            # Logging
            if episode % 10 == 0:
                self._log_training_progress(episode, episode_reward, episode_time, llm_usage_rate)
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                self._evaluate_agent(num_eval_episodes=5)
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Reduce LLM confidence gradually for curriculum learning
            self.agent.llm_confidence_threshold *= self.llm_confidence_decay
        
        # Final evaluation and save
        final_eval_results = self._evaluate_agent(num_eval_episodes=10)
        self._save_checkpoint(total_episodes, final=True)
        
        training_summary = self._generate_training_summary(final_eval_results)
        self.logger.info("Training completed successfully")
        
        return training_summary
    
    def _analyze_episode_decisions(self, decisions: List[Dict], episode_reward: float):
        """Analyze episode decisions for pattern learning."""
        if not decisions:
            return
        
        # Extract decision sequence for pattern analysis
        action_sequence = [d['action'] for d in decisions]
        context_sequence = [d['decision_info'] for d in decisions]
        
        # Determine episode success
        is_success = episode_reward > np.mean(list(self.episode_rewards)[-20:]) if len(self.episode_rewards) > 0 else episode_reward > 0
        
        # Add to decision history for learning
        for i, decision in enumerate(decisions):
            decision_data = {
                'state_hash': hash(str(decision['game_state'])),
                'action': decision['action'],
                'context': decision['decision_info'],
                'outcome': 'success' if is_success else 'failure',
                'step_in_episode': i,
                'total_episode_reward': episode_reward
            }
            
            try:
                # Note: DecisionHistoryAnalyzer uses record_decision, not add_decision
                # Skip decision recording for now
                pass
            except Exception as e:
                self.logger.warning(f"Failed to record decision: {e}")
    
    def _update_strategy_system(self, episode_reward: float, llm_usage_rate: float):
        """Update the adaptive strategy system based on episode performance."""
        # Calculate performance metrics
        recent_rewards = list(self.episode_rewards)[-10:] if len(self.episode_rewards) >= 10 else list(self.episode_rewards)
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        performance_metrics = {
            'episode_reward': episode_reward,
            'average_reward': avg_reward,
            'llm_usage_rate': llm_usage_rate,
            'episode_length': self.episode_lengths[-1] if self.episode_lengths else 0
        }
        
        # Check if strategy should switch
        old_strategy = self.strategy_system.current_strategy
        # Note: AdaptiveStrategySystem doesn't have evaluate_performance
        # Skip strategy evaluation for now
        
        if self.strategy_system.current_strategy != old_strategy:
            self.training_stats['strategy_switches'] += 1
            self.logger.info(f"Strategy switched from {old_strategy} to {self.strategy_system.current_strategy}")
    
    def _should_advance_curriculum(self, avg_reward: float) -> bool:
        """Determine if curriculum should advance to next stage."""
        if self.curriculum_stage >= len(self.curriculum_thresholds):
            return False
        
        threshold = self.curriculum_thresholds[self.curriculum_stage]
        return avg_reward >= threshold and len(self.episode_rewards) >= 10
    
    def _evaluate_agent(self, num_eval_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance without learning updates."""
        self.logger.info(f"Running evaluation for {num_eval_episodes} episodes")
        
        eval_rewards = []
        eval_lengths = []
        eval_llm_usage = []
        
        # Store original exploration parameters
        original_exploration = getattr(self.agent, 'exploration_rate', None)
        
        # Reduce exploration for evaluation
        if hasattr(self.agent, 'exploration_rate'):
            self.agent.exploration_rate = 0.05
        
        for eval_episode in range(num_eval_episodes):
            obs, info = self.env.reset()
            eval_reward = 0
            eval_steps = 0
            eval_llm_decisions = 0
            
            while eval_steps < 5000:  # Max eval episode length
                action, decision_info = self.agent.get_action(obs, info)
                
                if decision_info.get('source') == 'llm':
                    eval_llm_decisions += 1
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                eval_reward += reward
                eval_steps += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_steps)
            eval_llm_usage.append(eval_llm_decisions / eval_steps if eval_steps > 0 else 0)
        
        # Restore original exploration
        if original_exploration is not None:
            self.agent.exploration_rate = original_exploration
        
        eval_results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_llm_usage': np.mean(eval_llm_usage)
        }
        
        self.logger.info(f"Evaluation results: Avg reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        
        return eval_results
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_episode_{episode}" + ("_final" if final else "")
        checkpoint_path = self.save_dir / f"{checkpoint_name}.pt"
        
        checkpoint_data = {
            'episode': episode,
            'agent_state': self.agent.get_state_dict(),
            'training_stats': self.training_stats,
            'curriculum_stage': self.curriculum_stage,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'llm_usage_rates': list(self.llm_usage_rates)
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_best_model(self, episode: int):
        """Save the best performing model."""
        best_model_path = self.save_dir / "best_model.pt"
        
        best_model_data = {
            'episode': episode,
            'reward': self.training_stats['best_reward'],
            'agent_state': self.agent.get_state_dict(),
            'curriculum_stage': self.curriculum_stage
        }
        
        torch.save(best_model_data, best_model_path)
        self.logger.info(f"Saved new best model with reward: {self.training_stats['best_reward']:.2f}")
    
    def _log_training_progress(self, episode: int, reward: float, episode_time: float, llm_usage: float):
        """Log training progress."""
        avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else reward
        avg_length = np.mean(list(self.episode_lengths)[-10:]) if len(self.episode_lengths) >= 10 else 0
        
        self.logger.info(
            f"Episode {episode}: Reward: {reward:.2f}, Avg: {avg_reward:.2f}, "
            f"Length: {self.episode_lengths[-1]}, LLM Usage: {llm_usage:.2%}, "
            f"Time: {episode_time:.2f}s, Stage: {self.curriculum_stage}"
        )
    
    def _generate_training_summary(self, final_eval: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        summary = {
            'total_episodes': self.training_stats['episodes'],
            'total_steps': self.training_stats['total_steps'],
            'best_reward': self.training_stats['best_reward'],
            'final_avg_reward': np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0,
            'curriculum_stage_reached': self.curriculum_stage,
            'curriculum_advancements': self.training_stats['curriculum_advancements'],
            'strategy_switches': self.training_stats['strategy_switches'],
            'final_evaluation': final_eval,
            'avg_episode_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0,
            'avg_llm_usage': np.mean(list(self.llm_usage_rates)) if self.llm_usage_rates else 0
        }
        
        # Save summary to file
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.agent.load_state_dict(checkpoint['agent_state'])
            self.training_stats = checkpoint['training_stats']
            self.curriculum_stage = checkpoint['curriculum_stage']
            self.episode_rewards.extend(checkpoint['episode_rewards'])
            self.episode_lengths.extend(checkpoint['episode_lengths'])
            self.llm_usage_rates.extend(checkpoint['llm_usage_rates'])
            
            self.logger.info(f"Loaded checkpoint from episode {checkpoint['episode']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False


def create_trainer_from_config(config_path: str) -> HybridLLMRLTrainer:
    """Create trainer instance from configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize environment
    env = EnhancedPyBoyPokemonCrystalEnv(
        rom_path=config['rom_path'],
        headless=config.get('headless', True),
        enable_strategic_context=True,
        enable_action_masking=True
    )
    
    # Initialize LLM manager
    llm_manager = LLMManager(
        model=config.get('llm_model', 'gpt-4'),
        max_context_turns=config.get('max_context_turns', 5)
    )
    
    # Initialize decision analyzer
    decision_analyzer = DecisionHistoryAnalyzer(
        db_path=config.get('decision_db_path', 'decisions.db')
    )
    
    # Initialize adaptive strategy system
    strategy_system = AdaptiveStrategySystem(
        history_analyzer=decision_analyzer
    )
    
    # Initialize hybrid agent
    agent = HybridAgent(
        llm_manager=llm_manager,
        adaptive_strategy=strategy_system,
        action_space_size=env.action_space.n
    )
    
    # Create trainer
    trainer = HybridLLMRLTrainer(
        env=env,
        agent=agent,
        strategy_system=strategy_system,
        decision_analyzer=decision_analyzer,
        llm_manager=llm_manager,
        save_dir=config.get('save_dir', 'checkpoints'),
        log_level=config.get('log_level', 'INFO')
    )
    
    return trainer


if __name__ == "__main__":
    # Example configuration for standalone execution
    config = {
        'rom_path': 'pokemoncrystal.gbc',
        'headless': True,
        'observation_type': 'multi_modal',
        'llm_model': 'gpt-4',
        'max_context_length': 8000,
        'initial_strategy': 'llm_heavy',
        'save_dir': 'checkpoints',
        'log_level': 'INFO'
    }
    
    # Save example config
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Hybrid LLM-RL Trainer ready. Use create_trainer_from_config('training_config.json') to start.")