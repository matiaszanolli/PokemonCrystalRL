#!/usr/bin/env python3
"""
Hybrid LLM-RL Trainer

This module provides a comprehensive training system that integrates:
- Hybrid LLM-RL decision making
- Temporal memory for experience tracking
- Curriculum learning progression
- Real-time performance monitoring
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from agents.hybrid_llm_rl_agent import HybridLLMRLAgent, DecisionMode, DecisionMetrics
from agents.llm_agent import LLMAgent
from agents.dqn_agent import DQNAgent
from core.temporal_memory import TemporalMemoryBuffer
from core.experience_memory import ExperienceMemory
from training.curriculum_learning import CurriculumManager
from core.game_intelligence import GameIntelligence
from core.save_state_library import SaveStateLibrary
from environments.enhanced_pyboy_env import EnhancedPyBoyPokemonCrystalEnv
from rewards.calculator import PokemonRewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for hybrid training."""
    # Training parameters
    max_episodes: int = 100
    max_actions_per_episode: int = 1000

    # Hybrid agent configuration
    llm_weight: float = 0.7
    rl_weight: float = 0.3
    exploration_rate: float = 0.1

    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_config: Optional[str] = None

    # RL training parameters
    rl_learning_rate: float = 1e-4
    rl_batch_size: int = 32
    rl_memory_size: int = 50000
    rl_target_update_freq: int = 1000
    rl_training_freq: int = 4

    # Temporal memory parameters
    temporal_buffer_size: int = 100000
    temporal_episode_length: int = 500

    # Adaptive parameters
    weight_adaptation_freq: int = 10  # Episodes between weight adaptations
    performance_window: int = 20      # Episodes for performance calculation

    # Monitoring
    log_interval: int = 10           # Actions between detailed logs
    save_interval: int = 100         # Episodes between model saves

    # Web monitoring
    enable_web: bool = True
    web_port: int = 8080


class HybridLLMRLTrainer:
    """
    Advanced trainer that combines LLM strategic reasoning with RL optimization
    using temporal memory and curriculum learning.
    """

    def __init__(
        self,
        rom_path: str,
        config: TrainingConfig,
        save_state_library: Optional[SaveStateLibrary] = None,
        progress_callback: Optional[Callable] = None,
        websocket_handler = None
    ):
        self.rom_path = rom_path
        self.config = config
        self.save_state_library = save_state_library
        self.progress_callback = progress_callback
        self.websocket_handler = websocket_handler

        # Training state
        self.current_episode = 0
        self.total_actions = 0
        self.current_action = 0
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None

        # Initialize core components
        self._initialize_components()

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_modes: List[Dict[str, int]] = []
        self.performance_history: List[Dict[str, Any]] = []

        logger.info("🚀 Hybrid LLM-RL Trainer initialized")
        logger.info(f"   ROM: {rom_path}")
        logger.info(f"   Config: {config}")

    def _initialize_components(self):
        """Initialize all training components."""
        # Initialize environment
        self.env = EnhancedPyBoyPokemonCrystalEnv(rom_path=self.rom_path, headless=True)

        # Initialize LLM agent
        self.llm_agent = LLMAgent(
            model_name="smollm2:1.7b",
            base_url="http://localhost:11434"
        )

        # Initialize RL agent
        self.rl_agent = DQNAgent(
            state_size=32,  # Feature vector size
            action_size=8,  # PyBoy action space
            learning_rate=self.config.rl_learning_rate,
            gamma=0.99,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=self.config.rl_memory_size,
            batch_size=self.config.rl_batch_size,
            target_update=self.config.rl_target_update_freq
        )

        # Initialize experience memory and temporal memory
        self.experience_memory = ExperienceMemory()
        self.temporal_memory = TemporalMemoryBuffer(
            base_memory=self.experience_memory,
            buffer_size=self.config.temporal_buffer_size,
            episode_buffer_size=getattr(self.config, 'temporal_episode_length', 1000)
        )

        # Initialize game intelligence
        self.game_intelligence = GameIntelligence()

        # Initialize curriculum manager if enabled
        self.curriculum_manager = None
        if self.config.enable_curriculum and self.save_state_library:
            self.curriculum_manager = CurriculumManager(
                library=self.save_state_library,
                curriculum_config=self.config.curriculum_config
            )

        # Initialize hybrid agent
        hybrid_config = {
            'llm_weight': self.config.llm_weight,
            'rl_weight': self.config.rl_weight,
            'exploration_rate': self.config.exploration_rate
        }

        self.hybrid_agent = HybridLLMRLAgent(
            llm_agent=self.llm_agent,
            rl_agent=self.rl_agent,
            temporal_memory=self.temporal_memory,
            curriculum_manager=self.curriculum_manager,
            game_intelligence=self.game_intelligence,
            config=hybrid_config
        )

        # Initialize reward calculator
        self.reward_calculator = PokemonRewardCalculator()

        logger.info("✅ All components initialized successfully")

    def start_training(self) -> threading.Thread:
        """Start training in a separate thread."""
        if self.is_training:
            logger.warning("Training already in progress")
            return self.training_thread

        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        logger.info("🎯 Training started in background thread")
        return self.training_thread

    def stop_training(self):
        """Stop training gracefully."""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        logger.info("⏹️ Training stopped")

    def _training_loop(self):
        """Main training loop."""
        try:
            logger.info(f"🎓 Starting hybrid training for {self.config.max_episodes} episodes")

            for episode in range(self.config.max_episodes):
                if not self.is_training:
                    break

                self.current_episode = episode
                episode_start_time = time.time()

                # Run single episode
                episode_stats = self._run_episode()

                # Update curriculum if enabled
                if self.curriculum_manager:
                    success = episode_stats['reward'] > 0
                    advanced = self.curriculum_manager.record_episode_result(
                        success=success,
                        reward=episode_stats['reward']
                    )
                    if advanced:
                        logger.info(f"🎓 Curriculum advanced to: {self.curriculum_manager.get_current_stage().value}")

                # Update training metrics
                self._update_training_metrics(episode_stats)

                # Adapt agent weights periodically
                if episode % self.config.weight_adaptation_freq == 0:
                    self.hybrid_agent.adapt_weights()
                    logger.info(f"🔄 Adapted weights: {self.hybrid_agent.get_decision_summary()['current_weights']}")

                # Log progress
                if episode % 5 == 0:
                    self._log_progress(episode, episode_stats, time.time() - episode_start_time)

                # Save model periodically
                if episode % self.config.save_interval == 0:
                    self._save_models(episode)

                # Callback for external monitoring
                if self.progress_callback:
                    self.progress_callback(episode, episode_stats, self.get_training_summary())

            logger.info("🏁 Training completed successfully")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            self.is_training = False

    def _run_episode(self) -> Dict[str, Any]:
        """Run a single training episode."""
        # Select save state for this episode
        save_state_path = None
        if self.curriculum_manager:
            save_state = self.curriculum_manager.select_save_state()
            if save_state:
                save_state_path = save_state.file_path
                logger.debug(f"📁 Using save state: {save_state.name}")

        # Reset environment
        initial_state = self.env.reset(save_state_path)

        # Initialize episode tracking
        episode_reward = 0.0
        episode_actions = 0
        episode_modes = {mode.value: 0 for mode in DecisionMode}
        decision_times = []

        # Start new episode in temporal memory
        self.temporal_memory.start_episode()

        for action_num in range(self.config.max_actions_per_episode):
            if not self.is_training:
                break

            action_start_time = time.time()

            # Get current game state (read directly from environment's internal method)
            game_state = self.env._read_game_state()

            # Create action context
            action_context = {
                'episode': self.current_episode,
                'action_num': action_num,
                'llm_context': self._build_llm_context(game_state),
                'curriculum_stage': self.curriculum_manager.get_current_stage().value if self.curriculum_manager else None
            }

            # Get action from hybrid agent
            action_space = list(range(8))  # PyBoy action space
            action, decision_info = self.hybrid_agent.decide_action(
                game_state, action_space, action_context
            )

            # Execute action in environment (handle gymnasium API)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                # Gymnasium API: (obs, reward, terminated, truncated, info)
                next_state, env_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Legacy API: (obs, reward, done, info)
                next_state, env_reward, done, info = step_result

            # Calculate comprehensive reward
            reward, reward_breakdown = self.reward_calculator.calculate_reward(
                next_state, game_state  # current_state, previous_state
            )

            # Update hybrid agent with reward
            self.hybrid_agent.update_with_reward(action, reward, next_state)

            # Track episode metrics
            episode_reward += reward
            episode_actions += 1
            episode_modes[decision_info['mode']] += 1
            decision_times.append(decision_info['decision_time'])
            self.total_actions += 1
            self.current_action = action_num

            # Broadcast real-time updates
            self._broadcast_training_update(action, reward, decision_info, next_state)
            self._broadcast_decision_update(action, decision_info, reward)

            # Log detailed information periodically
            if action_num % self.config.log_interval == 0:
                self._log_action_details(action_num, action, reward, decision_info, game_state)

            # Check termination conditions
            if done or reward < -20:  # Terminal states
                break

        # End episode in temporal memory
        self.temporal_memory.end_episode(episode_reward)

        return {
            'episode': self.current_episode,
            'reward': episode_reward,
            'actions': episode_actions,
            'avg_decision_time': sum(decision_times) / len(decision_times) if decision_times else 0,
            'mode_distribution': episode_modes,
            'final_state': game_state
        }

    def _build_llm_context(self, game_state: Dict[str, Any]) -> str:
        """Build rich context for LLM decision making."""
        context_parts = []

        # Basic game state
        context_parts.append(f"Player at ({game_state.get('player_x', 0)}, {game_state.get('player_y', 0)})")
        context_parts.append(f"HP: {game_state.get('player_hp', 0)}")
        context_parts.append(f"Level: {game_state.get('player_level', 1)}")
        context_parts.append(f"Badges: {game_state.get('badges', 0)}")

        # Screen state
        screen_state = game_state.get('screen_state', 'unknown')
        context_parts.append(f"Screen: {screen_state}")

        # Curriculum context
        if self.curriculum_manager:
            stage = self.curriculum_manager.get_current_stage().value
            context_parts.append(f"Curriculum: {stage}")

        # Recent performance
        recent_summary = self.hybrid_agent.get_decision_summary()
        last_mode = recent_summary.get('last_mode', 'unknown')
        context_parts.append(f"Last decision mode: {last_mode}")

        return " | ".join(context_parts)

    def _update_training_metrics(self, episode_stats: Dict[str, Any]):
        """Update training metrics and history."""
        self.episode_rewards.append(episode_stats['reward'])
        self.episode_lengths.append(episode_stats['actions'])
        self.episode_modes.append(episode_stats['mode_distribution'])

        # Keep limited history
        if len(self.episode_rewards) > 1000:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.episode_modes.pop(0)

        # Update performance history
        performance_entry = {
            'episode': episode_stats['episode'],
            'reward': episode_stats['reward'],
            'actions': episode_stats['actions'],
            'hybrid_summary': self.hybrid_agent.get_decision_summary(),
            'curriculum_status': self.curriculum_manager.get_curriculum_status() if self.curriculum_manager else None
        }
        self.performance_history.append(performance_entry)

        # Keep limited performance history
        if len(self.performance_history) > 500:
            self.performance_history.pop(0)

    def _log_progress(self, episode: int, episode_stats: Dict[str, Any], episode_time: float):
        """Log training progress."""
        reward = episode_stats['reward']
        actions = episode_stats['actions']

        # Calculate recent averages
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0

        # Get hybrid agent summary
        hybrid_summary = self.hybrid_agent.get_decision_summary()

        logger.info(f"Episode {episode:3d} | "
                   f"Reward: {reward:6.1f} | "
                   f"Actions: {actions:3d} | "
                   f"Time: {episode_time:.1f}s | "
                   f"Avg: {avg_reward:.1f}")

        logger.info(f"         | "
                   f"Mode dist: LLM={hybrid_summary['mode_distribution']['llm']:.2f} "
                   f"RL={hybrid_summary['mode_distribution']['rl']:.2f} "
                   f"Hybrid={hybrid_summary['mode_distribution']['hybrid']:.2f}")

        # Curriculum status
        if self.curriculum_manager:
            status = self.curriculum_manager.get_curriculum_status()
            logger.info(f"         | "
                       f"Curriculum: {status['current_stage']} "
                       f"({status['level_progress']['episodes']}/{status['level_progress']['max_episodes']})")

    def _log_action_details(
        self,
        action_num: int,
        action: int,
        reward: float,
        decision_info: Dict[str, Any],
        game_state: Dict[str, Any]
    ):
        """Log detailed action information."""
        mode = decision_info['mode']
        confidence = decision_info['confidence']

        logger.debug(f"Action {action_num:3d}: {action} | "
                    f"Mode: {mode} | "
                    f"Confidence: {confidence:.2f} | "
                    f"Reward: {reward:5.1f} | "
                    f"Pos: ({game_state.get('player_x', 0)}, {game_state.get('player_y', 0)})")

    def _save_models(self, episode: int):
        """Save RL model and training state."""
        try:
            model_path = f"models/hybrid_rl_model_episode_{episode}.pth"
            self.rl_agent.save_model(model_path)
            logger.info(f"💾 Saved RL model: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        # Calculate statistics
        total_episodes = len(self.episode_rewards)
        avg_reward = sum(self.episode_rewards) / max(1, total_episodes)
        avg_length = sum(self.episode_lengths) / max(1, total_episodes)

        # Recent performance (last 20 episodes)
        recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        recent_avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))

        # Hybrid agent summary
        hybrid_summary = self.hybrid_agent.get_decision_summary()

        # Curriculum summary
        curriculum_summary = None
        if self.curriculum_manager:
            curriculum_summary = self.curriculum_manager.get_curriculum_status()

        return {
            'training_status': 'active' if self.is_training else 'stopped',
            'current_episode': self.current_episode,
            'total_actions': self.total_actions,
            'episodes_completed': total_episodes,
            'performance': {
                'avg_reward': avg_reward,
                'recent_avg_reward': recent_avg_reward,
                'avg_episode_length': avg_length,
                'total_reward': sum(self.episode_rewards)
            },
            'hybrid_agent': hybrid_summary,
            'curriculum': curriculum_summary,
            'config': {
                'max_episodes': self.config.max_episodes,
                'max_actions_per_episode': self.config.max_actions_per_episode,
                'llm_weight': self.config.llm_weight,
                'rl_weight': self.config.rl_weight
            },
            'temporal_memory': {
                'buffer_size': len(self.temporal_memory.experience_buffer),
                'episodes_completed': len(self.temporal_memory.episodes),
                'total_experiences': len(self.experience_memory.experiences)
            }
        }

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for web monitoring."""
        if not self.performance_history:
            return {'status': 'no_data'}

        latest = self.performance_history[-1]

        return {
            'status': 'active' if self.is_training else 'idle',
            'current_episode': self.current_episode,
            'latest_reward': latest['reward'],
            'latest_actions': latest['actions'],
            'hybrid_metrics': latest['hybrid_summary'],
            'curriculum_status': latest['curriculum_status'],
            'progress': {
                'episode_progress': self.current_episode / self.config.max_episodes,
                'action_progress': self.total_actions / (self.config.max_episodes * self.config.max_actions_per_episode)
            }
        }

    def _broadcast_training_update(self, action: int, reward: float, decision_info: Dict, game_state: Dict):
        """Broadcast real-time training update via WebSocket."""
        logger.info(f"🔗 _broadcast_training_update called: websocket_handler={self.websocket_handler is not None}")
        if not self.websocket_handler:
            logger.warning("No websocket handler available for broadcasting")
            return

        try:
            # Calculate actions per second
            actions_per_second = 0.0
            if hasattr(self, '_last_action_time'):
                current_time = time.time()
                time_diff = current_time - self._last_action_time
                if time_diff > 0:
                    actions_per_second = 1.0 / time_diff
            self._last_action_time = time.time()

            # Get hybrid agent metrics
            hybrid_summary = self.hybrid_agent.get_decision_summary()

            # Prepare update data
            update_data = {
                'episode': self.current_episode,
                'max_episodes': self.config.max_episodes,
                'action': self.current_action,
                'total_actions': self.total_actions,
                'actions_per_second': actions_per_second,
                'is_training': self.is_training,
                'current_mode': decision_info.get('mode', 'hybrid'),
                'latest_reward': reward,
                'metrics': {
                    'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0,
                    'reward_change': reward,
                    'llm_success_rate': hybrid_summary.get('success_rates', {}).get('llm', 0),
                    'rl_success_rate': hybrid_summary.get('success_rates', {}).get('rl', 0),
                    'exploration_rate': hybrid_summary.get('mode_distribution', {}).get('exploration', 0),
                    'llm_decisions': hybrid_summary.get('mode_distribution', {}).get('llm', 0) * hybrid_summary.get('total_decisions', 1),
                    'rl_decisions': hybrid_summary.get('mode_distribution', {}).get('rl', 0) * hybrid_summary.get('total_decisions', 1),
                    'hybrid_decisions': hybrid_summary.get('mode_distribution', {}).get('hybrid', 0) * hybrid_summary.get('total_decisions', 1),
                    'exploration_decisions': hybrid_summary.get('mode_distribution', {}).get('exploration', 0) * hybrid_summary.get('total_decisions', 1),
                    'total_decisions': hybrid_summary.get('total_decisions', 0),
                    'llm_weight': hybrid_summary.get('current_weights', {}).get('llm_weight', 0.7),
                    'rl_weight': hybrid_summary.get('current_weights', {}).get('rl_weight', 0.3),
                    'mode_switches': hybrid_summary.get('mode_switches', 0),
                    'avg_confidence': hybrid_summary.get('avg_confidences', {}).get(decision_info.get('mode', 'hybrid'), 0)
                },
                'temporal_memory': {
                    'buffer_size': len(self.temporal_memory.experience_buffer) if self.temporal_memory else 0,
                    'episodes_stored': len(self.temporal_memory.episodes) if self.temporal_memory else 0,
                    'max_buffer_size': self.config.temporal_buffer_size,
                    'avg_novelty': 0.5  # Placeholder - could be calculated from recent experiences
                },
                'curriculum': {
                    'current_stage': self.curriculum_manager.get_current_stage().value if self.curriculum_manager else 'tutorial',
                    'progress': min(1.0, self.current_episode / 10),  # Simplified progress calculation
                    'episodes': self.current_episode,
                    'max_episodes': 10,  # Simplified - could get from curriculum config
                    'success_rate': 0.0  # Placeholder - could calculate from recent episodes
                }
            }

            # Use simple sync broadcast to avoid asyncio threading issues
            if hasattr(self.websocket_handler, 'broadcast_sync'):
                logger.info(f"📡 Broadcasting training update: episode={update_data.get('episode', 'unknown')}")
                self.websocket_handler.broadcast_sync(update_data)
            elif hasattr(self.websocket_handler, 'broadcast_hybrid_update'):
                # Fallback: use thread-safe async call
                import threading
                def async_broadcast():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.websocket_handler.broadcast_hybrid_update(update_data))
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Async broadcast error: {e}")

                # Run in separate thread to avoid blocking
                thread = threading.Thread(target=async_broadcast, daemon=True)
                thread.start()

        except Exception as e:
            logger.warning(f"Failed to broadcast training update: {e}")

    def _broadcast_decision_update(self, action: int, decision_info: Dict, reward: float):
        """Broadcast LLM/RL decision update via WebSocket."""
        if not self.websocket_handler:
            return

        try:
            # Map action to human-readable format
            action_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'A', 5: 'B', 6: 'START', 7: 'SELECT'}
            action_name = action_map.get(action, f'Action {action}')

            decision_data = {
                'mode': decision_info.get('mode', 'hybrid'),
                'action': action,
                'action_name': action_name,
                'confidence': decision_info.get('confidence', 0.0),
                'reasoning': f"Mode: {decision_info.get('mode', 'hybrid')}, Confidence: {decision_info.get('confidence', 0.0):.2f}",
                'reward': reward,
                'timestamp': time.time()
            }

            # Async broadcast
            if hasattr(self.websocket_handler, 'broadcast_decision'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.websocket_handler.broadcast_decision(decision_data))
                loop.close()

        except Exception as e:
            logger.warning(f"Failed to broadcast decision update: {e}")

    def _broadcast_episode_completed(self, episode: int, reward: float, actions: int, modes: Dict):
        """Broadcast episode completion via WebSocket."""
        if not self.websocket_handler:
            return

        try:
            # Log message
            message = f"Episode {episode} completed: Reward={reward:.1f}, Actions={actions}, Modes={modes}"

            # Async broadcast
            if hasattr(self.websocket_handler, 'broadcast_log_entry'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.websocket_handler.broadcast_log_entry('success', message))
                loop.close()

        except Exception as e:
            logger.warning(f"Failed to broadcast episode completion: {e}")