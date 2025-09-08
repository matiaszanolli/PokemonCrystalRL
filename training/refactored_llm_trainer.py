#!/usr/bin/env python3
"""
Refactored LLM Trainer - Slim orchestration class using extracted components

This refactored trainer uses the extracted components for better separation of concerns:
- EmulationManager: PyBoy emulation lifecycle
- LLMDecisionEngine: AI decision making
- RewardCalculator: Multi-factor reward calculation
- StatisticsTracker: Training metrics and analytics

The core LLMTrainer class now focuses on orchestrating the training loop
and coordinating between components.
"""

import time
import signal
import sys
import os
import logging
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import extracted components
from .components import (
    EmulationManager, EmulationConfig,
    LLMDecisionEngine, LLMConfig, 
    RewardCalculator, RewardConfig,
    StatisticsTracker, TrainingSession
)

# Import optional monitoring components
try:
    from trainer.compat import WebMonitor
except ImportError:
    print("⚠️  WebMonitor not available - web monitoring disabled")
    WebMonitor = None

try:
    from core.dqn_agent import DQNAgent, HybridAgent
except ImportError:
    print("⚠️  DQN agents not available")
    DQNAgent = HybridAgent = None


@dataclass
class LLMTrainerConfig:
    """Configuration for the LLM Trainer."""
    # Core settings
    rom_path: str
    max_actions: int = 5000
    save_state_path: Optional[str] = None
    log_dir: str = "logs"
    show_progress: bool = True
    
    # Web monitoring
    enable_web: bool = True
    web_port: int = 8080
    web_host: str = "localhost"
    
    # DQN settings
    enable_dqn: bool = False
    dqn_model_path: Optional[str] = None
    dqn_learning_rate: float = 1e-4
    dqn_batch_size: int = 32
    dqn_memory_size: int = 50000
    dqn_training_frequency: int = 4
    dqn_save_frequency: int = 500


class RefactoredLLMTrainer:
    """Refactored LLM Trainer using extracted components for clean architecture."""
    
    def __init__(self, config: LLMTrainerConfig):
        self.config = config
        self.logger = logging.getLogger("RefactoredLLMTrainer")
        self.running = False
        
        # Initialize components
        self._setup_components()
        
        # DQN agent (optional)
        self.dqn_agent = None
        self.hybrid_agent = None
        if config.enable_dqn:
            self._setup_dqn()
        
        # Web monitor (optional)
        self.web_monitor = None
        if config.enable_web:
            self._setup_web_monitor()
        
        # Training control
        self.training_thread = None
        self._shutdown_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Refactored LLM Trainer initialized successfully")
    
    def _setup_components(self):
        """Initialize the extracted components."""
        # Emulation manager
        emulation_config = EmulationConfig(
            rom_path=self.config.rom_path,
            save_state_path=self.config.save_state_path,
            headless=True,
            debug_mode=False
        )
        self.emulation_manager = EmulationManager(emulation_config)
        
        # LLM decision engine
        llm_config = LLMConfig(
            model="smollm2:1.7b",
            base_url="http://localhost:11434",
            temperature=0.7,
            interval=20
        )
        self.llm_engine = LLMDecisionEngine(llm_config)
        
        # Reward calculator
        reward_config = RewardConfig()
        self.reward_calculator = RewardCalculator(reward_config)
        
        # Statistics tracker
        self.stats_tracker = StatisticsTracker(
            session_name=f"refactored_session_{int(time.time())}"
        )
        
        self.logger.info("All components initialized successfully")
    
    def _setup_dqn(self):
        """Setup DQN agent if enabled."""
        if not DQNAgent:
            self.logger.warning("DQN agent not available")
            return
        
        try:
            self.dqn_agent = DQNAgent(
                state_size=32,
                action_size=8,
                learning_rate=self.config.dqn_learning_rate,
                gamma=0.99,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                memory_size=self.config.dqn_memory_size,
                batch_size=self.config.dqn_batch_size,
                target_update=1000
            )
            
            # Load existing model if provided
            if self.config.dqn_model_path and os.path.exists(self.config.dqn_model_path):
                self.dqn_agent.load_model(self.config.dqn_model_path)
                self.logger.info(f"Loaded DQN model from {self.config.dqn_model_path}")
            
            # Create hybrid agent if available
            if HybridAgent and self.llm_engine.llm_agent:
                self.hybrid_agent = HybridAgent(
                    dqn_agent=self.dqn_agent,
                    llm_agent=self.llm_engine.llm_agent,
                    dqn_weight=0.2,
                    exploration_bonus=0.1
                )
                self.logger.info("Hybrid agent created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup DQN: {e}")
            self.dqn_agent = None
    
    def _setup_web_monitor(self):
        """Setup web monitor if enabled."""
        if not WebMonitor:
            self.logger.warning("WebMonitor not available")
            return
        
        try:
            self.web_monitor = WebMonitor(
                trainer=self,
                port=self.config.web_port, 
                host=self.config.web_host
            )
            
            if self.web_monitor.start():
                self.logger.info(f"Web monitor started at http://{self.config.web_host}:{self.config.web_port}")
            else:
                self.logger.error("Failed to start web monitor")
                self.web_monitor = None
                
        except Exception as e:
            self.logger.error(f"Failed to setup web monitor: {e}")
            self.web_monitor = None
    
    def start_training(self):
        """Start the training process."""
        if self.running:
            self.logger.warning("Training already running")
            return
        
        self.logger.info("Starting training...")
        
        # Initialize emulation
        if not self.emulation_manager.initialize():
            raise RuntimeError("Failed to initialize emulation")
        
        # Update web monitor with emulation instance
        if self.web_monitor and self.emulation_manager.get_instance():
            try:
                self.web_monitor.update_pyboy(self.emulation_manager.get_instance())
            except Exception as e:
                self.logger.warning(f"Failed to update web monitor: {e}")
        
        self.running = True
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        self.logger.info("Training started successfully")
    
    def _training_loop(self):
        """Main training loop using extracted components."""
        action_count = 0
        
        try:
            while self.running and action_count < self.config.max_actions:
                if self._shutdown_event.is_set():
                    break
                
                # Get current game state
                game_state = self._get_game_state()
                
                # Get screen data
                screen_data = self.emulation_manager.get_screen_array()
                
                # Decide on action
                if self.llm_engine.should_use_llm(action_count):
                    action, decision_meta = self.llm_engine.get_decision(
                        game_state=game_state,
                        screen_data=screen_data,
                        context={'action_count': action_count}
                    )
                else:
                    # Use fallback action
                    action, decision_meta = self.llm_engine.get_decision(
                        game_state=game_state,
                        screen_data=None,
                        context={'action_count': action_count}
                    )
                
                # Execute action
                if self.emulation_manager.execute_action(action, frames=24):
                    # Calculate reward
                    reward = self.reward_calculator.calculate_reward(
                        current_state=game_state,
                        action=action,
                        screen_analysis=self._analyze_screen(screen_data)
                    )
                    
                    # Record statistics
                    self.stats_tracker.record_action(
                        action=action,
                        source=decision_meta['source'],
                        game_state=game_state,
                        reward=reward,
                        metadata=decision_meta
                    )
                    
                    # Record LLM decision if used
                    if decision_meta['source'] == 'llm':
                        self.stats_tracker.record_llm_decision(
                            action=action,
                            response_time=decision_meta['response_time'],
                            confidence=decision_meta['confidence'],
                            success=decision_meta['success']
                        )
                    
                    # Train DQN if enabled
                    if self.dqn_agent and action_count % self.config.dqn_training_frequency == 0:
                        self._train_dqn_step(game_state, action, reward)
                    
                    # Save DQN model periodically
                    if self.dqn_agent and action_count % self.config.dqn_save_frequency == 0:
                        self._save_dqn_model()
                    
                    action_count += 1
                    
                    # Show progress
                    if self.config.show_progress and action_count % 100 == 0:
                        self._show_progress(action_count)
                
                else:
                    self.logger.warning("Failed to execute action, retrying...")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Training loop error: {e}")
            
        finally:
            self.running = False
            self.logger.info(f"Training completed after {action_count} actions")
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Extract current game state from emulation."""
        pyboy = self.emulation_manager.get_instance()
        if not pyboy:
            return {}
        
        try:
            # Import memory addresses if available
            try:
                from config.memory_addresses import MEMORY_ADDRESSES
                memory = pyboy.memory
                
                return {
                    'party_count': memory[MEMORY_ADDRESSES.get('party_count', 0)],
                    'player_map': memory[MEMORY_ADDRESSES.get('player_map', 0)],
                    'player_x': memory[MEMORY_ADDRESSES.get('player_x', 0)],
                    'player_y': memory[MEMORY_ADDRESSES.get('player_y', 0)],
                    'money': (
                        memory[MEMORY_ADDRESSES.get('money_low', 0)] + 
                        (memory[MEMORY_ADDRESSES.get('money_mid', 0)] << 8) +
                        (memory[MEMORY_ADDRESSES.get('money_high', 0)] << 16)
                    ),
                    'badges': bin(memory[MEMORY_ADDRESSES.get('badges', 0)]).count('1'),
                    'in_battle': memory[MEMORY_ADDRESSES.get('in_battle', 0)],
                    'player_level': memory[MEMORY_ADDRESSES.get('player_level', 0)]
                }
            except ImportError:
                # Fallback to basic state
                return {
                    'frame_count': pyboy.frame_count,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to get game state: {e}")
            return {}
    
    def _analyze_screen(self, screen_data) -> Dict[str, Any]:
        """Analyze screen data for state detection."""
        if screen_data is None:
            return {'state': 'unknown'}
        
        # Basic screen analysis - could be expanded
        try:
            import numpy as np
            mean_brightness = np.mean(screen_data)
            
            if mean_brightness > 200:
                return {'state': 'dialogue'}
            elif mean_brightness < 50:
                return {'state': 'loading'}
            else:
                return {'state': 'overworld'}
                
        except Exception:
            return {'state': 'unknown'}
    
    def _train_dqn_step(self, game_state: Dict[str, Any], action: int, reward: float):
        """Train DQN agent with current step."""
        if not self.dqn_agent:
            return
        
        try:
            # Create state representation for DQN
            # This would need to be expanded based on actual state encoding
            state_vector = [
                game_state.get('player_x', 0),
                game_state.get('player_y', 0),
                game_state.get('player_map', 0),
                game_state.get('badges', 0),
                # Add more state features...
            ]
            
            # Pad to required size
            while len(state_vector) < 32:
                state_vector.append(0)
            
            # Train the agent (would need proper implementation)
            # self.dqn_agent.train_step(state_vector, action, reward, next_state_vector)
            
        except Exception as e:
            self.logger.warning(f"DQN training step failed: {e}")
    
    def _save_dqn_model(self):
        """Save DQN model."""
        if not self.dqn_agent:
            return
        
        try:
            model_path = f"{self.config.log_dir}/dqn_model_{int(time.time())}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.dqn_agent.save_model(model_path)
            self.logger.info(f"DQN model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save DQN model: {e}")
    
    def _show_progress(self, action_count: int):
        """Display training progress."""
        stats = self.stats_tracker.get_current_stats()
        
        print(f"\n🎮 Training Progress - Action {action_count}/{self.config.max_actions}")
        print(f"   ⚡ Actions/sec: {stats['actions_per_second']:.1f}")
        print(f"   💰 Total reward: {stats['total_reward']:.2f}")
        print(f"   🎯 LLM calls: {stats['llm_calls']}")
        print(f"   🏆 Badges: {stats['current_badges']}")
        print(f"   📊 Success rate: {stats['success_rate']:.2%}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics for web monitor."""
        base_stats = self.stats_tracker.get_current_stats()
        
        # Add component statistics
        base_stats.update({
            'emulation_stats': {
                'is_alive': self.emulation_manager.is_alive(),
                'frame_count': self.emulation_manager.get_frame_count()
            },
            'llm_stats': self.llm_engine.get_statistics(),
            'reward_stats': self.reward_calculator.get_statistics()
        })
        
        return base_stats
    
    def stop_training(self):
        """Stop the training process gracefully."""
        self.logger.info("Stopping training...")
        self.running = False
        self._shutdown_event.set()
        
        # Wait for training thread
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        # End session and save statistics
        session = self.stats_tracker.end_session()
        
        # Save final statistics
        stats_path = f"{self.config.log_dir}/final_stats_{int(time.time())}.json"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        self.stats_tracker.save_statistics(stats_path)
        
        # Shutdown components
        self._shutdown_components()
        
        self.logger.info("Training stopped successfully")
        return session
    
    def _shutdown_components(self):
        """Shutdown all components."""
        # Stop web monitor
        if self.web_monitor:
            try:
                self.web_monitor.stop()
            except Exception as e:
                self.logger.error(f"Error stopping web monitor: {e}")
        
        # Shutdown LLM engine
        self.llm_engine.shutdown()
        
        # Shutdown emulation
        self.emulation_manager.shutdown()
        
        self.logger.info("All components shut down")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_training()
        sys.exit(0)
    
    # Backward compatibility methods for existing interfaces
    def initialize_pyboy(self):
        """Backward compatibility wrapper."""
        return self.emulation_manager.initialize()
    
    @property
    def pyboy(self):
        """Backward compatibility property."""
        return self.emulation_manager.get_instance()
    
    def graceful_shutdown(self):
        """Backward compatibility wrapper."""
        return self.stop_training()


# Factory function for creating trainer with old interface
def create_llm_trainer(rom_path, **kwargs) -> RefactoredLLMTrainer:
    """Factory function to create trainer with backward-compatible interface."""
    config = LLMTrainerConfig(
        rom_path=rom_path,
        max_actions=kwargs.get('max_actions', 5000),
        save_state_path=kwargs.get('save_state'),
        log_dir=kwargs.get('log_dir', 'logs'),
        show_progress=kwargs.get('show_progress', True),
        enable_web=kwargs.get('enable_web', True),
        web_port=kwargs.get('web_port', 8080),
        web_host=kwargs.get('web_host', 'localhost'),
        enable_dqn=kwargs.get('enable_dqn', False),
        dqn_model_path=kwargs.get('dqn_model_path'),
        dqn_learning_rate=kwargs.get('dqn_learning_rate', 1e-4),
        dqn_batch_size=kwargs.get('dqn_batch_size', 32),
        dqn_memory_size=kwargs.get('dqn_memory_size', 50000),
        dqn_training_frequency=kwargs.get('dqn_training_frequency', 4),
        dqn_save_frequency=kwargs.get('dqn_save_frequency', 500)
    )
    
    return RefactoredLLMTrainer(config)


if __name__ == "__main__":
    # Example usage
    config = LLMTrainerConfig(
        rom_path="pokemon_crystal.gb",
        max_actions=1000,
        enable_web=True,
        enable_dqn=False
    )
    
    trainer = RefactoredLLMTrainer(config)
    
    try:
        trainer.start_training()
        
        # Keep running until completion or interrupt
        while trainer.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        trainer.stop_training()