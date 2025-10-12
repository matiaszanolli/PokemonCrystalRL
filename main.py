#!/usr/bin/env python3
"""
Pokemon Crystal RL Training Entry Point

Main script for running the Pokemon Crystal LLM-enhanced training system.
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Optional

from pyboy import PyBoy
# WebMonitor handled internally by UnifiedPokemonTrainer
from core.game_intelligence import GameIntelligence
from core.experience_memory import ExperienceMemory
from agents.dqn_agent import DQNAgent
from agents.hybrid_agent import HybridAgent
from training.hybrid_llm_rl_trainer import HybridLLMRLTrainer, TrainingConfig
from core.strategic_context_builder import StrategicContextBuilder

from agents.llm_agent import LLMAgent
from training.unified_pokemon_trainer import create_llm_trainer as LLMTrainer
from rewards.calculator import PokemonRewardCalculator

# Curriculum Learning
from core.save_state_library import SaveStateLibrary
from training.curriculum_learning import CurriculumManager

from utils.memory_reader import build_observation
from utils.screen_analyzer import analyze_screen_state
from utils.action_parser import (
    get_context_specific_action,
    is_action_allowed,
    get_allowed_action
)
from utils.reward_helpers import get_reward_summary

from config.constants import (
    TRAINING_PARAMS,
    REWARD_VALUES,
    SCREEN_STATES,
)

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments_from_dict(config: Optional[Dict] = None) -> argparse.Namespace:
    """
    Parse arguments from dictionary for programmatic access (testing).

    Args:
        config: Dictionary of configuration values. If None, parses from command line.

    Returns:
        Parsed arguments namespace

    Example:
        >>> args = parse_arguments_from_dict({
        ...     'rom_path': 'test.gbc',
        ...     'max_actions': 10,
        ...     'headless': True
        ... })
    """
    parser = _create_argument_parser()

    if config is None:
        return parser.parse_args()

    # Convert config dict to argument list
    arg_list = []

    # Add positional argument
    if 'rom_path' in config:
        arg_list.append(config['rom_path'])

    # Add optional arguments
    for key, value in config.items():
        if key == 'rom_path':
            continue  # Already added

        # Convert underscore to hyphen for CLI args
        arg_name = f"--{key.replace('_', '-')}"

        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                arg_list.append(arg_name)
        else:
            arg_list.extend([arg_name, str(value)])

    return parser.parse_args(arg_list)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser (extracted for reusability)."""
    parser = argparse.ArgumentParser(description="Pokemon Crystal RL Training")
    
    # Required arguments
    parser.add_argument("rom_path", help="Path to Pokemon Crystal ROM file")
    
    # Optional arguments
    parser.add_argument("--save-state", help="Path to save state file to load")
    parser.add_argument("--max-actions", type=int, default=5000, help="Maximum number of actions to take")
    parser.add_argument("--llm-model", default="smollm2:1.7b", help="LLM model name")
    parser.add_argument("--llm-base-url", default="http://localhost:11434", help="LLM API base URL")
    parser.add_argument("--llm-interval", type=int, default=TRAINING_PARAMS['LLM_INTERVAL'],
                       help="Actions between LLM decisions")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                       help="LLM temperature for decision making")
    
    # Web monitoring options
    parser.add_argument("--enable-web", action="store_true", help="Enable web monitoring")
    parser.add_argument("--web-port", type=int, default=8080, help="Web monitor port")
    parser.add_argument("--web-host", default="localhost", help="Web monitor host")
    
    # DQN options
    parser.add_argument("--enable-dqn", action="store_true", help="Enable DQN hybrid agent")
    parser.add_argument("--dqn-model", help="Path to DQN model file")
    parser.add_argument("--dqn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-memory-size", type=int, default=50000)
    parser.add_argument("--dqn-training-freq", type=int, default=4)
    parser.add_argument("--dqn-save-freq", type=int, default=500)

    # Advanced Hybrid LLM-RL Training options
    parser.add_argument("--enable-hybrid-llm-rl", action="store_true",
                       help="Enable advanced hybrid LLM-RL training with temporal memory")
    parser.add_argument("--llm-weight", type=float, default=0.7,
                       help="Weight for LLM decisions in hybrid mode (0.0-1.0)")
    parser.add_argument("--rl-weight", type=float, default=0.3,
                       help="Weight for RL decisions in hybrid mode (0.0-1.0)")
    parser.add_argument("--exploration-rate", type=float, default=0.1,
                       help="Exploration rate for hybrid agent")
    parser.add_argument("--temporal-buffer-size", type=int, default=100000,
                       help="Temporal memory buffer size")
    parser.add_argument("--max-episodes", type=int, default=100,
                       help="Maximum episodes for hybrid training")
    
    # Curriculum Learning options
    parser.add_argument("--enable-curriculum", action="store_true",
                       help="Enable curriculum learning with save state library")
    parser.add_argument("--library-path", default="save_states",
                       help="Path to save state library")
    parser.add_argument("--curriculum-config",
                       help="Path to curriculum configuration file")
    parser.add_argument("--curriculum-episodes", type=int, default=5,
                       help="Number of episodes per curriculum level")

    # Logging options
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output")

    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments (CLI entry point)."""
    parser = _create_argument_parser()
    return parser.parse_args()

def initialize_training_systems(args: argparse.Namespace) -> Dict:
    """Initialize all training subsystems."""
    # Initialize core components
    llm_agent = LLMAgent(args.llm_model, args.llm_base_url)
    reward_calculator = PokemonRewardCalculator()
    game_intelligence = GameIntelligence()
    experience_memory = ExperienceMemory()
    context_builder = StrategicContextBuilder()
    
    # Web monitor is handled internally by the trainer
    
    # Initialize DQN components if enabled
    dqn_agent = None
    hybrid_agent = None
    if args.enable_dqn:
        dqn_agent = DQNAgent(
            state_size=32,
            action_size=8,
            learning_rate=args.dqn_learning_rate,
            gamma=0.99,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=args.dqn_memory_size,
            batch_size=args.dqn_batch_size,
            target_update=1000
        )
        
        # Load existing DQN model if provided
        if args.dqn_model and os.path.exists(args.dqn_model):
            dqn_agent.load_model(args.dqn_model)
            logger.info(f"Loaded DQN model from {args.dqn_model}")
        
        # Create hybrid agent
        hybrid_agent = HybridAgent(
            dqn_agent=dqn_agent,
            llm_agent=llm_agent,
            dqn_weight=0.2,  # Start with low DQN influence
            exploration_bonus=0.1
        )
    
    # Create trainer instance
    trainer = LLMTrainer(
        rom_path=args.rom_path,
        max_actions=args.max_actions,
        save_state=args.save_state,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_interval=args.llm_interval,
        llm_temperature=args.llm_temperature,
        enable_web=args.enable_web,
        web_port=args.web_port,
        web_host=args.web_host,
        enable_dqn=args.enable_dqn,
        dqn_model_path=args.dqn_model,
        dqn_learning_rate=args.dqn_learning_rate,
        dqn_batch_size=args.dqn_batch_size,
        dqn_memory_size=args.dqn_memory_size,
        dqn_training_frequency=args.dqn_training_freq,
        dqn_save_frequency=args.dqn_save_freq,
        log_dir=args.log_dir,
        show_progress=not args.quiet
    )
    
    return {
        'trainer': trainer,
        'llm_agent': llm_agent,
        'reward_calculator': reward_calculator,
        'game_intelligence': game_intelligence,
        'experience_memory': experience_memory,
        'context_builder': context_builder,
        'web_monitor': None,  # Handled internally by trainer
        'dqn_agent': dqn_agent,
        'hybrid_agent': hybrid_agent
    }

def graceful_shutdown(systems: Dict, signum: Optional[int] = None, frame: Optional[object] = None) -> None:
    """Handle graceful shutdown of all systems."""
    logger.info("Initiating graceful shutdown...")
    
    # Stop web monitor
    if systems.get('web_monitor'):
        try:
            systems['web_monitor'].stop()
            logger.info("Web monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping web monitor: {e}")
    
    # Stop trainer
    if systems.get('trainer'):
        try:
            systems['trainer'].stop_training()
            logger.info("Trainer stopped")
        except Exception as e:
            logger.error(f"Error stopping trainer: {e}")
    
    logger.info("Shutdown complete")
    if signum is not None:
        sys.exit(0)

def run_curriculum_training(args):
    """Run curriculum learning training mode."""
    from examples.run_curriculum_training import CurriculumTrainer

    logger.info("🎓 Starting Curriculum Learning Training")

    # Verify library exists
    if not os.path.exists(args.library_path):
        logger.error(f"❌ Save state library not found: {args.library_path}")
        logger.info("   Create save states using: python3 scripts/manage_save_states.py add ...")
        return 1

    try:
        # Initialize curriculum trainer
        trainer = CurriculumTrainer(
            rom_path=args.rom_path,
            library_path=args.library_path,
            curriculum_config=args.curriculum_config,
            llm_model=args.llm_model,
            enable_web=args.enable_web,
            web_port=args.web_port
        )

        # Run curriculum training
        trainer.run_training(
            num_episodes=args.curriculum_episodes,
            max_actions_per_episode=args.max_actions
        )

        return 0

    except Exception as e:
        logger.error(f"❌ Curriculum training failed: {e}")
        raise


def run_hybrid_llm_rl_training(args):
    """Run advanced hybrid LLM-RL training mode."""
    from core.save_state_library import SaveStateLibrary

    logger.info("🤖 Starting Advanced Hybrid LLM-RL Training")
    logger.info(f"   LLM Weight: {args.llm_weight}, RL Weight: {args.rl_weight}")
    logger.info(f"   Episodes: {args.max_episodes}, Actions per episode: {args.max_actions}")

    try:
        # Load save state library if curriculum is enabled
        save_state_library = None
        if args.enable_curriculum:
            if not os.path.exists(args.library_path):
                logger.warning(f"Save state library not found: {args.library_path}")
                logger.info("   Creating empty library. Consider adding save states.")
                os.makedirs(args.library_path, exist_ok=True)

            save_state_library = SaveStateLibrary(args.library_path)
            states = save_state_library.list_save_states()
            logger.info(f"📁 Loaded save state library with {len(states)} states")

        # Create training configuration
        config = TrainingConfig(
            max_episodes=args.max_episodes,
            max_actions_per_episode=args.max_actions,
            llm_weight=args.llm_weight,
            rl_weight=args.rl_weight,
            exploration_rate=args.exploration_rate,
            enable_curriculum=args.enable_curriculum,
            curriculum_config=args.curriculum_config,
            rl_learning_rate=args.dqn_learning_rate,
            rl_batch_size=args.dqn_batch_size,
            rl_memory_size=args.dqn_memory_size,
            temporal_buffer_size=args.temporal_buffer_size,
            enable_web=args.enable_web,
            web_port=args.web_port
        )

        # Initialize hybrid trainer
        trainer = HybridLLMRLTrainer(
            rom_path=args.rom_path,
            config=config,
            save_state_library=save_state_library
        )

        # Enable WebSocket integration if web monitoring is enabled
        if args.enable_web:
            try:
                from web_dashboard.server import UnifiedWebServer
                from web_dashboard.websocket_handler import WebSocketHandler

                # Create WebSocket handler
                websocket_handler = WebSocketHandler(trainer=trainer)
                trainer.websocket_handler = websocket_handler

                # Start web server with WebSocket support
                web_server = UnifiedWebServer(
                    trainer=trainer,
                    websocket_handler=websocket_handler,
                    host="localhost",
                    http_port=args.web_port
                )
                web_server.start()

                logger.info(f"🌐 Hybrid dashboard available at http://localhost:{args.web_port}/hybrid")
                logger.info("   Real-time LLM-RL decision monitoring with live metrics")

            except Exception as e:
                logger.warning(f"Failed to start web monitoring: {e}")
                logger.info("Training will continue without web dashboard")

        logger.info("🚀 Hybrid trainer initialized successfully")

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received")
            trainer.stop_training()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start training
        training_thread = trainer.start_training()

        # Monitor progress
        while trainer.is_training:
            try:
                time.sleep(5)
                summary = trainer.get_real_time_stats()

                if summary.get('status') == 'active':
                    episode = summary.get('current_episode', 0)
                    reward = summary.get('latest_reward', 0)
                    hybrid_metrics = summary.get('hybrid_metrics', {})

                    mode_dist = hybrid_metrics.get('mode_distribution', {})
                    logger.info(f"Episode {episode}: Reward={reward:.1f}, "
                               f"LLM={mode_dist.get('llm', 0):.2f}, "
                               f"RL={mode_dist.get('rl', 0):.2f}")

            except KeyboardInterrupt:
                logger.info("🛑 Stopping training...")
                break
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

        # Wait for completion
        training_thread.join()

        # Print final results
        final_summary = trainer.get_training_summary()
        logger.info("🏁 Hybrid Training Completed!")
        logger.info(f"   Episodes: {final_summary['episodes_completed']}")
        logger.info(f"   Total actions: {final_summary['total_actions']}")
        logger.info(f"   Avg reward: {final_summary['performance']['avg_reward']:.2f}")

        hybrid_summary = final_summary['hybrid_agent']
        logger.info(f"   Mode distribution: {hybrid_summary['mode_distribution']}")
        logger.info(f"   Success rates: {hybrid_summary['success_rates']}")

        return 0

    except Exception as e:
        logger.error(f"❌ Hybrid LLM-RL training failed: {e}")
        raise


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_dir)

    # Check for advanced hybrid LLM-RL training mode
    if args.enable_hybrid_llm_rl:
        return run_hybrid_llm_rl_training(args)

    # Check for curriculum learning mode
    if args.enable_curriculum:
        return run_curriculum_training(args)

    logger.info("Starting Pokemon Crystal RL training...")

    # Initialize all systems
    systems = initialize_training_systems(args)

    # Setup graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: graceful_shutdown(systems, s, f))
    signal.signal(signal.SIGTERM, lambda s, f: graceful_shutdown(systems, s, f))

    try:
        # Web monitor is started internally by the trainer
        # Run training
        systems['trainer'].start_training()

        # Wait for training to complete
        trainer = systems['trainer']
        if hasattr(trainer, 'training_thread') and trainer.training_thread:
            trainer.training_thread.join()

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
        
    finally:
        # Always ensure clean shutdown
        graceful_shutdown(systems)

if __name__ == "__main__":
    main()
