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
from datetime import datetime
from typing import Dict, Optional

from pyboy import PyBoy
from core.web_monitor import WebMonitor
from core.game_intelligence import GameIntelligence
from core.experience_memory import ExperienceMemory
from agents.dqn_agent import DQNAgent
from agents.hybrid_agent import HybridAgent
from core.strategic_context_builder import StrategicContextBuilder

from agents.llm_agent import LLMAgent
from training.llm_pokemon_trainer import LLMTrainer
from rewards.calculator import PokemonRewardCalculator

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

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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
    
    # Logging options
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output")
    
    return parser.parse_args()

def initialize_training_systems(args: argparse.Namespace) -> Dict:
    """Initialize all training subsystems."""
    # Initialize core components
    llm_agent = LLMAgent(args.llm_model, args.llm_base_url)
    reward_calculator = PokemonRewardCalculator()
    game_intelligence = GameIntelligence()
    experience_memory = ExperienceMemory()
    context_builder = StrategicContextBuilder()
    
    # Initialize web monitor if enabled
    web_monitor = None
    if args.enable_web:
        web_monitor = WebMonitor(
            host=args.web_host,
            port=args.web_port,
            show_plots=not args.quiet
        )
    
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
        'web_monitor': web_monitor,
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
            systems['trainer'].shutdown()
            logger.info("Trainer stopped")
        except Exception as e:
            logger.error(f"Error stopping trainer: {e}")
    
    logger.info("Shutdown complete")
    if signum is not None:
        sys.exit(0)

def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger.info("Starting Pokemon Crystal RL training...")
    
    # Initialize all systems
    systems = initialize_training_systems(args)
    
    # Setup graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: graceful_shutdown(systems, s, f))
    signal.signal(signal.SIGTERM, lambda s, f: graceful_shutdown(systems, s, f))
    
    try:
        # Start web monitor if enabled
        if systems['web_monitor']:
            systems['web_monitor'].start()
            logger.info(f"Web monitor started at http://{args.web_host}:{args.web_port}")
        
        # Run training
        systems['trainer'].run()
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
        
    finally:
        # Always ensure clean shutdown
        graceful_shutdown(systems)

if __name__ == "__main__":
    main()
