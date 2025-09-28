#!/usr/bin/env python3
"""
Hybrid LLM-RL Training Example

Demonstrates the complete hybrid training system that combines:
- LLM strategic reasoning for novel situations and high-level planning
- RL tactical optimization for pattern recognition and execution
- Temporal memory for bridging decisions across time scales
- Curriculum learning for progressive skill development
"""

import argparse
import logging
import os
import sys
import time
import signal
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from trainer.hybrid_llm_rl_trainer import HybridLLMRLTrainer, TrainingConfig
from core.save_state_library import SaveStateLibrary
from web_dashboard.server import WebServer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid LLM-RL Training Demo")

    # Required arguments
    parser.add_argument("rom_path", help="Path to Pokemon Crystal ROM file")

    # Training configuration
    parser.add_argument("--max-episodes", type=int, default=50,
                       help="Maximum number of training episodes")
    parser.add_argument("--max-actions", type=int, default=500,
                       help="Maximum actions per episode")

    # Hybrid agent configuration
    parser.add_argument("--llm-weight", type=float, default=0.7,
                       help="Weight for LLM decisions (0.0-1.0)")
    parser.add_argument("--rl-weight", type=float, default=0.3,
                       help="Weight for RL decisions (0.0-1.0)")
    parser.add_argument("--exploration-rate", type=float, default=0.1,
                       help="Exploration rate for hybrid agent")

    # Curriculum learning
    parser.add_argument("--library-path", default="save_states",
                       help="Path to save state library")
    parser.add_argument("--curriculum-config",
                       help="Path to curriculum configuration file")
    parser.add_argument("--disable-curriculum", action="store_true",
                       help="Disable curriculum learning")

    # RL parameters
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4,
                       help="RL learning rate")
    parser.add_argument("--rl-batch-size", type=int, default=32,
                       help="RL batch size")
    parser.add_argument("--rl-memory-size", type=int, default=50000,
                       help="RL experience replay memory size")

    # Temporal memory
    parser.add_argument("--temporal-buffer-size", type=int, default=100000,
                       help="Temporal memory buffer size")

    # Monitoring and output
    parser.add_argument("--enable-web", action="store_true",
                       help="Enable web monitoring dashboard")
    parser.add_argument("--web-port", type=int, default=8080,
                       help="Web dashboard port")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--save-interval", type=int, default=25,
                       help="Episodes between model saves")

    return parser.parse_args()


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    return TrainingConfig(
        # Training parameters
        max_episodes=args.max_episodes,
        max_actions_per_episode=args.max_actions,

        # Hybrid agent configuration
        llm_weight=args.llm_weight,
        rl_weight=args.rl_weight,
        exploration_rate=args.exploration_rate,

        # Curriculum learning
        enable_curriculum=not args.disable_curriculum,
        curriculum_config=args.curriculum_config,

        # RL parameters
        rl_learning_rate=args.rl_learning_rate,
        rl_batch_size=args.rl_batch_size,
        rl_memory_size=args.rl_memory_size,

        # Temporal memory
        temporal_buffer_size=args.temporal_buffer_size,

        # Monitoring
        save_interval=args.save_interval,
        enable_web=args.enable_web,
        web_port=args.web_port
    )


def load_save_state_library(library_path: str) -> SaveStateLibrary:
    """Load save state library for curriculum learning."""
    if not os.path.exists(library_path):
        logger.warning(f"Save state library not found: {library_path}")
        logger.info("Creating empty library. Consider adding save states with:")
        logger.info("  python3 scripts/manage_save_states.py add ...")
        os.makedirs(library_path, exist_ok=True)

    library = SaveStateLibrary(library_path)
    states = library.list_save_states()

    logger.info(f"📁 Loaded save state library with {len(states)} states")

    if states:
        # Show library summary
        scenarios = set(state.scenario.value for state in states)
        difficulties = set(state.difficulty.value for state in states)
        phases = set(state.phase.value for state in states)

        logger.info(f"   Scenarios: {', '.join(sorted(scenarios))}")
        logger.info(f"   Difficulties: {', '.join(sorted(difficulties))}")
        logger.info(f"   Phases: {', '.join(sorted(phases))}")

    return library


class HybridTrainingDemo:
    """Demo class for hybrid training system."""

    def __init__(self, args):
        self.args = args
        self.config = create_training_config(args)
        self.trainer = None
        self.web_server = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("🛑 Shutdown signal received")
        self.stop()
        sys.exit(0)

    def run(self):
        """Run the hybrid training demonstration."""
        logger.info("🚀 Starting Hybrid LLM-RL Training Demo")
        logger.info("=" * 60)

        # Verify ROM file
        if not os.path.exists(self.args.rom_path):
            logger.error(f"❌ ROM file not found: {self.args.rom_path}")
            return 1

        # Load save state library
        save_state_library = None
        if self.config.enable_curriculum:
            save_state_library = load_save_state_library(self.args.library_path)

        # Initialize trainer
        self.trainer = HybridLLMRLTrainer(
            rom_path=self.args.rom_path,
            config=self.config,
            save_state_library=save_state_library,
            progress_callback=self._progress_callback
        )

        # Start web monitoring if enabled
        if self.config.enable_web:
            self._start_web_monitoring()

        try:
            # Print configuration summary
            self._print_config_summary()

            # Start training
            training_thread = self.trainer.start_training()

            # Monitor training progress
            self._monitor_training()

            # Wait for completion
            training_thread.join()

            # Print final results
            self._print_final_results()

            return 0

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return 1

        finally:
            self.stop()

    def _start_web_monitoring(self):
        """Start web monitoring dashboard."""
        try:
            self.web_server = WebServer(
                port=self.config.web_port,
                host="localhost"
            )
            self.web_server.start()

            logger.info(f"🌐 Web dashboard available at http://localhost:{self.config.web_port}")
            logger.info("   Real-time training metrics and hybrid decision analysis")

        except Exception as e:
            logger.warning(f"Failed to start web server: {e}")
            self.web_server = None

    def _progress_callback(self, episode: int, episode_stats: dict, training_summary: dict):
        """Callback for training progress updates."""
        # This could be used for external monitoring, logging, etc.
        if episode % 10 == 0:
            hybrid_summary = training_summary['hybrid_agent']
            logger.info(f"Progress Update - Episode {episode}")
            logger.info(f"  Recent avg reward: {training_summary['performance']['recent_avg_reward']:.2f}")
            logger.info(f"  Decision mode distribution: {hybrid_summary['mode_distribution']}")
            logger.info(f"  Success rates: {hybrid_summary['success_rates']}")

    def _print_config_summary(self):
        """Print training configuration summary."""
        logger.info("🔧 Training Configuration:")
        logger.info(f"   Episodes: {self.config.max_episodes}")
        logger.info(f"   Actions per episode: {self.config.max_actions_per_episode}")
        logger.info(f"   LLM weight: {self.config.llm_weight}")
        logger.info(f"   RL weight: {self.config.rl_weight}")
        logger.info(f"   Exploration rate: {self.config.exploration_rate}")
        logger.info(f"   Curriculum learning: {'Enabled' if self.config.enable_curriculum else 'Disabled'}")
        logger.info(f"   Web monitoring: {'Enabled' if self.config.enable_web else 'Disabled'}")
        logger.info("")

    def _monitor_training(self):
        """Monitor training progress in real-time."""
        logger.info("🎯 Training started - monitoring progress...")
        logger.info("   Press Ctrl+C to stop training gracefully")
        logger.info("")

        start_time = time.time()
        last_episode = -1

        while self.trainer.is_training:
            try:
                time.sleep(5)  # Update every 5 seconds

                summary = self.trainer.get_training_summary()
                current_episode = summary['current_episode']

                # Print progress update
                if current_episode != last_episode:
                    elapsed = time.time() - start_time
                    progress = current_episode / self.config.max_episodes
                    eta = (elapsed / max(1, current_episode)) * (self.config.max_episodes - current_episode)

                    logger.info(f"⏱️  Episode {current_episode}/{self.config.max_episodes} "
                               f"({progress:.1%}) - ETA: {eta/60:.1f}m")

                    last_episode = current_episode

            except KeyboardInterrupt:
                logger.info("🛑 Stopping training...")
                break
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

    def _print_final_results(self):
        """Print final training results."""
        if not self.trainer:
            return

        summary = self.trainer.get_training_summary()

        logger.info("")
        logger.info("🏁 Training Completed!")
        logger.info("=" * 50)

        # Performance metrics
        perf = summary['performance']
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Episodes completed: {summary['episodes_completed']}")
        logger.info(f"   Total actions: {summary['total_actions']}")
        logger.info(f"   Average reward: {perf['avg_reward']:.2f}")
        logger.info(f"   Recent average reward: {perf['recent_avg_reward']:.2f}")
        logger.info(f"   Average episode length: {perf['avg_episode_length']:.1f}")

        # Hybrid agent analysis
        hybrid = summary['hybrid_agent']
        logger.info(f"🤖 Hybrid Agent Analysis:")
        logger.info(f"   Total decisions: {hybrid['total_decisions']}")
        logger.info(f"   Mode distribution: {hybrid['mode_distribution']}")
        logger.info(f"   Success rates: {hybrid['success_rates']}")
        logger.info(f"   Mode switches: {hybrid['mode_switches']}")

        # Curriculum progress
        if summary['curriculum']:
            curr = summary['curriculum']
            logger.info(f"🎓 Curriculum Progress:")
            logger.info(f"   Current stage: {curr['current_stage']}")
            logger.info(f"   Level progress: {curr['level_progress']['episodes']}/{curr['level_progress']['max_episodes']}")

        # Temporal memory stats
        temporal = summary['temporal_memory']
        logger.info(f"🧠 Temporal Memory:")
        logger.info(f"   Buffer size: {temporal['buffer_size']}")
        logger.info(f"   Episodes stored: {temporal['episodes_completed']}")
        logger.info(f"   Total experiences: {temporal['total_experiences']}")

    def stop(self):
        """Stop all components gracefully."""
        if self.trainer:
            self.trainer.stop_training()

        if self.web_server:
            self.web_server.stop()


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)

    demo = HybridTrainingDemo(args)
    return demo.run()


if __name__ == "__main__":
    sys.exit(main())