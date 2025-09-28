#!/usr/bin/env python3
"""
Curriculum Learning Training Example

Demonstrates how to use the curriculum learning system for progressive
Pokemon Crystal RL training with automated save state selection.
"""

import argparse
import logging
import time
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.save_state_library import SaveStateLibrary
from training.curriculum_learning import CurriculumManager
from training.components.emulation_manager import EmulationManager, EmulationConfig
from training.components.statistics_tracker import StatisticsTracker
from training.components.llm_decision_engine import LLMDecisionEngine, LLMConfig


class CurriculumTrainer:
    """
    Curriculum-based trainer that uses save state library and progressive difficulty.
    """

    def __init__(self,
                 rom_path: str,
                 library_path: str = "save_states",
                 curriculum_config: str = None,
                 llm_model: str = "smollm2:1.7b",
                 enable_web: bool = False,
                 web_port: int = 8080):
        """
        Initialize curriculum trainer.

        Args:
            rom_path: Path to Pokemon Crystal ROM
            library_path: Path to save state library
            curriculum_config: Path to curriculum configuration file
            llm_model: LLM model to use for decision making
            enable_web: Whether to enable web monitoring
            web_port: Port for web dashboard
        """
        self.rom_path = rom_path
        self.logger = logging.getLogger("CurriculumTrainer")

        # Initialize save state library
        self.library = SaveStateLibrary(library_path)
        library_info = self.library.export_library_info()
        self.logger.info(f"📚 Loaded save state library: {library_info['total_states']} states available")

        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(
            library=self.library,
            curriculum_config=curriculum_config
        )

        # Initialize components
        self.emulation_manager = None
        self.statistics_tracker = StatisticsTracker()
        self.llm_engine = None

        # Configuration
        self.llm_model = llm_model
        self.enable_web = enable_web
        self.web_port = web_port

        # Training state
        self.current_save_state = None
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_start_time = None

    def start_episode(self) -> bool:
        """
        Start a new training episode with curriculum-selected save state.

        Returns:
            bool: True if episode started successfully
        """
        try:
            # Select save state based on current curriculum level
            self.current_save_state = self.curriculum_manager.select_save_state()
            if not self.current_save_state:
                self.logger.error("❌ No suitable save state found for current curriculum level")
                return False

            # Record usage
            self.library.record_usage(self.current_save_state.id)

            # Initialize emulation with selected save state
            emulation_config = EmulationConfig(
                rom_path=self.rom_path,
                save_state_path=self.current_save_state.file_path,
                headless=True,
                debug_mode=False
            )

            self.emulation_manager = EmulationManager(emulation_config)
            if not self.emulation_manager.initialize():
                self.logger.error("❌ Failed to initialize emulation")
                return False

            # Initialize LLM engine
            llm_config = LLMConfig(
                model_name=self.llm_model,
                base_url="http://localhost:11434",
                decision_interval=10,
                context_window=2000
            )

            self.llm_engine = LLMDecisionEngine(
                config=llm_config,
                emulation_manager=self.emulation_manager,
                statistics_tracker=self.statistics_tracker
            )

            # Reset episode tracking
            self.episode_count += 1
            self.total_reward = 0.0
            self.episode_start_time = time.time()

            # Log episode info
            curriculum_status = self.curriculum_manager.get_curriculum_status()
            self.logger.info(f"🎮 Episode {self.episode_count} Started")
            self.logger.info(f"   📍 Save State: {self.current_save_state.name}")
            self.logger.info(f"   🎯 Stage: {curriculum_status['current_stage']}")
            self.logger.info(f"   📊 Level Progress: {curriculum_status['level_progress']['episodes']}/{curriculum_status['level_progress']['max_episodes']} episodes")
            self.logger.info(f"   ✅ Success Rate: {curriculum_status['level_progress']['success_rate']:.1%}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start episode: {e}")
            return False

    def run_episode(self, max_actions: int = 1000) -> bool:
        """
        Run a single training episode.

        Args:
            max_actions: Maximum actions per episode

        Returns:
            bool: True if episode completed successfully
        """
        if not self.emulation_manager or not self.llm_engine:
            self.logger.error("❌ Episode not properly initialized")
            return False

        try:
            self.logger.info(f"🚀 Running episode with max {max_actions} actions...")

            action_count = 0
            last_progress_report = 0

            while action_count < max_actions:
                # Get LLM decision
                action, confidence, reasoning = self.llm_engine.get_decision()
                if action is None:
                    self.logger.warning("⚠️ LLM decision failed, using random action")
                    action = 0  # No action

                # Execute action
                if not self.emulation_manager.execute_action(action):
                    self.logger.error("❌ Action execution failed")
                    break

                # Update statistics
                self.statistics_tracker.record_action(action)
                action_count += 1

                # Periodic progress report
                if action_count - last_progress_report >= 100:
                    self.logger.info(f"   📈 Progress: {action_count}/{max_actions} actions")
                    last_progress_report = action_count

            # Calculate episode results
            episode_duration = time.time() - self.episode_start_time
            stats = self.statistics_tracker.get_summary()
            self.total_reward = stats.get('total_reward', 0.0)

            # Determine success (you can customize this logic)
            success = self._evaluate_episode_success(stats)

            self.logger.info(f"🏁 Episode {self.episode_count} Complete")
            self.logger.info(f"   ⏱️ Duration: {episode_duration:.1f}s")
            self.logger.info(f"   🎯 Actions: {action_count}")
            self.logger.info(f"   🏆 Total Reward: {self.total_reward:.1f}")
            self.logger.info(f"   ✅ Success: {'Yes' if success else 'No'}")

            # Record episode result in curriculum
            advanced = self.curriculum_manager.record_episode_result(success, self.total_reward)
            if advanced:
                self.logger.info("🎓 CURRICULUM ADVANCEMENT! Moving to next level.")

            # Record usage result in save state library
            self.library.record_usage(self.current_save_state.id, success)

            return True

        except Exception as e:
            self.logger.error(f"❌ Episode execution failed: {e}")
            return False

        finally:
            # Cleanup
            if self.emulation_manager:
                self.emulation_manager.shutdown()

    def _evaluate_episode_success(self, stats: dict) -> bool:
        """
        Evaluate whether an episode was successful.
        Customize this logic based on your training objectives.

        Args:
            stats: Episode statistics

        Returns:
            bool: True if episode was successful
        """
        # Simple success criteria - customize as needed
        total_reward = stats.get('total_reward', 0.0)
        level_gains = stats.get('level_gains', 0)
        badges_gained = stats.get('badges_gained', 0)

        # Consider episode successful if:
        # - Positive total reward, OR
        # - Gained levels, OR
        # - Gained badges
        return total_reward > 50.0 or level_gains > 0 or badges_gained > 0

    def run_training(self, num_episodes: int = 10, max_actions_per_episode: int = 1000):
        """
        Run curriculum training for specified number of episodes.

        Args:
            num_episodes: Number of episodes to run
            max_actions_per_episode: Maximum actions per episode
        """
        self.logger.info(f"🎓 Starting Curriculum Training")
        self.logger.info(f"   📚 Library: {self.library.export_library_info()['total_states']} save states")
        self.logger.info(f"   🎯 Episodes: {num_episodes}")
        self.logger.info(f"   🎮 Max Actions: {max_actions_per_episode}")

        successful_episodes = 0
        failed_episodes = 0

        for episode_num in range(num_episodes):
            self.logger.info(f"\n{'='*60}")
            curriculum_status = self.curriculum_manager.get_curriculum_status()
            self.logger.info(f"📊 Curriculum Status: Level {curriculum_status['current_level']}, Stage: {curriculum_status['current_stage']}")

            # Start episode
            if not self.start_episode():
                failed_episodes += 1
                self.logger.error(f"❌ Failed to start episode {episode_num + 1}")
                continue

            # Run episode
            if self.run_episode(max_actions_per_episode):
                successful_episodes += 1
            else:
                failed_episodes += 1

            # Show curriculum progress
            curriculum_status = self.curriculum_manager.get_curriculum_status()
            advancement_status = curriculum_status['advancement_status']

            self.logger.info(f"📈 Curriculum Progress:")
            self.logger.info(f"   Current Stage: {curriculum_status['current_stage']}")
            self.logger.info(f"   Episode Progress: {curriculum_status['level_progress']['episodes']}/{curriculum_status['level_progress']['max_episodes']}")
            self.logger.info(f"   Success Rate: {curriculum_status['level_progress']['success_rate']:.1%}")

            if advancement_status['ready_to_advance']:
                self.logger.info(f"   🎓 Ready to advance to next level!")
            else:
                if advancement_status['episodes_until_eligible'] > 0:
                    self.logger.info(f"   📅 Episodes until eligible: {advancement_status['episodes_until_eligible']}")
                if advancement_status['success_rate_deficit'] > 0:
                    self.logger.info(f"   📊 Success rate needed: +{advancement_status['success_rate_deficit']:.1%}")

        # Final summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎓 Curriculum Training Complete!")
        self.logger.info(f"   ✅ Successful Episodes: {successful_episodes}")
        self.logger.info(f"   ❌ Failed Episodes: {failed_episodes}")
        self.logger.info(f"   📊 Success Rate: {successful_episodes / num_episodes:.1%}")

        final_status = self.curriculum_manager.get_curriculum_status()
        self.logger.info(f"   🎯 Final Stage: {final_status['current_stage']}")
        self.logger.info(f"   📈 Completion: {final_status['overall_progress']['completion_percentage']:.1f}%")


def main():
    """Main entry point for curriculum training."""
    parser = argparse.ArgumentParser(description="Pokemon Crystal Curriculum Learning Training")
    parser.add_argument("rom_path", help="Path to Pokemon Crystal ROM file")
    parser.add_argument("--library-path", default="save_states", help="Path to save state library")
    parser.add_argument("--curriculum-config", help="Path to curriculum configuration file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-actions", type=int, default=1000, help="Maximum actions per episode")
    parser.add_argument("--llm-model", default="smollm2:1.7b", help="LLM model to use")
    parser.add_argument("--enable-web", action="store_true", help="Enable web monitoring")
    parser.add_argument("--web-port", type=int, default=8080, help="Web dashboard port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Verify ROM file
    if not os.path.exists(args.rom_path):
        print(f"❌ ROM file not found: {args.rom_path}")
        return 1

    # Verify save state library
    if not os.path.exists(args.library_path):
        print(f"❌ Save state library not found: {args.library_path}")
        print(f"   Create one using: python3 scripts/manage_save_states.py add ...")
        return 1

    try:
        # Initialize trainer
        trainer = CurriculumTrainer(
            rom_path=args.rom_path,
            library_path=args.library_path,
            curriculum_config=args.curriculum_config,
            llm_model=args.llm_model,
            enable_web=args.enable_web,
            web_port=args.web_port
        )

        # Run training
        trainer.run_training(args.episodes, args.max_actions)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())