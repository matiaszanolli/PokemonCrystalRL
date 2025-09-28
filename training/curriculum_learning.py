#!/usr/bin/env python3
"""
Curriculum Learning System - Progressive difficulty training with save state library integration

This module implements curriculum learning for Pokemon Crystal RL training, allowing
agents to progressively learn from easier to harder scenarios using the save state library.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from core.save_state_library import (
    SaveStateLibrary, GamePhase, TrainingScenario, Difficulty,
    SaveStateMetadata
)


class CurriculumStage(Enum):
    """Curriculum learning stages with increasing complexity."""
    TUTORIAL = "tutorial"
    BASIC_MECHANICS = "basic_mechanics"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CurriculumLevel:
    """Configuration for a single curriculum level."""
    stage: CurriculumStage
    scenarios: List[TrainingScenario]
    difficulty: Difficulty
    min_success_rate: float = 0.7  # Required success rate to advance
    min_episodes: int = 10  # Minimum episodes before advancing
    max_episodes: int = 50  # Maximum episodes before forced advancement
    phase_filter: Optional[GamePhase] = None
    badges_range: Optional[Tuple[int, int]] = None
    tags: Optional[List[str]] = None


@dataclass
class CurriculumProgress:
    """Tracks progress through curriculum levels."""
    current_level: int = 0
    total_episodes: int = 0
    level_episodes: int = 0
    level_successes: int = 0
    success_rate: float = 0.0
    last_advancement: Optional[str] = None
    stage_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.stage_history is None:
            self.stage_history = []


class CurriculumManager:
    """Manages curriculum learning progression and save state selection."""

    def __init__(self,
                 library: SaveStateLibrary,
                 curriculum_config: Optional[str] = None,
                 progress_file: str = "curriculum_progress.json"):
        """
        Initialize curriculum manager.

        Args:
            library: Save state library for scenario selection
            curriculum_config: Path to curriculum configuration file
            progress_file: File to persist curriculum progress
        """
        self.library = library
        self.progress_file = Path(progress_file)
        self.logger = logging.getLogger("CurriculumManager")

        # Load or create default curriculum
        if curriculum_config and Path(curriculum_config).exists():
            self.curriculum = self._load_curriculum_config(curriculum_config)
        else:
            self.curriculum = self._create_default_curriculum()

        # Load or initialize progress
        self.progress = self._load_progress()

        self.logger.info(f"Curriculum Manager initialized with {len(self.curriculum)} levels")
        self.logger.info(f"Current stage: {self.get_current_stage().value} (Level {self.progress.current_level})")

    def _create_default_curriculum(self) -> List[CurriculumLevel]:
        """Create a default curriculum progression."""
        return [
            # Stage 1: Tutorial - Basic movement and interaction
            CurriculumLevel(
                stage=CurriculumStage.TUTORIAL,
                scenarios=[TrainingScenario.FIRST_POKEMON, TrainingScenario.EXPLORATION],
                difficulty=Difficulty.EASY,
                phase_filter=GamePhase.TUTORIAL,
                min_success_rate=0.6,
                min_episodes=5,
                max_episodes=20,
                tags=["beginner", "movement"]
            ),

            # Stage 2: Basic Mechanics - Simple battles and progression
            CurriculumLevel(
                stage=CurriculumStage.BASIC_MECHANICS,
                scenarios=[TrainingScenario.WILD_ENCOUNTER, TrainingScenario.TEAM_BUILDING],
                difficulty=Difficulty.EASY,
                phase_filter=GamePhase.EARLY_GAME,
                badges_range=(0, 1),
                min_success_rate=0.65,
                min_episodes=8,
                max_episodes=25
            ),

            # Stage 3: Intermediate - Gym battles and strategy
            CurriculumLevel(
                stage=CurriculumStage.INTERMEDIATE,
                scenarios=[TrainingScenario.GYM_BATTLE, TrainingScenario.BATTLE_TRAINING],
                difficulty=Difficulty.MEDIUM,
                phase_filter=GamePhase.EARLY_GAME,
                badges_range=(1, 3),
                min_success_rate=0.7,
                min_episodes=10,
                max_episodes=30
            ),

            # Stage 4: Advanced - Complex battles and progression
            CurriculumLevel(
                stage=CurriculumStage.ADVANCED,
                scenarios=[TrainingScenario.GYM_BATTLE, TrainingScenario.PROGRESSION],
                difficulty=Difficulty.MEDIUM,
                phase_filter=GamePhase.MID_GAME,
                badges_range=(3, 6),
                min_success_rate=0.75,
                min_episodes=15,
                max_episodes=40
            ),

            # Stage 5: Expert - Elite Four and endgame content
            CurriculumLevel(
                stage=CurriculumStage.EXPERT,
                scenarios=[TrainingScenario.ELITE_FOUR, TrainingScenario.COMPLETIONIST],
                difficulty=Difficulty.HARD,
                phase_filter=GamePhase.LATE_GAME,
                badges_range=(6, 16),
                min_success_rate=0.8,
                min_episodes=20,
                max_episodes=50
            )
        ]

    def _load_curriculum_config(self, config_path: str) -> List[CurriculumLevel]:
        """Load curriculum configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            curriculum = []
            for level_data in data['levels']:
                # Convert string enums back to enum objects
                level_data['stage'] = CurriculumStage(level_data['stage'])
                level_data['scenarios'] = [TrainingScenario(s) for s in level_data['scenarios']]
                level_data['difficulty'] = Difficulty(level_data['difficulty'])
                if 'phase_filter' in level_data and level_data['phase_filter']:
                    level_data['phase_filter'] = GamePhase(level_data['phase_filter'])

                curriculum.append(CurriculumLevel(**level_data))

            self.logger.info(f"Loaded curriculum with {len(curriculum)} levels from {config_path}")
            return curriculum

        except Exception as e:
            self.logger.error(f"Failed to load curriculum config: {e}")
            self.logger.info("Using default curriculum instead")
            return self._create_default_curriculum()

    def _load_progress(self) -> CurriculumProgress:
        """Load curriculum progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return CurriculumProgress(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load progress file: {e}")

        return CurriculumProgress()

    def _save_progress(self) -> None:
        """Save current progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(self.progress), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level configuration."""
        if self.progress.current_level >= len(self.curriculum):
            return self.curriculum[-1]  # Stay at final level
        return self.curriculum[self.progress.current_level]

    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.get_current_level().stage

    def select_save_state(self) -> Optional[SaveStateMetadata]:
        """
        Select an appropriate save state for the current curriculum level.

        Returns:
            SaveStateMetadata or None if no suitable states found
        """
        level = self.get_current_level()

        # Get recommendations from save state library
        suitable_states = []

        for scenario in level.scenarios:
            states = self.library.get_recommendations(
                target_scenario=scenario,
                target_difficulty=level.difficulty,
                badges_range=level.badges_range
            )

            # Apply additional filters
            for state in states:
                if level.phase_filter and state.phase != level.phase_filter:
                    continue
                if level.tags and not any(tag in state.tags for tag in level.tags):
                    continue

                suitable_states.append(state)

        if not suitable_states:
            self.logger.warning(f"No suitable save states found for level {self.progress.current_level}")
            return None

        # Prefer states with lower usage count to ensure variety
        suitable_states.sort(key=lambda s: s.usage_count)
        selected = suitable_states[0]

        self.logger.info(f"Selected save state: {selected.name} ({selected.id})")
        return selected

    def record_episode_result(self, success: bool, reward: float = None) -> bool:
        """
        Record the result of a training episode and check for advancement.

        Args:
            success: Whether the episode was successful
            reward: Optional reward value for the episode

        Returns:
            bool: True if advanced to next level
        """
        self.progress.total_episodes += 1
        self.progress.level_episodes += 1

        if success:
            self.progress.level_successes += 1

        # Calculate success rate
        if self.progress.level_episodes > 0:
            self.progress.success_rate = self.progress.level_successes / self.progress.level_episodes

        # Check for advancement
        level = self.get_current_level()
        should_advance = False

        if (self.progress.level_episodes >= level.min_episodes and
            self.progress.success_rate >= level.min_success_rate):
            should_advance = True
            self.logger.info(f"Advancement criteria met: {self.progress.success_rate:.2%} success rate")
        elif self.progress.level_episodes >= level.max_episodes:
            should_advance = True
            self.logger.info(f"Maximum episodes reached: {self.progress.level_episodes}/{level.max_episodes}")

        if should_advance and self.progress.current_level < len(self.curriculum) - 1:
            self._advance_level()
            self._save_progress()
            return True

        self._save_progress()
        return False

    def _advance_level(self) -> None:
        """Advance to the next curriculum level."""
        old_level = self.progress.current_level
        old_stage = self.get_current_stage()

        # Record stage completion
        stage_record = {
            "level": old_level,
            "stage": old_stage.value,
            "episodes": self.progress.level_episodes,
            "successes": self.progress.level_successes,
            "success_rate": self.progress.success_rate,
            "completed_at": time.time()
        }
        self.progress.stage_history.append(stage_record)

        # Advance to next level
        self.progress.current_level += 1
        self.progress.level_episodes = 0
        self.progress.level_successes = 0
        self.progress.success_rate = 0.0
        self.progress.last_advancement = time.time()

        new_stage = self.get_current_stage()

        self.logger.info(f"🎓 CURRICULUM ADVANCEMENT!")
        self.logger.info(f"   From: {old_stage.value} (Level {old_level})")
        self.logger.info(f"   To: {new_stage.value} (Level {self.progress.current_level})")
        self.logger.info(f"   Previous stage: {stage_record['success_rate']:.1%} success in {stage_record['episodes']} episodes")

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get comprehensive curriculum status information."""
        level = self.get_current_level()

        return {
            "current_level": self.progress.current_level,
            "total_levels": len(self.curriculum),
            "current_stage": level.stage.value,
            "current_scenarios": [s.value for s in level.scenarios],
            "current_difficulty": level.difficulty.value,
            "level_progress": {
                "episodes": self.progress.level_episodes,
                "successes": self.progress.level_successes,
                "success_rate": self.progress.success_rate,
                "min_episodes": level.min_episodes,
                "max_episodes": level.max_episodes,
                "target_success_rate": level.min_success_rate
            },
            "overall_progress": {
                "total_episodes": self.progress.total_episodes,
                "stages_completed": len(self.progress.stage_history),
                "completion_percentage": (self.progress.current_level / len(self.curriculum)) * 100
            },
            "advancement_status": self._get_advancement_status()
        }

    def _get_advancement_status(self) -> Dict[str, Any]:
        """Get advancement status and requirements."""
        level = self.get_current_level()

        episodes_needed = max(0, level.min_episodes - self.progress.level_episodes)
        success_rate_needed = max(0, level.min_success_rate - self.progress.success_rate)
        episodes_remaining = max(0, level.max_episodes - self.progress.level_episodes)

        ready_to_advance = (
            self.progress.level_episodes >= level.min_episodes and
            self.progress.success_rate >= level.min_success_rate
        )

        return {
            "ready_to_advance": ready_to_advance,
            "episodes_until_eligible": episodes_needed,
            "success_rate_deficit": success_rate_needed,
            "episodes_until_forced": episodes_remaining,
            "is_final_level": self.progress.current_level >= len(self.curriculum) - 1
        }

    def export_curriculum_config(self, output_path: str) -> None:
        """Export current curriculum configuration to JSON file."""
        config_data = {
            "description": "Pokemon Crystal RL Curriculum Learning Configuration",
            "levels": []
        }

        for level in self.curriculum:
            level_data = asdict(level)
            # Convert enums to strings for JSON serialization
            level_data['stage'] = level.stage.value
            level_data['scenarios'] = [s.value for s in level.scenarios]
            level_data['difficulty'] = level.difficulty.value
            if level.phase_filter:
                level_data['phase_filter'] = level.phase_filter.value

            config_data['levels'].append(level_data)

        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Exported curriculum configuration to {output_path}")


def create_default_curriculum_config(output_path: str = "curriculum_config.json") -> None:
    """Create and save a default curriculum configuration file."""
    # Create a temporary manager to generate default curriculum
    from core.save_state_library import SaveStateLibrary
    temp_library = SaveStateLibrary("save_states")
    manager = CurriculumManager(temp_library)
    manager.export_curriculum_config(output_path)
    print(f"✅ Created default curriculum configuration: {output_path}")


if __name__ == "__main__":
    create_default_curriculum_config()