#!/usr/bin/env python3
"""
Save State Library - Organized collection of curated save states for training

This module provides a centralized library for managing save states with metadata,
categorization, and easy access for different training scenarios.
"""

import os
import json
import shutil
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game progression phases for save state categorization."""
    TUTORIAL = "tutorial"
    EARLY_GAME = "early_game"
    MID_GAME = "mid_game"
    LATE_GAME = "late_game"
    POST_GAME = "post_game"


class TrainingScenario(Enum):
    """Training scenario types for save state usage."""
    FIRST_POKEMON = "first_pokemon"
    GYM_BATTLE = "gym_battle"
    WILD_ENCOUNTER = "wild_encounter"
    ELITE_FOUR = "elite_four"
    EXPLORATION = "exploration"
    TEAM_BUILDING = "team_building"
    SPEEDRUN = "speedrun"
    COMPLETIONIST = "completionist"
    BATTLE_TRAINING = "battle_training"
    PROGRESSION = "progression"


class Difficulty(Enum):
    """Difficulty levels for training scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class GameState:
    """Metadata about the game state at the save point."""
    player_name: str
    location: str
    map_id: int
    badges: int
    level: int
    party_size: int
    money: int
    playtime_hours: int
    story_progress: str


@dataclass
class SaveStateMetadata:
    """Complete metadata for a save state entry."""
    # Identification
    id: str
    name: str
    description: str

    # Classification
    phase: GamePhase
    scenario: TrainingScenario
    difficulty: Difficulty

    # Game state
    game_state: GameState

    # Training context
    training_objectives: List[str]
    expected_actions: int
    recommended_models: List[str]

    # File management
    file_path: str
    file_size: int
    checksum: str
    created_at: str
    created_by: str

    # Usage tracking
    usage_count: int = 0
    last_used: Optional[str] = None
    success_rate: Optional[float] = None

    # Tags for filtering
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SaveStateLibrary:
    """Manages a library of organized save states for training scenarios."""

    def __init__(self, library_path: str = "save_states"):
        self.library_path = Path(library_path)
        self.metadata_file = self.library_path / "library.json"
        self.states_dir = self.library_path / "states"

        # Create directories if they don't exist
        self.library_path.mkdir(exist_ok=True)
        self.states_dir.mkdir(exist_ok=True)

        # Load existing metadata
        self.metadata: Dict[str, SaveStateMetadata] = {}
        self._load_metadata()

        logger.info(f"Save State Library initialized: {len(self.metadata)} states available")

    def _load_metadata(self) -> None:
        """Load metadata from the library file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)

                for state_id, state_data in data.items():
                    # Convert string enums back to enum objects
                    state_data['phase'] = GamePhase(state_data['phase'])
                    state_data['scenario'] = TrainingScenario(state_data['scenario'])
                    state_data['difficulty'] = Difficulty(state_data['difficulty'])

                    # Convert game_state dict back to GameState object
                    state_data['game_state'] = GameState(**state_data['game_state'])

                    self.metadata[state_id] = SaveStateMetadata(**state_data)

                logger.info(f"Loaded {len(self.metadata)} save states from library")

            except Exception as e:
                logger.error(f"Failed to load save state library: {e}")
                self.metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata to the library file."""
        try:
            # Convert to serializable format
            data = {}
            for state_id, metadata in self.metadata.items():
                metadata_dict = asdict(metadata)
                # Convert enums to strings
                metadata_dict['phase'] = metadata.phase.value
                metadata_dict['scenario'] = metadata.scenario.value
                metadata_dict['difficulty'] = metadata.difficulty.value
                data[state_id] = metadata_dict

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("Save state library metadata saved")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_save_state(self,
                      source_path: str,
                      name: str,
                      description: str,
                      phase: GamePhase,
                      scenario: TrainingScenario,
                      difficulty: Difficulty,
                      game_state: GameState,
                      training_objectives: List[str],
                      expected_actions: int = 1000,
                      recommended_models: List[str] = None,
                      tags: List[str] = None,
                      created_by: str = "user") -> str:
        """Add a new save state to the library.

        Args:
            source_path: Path to the source save state file
            name: Human-readable name for the save state
            description: Detailed description of the scenario
            phase: Game progression phase
            scenario: Training scenario type
            difficulty: Difficulty level
            game_state: Current game state information
            training_objectives: List of training objectives
            expected_actions: Expected number of actions for this scenario
            recommended_models: List of recommended LLM models
            tags: Additional tags for filtering
            created_by: Creator identifier

        Returns:
            str: Unique ID of the added save state
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source save state not found: {source_path}")

        # Generate unique ID
        state_id = f"{scenario.value}_{difficulty.value}_{game_state.badges}badges_{int(datetime.now().timestamp())}"

        # Copy file to library
        dest_path = self.states_dir / f"{state_id}.state"
        shutil.copy2(source_path, dest_path)

        # Calculate file info
        file_size = os.path.getsize(dest_path)
        checksum = self._calculate_checksum(dest_path)

        # Create metadata
        metadata = SaveStateMetadata(
            id=state_id,
            name=name,
            description=description,
            phase=phase,
            scenario=scenario,
            difficulty=difficulty,
            game_state=game_state,
            training_objectives=training_objectives,
            expected_actions=expected_actions,
            recommended_models=recommended_models or [],
            file_path=str(dest_path),
            file_size=file_size,
            checksum=checksum,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            tags=tags or []
        )

        # Add to library
        self.metadata[state_id] = metadata
        self._save_metadata()

        logger.info(f"Added save state '{name}' with ID: {state_id}")
        return state_id

    def get_save_state(self, state_id: str) -> Optional[SaveStateMetadata]:
        """Get save state metadata by ID."""
        return self.metadata.get(state_id)

    def list_save_states(self,
                        phase: Optional[GamePhase] = None,
                        scenario: Optional[TrainingScenario] = None,
                        difficulty: Optional[Difficulty] = None,
                        min_badges: Optional[int] = None,
                        max_badges: Optional[int] = None,
                        tags: Optional[List[str]] = None) -> List[SaveStateMetadata]:
        """List save states with optional filtering.

        Args:
            phase: Filter by game phase
            scenario: Filter by training scenario
            difficulty: Filter by difficulty level
            min_badges: Minimum number of badges
            max_badges: Maximum number of badges
            tags: Filter by tags (must have all specified tags)

        Returns:
            List of matching save state metadata
        """
        results = []

        for metadata in self.metadata.values():
            # Apply filters
            if phase and metadata.phase != phase:
                continue
            if scenario and metadata.scenario != scenario:
                continue
            if difficulty and metadata.difficulty != difficulty:
                continue
            if min_badges is not None and metadata.game_state.badges < min_badges:
                continue
            if max_badges is not None and metadata.game_state.badges > max_badges:
                continue
            if tags and not all(tag in metadata.tags for tag in tags):
                continue

            results.append(metadata)

        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results

    def get_recommendations(self,
                          target_scenario: TrainingScenario,
                          target_difficulty: Difficulty = Difficulty.MEDIUM,
                          badges_range: Optional[tuple] = None) -> List[SaveStateMetadata]:
        """Get recommended save states for a specific training goal.

        Args:
            target_scenario: The training scenario to optimize for
            target_difficulty: Preferred difficulty level
            badges_range: Optional (min, max) badges range

        Returns:
            List of recommended save states
        """
        # Start with exact matches
        exact_matches = self.list_save_states(
            scenario=target_scenario,
            difficulty=target_difficulty
        )

        if badges_range:
            exact_matches = [
                state for state in exact_matches
                if badges_range[0] <= state.game_state.badges <= badges_range[1]
            ]

        if exact_matches:
            return exact_matches[:5]  # Top 5 exact matches

        # Fall back to similar scenarios
        similar_matches = self.list_save_states(scenario=target_scenario)
        return similar_matches[:3]  # Top 3 similar matches

    def record_usage(self, state_id: str, success: bool = None) -> None:
        """Record usage of a save state and update statistics.

        Args:
            state_id: ID of the used save state
            success: Whether the training session was successful
        """
        if state_id not in self.metadata:
            logger.warning(f"Unknown save state ID: {state_id}")
            return

        metadata = self.metadata[state_id]
        metadata.usage_count += 1
        metadata.last_used = datetime.now().isoformat()

        # Update success rate if provided
        if success is not None:
            if metadata.success_rate is None:
                metadata.success_rate = 1.0 if success else 0.0
            else:
                # Simple moving average (could be improved with more sophisticated tracking)
                metadata.success_rate = (metadata.success_rate * 0.9) + (0.1 if success else 0.0)

        self._save_metadata()
        logger.debug(f"Recorded usage for save state: {state_id}")

    def remove_save_state(self, state_id: str) -> bool:
        """Remove a save state from the library.

        Args:
            state_id: ID of the save state to remove

        Returns:
            bool: True if removed successfully
        """
        if state_id not in self.metadata:
            logger.warning(f"Save state not found: {state_id}")
            return False

        metadata = self.metadata[state_id]

        # Remove file
        try:
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
        except Exception as e:
            logger.error(f"Failed to remove save state file: {e}")

        # Remove from metadata
        del self.metadata[state_id]
        self._save_metadata()

        logger.info(f"Removed save state: {state_id}")
        return True

    def verify_integrity(self) -> Dict[str, List[str]]:
        """Verify the integrity of all save states in the library.

        Returns:
            Dict with 'valid', 'missing_files', 'checksum_mismatch' lists
        """
        results = {
            'valid': [],
            'missing_files': [],
            'checksum_mismatch': []
        }

        for state_id, metadata in self.metadata.items():
            if not os.path.exists(metadata.file_path):
                results['missing_files'].append(state_id)
                continue

            current_checksum = self._calculate_checksum(metadata.file_path)
            if current_checksum != metadata.checksum:
                results['checksum_mismatch'].append(state_id)
                continue

            results['valid'].append(state_id)

        logger.info(f"Library integrity check: {len(results['valid'])} valid, "
                   f"{len(results['missing_files'])} missing, "
                   f"{len(results['checksum_mismatch'])} corrupted")

        return results

    def export_library_info(self) -> Dict[str, Any]:
        """Export library information for external use.

        Returns:
            Dict with library statistics and state summaries
        """
        phase_counts = {}
        scenario_counts = {}
        difficulty_counts = {}

        for metadata in self.metadata.values():
            phase_counts[metadata.phase.value] = phase_counts.get(metadata.phase.value, 0) + 1
            scenario_counts[metadata.scenario.value] = scenario_counts.get(metadata.scenario.value, 0) + 1
            difficulty_counts[metadata.difficulty.value] = difficulty_counts.get(metadata.difficulty.value, 0) + 1

        return {
            'total_states': len(self.metadata),
            'library_path': str(self.library_path),
            'phases': phase_counts,
            'scenarios': scenario_counts,
            'difficulties': difficulty_counts,
            'total_size_mb': sum(m.file_size for m in self.metadata.values()) / (1024 * 1024),
            'most_used': sorted(
                [(m.id, m.usage_count, m.name) for m in self.metadata.values()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }


# Convenience functions for common operations

def create_library_from_current_state(library: SaveStateLibrary,
                                     current_state_path: str,
                                     name: str,
                                     description: str,
                                     scenario: TrainingScenario,
                                     difficulty: Difficulty = Difficulty.MEDIUM,
                                     **kwargs) -> str:
    """Create a library entry from the current save state file.

    This is a convenience function that can be used to quickly add
    commonly used save states to the library.
    """
    # Basic game state - would need to be extracted from actual save state
    # For now, using placeholder values
    game_state = GameState(
        player_name="Player",
        location="Unknown",
        map_id=0,
        badges=0,
        level=5,
        party_size=1,
        money=3000,
        playtime_hours=0,
        story_progress="Beginning"
    )

    phase = GamePhase.EARLY_GAME  # Default, could be inferred from game state

    return library.add_save_state(
        source_path=current_state_path,
        name=name,
        description=description,
        phase=phase,
        scenario=scenario,
        difficulty=difficulty,
        game_state=game_state,
        training_objectives=[f"Train for {scenario.value}"],
        **kwargs
    )