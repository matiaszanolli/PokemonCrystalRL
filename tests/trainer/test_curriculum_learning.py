#!/usr/bin/env python3
"""
Test Suite for Curriculum Learning System

Tests the curriculum learning implementation including CurriculumManager,
save state selection, progression logic, and integration with training.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from training.curriculum_learning import (
    CurriculumManager, CurriculumStage, CurriculumLevel, CurriculumProgress,
    create_default_curriculum_config
)
from core.save_state_library import (
    SaveStateLibrary, SaveStateMetadata, GameState, GamePhase,
    TrainingScenario, Difficulty
)


class TestCurriculumManager:
    """Test the CurriculumManager implementation."""

    @pytest.fixture
    def temp_library(self):
        """Create a temporary save state library for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = SaveStateLibrary(temp_dir)

            # Add some test save states
            game_state_tutorial = GameState(
                player_name="TestPlayer",
                location="New Bark Town",
                map_id=1,
                badges=0,
                level=5,
                party_size=1,
                money=3000,
                playtime_hours=0,
                story_progress="Beginning"
            )

            game_state_gym = GameState(
                player_name="TestPlayer",
                location="Violet City",
                map_id=10,
                badges=1,
                level=15,
                party_size=2,
                money=5000,
                playtime_hours=2,
                story_progress="First Gym"
            )

            # Create dummy save state files
            tutorial_save = Path(temp_dir) / "tutorial.state"
            gym_save = Path(temp_dir) / "gym.state"
            tutorial_save.write_bytes(b"fake_tutorial_data")
            gym_save.write_bytes(b"fake_gym_data")

            # Add states to library
            library.add_save_state(
                source_path=str(tutorial_save),
                name="Tutorial Start",
                description="Beginning of the game",
                phase=GamePhase.TUTORIAL,
                scenario=TrainingScenario.FIRST_POKEMON,
                difficulty=Difficulty.EASY,
                game_state=game_state_tutorial,
                training_objectives=["Learn basic movement", "Get first Pokemon"],
                tags=["beginner", "movement"]
            )

            library.add_save_state(
                source_path=str(gym_save),
                name="First Gym Battle",
                description="Ready for first gym",
                phase=GamePhase.EARLY_GAME,
                scenario=TrainingScenario.GYM_BATTLE,
                difficulty=Difficulty.MEDIUM,
                game_state=game_state_gym,
                training_objectives=["Battle strategy", "Type advantages"]
            )

            yield library

    @pytest.fixture
    def curriculum_manager(self, temp_library):
        """Create CurriculumManager instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_file = Path(temp_dir) / "test_progress.json"
            manager = CurriculumManager(
                library=temp_library,
                progress_file=str(progress_file)
            )
            yield manager

    def test_initialization(self, curriculum_manager):
        """Test CurriculumManager initializes correctly."""
        assert curriculum_manager is not None
        assert curriculum_manager.library is not None
        assert curriculum_manager.curriculum is not None
        assert len(curriculum_manager.curriculum) == 5  # Default curriculum levels
        assert curriculum_manager.progress is not None

    def test_default_curriculum_structure(self, curriculum_manager):
        """Test default curriculum has proper structure."""
        curriculum = curriculum_manager.curriculum

        # Check stages progress from tutorial to expert
        expected_stages = [
            CurriculumStage.TUTORIAL,
            CurriculumStage.BASIC_MECHANICS,
            CurriculumStage.INTERMEDIATE,
            CurriculumStage.ADVANCED,
            CurriculumStage.EXPERT
        ]

        for i, expected_stage in enumerate(expected_stages):
            assert curriculum[i].stage == expected_stage

        # Check difficulty progression
        difficulties = [level.difficulty for level in curriculum]
        assert difficulties[0] == Difficulty.EASY  # Tutorial
        assert difficulties[-1] == Difficulty.HARD  # Expert

    def test_get_current_level(self, curriculum_manager):
        """Test getting current curriculum level."""
        # Should start at level 0
        current_level = curriculum_manager.get_current_level()
        assert current_level.stage == CurriculumStage.TUTORIAL
        assert current_level.difficulty == Difficulty.EASY

        # Advance progress and test
        curriculum_manager.progress.current_level = 2
        current_level = curriculum_manager.get_current_level()
        assert current_level.stage == CurriculumStage.INTERMEDIATE

    def test_get_current_stage(self, curriculum_manager):
        """Test getting current curriculum stage."""
        assert curriculum_manager.get_current_stage() == CurriculumStage.TUTORIAL

        curriculum_manager.progress.current_level = 1
        assert curriculum_manager.get_current_stage() == CurriculumStage.BASIC_MECHANICS

    def test_select_save_state(self, curriculum_manager):
        """Test save state selection based on curriculum level."""
        # Start at tutorial level - should select tutorial save state
        selected = curriculum_manager.select_save_state()
        assert selected is not None
        assert selected.name == "Tutorial Start"
        assert selected.scenario == TrainingScenario.FIRST_POKEMON

        # Advance to intermediate level
        curriculum_manager.progress.current_level = 2
        # This should find the gym battle state since it matches scenario
        # (though filtering might be strict)

    def test_record_episode_result_success(self, curriculum_manager):
        """Test recording successful episode results."""
        initial_level = curriculum_manager.progress.current_level

        # Record multiple successful episodes
        for i in range(5):
            advanced = curriculum_manager.record_episode_result(success=True, reward=100.0)
            # Shouldn't advance until min_episodes reached with good success rate
            if i < 4:  # Before minimum episodes
                assert not advanced

        # Check progress tracking
        assert curriculum_manager.progress.level_episodes == 5
        assert curriculum_manager.progress.level_successes == 5
        assert curriculum_manager.progress.success_rate == 1.0

    def test_record_episode_result_advancement(self, curriculum_manager):
        """Test curriculum level advancement."""
        initial_level = curriculum_manager.progress.current_level

        # Simulate reaching advancement criteria
        level = curriculum_manager.get_current_level()

        # Record enough successful episodes
        for i in range(level.min_episodes):
            advanced = curriculum_manager.record_episode_result(success=True, reward=100.0)

        # Should advance now
        assert advanced
        assert curriculum_manager.progress.current_level == initial_level + 1

    def test_record_episode_result_forced_advancement(self, curriculum_manager):
        """Test forced advancement at max episodes."""
        level = curriculum_manager.get_current_level()
        initial_level = curriculum_manager.progress.current_level

        # Record max episodes with poor success rate
        for i in range(level.max_episodes):
            # Mix of success/failure to keep success rate low
            success = i % 3 == 0  # 33% success rate
            advanced = curriculum_manager.record_episode_result(success=success, reward=50.0)

        # Should force advancement despite poor success rate
        assert advanced
        assert curriculum_manager.progress.current_level == initial_level + 1

    def test_get_curriculum_status(self, curriculum_manager):
        """Test curriculum status reporting."""
        status = curriculum_manager.get_curriculum_status()

        # Check required fields
        assert 'current_level' in status
        assert 'total_levels' in status
        assert 'current_stage' in status
        assert 'current_scenarios' in status
        assert 'current_difficulty' in status
        assert 'level_progress' in status
        assert 'overall_progress' in status
        assert 'advancement_status' in status

        # Check initial values
        assert status['current_level'] == 0
        assert status['current_stage'] == 'tutorial'
        assert status['total_levels'] == 5

    def test_advancement_status(self, curriculum_manager):
        """Test advancement status calculation."""
        status = curriculum_manager.get_curriculum_status()
        advancement = status['advancement_status']

        # Initially should not be ready to advance
        assert not advancement['ready_to_advance']
        assert advancement['episodes_until_eligible'] > 0
        assert not advancement['is_final_level']

    def test_progress_persistence(self, temp_library):
        """Test that curriculum progress persists between sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_file = Path(temp_dir) / "persist_progress.json"

            # Create first manager and advance progress
            manager1 = CurriculumManager(
                library=temp_library,
                progress_file=str(progress_file)
            )

            # Advance to level 1
            for _ in range(5):
                manager1.record_episode_result(success=True, reward=100.0)

            # Create second manager with same progress file
            manager2 = CurriculumManager(
                library=temp_library,
                progress_file=str(progress_file)
            )

            # Should load previous progress
            assert manager2.progress.current_level == manager1.progress.current_level
            assert manager2.progress.total_episodes == manager1.progress.total_episodes

    def test_curriculum_config_export(self, curriculum_manager):
        """Test exporting curriculum configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name

        try:
            curriculum_manager.export_curriculum_config(export_path)

            # Verify file was created and has valid JSON
            assert os.path.exists(export_path)

            with open(export_path, 'r') as f:
                config_data = json.load(f)

            assert 'description' in config_data
            assert 'levels' in config_data
            assert len(config_data['levels']) == 5

            # Check first level structure
            first_level = config_data['levels'][0]
            assert first_level['stage'] == 'tutorial'
            assert first_level['difficulty'] == 'easy'
            assert 'scenarios' in first_level

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_curriculum_config_loading(self, temp_library):
        """Test loading custom curriculum configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

            # Create custom config
            custom_config = {
                "description": "Custom Test Curriculum",
                "levels": [
                    {
                        "stage": "tutorial",
                        "scenarios": ["exploration"],
                        "difficulty": "easy",
                        "min_success_rate": 0.5,
                        "min_episodes": 3,
                        "max_episodes": 10,
                        "phase_filter": "tutorial",
                        "badges_range": None,
                        "tags": ["test"]
                    }
                ]
            }

            json.dump(custom_config, f)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                progress_file = Path(temp_dir) / "test_progress.json"

                manager = CurriculumManager(
                    library=temp_library,
                    curriculum_config=config_path,
                    progress_file=str(progress_file)
                )

                # Should load custom curriculum
                assert len(manager.curriculum) == 1
                assert manager.curriculum[0].min_episodes == 3
                assert manager.curriculum[0].tags == ["test"]

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestCurriculumDataStructures:
    """Test curriculum learning data structures."""

    def test_curriculum_level_creation(self):
        """Test CurriculumLevel creation and attributes."""
        level = CurriculumLevel(
            stage=CurriculumStage.INTERMEDIATE,
            scenarios=[TrainingScenario.GYM_BATTLE],
            difficulty=Difficulty.MEDIUM,
            min_success_rate=0.7,
            min_episodes=10,
            max_episodes=30
        )

        assert level.stage == CurriculumStage.INTERMEDIATE
        assert TrainingScenario.GYM_BATTLE in level.scenarios
        assert level.difficulty == Difficulty.MEDIUM
        assert level.min_success_rate == 0.7

    def test_curriculum_progress_initialization(self):
        """Test CurriculumProgress initialization."""
        progress = CurriculumProgress()

        assert progress.current_level == 0
        assert progress.total_episodes == 0
        assert progress.level_episodes == 0
        assert progress.level_successes == 0
        assert progress.success_rate == 0.0
        assert progress.stage_history == []

    def test_curriculum_stages_enum(self):
        """Test CurriculumStage enum values."""
        stages = list(CurriculumStage)
        expected = ['tutorial', 'basic_mechanics', 'intermediate', 'advanced', 'expert']

        for stage, expected_value in zip(stages, expected):
            assert stage.value == expected_value


class TestCurriculumIntegration:
    """Test curriculum learning integration with other systems."""

    def test_create_default_config_function(self):
        """Test standalone config creation function."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            create_default_curriculum_config(config_path)

            assert os.path.exists(config_path)

            with open(config_path, 'r') as f:
                config = json.load(f)

            assert 'levels' in config
            assert len(config['levels']) > 0

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    @patch('training.curriculum_learning.SaveStateLibrary')
    def test_curriculum_manager_with_mock_library(self, mock_library_class):
        """Test CurriculumManager with mocked save state library."""
        # Setup mock library
        mock_library = Mock()
        mock_library_class.return_value = mock_library

        # Mock save state for selection
        mock_save_state = Mock()
        mock_save_state.name = "Test State"
        mock_save_state.scenario = TrainingScenario.EXPLORATION
        mock_library.get_recommendations.return_value = [mock_save_state]

        with tempfile.TemporaryDirectory() as temp_dir:
            progress_file = Path(temp_dir) / "mock_progress.json"

            manager = CurriculumManager(
                library=mock_library,
                progress_file=str(progress_file)
            )

            # Test save state selection
            selected = manager.select_save_state()
            assert selected == mock_save_state

            # Verify library was called correctly
            mock_library.get_recommendations.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])