#!/usr/bin/env python3
"""
Test Suite for Temporal Memory System

Tests the temporal memory implementation that bridges LLM reasoning
with RL training through temporal state tracking and experience buffering.
"""

import pytest
import tempfile
import numpy as np
import time
import json
from unittest.mock import Mock, patch
from pathlib import Path
from collections import deque

from core.temporal_memory import (
    TemporalMemoryBuffer, TemporalState, TemporalExperience, Episode,
    DecisionSource
)
from core.experience_memory import ExperienceMemory, ExperienceEntry


class TestTemporalState:
    """Test the TemporalState data structure."""

    def test_temporal_state_creation(self):
        """Test basic TemporalState creation."""
        state = TemporalState(
            position=(10, 20),
            map_id=5,
            hp_ratio=0.8,
            level=15,
            badges=2,
            party_size=3,
            money=5000,
            screen_state="overworld",
            game_phase="early_game",
            location_type="route",
            in_battle=False,
            in_menu=False,
            timestep=1000,
            episode_time=45.5
        )

        assert state.position == (10, 20)
        assert state.map_id == 5
        assert state.hp_ratio == 0.8
        assert state.level == 15
        assert state.badges == 2
        assert state.party_size == 3
        assert state.money == 5000
        assert state.screen_state == "overworld"
        assert state.game_phase == "early_game"
        assert state.location_type == "route"
        assert not state.in_battle
        assert not state.in_menu
        assert state.timestep == 1000
        assert state.episode_time == 45.5

    def test_feature_vector_generation(self):
        """Test automatic feature vector generation."""
        state = TemporalState(
            position=(128, 80),
            map_id=1,
            hp_ratio=0.75,
            level=10,
            badges=1,
            party_size=2,
            money=1000,
            screen_state="overworld",
            game_phase="early_game",
            location_type="route",
            in_battle=False,
            in_menu=False,
            timestep=500,
            episode_time=30.0
        )

        # Feature vector should be generated automatically
        assert hasattr(state, 'feature_vector')
        assert isinstance(state.feature_vector, np.ndarray)
        assert state.feature_vector.dtype == np.float32
        assert len(state.feature_vector) > 0

        # Check some specific feature values (allowing for floating point precision)
        assert abs(state.feature_vector[0] - 128 / 255.0) < 0.001  # Normalized position X
        assert abs(state.feature_vector[1] - 80 / 255.0) < 0.001   # Normalized position Y
        assert abs(state.feature_vector[2] - 1 / 100.0) < 0.001    # Normalized map ID
        assert abs(state.feature_vector[3] - 0.75) < 0.001         # HP ratio
        assert abs(state.feature_vector[4] - 10 / 100.0) < 0.001   # Normalized level
        assert abs(state.feature_vector[5] - 1 / 16.0) < 0.001     # Normalized badges
        assert abs(state.feature_vector[6] - 2 / 6.0) < 0.001      # Normalized party size

    def test_battle_state_encoding(self):
        """Test battle and menu state encoding in feature vector."""
        battle_state = TemporalState(
            position=(0, 0), map_id=1, hp_ratio=0.5, level=5, badges=0,
            party_size=1, money=0, screen_state="battle", game_phase="tutorial",
            location_type="gym", in_battle=True, in_menu=False,
            timestep=0, episode_time=0
        )

        menu_state = TemporalState(
            position=(0, 0), map_id=1, hp_ratio=0.5, level=5, badges=0,
            party_size=1, money=0, screen_state="menu", game_phase="tutorial",
            location_type="gym", in_battle=False, in_menu=True,
            timestep=0, episode_time=0
        )

        # Battle flag should be encoded in feature vector
        battle_feature_idx = 8  # Based on implementation
        assert battle_state.feature_vector[battle_feature_idx] == 1.0

        # Menu flag should be encoded in feature vector
        menu_feature_idx = 9  # Based on implementation
        assert menu_state.feature_vector[menu_feature_idx] == 1.0


class TestTemporalExperience:
    """Test the TemporalExperience data structure."""

    def test_temporal_experience_creation(self):
        """Test creating TemporalExperience with basic data."""
        state = TemporalState(
            position=(10, 20), map_id=1, hp_ratio=1.0, level=1, badges=0,
            party_size=1, money=0, screen_state="overworld", game_phase="tutorial",
            location_type="town", in_battle=False, in_menu=False,
            timestep=0, episode_time=0
        )

        experience = TemporalExperience(
            state=state,
            action=1,  # Up
            reward=10.0,
            next_state=None,
            done=False,
            decision_source=DecisionSource.LLM,
            llm_reasoning="Move north to explore",
            llm_confidence=0.8
        )

        assert experience.state == state
        assert experience.action == 1
        assert experience.reward == 10.0
        assert experience.next_state is None
        assert not experience.done
        assert experience.decision_source == DecisionSource.LLM
        assert experience.llm_reasoning == "Move north to explore"
        assert experience.llm_confidence == 0.8
        assert isinstance(experience.timestamp, float)

    def test_experience_with_rl_decision(self):
        """Test TemporalExperience from RL decision."""
        state = TemporalState(
            position=(5, 5), map_id=2, hp_ratio=0.8, level=5, badges=1,
            party_size=2, money=500, screen_state="overworld", game_phase="early",
            location_type="route", in_battle=False, in_menu=False,
            timestep=100, episode_time=60
        )

        experience = TemporalExperience(
            state=state,
            action=5,  # A button
            reward=50.0,
            next_state=None,
            done=True,
            decision_source=DecisionSource.RL
        )

        assert experience.decision_source == DecisionSource.RL
        assert experience.llm_reasoning is None
        assert experience.llm_confidence is None


class TestEpisode:
    """Test the Episode data structure."""

    def test_episode_creation(self):
        """Test creating Episode with metadata."""
        episode = Episode(
            episode_id="test_episode_1",
            save_state_id="save_001",
            curriculum_level=2,
            experiences=[],
            total_reward=125.0,
            total_steps=50,
            success=True,
            duration=300.0,
            start_time=time.time(),
            metadata={"difficulty": "medium", "objective": "reach_gym"}
        )

        assert episode.episode_id == "test_episode_1"
        assert episode.save_state_id == "save_001"
        assert episode.curriculum_level == 2
        assert episode.experiences == []
        assert episode.total_reward == 125.0
        assert episode.total_steps == 50
        assert episode.success
        assert episode.duration == 300.0
        assert episode.metadata["difficulty"] == "medium"


class TestTemporalMemoryBuffer:
    """Test the TemporalMemoryBuffer functionality."""

    @pytest.fixture
    def base_memory(self):
        """Create base ExperienceMemory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_file = Path(tmpdir) / "test_memory.json"
            return ExperienceMemory(str(memory_file))

    @pytest.fixture
    def temporal_memory(self, base_memory):
        """Create TemporalMemoryBuffer for testing."""
        return TemporalMemoryBuffer(
            base_memory=base_memory,
            buffer_size=1000,
            episode_buffer_size=100
        )

    def test_initialization(self, temporal_memory):
        """Test TemporalMemoryBuffer initializes correctly."""
        assert temporal_memory.base_memory is not None
        assert isinstance(temporal_memory.experience_buffer, deque)
        assert isinstance(temporal_memory.episodes, deque)
        assert temporal_memory.current_episode is None
        assert temporal_memory.current_episode_experiences == []
        assert temporal_memory.temporal_window == 10  # Default value

    def test_start_episode(self, temporal_memory):
        """Test starting a new episode."""
        episode_id = temporal_memory.start_episode(
            save_state_id="test_save",
            curriculum_level=1,
            metadata={"test": True}
        )

        assert isinstance(episode_id, str)
        assert temporal_memory.current_episode is not None
        assert temporal_memory.current_episode.episode_id == episode_id
        assert temporal_memory.current_episode.save_state_id == "test_save"
        assert temporal_memory.current_episode.curriculum_level == 1
        assert temporal_memory.current_episode.metadata["test"]

    def test_record_temporal_experience(self, temporal_memory):
        """Test recording temporal experiences."""
        temporal_memory.start_episode()

        game_state = {
            'player_x': 10, 'player_y': 20, 'map_id': 1, 'hp_current': 80,
            'hp_max': 100, 'level': 5, 'badges': 0, 'party_size': 1,
            'money': 100, 'in_battle': False
        }
        screen_analysis = {'state': 'overworld'}

        temporal_memory.record_temporal_experience(
            game_state=game_state,
            screen_analysis=screen_analysis,
            action=1,  # Up
            reward=5.0,
            decision_source=DecisionSource.LLM,
            llm_reasoning="Exploring north",
            llm_confidence=0.9
        )

        # Check experience was recorded
        assert len(temporal_memory.current_episode_experiences) == 1
        assert len(temporal_memory.experience_buffer) == 1
        assert temporal_memory.current_episode.total_reward == 5.0
        assert temporal_memory.current_episode.total_steps == 1

        experience = temporal_memory.current_episode_experiences[0]
        assert experience.action == 1
        assert experience.reward == 5.0
        assert experience.decision_source == DecisionSource.LLM
        assert experience.llm_reasoning == "Exploring north"
        assert experience.llm_confidence == 0.9

    def test_end_episode(self, temporal_memory):
        """Test ending an episode."""
        temporal_memory.start_episode()

        # Add some experiences
        for i in range(3):
            temporal_memory.record_temporal_experience(
                game_state={'player_x': i, 'player_y': i, 'map_id': 1, 'hp_current': 100,
                           'hp_max': 100, 'level': 1, 'badges': 0, 'party_size': 1,
                           'money': 0, 'in_battle': False},
                screen_analysis={'state': 'overworld'},
                action=i,
                reward=10.0,
                decision_source=DecisionSource.LLM
            )

        completed_episode = temporal_memory.end_episode(success=True)

        assert completed_episode is not None
        assert completed_episode.success
        assert len(completed_episode.experiences) == 3
        assert completed_episode.total_reward == 30.0
        assert completed_episode.total_steps == 3
        assert len(temporal_memory.episodes) == 1
        assert temporal_memory.current_episode is None

    def test_get_rl_batch(self, temporal_memory):
        """Test getting RL training batch."""
        temporal_memory.start_episode()

        # Add enough experiences for a batch
        for i in range(35):
            temporal_memory.record_temporal_experience(
                game_state={'player_x': i, 'player_y': i, 'map_id': 1, 'hp_current': 100,
                           'hp_max': 100, 'level': 1, 'badges': 0, 'party_size': 1,
                           'money': 0, 'in_battle': False},
                screen_analysis={'state': 'overworld'},
                action=i % 8,
                reward=float(i),
                decision_source=DecisionSource.LLM if i % 2 == 0 else DecisionSource.RL
            )

        batch = temporal_memory.get_rl_batch(batch_size=32)

        assert batch is not None
        assert 'states' in batch
        assert 'actions' in batch
        assert 'rewards' in batch
        assert 'next_states' in batch
        assert 'dones' in batch
        assert 'decision_sources' in batch
        assert 'llm_confidences' in batch

        assert batch['states'].shape == (32, len(temporal_memory.experience_buffer[0].state.feature_vector))
        assert batch['actions'].shape == (32,)
        assert batch['rewards'].shape == (32,)
        assert len(batch['decision_sources']) == 32

    def test_get_temporal_stats(self, temporal_memory):
        """Test getting temporal memory statistics."""
        # Create some episodes first
        for episode_num in range(3):
            temporal_memory.start_episode()
            for i in range(5):
                temporal_memory.record_temporal_experience(
                    game_state={'player_x': i, 'player_y': i, 'map_id': 1, 'hp_current': 100,
                               'hp_max': 100, 'level': 1, 'badges': 0, 'party_size': 1,
                               'money': 0, 'in_battle': False},
                    screen_analysis={'state': 'overworld'},
                    action=i,
                    reward=5.0 if episode_num > 0 else -1.0,
                    decision_source=DecisionSource.LLM
                )
            temporal_memory.end_episode(success=episode_num > 0)

        stats = temporal_memory.get_temporal_stats()

        assert 'temporal' in stats
        temporal_stats = stats['temporal']
        assert temporal_stats['total_episodes'] == 3
        assert temporal_stats['successful_episodes'] == 2
        assert temporal_stats['success_rate'] == 2/3
        assert temporal_stats['total_experiences'] == 15


class TestDecisionSource:
    """Test the DecisionSource enum."""

    def test_decision_source_values(self):
        """Test DecisionSource enum values."""
        assert DecisionSource.LLM.value == "llm"
        assert DecisionSource.RL.value == "rl"
        assert DecisionSource.RULE_BASED.value == "rule_based"
        assert DecisionSource.HYBRID.value == "hybrid"

    def test_decision_source_in_experience(self):
        """Test using DecisionSource in TemporalExperience."""
        state = TemporalState(
            position=(0, 0), map_id=1, hp_ratio=1.0, level=1, badges=0,
            party_size=1, money=0, screen_state="overworld", game_phase="tutorial",
            location_type="town", in_battle=False, in_menu=False,
            timestep=0, episode_time=0
        )

        llm_experience = TemporalExperience(
            state=state, action=0, reward=0.0, next_state=None,
            done=False, decision_source=DecisionSource.LLM
        )

        rl_experience = TemporalExperience(
            state=state, action=1, reward=0.0, next_state=None,
            done=False, decision_source=DecisionSource.RL
        )

        assert llm_experience.decision_source == DecisionSource.LLM
        assert rl_experience.decision_source == DecisionSource.RL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])