#!/usr/bin/env python3
"""
Test Suite for Training Strategic Context Builder

Tests the actual StrategicContextBuilder used in training.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from training.components.strategic_context_builder import StrategicContextBuilder


class TestTrainingStrategicContextBuilder:
    """Test the training strategic context builder implementation"""

    @pytest.fixture
    def context_builder(self):
        """Create StrategicContextBuilder instance"""
        return StrategicContextBuilder()

    @pytest.fixture
    def sample_game_state(self):
        """Create sample game state data"""
        return {
            'player_x': 10,
            'player_y': 15,
            'player_map': 1,
            'hp_current': 45,
            'hp_max': 50,
            'level': 5,
            'party_size': 1,
            'badges': 0,
            'money': 3000
        }

    def test_initialization(self, context_builder):
        """Test StrategicContextBuilder initializes correctly"""
        assert context_builder is not None
        assert hasattr(context_builder, 'location_database')
        assert hasattr(context_builder, 'important_locations')
        assert isinstance(context_builder.important_locations, dict)

    def test_build_enhanced_context_basic(self, context_builder, sample_game_state):
        """Test basic enhanced context building"""
        context = context_builder.build_enhanced_context(sample_game_state, action_count=10)

        assert isinstance(context, dict)
        # Check for expected keys in the enhanced context
        assert any(key in context for key in ['situation', 'position', 'progress', 'objective'])

    def test_build_enhanced_context_with_rewards(self, context_builder, sample_game_state):
        """Test enhanced context building with reward history"""
        recent_rewards = [1.0, 0.5, -0.2, 0.8]

        context = context_builder.build_enhanced_context(
            sample_game_state,
            action_count=20,
            recent_rewards=recent_rewards
        )

        assert isinstance(context, dict)

    def test_get_current_quest_objectives(self, context_builder, sample_game_state):
        """Test quest objectives retrieval"""
        objectives = context_builder.get_current_quest_objectives(sample_game_state)

        assert isinstance(objectives, list)
        # Should return a list even if empty

    def test_important_locations_attribute(self, context_builder):
        """Test that important_locations attribute exists and works"""
        # This was the critical bug we fixed
        assert hasattr(context_builder, 'important_locations')
        assert isinstance(context_builder.important_locations, dict)

        # Should be able to access without AttributeError
        locations = context_builder.important_locations
        assert locations is not None

    def test_location_lookup(self, context_builder):
        """Test location lookup functionality"""
        # Test that we can look up locations without errors
        map_id = 1
        location_name = context_builder.important_locations.get(map_id, "Unknown")
        assert isinstance(location_name, str)

    def test_context_with_different_map_ids(self, context_builder):
        """Test enhanced context building with different map IDs"""
        test_states = [
            {'player_map': 0, 'player_x': 5, 'player_y': 5},
            {'player_map': 1, 'player_x': 10, 'player_y': 10},
            {'player_map': 2, 'player_x': 15, 'player_y': 15},
        ]

        for state in test_states:
            context = context_builder.build_enhanced_context(state, action_count=5)
            assert isinstance(context, dict)

    def test_navigation_advice(self, context_builder, sample_game_state):
        """Test navigation advice functionality"""
        advice = context_builder.get_navigation_advice(sample_game_state)

        assert isinstance(advice, dict)
        # Should return navigation advice structure

    def test_no_attribute_errors(self, context_builder, sample_game_state):
        """Test that no AttributeError is raised during normal operation"""
        try:
            context = context_builder.build_enhanced_context(sample_game_state, action_count=10)
            objectives = context_builder.get_current_quest_objectives(sample_game_state)

            # Access the important_locations attribute that was causing issues
            _ = context_builder.important_locations

        except AttributeError as e:
            pytest.fail(f"AttributeError raised: {e}")

    def test_handles_missing_game_state_fields(self, context_builder):
        """Test graceful handling of missing game state fields"""
        minimal_state = {'player_x': 0, 'player_y': 0}

        # Should not crash even with minimal state
        context = context_builder.build_enhanced_context(minimal_state, action_count=1)
        assert isinstance(context, dict)

    def test_empty_reward_history(self, context_builder, sample_game_state):
        """Test handling of empty reward history"""
        context = context_builder.build_enhanced_context(
            sample_game_state,
            action_count=5,
            recent_rewards=[]
        )

        assert isinstance(context, dict)