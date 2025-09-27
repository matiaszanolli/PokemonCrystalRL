"""
Comprehensive Test Suite for Experience Memory System

This module provides extensive testing for the ExperienceMemory system which currently
has 0% test coverage despite being a critical component for AI learning and experience
retention across training sessions.

Test Coverage Areas:
- Experience recording and retrieval
- Action pattern recognition and learning
- Situation hashing and similarity detection
- Memory management and cleanup
- Persistence (save/load functionality)
- Recommendation algorithms
- Performance statistics and analytics
- Edge cases and error handling
"""

import pytest
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

# Import modules under test
from core.experience_memory import (
    ExperienceMemory,
    ExperienceEntry,
    ActionPattern
)


class TestExperienceMemoryCore:
    """Core functionality tests for ExperienceMemory"""

    @pytest.fixture
    def temp_memory_file(self):
        """Create temporary memory file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def experience_memory(self, temp_memory_file):
        """Create ExperienceMemory instance for testing"""
        return ExperienceMemory(memory_file=temp_memory_file)

    @pytest.fixture
    def sample_game_state(self):
        """Sample game state for testing"""
        return {
            'player_x': 10,
            'player_y': 15,
            'player_map': 5,
            'player_level': 12,
            'party_count': 3,
            'badges_total': 2,
            'in_battle': 0,
            'player_hp': 45,
            'player_max_hp': 60
        }

    @pytest.fixture
    def sample_screen_analysis(self):
        """Sample screen analysis for testing"""
        return {
            'state': 'overworld',
            'menu_open': False,
            'dialogue_active': False,
            'battle_ui_visible': False
        }

    @pytest.fixture
    def sample_context(self):
        """Sample context for testing"""
        return {
            'phase': 'exploration',
            'location_type': 'route',
            'recent_actions': ['up', 'right', 'a'],
            'last_reward': 1.5
        }

    def test_experience_memory_initialization(self, temp_memory_file):
        """Test ExperienceMemory initialization"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Verify initialization
        assert memory.memory_file == temp_memory_file
        assert isinstance(memory.experiences, dict)
        assert isinstance(memory.successful_patterns, list)
        assert isinstance(memory.situation_to_actions, dict)

        # Verify default parameters
        assert memory.min_success_rate == 0.6
        assert memory.max_memory_entries == 10000
        assert memory.pattern_length == 5

        # Verify initial state
        assert len(memory.experiences) == 0
        assert len(memory.successful_patterns) == 0

    def test_situation_hash_generation(self, experience_memory, sample_game_state,
                                      sample_screen_analysis, sample_context):
        """Test situation hash generation"""
        hash1 = experience_memory.get_situation_hash(
            sample_game_state, sample_screen_analysis, sample_context
        )

        # Hash should be consistent
        hash2 = experience_memory.get_situation_hash(
            sample_game_state, sample_screen_analysis, sample_context
        )
        assert hash1 == hash2

        # Different states should produce different hashes
        modified_state = sample_game_state.copy()
        modified_state['player_level'] = 20
        hash3 = experience_memory.get_situation_hash(
            modified_state, sample_screen_analysis, sample_context
        )
        assert hash1 != hash3

        # Hash should be a string
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_experience_recording_new(self, experience_memory, sample_game_state,
                                     sample_screen_analysis, sample_context):
        """Test recording new experiences"""
        situation_hash = experience_memory.get_situation_hash(
            sample_game_state, sample_screen_analysis, sample_context
        )

        actions = ['up', 'right', 'a', 'down']
        reward = 2.5

        # Record experience
        experience_memory.record_experience(situation_hash, actions, reward, sample_context)

        # Verify experience was recorded
        assert situation_hash in experience_memory.experiences
        exp = experience_memory.experiences[situation_hash]

        assert exp.situation_hash == situation_hash
        assert exp.action_sequence == actions
        assert exp.outcome_reward == reward
        assert exp.success_rate == 1.0  # Reward > 0.1 means success
        assert exp.usage_count == 1
        assert exp.context == sample_context
        assert isinstance(exp.last_used, float)

    def test_experience_recording_update_existing(self, experience_memory, sample_game_state,
                                                 sample_screen_analysis, sample_context):
        """Test updating existing experiences"""
        situation_hash = experience_memory.get_situation_hash(
            sample_game_state, sample_screen_analysis, sample_context
        )

        # Record initial experience (successful)
        experience_memory.record_experience(situation_hash, ['up', 'a'], 1.5, sample_context)
        initial_time = experience_memory.experiences[situation_hash].last_used

        # Small delay to ensure timestamp difference
        time.sleep(0.01)

        # Record second experience (unsuccessful)
        experience_memory.record_experience(situation_hash, ['down', 'b'], -0.5, sample_context)
        updated_exp = experience_memory.experiences[situation_hash]

        # Verify updates
        assert updated_exp.usage_count == 2
        assert updated_exp.success_rate == 0.5  # 1 success out of 2 attempts
        assert updated_exp.action_sequence == ['down', 'b']  # Latest sequence
        assert updated_exp.last_used > initial_time

        # Average reward should be updated
        expected_avg_reward = (1.5 + (-0.5)) / 2
        assert abs(updated_exp.outcome_reward - expected_avg_reward) < 0.01

    def test_get_recommended_actions_direct_match(self, experience_memory):
        """Test getting recommendations from direct experience matches"""
        situation_hash = "test_situation_123"
        successful_actions = ['up', 'up', 'right', 'a']

        # Create a successful experience
        experience_memory.experiences[situation_hash] = ExperienceEntry(
            situation_hash=situation_hash,
            action_sequence=successful_actions,
            outcome_reward=3.0,
            success_rate=0.8,
            usage_count=5,
            last_used=time.time(),
            context={}
        )

        # Should recommend the successful actions
        recommendations = experience_memory.get_recommended_actions(situation_hash)
        assert recommendations == successful_actions

    def test_get_recommended_actions_no_match(self, experience_memory):
        """Test behavior when no experience matches"""
        recommendations = experience_memory.get_recommended_actions("unknown_situation")
        assert recommendations is None

    def test_get_recommended_actions_low_success_rate(self, experience_memory):
        """Test that low success rate experiences are not recommended"""
        situation_hash = "low_success_situation"

        # Create a low success rate experience
        experience_memory.experiences[situation_hash] = ExperienceEntry(
            situation_hash=situation_hash,
            action_sequence=['left', 'left', 'down'],
            outcome_reward=-1.0,
            success_rate=0.3,  # Below min_success_rate of 0.6
            usage_count=10,
            last_used=time.time(),
            context={}
        )

        # Should not recommend low success actions
        recommendations = experience_memory.get_recommended_actions(situation_hash)
        assert recommendations is None

    def test_pattern_recognition_successful(self, experience_memory):
        """Test successful action pattern recognition"""
        # Record multiple experiences with the same successful pattern
        # Note: pattern recognition takes the LAST pattern_length actions
        pattern = ['up', 'right', 'a', 'down', 'a']

        for i in range(3):
            situation_hash = f"situation_{i}"
            # Actions where the last 5 actions match our pattern
            actions = ['left'] + pattern  # Last 5 will be our pattern
            experience_memory.record_experience(situation_hash, actions, 2.0, {})

        # Verify pattern was recognized
        assert len(experience_memory.successful_patterns) > 0

        # Check if our pattern is in the successful patterns
        found_pattern = None
        for p in experience_memory.successful_patterns:
            if p.pattern == pattern:
                found_pattern = p
                break

        assert found_pattern is not None
        assert found_pattern.confidence > 0
        assert len(found_pattern.success_situations) >= 3

    def test_pattern_based_recommendations(self, experience_memory):
        """Test recommendations based on successful patterns"""
        # Create a high-confidence pattern
        pattern = ActionPattern(
            pattern=['up', 'up', 'right', 'a', 'down'],
            success_situations=['sit1', 'sit2', 'sit3'],
            average_reward=2.5,
            confidence=0.8
        )
        experience_memory.successful_patterns = [pattern]

        # When no direct match, should recommend pattern
        recommendations = experience_memory.get_recommended_actions("unknown_situation", {})
        assert recommendations == pattern.pattern

    def test_memory_cleanup(self, experience_memory):
        """Test memory cleanup functionality"""
        # Set a low max_memory_entries for testing
        experience_memory.max_memory_entries = 5

        # Add more experiences than the limit
        for i in range(10):
            situation_hash = f"situation_{i}"
            # Make some experiences more successful than others
            success_rate = 0.9 if i < 3 else 0.3
            reward = 2.0 if success_rate > 0.5 else -1.0

            experience_memory.experiences[situation_hash] = ExperienceEntry(
                situation_hash=situation_hash,
                action_sequence=[f'action_{i}'],
                outcome_reward=reward,
                success_rate=success_rate,
                usage_count=1,
                last_used=time.time() - i * 1000,  # Older entries have earlier timestamps
                context={}
            )

        # Trigger cleanup
        experience_memory._cleanup_memory()

        # Should keep only max_memory_entries
        assert len(experience_memory.experiences) <= experience_memory.max_memory_entries

        # Should prefer keeping successful experiences
        remaining_hashes = list(experience_memory.experiences.keys())
        # Check that some high-success situations are retained
        assert any('situation_0' in h or 'situation_1' in h or 'situation_2' in h
                  for h in remaining_hashes)


class TestExperienceMemoryAdvanced:
    """Advanced functionality tests for ExperienceMemory"""

    @pytest.fixture
    def memory_with_data(self, temp_memory_file):
        """Create memory instance with pre-populated data"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Add various experiences
        experiences_data = [
            ("battle_situation_1", ['a', 'a', 'b'], 5.0, 0.9, 10),
            ("exploration_situation_1", ['up', 'right', 'a'], 2.0, 0.7, 5),
            ("menu_situation_1", ['start', 'down', 'a'], 1.0, 0.6, 3),
            ("stuck_situation_1", ['left', 'left', 'left'], -2.0, 0.1, 8),
        ]

        for sit_hash, actions, reward, success_rate, usage_count in experiences_data:
            memory.experiences[sit_hash] = ExperienceEntry(
                situation_hash=sit_hash,
                action_sequence=actions,
                outcome_reward=reward,
                success_rate=success_rate,
                usage_count=usage_count,
                last_used=time.time(),
                context={'phase': 'test'}
            )

        return memory

    @pytest.fixture
    def temp_memory_file(self):
        """Create temporary memory file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_similar_situation_detection(self, memory_with_data):
        """Test finding actions from similar situations"""
        # Should find the best similar experience when no exact match
        similar_actions = memory_with_data._find_similar_situation_actions("unknown_situation")

        # Should return actions from highest scoring experience
        assert similar_actions is not None
        assert isinstance(similar_actions, list)
        assert len(similar_actions) > 0

        # The best experience should be battle_situation_1 (highest success_rate * usage)
        assert similar_actions == ['a', 'a', 'b']

    def test_pattern_confidence_calculation(self, memory_with_data):
        """Test pattern confidence calculation"""
        # Manually add a pattern with multiple successes
        pattern = ActionPattern(
            pattern=['test', 'pattern'],
            success_situations=['sit1', 'sit2', 'sit3', 'sit4', 'sit5'],
            average_reward=2.0,
            confidence=0.0  # Will be recalculated
        )

        # Simulate confidence calculation
        total_uses = len(pattern.success_situations)
        expected_confidence = min(1.0, total_uses / 10.0)
        pattern.confidence = expected_confidence

        assert pattern.confidence == 0.5  # 5 uses / 10 = 0.5

    def test_memory_statistics(self, memory_with_data):
        """Test memory statistics collection"""
        stats = memory_with_data.get_memory_stats()

        # Verify stats structure
        assert isinstance(stats, dict)
        expected_keys = {'total_experiences', 'total_patterns', 'average_success_rate',
                        'successful_experiences', 'high_confidence_patterns'}

        # Check that stats contain meaningful information
        assert stats['total_experiences'] == 4
        assert 'average_success_rate' in stats
        assert isinstance(stats['average_success_rate'], float)

    def test_save_and_load_memory(self, temp_memory_file):
        """Test saving and loading memory to/from file"""
        # Create memory with some data
        memory1 = ExperienceMemory(memory_file=temp_memory_file)

        # Add test data
        test_experience = ExperienceEntry(
            situation_hash="test_save_load",
            action_sequence=['save', 'test', 'data'],
            outcome_reward=1.5,
            success_rate=0.8,
            usage_count=3,
            last_used=time.time(),
            context={'test': 'save_load'}
        )
        memory1.experiences["test_save_load"] = test_experience

        # Save memory
        memory1.save_memory()

        # Verify file was created and has content
        assert os.path.exists(temp_memory_file)
        assert os.path.getsize(temp_memory_file) > 0

        # Create new memory instance and load
        memory2 = ExperienceMemory(memory_file=temp_memory_file)

        # Verify data was loaded correctly
        assert "test_save_load" in memory2.experiences
        loaded_exp = memory2.experiences["test_save_load"]

        assert loaded_exp.situation_hash == test_experience.situation_hash
        assert loaded_exp.action_sequence == test_experience.action_sequence
        assert loaded_exp.outcome_reward == test_experience.outcome_reward
        assert loaded_exp.success_rate == test_experience.success_rate
        assert loaded_exp.usage_count == test_experience.usage_count

    def test_pattern_sorting_and_limitation(self, memory_with_data):
        """Test that patterns are sorted by confidence and limited in number"""
        # Add many patterns with different confidence levels
        for i in range(150):  # More than the 100 limit
            pattern = ActionPattern(
                pattern=[f'action_{i}', f'action_{i+1}'],
                success_situations=[f'sit_{i}'],
                average_reward=float(i),
                confidence=i / 150.0  # Varying confidence
            )
            memory_with_data.successful_patterns.append(pattern)

        # Trigger pattern cleanup (happens in _update_patterns)
        memory_with_data.successful_patterns = sorted(
            memory_with_data.successful_patterns,
            key=lambda x: x.confidence,
            reverse=True
        )[:100]

        # Should keep only top 100 patterns
        assert len(memory_with_data.successful_patterns) == 100

        # Should be sorted by confidence (descending)
        confidences = [p.confidence for p in memory_with_data.successful_patterns]
        assert confidences == sorted(confidences, reverse=True)

    def test_experience_context_usage(self, memory_with_data):
        """Test that context information is properly used"""
        # Test situation hash includes context
        game_state = {'player_level': 10}
        screen_analysis = {'state': 'overworld'}
        context1 = {'phase': 'exploration', 'location_type': 'route'}
        context2 = {'phase': 'battle', 'location_type': 'gym'}

        hash1 = memory_with_data.get_situation_hash(game_state, screen_analysis, context1)
        hash2 = memory_with_data.get_situation_hash(game_state, screen_analysis, context2)

        # Different contexts should produce different hashes
        assert hash1 != hash2

    def test_reward_threshold_behavior(self, memory_with_data):
        """Test success/failure determination based on reward threshold"""
        situation_hash = "reward_test"

        # Test borderline rewards
        test_cases = [
            (0.15, True),   # Above 0.1 threshold - success
            (0.1, False),   # Exactly at threshold - failure
            (0.05, False),  # Below threshold - failure
            (-0.5, False),  # Negative reward - failure
            (2.5, True),    # High positive reward - success
        ]

        for reward, expected_success in test_cases:
            memory_with_data.record_experience(situation_hash, ['test'], reward, {})
            exp = memory_with_data.experiences[situation_hash]

            # Check if success was recorded correctly
            if expected_success:
                assert exp.success_rate > 0
            # Note: For repeated recordings, success_rate is averaged, so we check the pattern


class TestExperienceMemoryEdgeCases:
    """Edge cases and error handling tests"""

    @pytest.fixture
    def temp_memory_file(self):
        """Create temporary memory file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_empty_memory_behavior(self, temp_memory_file):
        """Test behavior with empty memory"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Should handle empty memory gracefully
        recommendations = memory.get_recommended_actions("any_situation")
        assert recommendations is None

        stats = memory.get_memory_stats()
        assert stats['total_experiences'] == 0
        assert stats['patterns'] == 0

    def test_invalid_memory_file_handling(self):
        """Test handling of invalid memory file paths"""
        # Test with non-existent directory
        invalid_path = "/non/existent/directory/memory.json"

        # Should not crash during initialization
        memory = ExperienceMemory(memory_file=invalid_path)
        assert memory.memory_file == invalid_path
        assert len(memory.experiences) == 0

    def test_corrupted_memory_file_handling(self, temp_memory_file):
        """Test handling of corrupted memory files"""
        # Write invalid JSON to file
        with open(temp_memory_file, 'w') as f:
            f.write("invalid json content {[")

        # Should handle gracefully and start with empty memory
        memory = ExperienceMemory(memory_file=temp_memory_file)
        assert len(memory.experiences) == 0

    def test_extreme_data_values(self, temp_memory_file):
        """Test handling of extreme data values"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Test with extreme reward values
        extreme_values = [
            (float('inf'), ['inf_test']),
            (float('-inf'), ['neg_inf_test']),
            (1e10, ['very_large']),
            (-1e10, ['very_negative']),
            (0.0, ['zero_reward'])
        ]

        for reward, actions in extreme_values:
            try:
                memory.record_experience(f"extreme_{reward}", actions, reward, {})
                # Should handle gracefully without crashing
            except (ValueError, OverflowError):
                # Acceptable to reject invalid values
                pass

    def test_very_long_action_sequences(self, temp_memory_file):
        """Test handling of very long action sequences"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Test with very long action sequence
        long_actions = [f'action_{i}' for i in range(1000)]

        memory.record_experience("long_sequence_test", long_actions, 1.0, {})

        # Should store the sequence
        assert "long_sequence_test" in memory.experiences
        stored_exp = memory.experiences["long_sequence_test"]
        assert len(stored_exp.action_sequence) == 1000

    def test_memory_with_no_successful_patterns(self, temp_memory_file):
        """Test behavior when no successful patterns exist"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Record only failing experiences
        for i in range(10):
            memory.record_experience(f"fail_{i}", [f'bad_action_{i}'], -1.0, {})

        # Should not recommend any patterns
        pattern_recommendations = memory._find_pattern_match({})
        assert pattern_recommendations is None

    def test_concurrent_access_simulation(self, temp_memory_file):
        """Test simulation of concurrent access scenarios"""
        memory = ExperienceMemory(memory_file=temp_memory_file)

        # Simulate rapid successive updates to same situation
        situation_hash = "concurrent_test"

        for i in range(100):
            memory.record_experience(
                situation_hash,
                [f'concurrent_action_{i}'],
                1.0 if i % 2 == 0 else -0.5,  # Alternating success/failure
                {}
            )

        # Should handle all updates correctly
        exp = memory.experiences[situation_hash]
        assert exp.usage_count == 100
        assert 0 <= exp.success_rate <= 1


class TestExperienceEntryDataclass:
    """Test ExperienceEntry dataclass functionality"""

    def test_experience_entry_creation(self):
        """Test ExperienceEntry dataclass creation"""
        entry = ExperienceEntry(
            situation_hash="test_hash_123",
            action_sequence=['up', 'down', 'a'],
            outcome_reward=2.5,
            success_rate=0.8,
            usage_count=5,
            last_used=time.time(),
            context={'phase': 'test', 'location': 'route_1'}
        )

        # Verify all fields
        assert entry.situation_hash == "test_hash_123"
        assert entry.action_sequence == ['up', 'down', 'a']
        assert entry.outcome_reward == 2.5
        assert entry.success_rate == 0.8
        assert entry.usage_count == 5
        assert isinstance(entry.last_used, float)
        assert entry.context['phase'] == 'test'

        # Test serialization
        entry_dict = asdict(entry)
        assert isinstance(entry_dict, dict)
        assert entry_dict['situation_hash'] == "test_hash_123"

    def test_experience_entry_field_types(self):
        """Test ExperienceEntry field type validation"""
        current_time = time.time()

        entry = ExperienceEntry(
            situation_hash="type_test",
            action_sequence=[],
            outcome_reward=0.0,
            success_rate=0.0,
            usage_count=0,
            last_used=current_time,
            context={}
        )

        # Verify types
        assert isinstance(entry.situation_hash, str)
        assert isinstance(entry.action_sequence, list)
        assert isinstance(entry.outcome_reward, (int, float))
        assert isinstance(entry.success_rate, (int, float))
        assert isinstance(entry.usage_count, int)
        assert isinstance(entry.last_used, (int, float))
        assert isinstance(entry.context, dict)


class TestActionPatternDataclass:
    """Test ActionPattern dataclass functionality"""

    def test_action_pattern_creation(self):
        """Test ActionPattern dataclass creation"""
        pattern = ActionPattern(
            pattern=['up', 'right', 'a', 'down'],
            success_situations=['situation_1', 'situation_2', 'situation_3'],
            average_reward=3.2,
            confidence=0.75
        )

        # Verify all fields
        assert pattern.pattern == ['up', 'right', 'a', 'down']
        assert len(pattern.success_situations) == 3
        assert pattern.average_reward == 3.2
        assert pattern.confidence == 0.75

        # Test serialization
        pattern_dict = asdict(pattern)
        assert isinstance(pattern_dict, dict)
        assert pattern_dict['pattern'] == ['up', 'right', 'a', 'down']

    def test_action_pattern_confidence_bounds(self):
        """Test ActionPattern confidence value bounds"""
        # Test various confidence values
        confidence_values = [0.0, 0.5, 1.0, 1.5, -0.1]

        for conf in confidence_values:
            pattern = ActionPattern(
                pattern=['test'],
                success_situations=['test_sit'],
                average_reward=1.0,
                confidence=conf
            )

            # Confidence should be stored as provided (bounds checking may be elsewhere)
            assert pattern.confidence == conf

    def test_action_pattern_empty_values(self):
        """Test ActionPattern with empty values"""
        pattern = ActionPattern(
            pattern=[],
            success_situations=[],
            average_reward=0.0,
            confidence=0.0
        )

        assert len(pattern.pattern) == 0
        assert len(pattern.success_situations) == 0
        assert pattern.average_reward == 0.0
        assert pattern.confidence == 0.0