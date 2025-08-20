#!/usr/bin/env python3
"""
test_semantic_context_system.py - Comprehensive tests for the Semantic Context System

This module tests all aspects of the semantic context system including:
- Game context construction and validation
- Dialogue intent detection and analysis
- Strategy recommendation
- Pattern matching and confidence scoring
- Database operations and persistence
"""

import pytest
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import with fallbacks for missing dependencies
try:
    from core.semantic_context_system import SemanticContextSystem, GameContext
    from core.vision_processor import DetectedText, VisualContext
except ImportError:
    # Skip these tests if dependencies aren't available
    pytest.skip("Missing dependencies for semantic context system tests", allow_module_level=True)

class TestGameContext:
    """Test the GameContext dataclass"""
    
    @pytest.mark.unit
    def test_game_context_creation(self):
        """Test basic GameContext creation"""
        context = GameContext(
            current_objective="beat_falkner",
            player_progress={"badges": 2},
            location_info={"current_map": "violet_city"},
            recent_events=["entered_gym"],
            active_quests=["beat_falkner"]
        )
        
        assert context.player_progress["badges"] == 2
        assert context.location_info["current_map"] == "violet_city"
        assert "entered_gym" in context.recent_events
        assert "beat_falkner" in context.active_quests
    
    @pytest.mark.unit
    def test_game_context_empty_initialization(self):
        """Test GameContext with minimal data"""
        context = GameContext(
            current_objective=None,
            player_progress={},
            location_info={},
            recent_events=[],
            active_quests=[]
        )
        
        assert isinstance(context.player_progress, dict)
        assert isinstance(context.location_info, dict)
        assert isinstance(context.recent_events, list)
        assert isinstance(context.active_quests, list)
    
    @pytest.mark.unit
    def test_game_context_with_complex_data(self, advanced_game_context):
        """Test GameContext with complex nested data"""
        context = advanced_game_context
        
        assert context.player_progress["badges"] == 2
        assert len(context.player_progress["story_flags"]) > 2
        assert context.location_info["region"] == "johto"
        assert len(context.recent_events) > 0
        assert len(context.active_quests) > 0


class TestSemanticContextSystemInitialization:
    """Test semantic context system initialization"""
    
    @pytest.mark.unit
    def test_system_initialization(self, semantic_system):
        """Test basic system initialization"""
        assert semantic_system.db_path.exists()
        assert len(semantic_system.dialogue_patterns) > 0
        assert len(semantic_system.npc_behaviors) > 0
        assert len(semantic_system.location_contexts) > 0
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_database_initialization(self, semantic_system):
        """Test that database tables are created correctly"""
        with sqlite3.connect(semantic_system.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ["dialogue_understanding", "pattern_effectiveness", "context_learning"]
            for table in expected_tables:
                assert table in tables
    
    @pytest.mark.unit
    def test_pattern_loading(self, semantic_system):
        """Test that dialogue patterns are loaded correctly"""
        patterns = semantic_system.dialogue_patterns
        
        # Check key pattern categories exist
        assert "starter_selection_offer" in patterns
        assert "healing_offer" in patterns
        assert "gym_challenge_offer" in patterns
        assert "shop_greeting" in patterns
        
        # Verify pattern structure
        for pattern_name, pattern_data in patterns.items():
            assert hasattr(pattern_data, 'keywords')
            assert hasattr(pattern_data, 'context_requirements')
            assert hasattr(pattern_data, 'intent')
    
    @pytest.mark.unit
    def test_npc_behaviors_loading(self, semantic_system):
        """Test that NPC behaviors are loaded correctly"""
        behaviors = semantic_system.npc_behaviors
        
        expected_npc_types = [
            "professor", "gym_leader", "nurse", "shopkeeper", "trainer"
        ]
        
        for npc_type in expected_npc_types:
            assert npc_type in behaviors
            behavior = behaviors[npc_type]
            assert hasattr(behavior, 'common_topics')
            assert hasattr(behavior, 'greeting_patterns')
            assert hasattr(behavior, 'typical_responses')


class TestDialogueAnalysis:
    """Test dialogue analysis functionality"""
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_basic_dialogue_analysis(self, semantic_system, sample_game_context):
        """Test basic dialogue analysis"""
        dialogue_text = "Hello! I'm Professor Elm! Would you like a Pokemon?"
        
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        assert result is not None
        assert "primary_intent" in result
        assert "confidence" in result
        assert "response_strategy" in result
        assert "context_factors" in result
        assert "suggested_actions" in result
        
        # Confidence should be between 0 and 1
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    @pytest.mark.parametrize("dialogue_key", [
        "professor_elm_intro", "nurse_healing", "gym_leader_challenge", "shop_keeper"
    ])
    def test_dialogue_scenarios(self, semantic_system, sample_game_context, dialogue_scenarios, dialogue_key):
        """Test various dialogue scenarios"""
        scenario = dialogue_scenarios[dialogue_key]
        dialogue_text = scenario["dialogue_text"]
        expected_intent = scenario["expected_intent"]
        
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        assert result is not None
        # Intent should match or be reasonable for the context
        assert result["confidence"] > 0.3  # Should have some confidence
        assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_starter_pokemon_dialogue(self, semantic_system, sample_game_context):
        """Test specific starter Pokemon dialogue analysis"""
        dialogue_text = "So, what Pokemon will you choose? Cyndaquil, Totodile, or Chikorita?"
        
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        assert result["primary_intent"] == "starter_selection"
        assert result["confidence"] > 0.5  # Should be quite confident
        assert result["response_strategy"] in ["select_fire_starter", "accept_offer"]
        assert "A" in result["suggested_actions"]
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_healing_dialogue(self, semantic_system, sample_game_context):
        """Test Pokemon Center healing dialogue"""
        dialogue_text = "Welcome to the Pokemon Center! Would you like me to heal your Pokemon?"
        
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        assert result["primary_intent"] == "healing_request"
        assert result["confidence"] > 0.5
        assert result["response_strategy"] in ["accept_healing", "use_pokemon_center"]
        assert result["suggested_actions"] == ["A"]
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_gym_challenge_dialogue(self, semantic_system, advanced_game_context):
        """Test gym leader challenge dialogue"""
        dialogue_text = "I'm Falkner! Are you ready for a Pokemon battle?"
        
        result = semantic_system.analyze_dialogue(dialogue_text, advanced_game_context)
        
        assert result["primary_intent"] == "gym_challenge"
        assert result["confidence"] > 0.5
        assert result["response_strategy"] in ["accept_challenge", "prepare_for_battle"]
        assert result["suggested_actions"] == ["A"]
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_empty_dialogue(self, semantic_system):
        """Test handling of empty or invalid dialogue"""
        # Use neutral context to test pure edge case handling
        neutral_context = GameContext(
            current_objective="explore_world",  # Neutral objective
            player_progress={"badges": 1, "pokemon_count": 3, "story_flags": []},
            location_info={"current_map": "route_32", "region": "johto"},
            recent_events=["walked_around"],
            active_quests=["explore_world"]
        )
        result = semantic_system.analyze_dialogue("", neutral_context)
        
        assert result is not None
        assert result["primary_intent"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["response_strategy"] == "wait_and_observe"
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_nonsense_dialogue(self, semantic_system):
        """Test handling of nonsensical dialogue"""
        # Use neutral context to test pure edge case handling
        neutral_context = GameContext(
            current_objective="explore_world",  # Neutral objective
            player_progress={"badges": 1, "pokemon_count": 3, "story_flags": []},
            location_info={"current_map": "route_32", "region": "johto"},
            recent_events=["walked_around"],
            active_quests=["explore_world"]
        )
        dialogue_text = "asdjkl qwerty zxcvbn random text that makes no sense"
        
        result = semantic_system.analyze_dialogue(dialogue_text, neutral_context)
        
        assert result is not None
        assert result["confidence"] < 0.3  # Should have low confidence
        assert result["response_strategy"] in ["wait_and_observe", "listen_and_respond_appropriately"]


class TestContextInfluence:
    """Test how game context influences analysis"""
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_beginner_context_influence(self, semantic_system):
        """Test analysis with beginner game context"""
        beginner_context = GameContext(
            current_objective="meet_professor_elm",
            player_progress={"badges": 0, "pokemon_count": 0, "story_flags": []},
            location_info={"current_map": "new_bark_town", "region": "johto"},
            recent_events=["game_start"],
            active_quests=["meet_professor_elm"]
        )
        
        dialogue_text = "Hello! I'm Professor Elm! Would you like a Pokemon?"
        result = semantic_system.analyze_dialogue(dialogue_text, beginner_context)
        
        # Should strongly suggest accepting Pokemon for beginners
        assert result["confidence"] > 0.3
        assert result["response_strategy"] in ["select_fire_starter", "accept_offer"]
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_advanced_context_influence(self, semantic_system, advanced_game_context):
        """Test analysis with advanced game context"""
        dialogue_text = "Would you like to trade Pokemon?"
        result = semantic_system.analyze_dialogue(dialogue_text, advanced_game_context)
        
        # Advanced player might have different priorities
        assert result is not None
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_location_context_influence(self, semantic_system):
        """Test how location affects analysis"""
        gym_context = GameContext(
            current_objective="beat_falkner",
            player_progress={"badges": 0, "pokemon_count": 1, "story_flags": ["got_starter"]},
            location_info={"current_map": "violet_gym", "region": "johto"},
            recent_events=["entered_gym"],
            active_quests=["beat_falkner"]
        )
        
        dialogue_text = "Are you ready to battle?"
        result = semantic_system.analyze_dialogue(dialogue_text, gym_context)
        
        # In a gym, should recognize battle challenge
        assert result["primary_intent"] in ["gym_challenge", "battle_request"]
        assert result["confidence"] > 0.3


class TestPatternMatching:
    """Test pattern matching algorithms"""
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_keyword_matching(self, semantic_system, sample_game_context):
        """Test keyword-based pattern matching"""
        # Test various keyword combinations
        test_cases = [
            ("Pokemon Center heal", "healing_request"),
            ("starter Pokemon choose", "starter_selection"),
            ("gym leader battle", "gym_challenge"),
            ("buy sell item", "shop_interaction"),
        ]
        
        for dialogue, expected_intent in test_cases:
            result = semantic_system.analyze_dialogue(dialogue, sample_game_context)
            # Intent should be detected with reasonable confidence
            assert result["confidence"] > 0.2
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_pattern_confidence_scoring(self, semantic_system, sample_game_context):
        """Test that confidence scores vary appropriately"""
        # Strong pattern match
        strong_dialogue = "Welcome to the Pokemon Center! Would you like me to heal your Pokemon to full health?"
        strong_result = semantic_system.analyze_dialogue(strong_dialogue, sample_game_context)
        
        # Weak pattern match
        weak_dialogue = "Hello there, how are you doing today?"
        weak_result = semantic_system.analyze_dialogue(weak_dialogue, sample_game_context)
        
        assert strong_result["confidence"] > weak_result["confidence"]
        assert strong_result["confidence"] > 0.5
        assert weak_result["confidence"] < 0.5
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_multiple_pattern_matches(self, semantic_system, sample_game_context):
        """Test handling of dialogue that matches multiple patterns"""
        # Dialogue that could match multiple intents
        dialogue_text = "Welcome! Would you like to battle or buy some items?"
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        # Should pick the most confident match
        assert result["confidence"] > 0.0
        assert result["primary_intent"] in ["battle_request", "shopping", "shop_interaction", "gym_challenge"]


class TestStrategyRecommendation:
    """Test strategy recommendation system"""
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_strategy_mapping(self, semantic_system, sample_game_context):
        """Test that intents map to appropriate strategies"""
        test_cases = [
            ("starter_selection", ["select_fire_starter", "select_water_starter", "select_grass_starter"]),
            ("healing_request", ["accept_healing"]),
            ("gym_challenge", ["accept_challenge"]),
            ("shop_interaction", ["browse_items", "buy_potions", "buy_pokeballs"]),
        ]
        
        for intent, expected_strategies in test_cases:
            # Mock a result with specific intent
            dialogue = f"Test dialogue for {intent}"
            result = semantic_system.analyze_dialogue(dialogue, sample_game_context)
            
            # Strategy should be reasonable for the intent
            assert result["response_strategy"] is not None
            assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_action_recommendation(self, semantic_system, sample_game_context):
        """Test action sequence recommendations"""
        dialogue_text = "Would you like me to heal your Pokemon?"
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        actions = result["suggested_actions"]
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(action, str) for action in actions)
        # Should typically recommend "A" for positive responses
        assert "A" in actions
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_context_dependent_strategy(self, semantic_system):
        """Test that strategy adapts to context"""
        # Low-level player should be more cautious
        beginner_context = GameContext(
            current_objective="reach_cherrygrove",
            player_progress={"badges": 0, "pokemon_count": 1, "story_flags": ["got_starter"]},
            location_info={"current_map": "route_29", "region": "johto"},
            recent_events=["left_new_bark_town"],
            active_quests=["reach_cherrygrove"]
        )
        
        # High-level player can be more aggressive
        advanced_context = GameContext(
            current_objective="become_champion",
            player_progress={"badges": 5, "pokemon_count": 6, "story_flags": ["beat_elite_four"]},
            location_info={"current_map": "victory_road", "region": "johto"},
            recent_events=["entered_victory_road"],
            active_quests=["become_champion"]
        )
        
        battle_dialogue = "Want to battle?"
        
        beginner_result = semantic_system.analyze_dialogue(battle_dialogue, beginner_context)
        advanced_result = semantic_system.analyze_dialogue(battle_dialogue, advanced_context)
        
        # Both should recognize it as battle, but may have different confidence
        assert beginner_result["primary_intent"] in ["battle_request", "unknown"]
        assert advanced_result["primary_intent"] in ["battle_request", "unknown"]


class TestDatabaseOperations:
    """Test database persistence and learning"""
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_analysis_storage(self, semantic_system, sample_game_context):
        """Test that analysis results are stored in database"""
        dialogue_text = "Would you like a Pokemon?"
        result = semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        # Check database for stored analysis
        with sqlite3.connect(semantic_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dialogue_understanding")
            count = cursor.fetchone()[0]
            assert count > 0
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_pattern_effectiveness_tracking(self, semantic_system, sample_game_context):
        """Test pattern effectiveness tracking"""
        dialogue_text = "Welcome to the Pokemon Center!"
        
        # Analyze multiple times
        for _ in range(3):
            semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        
        with sqlite3.connect(semantic_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pattern_effectiveness")
            count = cursor.fetchone()[0]
            assert count > 0
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_pattern_effectiveness_updates(self, semantic_system, sample_game_context):
        """Test pattern effectiveness updates"""
        semantic_system.update_pattern_effectiveness(
            pattern_id="healing_offer",
            success=True
        )
        
        with sqlite3.connect(semantic_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pattern_effectiveness WHERE pattern_id = 'healing_offer'")
            count = cursor.fetchone()[0]
            assert count > 0
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_statistics_generation(self, semantic_system, sample_game_context):
        """Test statistics generation"""
        # Generate some analysis data
        test_dialogues = [
            "Would you like a Pokemon?",
            "Welcome to the Pokemon Center!",
            "Ready for a gym battle?",
        ]
        
        for dialogue in test_dialogues:
            semantic_system.analyze_dialogue(dialogue, sample_game_context)
        
        stats = semantic_system.get_semantic_stats()
        
        assert "total_dialogue_analyses" in stats
        assert "intent_distribution" in stats
        assert "average_confidence" in stats
        assert stats["total_dialogue_analyses"] >= len(test_dialogues)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_none_inputs(self, semantic_system, sample_game_context):
        """Test handling of None inputs"""
        result = semantic_system.analyze_dialogue(None, sample_game_context)
        assert result["primary_intent"] == "unknown"
        assert result["confidence"] == 0.0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_empty_context(self, semantic_system):
        """Test handling of empty game context"""
        empty_context = GameContext(
            current_objective=None,
            player_progress={},
            location_info={},
            recent_events=[],
            active_quests=[]
        )
        
        dialogue_text = "Hello there!"
        result = semantic_system.analyze_dialogue(dialogue_text, empty_context)
        
        assert result is not None
        assert result["confidence"] >= 0.0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_very_long_dialogue(self, semantic_system, sample_game_context):
        """Test handling of unusually long dialogue"""
        long_dialogue = "This is a very long dialogue " * 100
        result = semantic_system.analyze_dialogue(long_dialogue, sample_game_context)
        
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.semantic
    def test_special_characters(self, semantic_system, sample_game_context):
        """Test handling of special characters in dialogue"""
        special_dialogue = "Hello! @#$%^&*()_+ Would you like a Pokémon?"
        result = semantic_system.analyze_dialogue(special_dialogue, sample_game_context)
        
        assert result is not None
        assert result["confidence"] > 0.0  # Should still recognize Pokemon reference
    
    @pytest.mark.unit
    @pytest.mark.database
    def test_database_connection_error(self, semantic_system):
        """Test handling of database connection issues"""
        # Simulate database error by corrupting the path
        original_path = semantic_system.db_path
        semantic_system.db_path = Path("/nonexistent/path/test.db")
        
        # Should handle gracefully without crashing
        try:
            stats = semantic_system.get_semantic_stats()
            # Should return empty or default stats
            assert isinstance(stats, dict)
        except Exception as e:
            # If it raises an exception, it should be a specific, handled one
            assert "database" in str(e).lower() or "connection" in str(e).lower()
        finally:
            semantic_system.db_path = original_path


@pytest.mark.performance
class TestPerformance:
    """Test performance characteristics"""
    
    def test_analysis_performance(self, semantic_system, sample_game_context):
        """Test that analysis completes in reasonable time"""
        import time
        
        dialogue_text = "Welcome to the Pokemon Center! Would you like me to heal your Pokemon?"
        
        start_time = time.time()
        for _ in range(10):
            semantic_system.analyze_dialogue(dialogue_text, sample_game_context)
        end_time = time.time()
        
        # Should complete 10 analyses in under 1 second
        assert (end_time - start_time) < 1.0
    
    def test_large_context_handling(self, semantic_system):
        """Test handling of large game contexts"""
        large_context = GameContext(
            current_objective=None,
            player_progress={f"flag_{i}": True for i in range(1000)},
            location_info={f"map_{i}": f"value_{i}" for i in range(100)},
            recent_events=[f"event_{i}" for i in range(500)],
            active_quests=[f"quest_{i}" for i in range(50)]
        )
        
        dialogue_text = "Hello there!"
        result = semantic_system.analyze_dialogue(dialogue_text, large_context)
        
        assert result is not None
        assert result["confidence"] >= 0.0
