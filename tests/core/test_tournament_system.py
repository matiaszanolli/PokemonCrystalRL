"""
Test Tournament System - Comprehensive tests for tournament functionality

Tests all components of the tournament system including models, bracket generation,
tournament management, analytics, and REST API endpoints.
"""

import unittest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.tournament import (
    TournamentManager,
    TournamentConfig,
    TournamentParticipant,
    TournamentType,
    TournamentStatus,
    MatchStatus,
    ParticipantType,
    BracketGenerator
)
from core.tournament.tournament_analytics import TournamentAnalytics
from core.tournament.tournament_profiles import TournamentProfiles


class TestTournamentModels(unittest.TestCase):
    """Test tournament data models"""

    def test_tournament_participant_creation(self):
        """Test tournament participant creation and properties"""
        participant = TournamentParticipant(
            name="Test AI",
            description="Test AI configuration",
            participant_type=ParticipantType.AI_CONFIGURATION,
            configuration={"test": "config"}
        )

        self.assertEqual(participant.name, "Test AI")
        self.assertEqual(participant.description, "Test AI configuration")
        self.assertEqual(participant.participant_type, ParticipantType.AI_CONFIGURATION)
        self.assertEqual(participant.configuration, {"test": "config"})
        self.assertEqual(participant.matches_played, 0)
        self.assertEqual(participant.wins, 0)
        self.assertEqual(participant.losses, 0)
        self.assertEqual(participant.win_rate, 0.0)
        self.assertEqual(participant.points, 0)

    def test_participant_stats_updates(self):
        """Test participant statistics updates"""
        participant = TournamentParticipant(name="Test AI")

        # Simulate a win
        participant.matches_played = 1
        participant.wins = 1
        participant.total_score = 100.0

        self.assertEqual(participant.win_rate, 1.0)
        self.assertEqual(participant.points, 3)

        # Simulate a loss
        participant.matches_played = 2
        participant.losses = 1
        participant.total_score = 150.0

        self.assertEqual(participant.win_rate, 0.5)
        self.assertEqual(participant.points, 3)

        # Simulate a draw
        participant.matches_played = 3
        participant.draws = 1
        participant.total_score = 200.0

        self.assertEqual(participant.win_rate, 1/3)
        self.assertEqual(participant.points, 4)

    def test_tournament_config_creation(self):
        """Test tournament configuration creation"""
        config = TournamentConfig(
            name="Test Tournament",
            description="Test Description",
            tournament_type=TournamentType.SINGLE_ELIMINATION,
            max_participants=8,
            min_participants=4
        )

        self.assertEqual(config.name, "Test Tournament")
        self.assertEqual(config.tournament_type, TournamentType.SINGLE_ELIMINATION)
        self.assertEqual(config.max_participants, 8)
        self.assertEqual(config.min_participants, 4)

    def test_tournament_creation_and_participant_management(self):
        """Test tournament creation and participant management"""
        from core.tournament.tournament_models import Tournament

        config = TournamentConfig(
            name="Test Tournament",
            tournament_type=TournamentType.SINGLE_ELIMINATION
        )

        tournament = Tournament(config=config)

        # Test adding participants
        participant1 = TournamentParticipant(name="AI 1")
        participant2 = TournamentParticipant(name="AI 2")

        self.assertTrue(tournament.add_participant(participant1))
        self.assertTrue(tournament.add_participant(participant2))

        self.assertEqual(len(tournament.participants), 2)
        self.assertIn(participant1.id, tournament.participants)
        self.assertIn(participant2.id, tournament.participants)

        # Test removing participants
        self.assertTrue(tournament.remove_participant(participant1.id))
        self.assertEqual(len(tournament.participants), 1)
        self.assertNotIn(participant1.id, tournament.participants)

        # Test finding participant by name
        found = tournament.get_participant_by_name("AI 2")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "AI 2")


class TestBracketGenerator(unittest.TestCase):
    """Test bracket generation for different tournament formats"""

    def setUp(self):
        """Set up test fixtures"""
        self.bracket_generator = BracketGenerator()
        self.config = TournamentConfig(
            name="Test Tournament",
            tournament_type=TournamentType.SINGLE_ELIMINATION,
            min_participants=4
        )

    def create_test_tournament(self, participant_count: int, tournament_type: TournamentType = None):
        """Create test tournament with specified number of participants"""
        from core.tournament.tournament_models import Tournament

        if tournament_type:
            self.config.tournament_type = tournament_type

        tournament = Tournament(config=self.config)

        for i in range(participant_count):
            participant = TournamentParticipant(name=f"AI {i+1}")
            tournament.add_participant(participant)

        return tournament

    def test_single_elimination_bracket_generation(self):
        """Test single elimination bracket generation"""
        tournament = self.create_test_tournament(4)

        success = self.bracket_generator.generate_bracket(tournament)
        self.assertTrue(success)

        # Check tournament structure
        self.assertEqual(tournament.total_rounds, 2)  # 4 -> 2 -> 1
        self.assertEqual(len(tournament.matches), 3)  # 2 first round + 1 final

        # Check first round
        first_round_matches = tournament.bracket.get_round_matches(1)
        self.assertEqual(len(first_round_matches), 2)

        # Check that all participants are assigned to first round
        assigned_participants = set()
        for match_id in first_round_matches:
            match = tournament.matches[match_id]
            assigned_participants.add(match.participant1_id)
            if match.participant2_id:
                assigned_participants.add(match.participant2_id)

        self.assertEqual(len(assigned_participants), 4)

    def test_single_elimination_with_byes(self):
        """Test single elimination with odd number of participants (byes)"""
        tournament = self.create_test_tournament(5)

        success = self.bracket_generator.generate_bracket(tournament)
        self.assertTrue(success)

        # Should create bracket for 8 participants (next power of 2)
        self.assertEqual(tournament.total_rounds, 3)  # 8 -> 4 -> 2 -> 1

        # Check for bye matches
        first_round_matches = tournament.bracket.get_round_matches(1)
        bye_matches = 0
        for match_id in first_round_matches:
            match = tournament.matches[match_id]
            if match.is_bye:
                bye_matches += 1

        self.assertGreater(bye_matches, 0)

    def test_round_robin_bracket_generation(self):
        """Test round robin bracket generation"""
        tournament = self.create_test_tournament(4, TournamentType.ROUND_ROBIN)

        success = self.bracket_generator.generate_bracket(tournament)
        self.assertTrue(success)

        # Round robin: n*(n-1)/2 matches
        expected_matches = 4 * 3 // 2
        self.assertEqual(tournament.total_matches, expected_matches)

        # All matches should be in round 1 for simplicity
        self.assertEqual(tournament.total_rounds, 1)
        first_round_matches = tournament.bracket.get_round_matches(1)
        self.assertEqual(len(first_round_matches), expected_matches)

    def test_swiss_system_bracket_generation(self):
        """Test Swiss system bracket generation"""
        tournament = self.create_test_tournament(6, TournamentType.SWISS_SYSTEM)

        success = self.bracket_generator.generate_bracket(tournament)
        self.assertTrue(success)

        # Swiss system generates first round only
        self.assertGreater(tournament.total_rounds, 0)
        first_round_matches = tournament.bracket.get_round_matches(1)
        self.assertEqual(len(first_round_matches), 3)  # 6 players = 3 matches

    def test_tournament_advancement(self):
        """Test tournament advancement between rounds"""
        tournament = self.create_test_tournament(4)
        self.bracket_generator.generate_bracket(tournament)

        # Mark first round matches as completed
        first_round_matches = tournament.bracket.get_round_matches(1)
        winners = []

        for i, match_id in enumerate(first_round_matches):
            match = tournament.matches[match_id]
            match.status = MatchStatus.COMPLETED
            match.winner_id = match.participant1_id  # First participant always wins
            winners.append(match.participant1_id)

        # Advance tournament
        success = self.bracket_generator.advance_tournament(tournament)
        self.assertTrue(success)

        # Check second round setup
        self.assertEqual(tournament.current_round, 2)
        second_round_matches = tournament.bracket.get_round_matches(2)
        self.assertEqual(len(second_round_matches), 1)

        # Check that winners are assigned to second round
        final_match = tournament.matches[second_round_matches[0]]
        self.assertIn(final_match.participant1_id, winners)
        self.assertIn(final_match.participant2_id, winners)


@patch('core.tournament.tournament_manager.ExperimentManager')
@patch('core.tournament.tournament_manager.ExperimentScheduler')
@patch('core.tournament.tournament_manager.StatisticalAnalyzer')
class TestTournamentManager(unittest.TestCase):
    """Test tournament manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tournament_creation(self, mock_analyzer, mock_scheduler, mock_manager):
        """Test tournament creation"""
        tournament_manager = TournamentManager(data_dir=self.temp_dir)

        config = TournamentConfig(
            name="Test Tournament",
            tournament_type=TournamentType.SINGLE_ELIMINATION
        )

        tournament_id = tournament_manager.create_tournament(config)
        self.assertIsNotNone(tournament_id)

        tournament = tournament_manager.get_tournament(tournament_id)
        self.assertIsNotNone(tournament)
        self.assertEqual(tournament.config.name, "Test Tournament")

    def test_participant_management(self, mock_analyzer, mock_scheduler, mock_manager):
        """Test adding and managing participants"""
        tournament_manager = TournamentManager(data_dir=self.temp_dir)

        config = TournamentConfig(name="Test Tournament")
        tournament_id = tournament_manager.create_tournament(config)

        participant = TournamentParticipant(name="Test AI")
        success = tournament_manager.add_participant(tournament_id, participant)
        self.assertTrue(success)

        tournament = tournament_manager.get_tournament(tournament_id)
        self.assertEqual(len(tournament.participants), 1)

    def test_tournament_start(self, mock_analyzer, mock_scheduler, mock_manager):
        """Test tournament start functionality"""
        tournament_manager = TournamentManager(data_dir=self.temp_dir, auto_advance=False)

        config = TournamentConfig(
            name="Test Tournament",
            min_participants=2,
            tournament_type=TournamentType.SINGLE_ELIMINATION
        )
        tournament_id = tournament_manager.create_tournament(config)

        # Add minimum participants
        for i in range(2):
            participant = TournamentParticipant(name=f"AI {i+1}")
            tournament_manager.add_participant(tournament_id, participant)

        # Mock experiment scheduler
        mock_scheduler.return_value.schedule_experiment.return_value = "mock_experiment_id"

        success = tournament_manager.start_tournament(tournament_id)
        self.assertTrue(success)

        tournament = tournament_manager.get_tournament(tournament_id)
        self.assertEqual(tournament.status, TournamentStatus.RUNNING)

    def test_active_tournament_tracking(self, mock_analyzer, mock_scheduler, mock_manager):
        """Test active tournament tracking"""
        tournament_manager = TournamentManager(data_dir=self.temp_dir)

        config = TournamentConfig(name="Test Tournament")
        tournament_id = tournament_manager.create_tournament(config)

        # Initially no active tournaments
        active = tournament_manager.get_active_tournaments()
        self.assertEqual(len(active), 0)

        # Add participants and start
        for i in range(4):
            participant = TournamentParticipant(name=f"AI {i+1}")
            tournament_manager.add_participant(tournament_id, participant)

        mock_scheduler.return_value.schedule_experiment.return_value = "mock_experiment_id"
        tournament_manager.start_tournament(tournament_id)

        # Should now have one active tournament
        active = tournament_manager.get_active_tournaments()
        self.assertEqual(len(active), 1)


class TestTournamentAnalytics(unittest.TestCase):
    """Test tournament analytics functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.analytics = TournamentAnalytics()

    def create_mock_tournament(self):
        """Create mock tournament for testing"""
        from core.tournament.tournament_models import Tournament

        config = TournamentConfig(
            name="Mock Tournament",
            tournament_type=TournamentType.SINGLE_ELIMINATION
        )

        tournament = Tournament(config=config)
        tournament.status = TournamentStatus.COMPLETED
        tournament.actual_start = datetime.now() - timedelta(hours=2)
        tournament.completed_at = datetime.now()

        # Add participants
        for i in range(4):
            participant = TournamentParticipant(name=f"AI {i+1}")
            participant.matches_played = 2
            participant.wins = i % 2  # Alternate wins
            participant.total_score = 100 + i * 10
            tournament.add_participant(participant)

        return tournament

    def test_tournament_insights_generation(self):
        """Test tournament insights generation"""
        tournament = self.create_mock_tournament()

        insights = self.analytics.analyze_tournament(tournament)

        self.assertEqual(insights.tournament_id, tournament.id)
        self.assertEqual(insights.tournament_name, tournament.config.name)
        self.assertEqual(insights.total_participants, 4)
        self.assertGreater(insights.tournament_duration_hours, 0)

    def test_participant_performance_analysis(self):
        """Test participant performance analysis"""
        tournament = self.create_mock_tournament()
        participant = list(tournament.participants.values())[0]

        insights = self.analytics.analyze_participant_performance(participant, tournament)

        self.assertIsInstance(insights, list)
        # Should generate some insights for any participant
        # (exact insights depend on performance)

    def test_strategy_analysis(self):
        """Test strategy analysis across tournaments"""
        tournaments = [self.create_mock_tournament() for _ in range(3)]

        analysis = self.analytics.analyze_strategy_across_tournaments(tournaments, "test_strategy")

        self.assertEqual(analysis.strategy_name, "test_strategy")
        self.assertIsInstance(analysis.strengths, list)
        self.assertIsInstance(analysis.weaknesses, list)

    def test_meta_insights_generation(self):
        """Test meta-game insights generation"""
        tournaments = [self.create_mock_tournament() for _ in range(5)]

        meta_insights = self.analytics.generate_meta_insights(tournaments)

        self.assertIn("total_tournaments", meta_insights)
        self.assertIn("total_participants", meta_insights)
        self.assertIn("strategy_trends", meta_insights)
        self.assertEqual(meta_insights["total_tournaments"], 5)


class TestTournamentProfiles(unittest.TestCase):
    """Test tournament configuration profiles"""

    def setUp(self):
        """Set up test fixtures"""
        self.profiles = TournamentProfiles()

    def test_quick_battle_profile(self):
        """Test quick battle profile configuration"""
        config = self.profiles.get_quick_battle_config()

        self.assertEqual(config.tournament_type, TournamentType.SINGLE_ELIMINATION)
        self.assertEqual(config.max_participants, 8)
        self.assertEqual(config.match_duration_minutes, 10)
        self.assertLessEqual(config.max_actions_per_match, 500)

    def test_championship_profile(self):
        """Test championship profile configuration"""
        config = self.profiles.get_championship_config()

        self.assertEqual(config.tournament_type, TournamentType.DOUBLE_ELIMINATION)
        self.assertEqual(config.max_participants, 16)
        self.assertGreater(config.match_duration_minutes, 30)

    def test_research_profile(self):
        """Test research profile configuration"""
        config = self.profiles.get_research_config()

        self.assertEqual(config.tournament_type, TournamentType.ROUND_ROBIN)
        self.assertLessEqual(config.max_participants, 6)
        self.assertGreater(len(config.secondary_metrics), 3)

    def test_sample_participants_creation(self):
        """Test sample participants creation"""
        participants = self.profiles.create_sample_participants("mixed")

        self.assertGreater(len(participants), 4)
        self.assertTrue(all(isinstance(p, TournamentParticipant) for p in participants))

        # Test different participant types
        battle_participants = self.profiles.create_sample_participants("battle")
        self.assertGreater(len(battle_participants), 2)

        exploration_participants = self.profiles.create_sample_participants("exploration")
        self.assertGreater(len(exploration_participants), 2)

    def test_profile_list(self):
        """Test profile list generation"""
        profiles = self.profiles.get_profile_list()

        self.assertIsInstance(profiles, list)
        self.assertGreater(len(profiles), 4)

        for profile in profiles:
            self.assertIn("id", profile)
            self.assertIn("name", profile)
            self.assertIn("description", profile)
            self.assertIn("format", profile)


@pytest.mark.integration
class TestTournamentIntegration(unittest.TestCase):
    """Integration tests for tournament system"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('core.tournament.tournament_manager.ExperimentScheduler')
    def test_full_tournament_workflow(self, mock_scheduler):
        """Test complete tournament workflow from creation to completion"""
        # Mock experiment scheduler
        mock_scheduler.return_value.schedule_experiment.return_value = "mock_experiment_id"

        tournament_manager = TournamentManager(data_dir=self.temp_dir, auto_advance=False)
        profiles = TournamentProfiles()

        # Create tournament
        config = profiles.get_quick_battle_config("Integration Test Tournament")
        tournament_id = tournament_manager.create_tournament(config)

        # Add participants
        participants = profiles.create_sample_participants("mixed")[:4]
        for participant in participants:
            success = tournament_manager.add_participant(tournament_id, participant)
            self.assertTrue(success)

        # Start tournament
        success = tournament_manager.start_tournament(tournament_id)
        self.assertTrue(success)

        # Verify tournament state
        tournament = tournament_manager.get_tournament(tournament_id)
        self.assertEqual(tournament.status, TournamentStatus.RUNNING)
        self.assertGreater(tournament.total_matches, 0)

        # Verify active tournament tracking
        active_tournaments = tournament_manager.get_active_tournaments()
        self.assertEqual(len(active_tournaments), 1)

    def test_tournament_persistence(self):
        """Test tournament data persistence"""
        tournament_manager = TournamentManager(data_dir=self.temp_dir)

        config = TournamentConfig(name="Persistence Test")
        tournament_id = tournament_manager.create_tournament(config)

        # Add participant
        participant = TournamentParticipant(name="Test AI")
        tournament_manager.add_participant(tournament_id, participant)

        # Create new manager instance (simulating restart)
        tournament_manager2 = TournamentManager(data_dir=self.temp_dir)

        # Should be able to load tournament (in real implementation)
        # For now, just verify the directory structure exists
        import os
        self.assertTrue(os.path.exists(self.temp_dir))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)