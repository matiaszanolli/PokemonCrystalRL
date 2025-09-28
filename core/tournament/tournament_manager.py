"""
Tournament Manager - Core tournament orchestration and execution

Manages tournament lifecycle, match scheduling, and integration with
the A/B testing automation framework for seamless experiment execution.
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .tournament_models import (
    Tournament,
    TournamentParticipant,
    TournamentMatch,
    TournamentConfig,
    TournamentStatus,
    MatchStatus,
    TournamentType,
    TournamentSummary
)
from .bracket_generator import BracketGenerator

# Import A/B testing infrastructure
from core.ab_testing.experiment_manager import ExperimentManager
from core.ab_testing.experiment_scheduler import ExperimentScheduler
from core.ab_testing.experiment_models import ExperimentConfig, ExperimentType
from core.ab_testing.statistical_analyzer import StatisticalAnalyzer
from core.event_system import EventType, Event, EventSubscriber, get_event_bus


class TournamentManager(EventSubscriber):
    """
    Main tournament management system.

    Handles:
    - Tournament creation and configuration
    - Bracket generation and match scheduling
    - Integration with A/B testing automation
    - Real-time tournament monitoring
    - Results analysis and ranking
    """

    def __init__(self,
                 data_dir: str = "tournaments",
                 auto_advance: bool = True):
        """
        Initialize tournament manager.

        Args:
            data_dir: Directory to store tournament data
            auto_advance: Automatically advance rounds when matches complete
        """
        self.logger = logging.getLogger("TournamentManager")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Core components
        self.bracket_generator = BracketGenerator()
        self.experiment_manager = ExperimentManager()
        self.experiment_scheduler = ExperimentScheduler()
        self.statistical_analyzer = StatisticalAnalyzer()

        # Tournament storage
        self.tournaments: Dict[str, Tournament] = {}
        self.active_tournaments: Dict[str, Tournament] = {}

        # Configuration
        self.auto_advance = auto_advance
        self.match_callbacks: List[Callable] = []

        # Threading
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._running = False

        # Event system integration
        self.event_bus = get_event_bus()
        self.subscribe_to_events()

        self.logger.info("Tournament manager initialized")

    def subscribe_to_events(self) -> None:
        """Subscribe to relevant events"""
        # Subscribe to experiment completion events
        self.event_bus.subscribe(EventType.BATTLE_COMPLETED, self)
        self.event_bus.subscribe(EventType.TRAINING_COMPLETED, self)

    def handle_event(self, event: Event) -> None:
        """Handle events from the event system"""
        try:
            if event.event_type in [EventType.BATTLE_COMPLETED, EventType.TRAINING_COMPLETED]:
                # Check if this event relates to a tournament match
                self._check_match_completion(event)
        except Exception as e:
            self.logger.error(f"Error handling event: {str(e)}")

    def create_tournament(self, config: TournamentConfig) -> str:
        """
        Create a new tournament.

        Args:
            config: Tournament configuration

        Returns:
            Tournament ID
        """
        try:
            with self._lock:
                tournament = Tournament(config=config)
                tournament.bracket.tournament_id = tournament.id

                self.tournaments[tournament.id] = tournament
                self._save_tournament(tournament)

                self.logger.info(f"Created tournament '{config.name}' ({tournament.id})")
                return tournament.id

        except Exception as e:
            self.logger.error(f"Error creating tournament: {str(e)}")
            raise

    def add_participant(self, tournament_id: str, participant: TournamentParticipant) -> bool:
        """
        Add participant to tournament.

        Args:
            tournament_id: Tournament ID
            participant: Participant to add

        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                if tournament_id not in self.tournaments:
                    self.logger.error(f"Tournament not found: {tournament_id}")
                    return False

                tournament = self.tournaments[tournament_id]

                if tournament.status != TournamentStatus.DRAFT:
                    self.logger.error(f"Cannot add participant to tournament in status: {tournament.status}")
                    return False

                success = tournament.add_participant(participant)
                if success:
                    self._save_tournament(tournament)
                    self.logger.info(f"Added participant '{participant.name}' to tournament '{tournament.config.name}'")

                return success

        except Exception as e:
            self.logger.error(f"Error adding participant: {str(e)}")
            return False

    def start_tournament(self, tournament_id: str) -> bool:
        """
        Start a tournament.

        Args:
            tournament_id: Tournament ID

        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                if tournament_id not in self.tournaments:
                    self.logger.error(f"Tournament not found: {tournament_id}")
                    return False

                tournament = self.tournaments[tournament_id]

                if not tournament.can_start:
                    self.logger.error(f"Tournament cannot be started: {tournament.status}, {len(tournament.participants)} participants")
                    return False

                # Generate bracket
                if not self.bracket_generator.generate_bracket(tournament):
                    self.logger.error("Failed to generate tournament bracket")
                    return False

                # Update tournament status
                tournament.status = TournamentStatus.RUNNING
                tournament.actual_start = datetime.now()
                self.active_tournaments[tournament_id] = tournament

                # Start monitoring if not already running
                if not self._running:
                    self._start_monitoring()

                # Schedule first round matches
                self._schedule_round_matches(tournament, 1)

                self._save_tournament(tournament)
                self.logger.info(f"Started tournament '{tournament.config.name}' with {len(tournament.participants)} participants")

                # Publish tournament start event
                self.event_bus.publish(Event(
                    event_type=EventType.TRAINING_STARTED,  # Reuse existing event type
                    data={
                        "tournament_id": tournament_id,
                        "tournament_name": tournament.config.name,
                        "participants": len(tournament.participants),
                        "total_rounds": tournament.total_rounds
                    }
                ))

                return True

        except Exception as e:
            self.logger.error(f"Error starting tournament: {str(e)}")
            return False

    def _schedule_round_matches(self, tournament: Tournament, round_number: int) -> None:
        """Schedule all matches for a tournament round"""
        try:
            round_matches = tournament.bracket.get_round_matches(round_number)

            for match_id in round_matches:
                match = tournament.matches[match_id]

                # Skip completed matches (byes)
                if match.is_completed:
                    continue

                # Create experiment configuration for this match
                experiment_config = self._create_match_experiment(tournament, match)

                if experiment_config:
                    # Schedule experiment through automation framework
                    experiment_id = self.experiment_scheduler.schedule_experiment(
                        experiment_config,
                        start_delay_seconds=tournament.config.round_delay_minutes * 60 if round_number > 1 else 0
                    )

                    if experiment_id:
                        match.experiment_id = experiment_id
                        match.scheduled_start = datetime.now() + timedelta(
                            minutes=tournament.config.round_delay_minutes if round_number > 1 else 0
                        )
                        self.logger.info(f"Scheduled match {match_id} as experiment {experiment_id}")
                    else:
                        self.logger.error(f"Failed to schedule experiment for match {match_id}")

            self.logger.info(f"Scheduled {len(round_matches)} matches for round {round_number}")

        except Exception as e:
            self.logger.error(f"Error scheduling round matches: {str(e)}")

    def _create_match_experiment(self, tournament: Tournament, match: TournamentMatch) -> Optional[ExperimentConfig]:
        """Create experiment configuration for a tournament match"""
        try:
            participant1 = tournament.participants[match.participant1_id]
            participant2 = tournament.participants.get(match.participant2_id)

            if not participant2:  # Bye match
                return None

            # Create experiment configuration
            experiment_config = ExperimentConfig(
                name=f"Tournament Match: {participant1.name} vs {participant2.name}",
                description=f"Round {match.round_number} match in tournament '{tournament.config.name}'",
                experiment_type=ExperimentType.CONFIGURATION_COMPARISON,

                # Use participant configurations as experiment variants
                variant_a_config=participant1.configuration,
                variant_b_config=participant2.configuration,

                # Match settings
                sample_size=1,  # Single match
                max_duration_minutes=tournament.config.match_duration_minutes,
                max_actions=tournament.config.max_actions_per_match,

                # Use tournament save state if specified
                save_state_path=tournament.config.save_state_path,

                # Metrics to collect
                primary_metric=tournament.config.primary_metric,
                secondary_metrics=tournament.config.secondary_metrics,

                # Tournament-specific metadata
                metadata={
                    "tournament_id": tournament.id,
                    "tournament_name": tournament.config.name,
                    "match_id": match.id,
                    "round_number": match.round_number,
                    "participant1_id": match.participant1_id,
                    "participant1_name": participant1.name,
                    "participant2_id": match.participant2_id,
                    "participant2_name": participant2.name
                }
            )

            return experiment_config

        except Exception as e:
            self.logger.error(f"Error creating match experiment: {str(e)}")
            return None

    def _check_match_completion(self, event: Event) -> None:
        """Check if an event indicates match completion"""
        try:
            # Look for experiment ID in event data
            experiment_id = event.data.get("experiment_id")
            if not experiment_id:
                return

            # Find tournament match with this experiment ID
            tournament_id = None
            match_id = None

            for tid, tournament in self.active_tournaments.items():
                for mid, match in tournament.matches.items():
                    if match.experiment_id == experiment_id:
                        tournament_id = tid
                        match_id = mid
                        break
                if tournament_id:
                    break

            if not tournament_id or not match_id:
                return

            # Process match completion
            self._process_match_completion(tournament_id, match_id, event.data)

        except Exception as e:
            self.logger.error(f"Error checking match completion: {str(e)}")

    def _process_match_completion(self, tournament_id: str, match_id: str, event_data: Dict[str, Any]) -> None:
        """Process completion of a tournament match"""
        try:
            with self._lock:
                tournament = self.active_tournaments[tournament_id]
                match = tournament.matches[match_id]

                if match.status == MatchStatus.COMPLETED:
                    return  # Already processed

                # Get experiment results
                experiment_results = self.experiment_manager.get_experiment_results(match.experiment_id)
                if not experiment_results:
                    self.logger.error(f"No results found for experiment {match.experiment_id}")
                    return

                # Determine winner based on primary metric
                primary_metric = tournament.config.primary_metric

                variant_a_score = experiment_results.variant_a_metrics.get(primary_metric, 0.0)
                variant_b_score = experiment_results.variant_b_metrics.get(primary_metric, 0.0)

                # Update match results
                match.status = MatchStatus.COMPLETED
                match.completed_at = datetime.now()
                match.participant1_score = variant_a_score
                match.participant2_score = variant_b_score

                # Determine winner
                if abs(variant_a_score - variant_b_score) < 0.001:  # Tie
                    match.winner_id = None
                    tournament.update_participant_stats(match.participant1_id, draw=True, score=variant_a_score)
                    tournament.update_participant_stats(match.participant2_id, draw=True, score=variant_b_score)
                elif variant_a_score > variant_b_score:
                    match.winner_id = match.participant1_id
                    tournament.update_participant_stats(match.participant1_id, win=True, score=variant_a_score)
                    tournament.update_participant_stats(match.participant2_id, win=False, score=variant_b_score)
                else:
                    match.winner_id = match.participant2_id
                    tournament.update_participant_stats(match.participant1_id, win=False, score=variant_a_score)
                    tournament.update_participant_stats(match.participant2_id, win=True, score=variant_b_score)

                tournament.completed_matches += 1

                self.logger.info(f"Match completed: {match.participant1_id} ({variant_a_score}) vs {match.participant2_id} ({variant_b_score}), winner: {match.winner_id}")

                # Check if round is complete and advance if needed
                if self.auto_advance:
                    self._check_round_advancement(tournament)

                self._save_tournament(tournament)

        except Exception as e:
            self.logger.error(f"Error processing match completion: {str(e)}")

    def _check_round_advancement(self, tournament: Tournament) -> None:
        """Check if current round is complete and advance tournament"""
        try:
            current_round_matches = tournament.bracket.get_round_matches(tournament.current_round)

            # Check if all matches in current round are completed
            all_completed = all(
                tournament.matches[match_id].is_completed
                for match_id in current_round_matches
            )

            if not all_completed:
                return

            self.logger.info(f"Round {tournament.current_round} completed for tournament {tournament.id}")

            # Try to advance to next round
            if self.bracket_generator.advance_tournament(tournament):
                if tournament.current_round <= tournament.total_rounds:
                    # Schedule next round matches
                    self._schedule_round_matches(tournament, tournament.current_round)
                    self.logger.info(f"Advanced to round {tournament.current_round}")
                else:
                    # Tournament completed
                    self._complete_tournament(tournament)

        except Exception as e:
            self.logger.error(f"Error checking round advancement: {str(e)}")

    def _complete_tournament(self, tournament: Tournament) -> None:
        """Complete a tournament and generate final results"""
        try:
            tournament.status = TournamentStatus.COMPLETED
            tournament.completed_at = datetime.now()

            # Remove from active tournaments
            if tournament.id in self.active_tournaments:
                del self.active_tournaments[tournament.id]

            # Generate tournament summary
            summary = self._generate_tournament_summary(tournament)

            self.logger.info(f"Tournament '{tournament.config.name}' completed. Champion: {tournament.champion_id}")

            # Publish tournament completion event
            self.event_bus.publish(Event(
                event_type=EventType.TRAINING_COMPLETED,  # Reuse existing event type
                data={
                    "tournament_id": tournament.id,
                    "tournament_name": tournament.config.name,
                    "champion_id": tournament.champion_id,
                    "total_matches": tournament.total_matches,
                    "duration_hours": (tournament.completed_at - tournament.actual_start).total_seconds() / 3600
                }
            ))

            self._save_tournament(tournament)

        except Exception as e:
            self.logger.error(f"Error completing tournament: {str(e)}")

    def _generate_tournament_summary(self, tournament: Tournament) -> TournamentSummary:
        """Generate summary statistics for completed tournament"""
        try:
            duration_hours = 0.0
            if tournament.actual_start and tournament.completed_at:
                duration_hours = (tournament.completed_at - tournament.actual_start).total_seconds() / 3600

            champion_name = ""
            champion_config = {}
            champion_win_rate = 0.0

            if tournament.champion_id and tournament.champion_id in tournament.participants:
                champion = tournament.participants[tournament.champion_id]
                champion_name = champion.name
                champion_config = champion.configuration
                champion_win_rate = champion.win_rate

            summary = TournamentSummary(
                tournament_id=tournament.id,
                tournament_name=tournament.config.name,
                tournament_type=tournament.config.tournament_type,
                total_participants=len(tournament.participants),
                total_matches=tournament.total_matches,
                duration_hours=duration_hours,
                champion_name=champion_name,
                champion_config=champion_config,
                champion_win_rate=champion_win_rate,
                completed_at=tournament.completed_at or datetime.now()
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating tournament summary: {str(e)}")
            return TournamentSummary()

    def get_tournament(self, tournament_id: str) -> Optional[Tournament]:
        """Get tournament by ID"""
        return self.tournaments.get(tournament_id)

    def list_tournaments(self) -> List[Tournament]:
        """Get list of all tournaments"""
        return list(self.tournaments.values())

    def get_active_tournaments(self) -> List[Tournament]:
        """Get list of active tournaments"""
        return list(self.active_tournaments.values())

    def _save_tournament(self, tournament: Tournament) -> None:
        """Save tournament data to disk"""
        try:
            file_path = self.data_dir / f"{tournament.id}.json"

            # Convert tournament to dict for JSON serialization
            tournament_data = {
                "id": tournament.id,
                "config": tournament.config.__dict__,
                "status": tournament.status.value,
                "current_round": tournament.current_round,
                "total_rounds": tournament.total_rounds,
                "participants": {pid: p.__dict__ for pid, p in tournament.participants.items()},
                "matches": {mid: m.__dict__ for mid, m in tournament.matches.items()},
                "bracket": tournament.bracket.__dict__,
                "final_rankings": tournament.final_rankings,
                "champion_id": tournament.champion_id,
                "runner_up_id": tournament.runner_up_id,
                "created_at": tournament.created_at.isoformat() if tournament.created_at else None,
                "updated_at": tournament.updated_at.isoformat() if tournament.updated_at else None
            }

            with open(file_path, 'w') as f:
                json.dump(tournament_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving tournament: {str(e)}")

    def _start_monitoring(self) -> None:
        """Start tournament monitoring thread"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_tournaments, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Started tournament monitoring")

    def _monitor_tournaments(self) -> None:
        """Monitor active tournaments for status updates"""
        while self._running:
            try:
                with self._lock:
                    for tournament in list(self.active_tournaments.values()):
                        # Check for timeout matches, stuck tournaments, etc.
                        self._check_tournament_health(tournament)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in tournament monitoring: {str(e)}")

    def _check_tournament_health(self, tournament: Tournament) -> None:
        """Check tournament health and handle issues"""
        # Implementation for timeout handling, stuck matches, etc.
        pass

    def stop(self) -> None:
        """Stop tournament manager"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Tournament manager stopped")