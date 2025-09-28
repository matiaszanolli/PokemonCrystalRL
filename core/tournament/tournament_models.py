"""
Tournament Models - Data structures for competitive AI tournaments

Defines the core data models for tournament management, including
brackets, matches, participants, and tournament configurations.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from datetime import datetime

from core.ab_testing.experiment_models import PerformanceMetrics, ExperimentConfig


class TournamentType(Enum):
    """Types of tournament formats"""
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS_SYSTEM = "swiss_system"
    LADDER = "ladder"


class TournamentStatus(Enum):
    """Status of a tournament"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MatchStatus(Enum):
    """Status of individual matches"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipantType(Enum):
    """Types of tournament participants"""
    AI_CONFIGURATION = "ai_configuration"
    PLUGIN_COMBINATION = "plugin_combination"
    AGENT_STRATEGY = "agent_strategy"
    HYBRID_SETUP = "hybrid_setup"


@dataclass
class TournamentParticipant:
    """Represents a tournament participant (AI configuration)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    participant_type: ParticipantType = ParticipantType.AI_CONFIGURATION

    # Configuration details
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_score: float = 0.0
    average_performance: Optional[PerformanceMetrics] = None

    # Tournament-specific stats
    elimination_round: Optional[int] = None  # Round eliminated (for elimination tournaments)
    ranking: Optional[int] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def points(self) -> int:
        """Calculate tournament points (wins=3, draws=1, losses=0)"""
        return self.wins * 3 + self.draws * 1


@dataclass
class TournamentMatch:
    """Represents a single match between participants"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tournament_id: str = ""
    round_number: int = 1
    match_number: int = 1

    # Participants
    participant1_id: str = ""
    participant2_id: str = ""

    # Results
    status: MatchStatus = MatchStatus.PENDING
    winner_id: Optional[str] = None
    participant1_score: Optional[float] = None
    participant2_score: Optional[float] = None

    # Performance data
    participant1_metrics: Optional[PerformanceMetrics] = None
    participant2_metrics: Optional[PerformanceMetrics] = None

    # Match configuration
    experiment_config: Optional[ExperimentConfig] = None
    experiment_id: Optional[str] = None

    # Timing
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_bye(self) -> bool:
        """Check if this is a bye match (one participant)"""
        return not self.participant2_id

    @property
    def is_completed(self) -> bool:
        """Check if match is completed"""
        return self.status == MatchStatus.COMPLETED


@dataclass
class TournamentBracket:
    """Represents tournament bracket structure"""
    tournament_id: str = ""
    rounds: List[List[str]] = field(default_factory=list)  # List of rounds, each containing match IDs

    # Bracket metadata
    total_rounds: int = 0
    current_round: int = 1
    bracket_type: str = "main"  # "main", "losers", "finals"

    def get_round_matches(self, round_number: int) -> List[str]:
        """Get match IDs for a specific round"""
        if 0 < round_number <= len(self.rounds):
            return self.rounds[round_number - 1]
        return []

    def add_round(self, match_ids: List[str]) -> None:
        """Add a new round to the bracket"""
        self.rounds.append(match_ids)
        self.total_rounds = len(self.rounds)


@dataclass
class TournamentConfig:
    """Configuration for tournament setup"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tournament_type: TournamentType = TournamentType.SINGLE_ELIMINATION

    # Participant limits
    max_participants: int = 16
    min_participants: int = 4

    # Match configuration
    match_duration_minutes: int = 30
    max_actions_per_match: int = 1000
    save_state_path: Optional[str] = None

    # Scoring configuration
    primary_metric: str = "reward"  # Primary metric for determining winners
    secondary_metrics: List[str] = field(default_factory=list)
    win_threshold: Optional[float] = None  # Minimum score to win

    # Tournament rules
    allow_ties: bool = True
    tiebreaker_metrics: List[str] = field(default_factory=lambda: ["actions_per_second", "progress_rate"])

    # Scheduling
    auto_advance: bool = True  # Automatically advance to next round
    round_delay_minutes: int = 5  # Delay between rounds

    # Automation integration
    use_experiment_framework: bool = True
    experiment_template: str = "tournament_match"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Tournament:
    """Main tournament data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: TournamentConfig = field(default_factory=TournamentConfig)

    # Tournament state
    status: TournamentStatus = TournamentStatus.DRAFT
    current_round: int = 0
    total_rounds: int = 0

    # Participants and matches
    participants: Dict[str, TournamentParticipant] = field(default_factory=dict)
    matches: Dict[str, TournamentMatch] = field(default_factory=dict)
    bracket: TournamentBracket = field(default_factory=TournamentBracket)

    # Results and rankings
    final_rankings: List[str] = field(default_factory=list)  # Participant IDs in ranking order
    champion_id: Optional[str] = None
    runner_up_id: Optional[str] = None

    # Statistics
    total_matches: int = 0
    completed_matches: int = 0
    average_match_duration: Optional[float] = None
    tournament_statistics: Dict[str, Any] = field(default_factory=dict)

    # Timing
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_participant(self, participant: TournamentParticipant) -> bool:
        """Add a participant to the tournament"""
        if len(self.participants) >= self.config.max_participants:
            return False

        self.participants[participant.id] = participant
        self.config.updated_at = datetime.now()
        return True

    def remove_participant(self, participant_id: str) -> bool:
        """Remove a participant from the tournament"""
        if participant_id in self.participants:
            del self.participants[participant_id]
            self.config.updated_at = datetime.now()
            return True
        return False

    def get_participant_by_name(self, name: str) -> Optional[TournamentParticipant]:
        """Find participant by name"""
        for participant in self.participants.values():
            if participant.name == name:
                return participant
        return None

    def update_participant_stats(self, participant_id: str, win: bool = False,
                               draw: bool = False, score: float = 0.0) -> None:
        """Update participant statistics after a match"""
        if participant_id not in self.participants:
            return

        participant = self.participants[participant_id]
        participant.matches_played += 1
        participant.total_score += score

        if win:
            participant.wins += 1
        elif draw:
            participant.draws += 1
        else:
            participant.losses += 1

        participant.updated_at = datetime.now()

    @property
    def is_completed(self) -> bool:
        """Check if tournament is completed"""
        return self.status == TournamentStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """Check if tournament is currently running"""
        return self.status == TournamentStatus.RUNNING

    @property
    def can_start(self) -> bool:
        """Check if tournament can be started"""
        return (self.status == TournamentStatus.SCHEDULED and
                len(self.participants) >= self.config.min_participants)


@dataclass
class TournamentSummary:
    """Summary statistics for completed tournaments"""
    tournament_id: str = ""
    tournament_name: str = ""
    tournament_type: TournamentType = TournamentType.SINGLE_ELIMINATION

    # Basic stats
    total_participants: int = 0
    total_matches: int = 0
    duration_hours: float = 0.0

    # Champion information
    champion_name: str = ""
    champion_config: Dict[str, Any] = field(default_factory=dict)
    champion_win_rate: float = 0.0

    # Performance insights
    most_effective_strategy: str = ""
    highest_scoring_match: Optional[str] = None
    closest_match: Optional[str] = None

    # Metadata
    completed_at: datetime = field(default_factory=datetime.now)