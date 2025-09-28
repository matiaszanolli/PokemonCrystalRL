"""
Tournament System - Competitive AI configuration battles

This module provides a comprehensive tournament management system for
competitive AI battles, leveraging the existing A/B testing automation
infrastructure for seamless experiment execution.
"""

from .tournament_models import (
    TournamentType,
    TournamentStatus,
    MatchStatus,
    ParticipantType,
    TournamentParticipant,
    TournamentMatch,
    TournamentBracket,
    TournamentConfig,
    Tournament,
    TournamentSummary
)
from .tournament_manager import TournamentManager
from .bracket_generator import BracketGenerator
from .tournament_profiles import TournamentProfiles
from .tournament_analytics import TournamentAnalytics

__all__ = [
    "TournamentType",
    "TournamentStatus",
    "MatchStatus",
    "ParticipantType",
    "TournamentParticipant",
    "TournamentMatch",
    "TournamentBracket",
    "TournamentConfig",
    "Tournament",
    "TournamentSummary",
    "TournamentManager",
    "BracketGenerator",
    "TournamentProfiles",
    "TournamentAnalytics"
]