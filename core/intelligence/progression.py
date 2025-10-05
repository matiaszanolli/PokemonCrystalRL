"""
Progression Tracking Module

Tracks game progress and determines objectives for Pokemon Crystal.
"""

from typing import Dict, List
import logging

from environments.state.analyzer import GamePhase
from .location import LocationType


class ProgressTracker:
    """Tracks game progress and determines next objectives"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_game_phase(self, game_state: Dict) -> GamePhase:
        """Determine current game progression phase"""
        party_count = game_state.get('party_count', 0)
        badges_total = game_state.get('badges_total', 0)

        if party_count == 0:
            return GamePhase.TUTORIAL
        elif badges_total == 0 and party_count > 0:
            return GamePhase.EARLY_GAME
        elif badges_total < 4:
            return GamePhase.GYM_BATTLES
        elif badges_total < 8:
            return GamePhase.LATE_GAME
        elif badges_total < 16:
            return GamePhase.POST_GAME
        else:
            return GamePhase.POST_GAME

    def get_immediate_goals(self, game_state: Dict, location_type: LocationType) -> List[str]:
        """Get immediate, actionable goals"""
        goals = []
        phase = self.get_game_phase(game_state)
        party_count = game_state.get('party_count', 0)

        # Health is only a priority if we actually have Pokemon
        if party_count > 0:
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
            if hp_ratio < 0.3:
                goals.append("URGENT: Heal Pokemon immediately")
            elif hp_ratio < 0.6:
                goals.append("Find healing when convenient")

        # Phase-specific goals
        if phase == GamePhase.TUTORIAL:
            goals.append("Get your first Pokemon from Professor Elm")

        elif phase == GamePhase.EARLY_GAME:
            goals.append("Level up your Pokemon to ~10")
            goals.append("Explore routes and catch more Pokemon")
            goals.append("Head to Violet City for first gym")

        elif phase == GamePhase.GYM_BATTLES:
            badges = game_state.get('badges_total', 0)
            goals.append(f"Prepare for gym #{badges + 1}")
            goals.append("Level Pokemon to ~15-20")

        return goals

    def get_strategic_goals(self, game_state: Dict) -> List[str]:
        """Get longer-term strategic goals"""
        goals = []
        phase = self.get_game_phase(game_state)

        if phase == GamePhase.TUTORIAL:
            goals.append("Complete Professor Elm's tasks")
            goals.append("Learn basic game mechanics")

        elif phase == GamePhase.EARLY_GAME:
            goals.append("Build a balanced party")
            goals.append("Earn first gym badge")

        elif phase == GamePhase.GYM_BATTLES:
            goals.append("Earn all 8 Johto badges")
            goals.append("Prepare for Elite Four")

        return goals
