"""
Tournament Configuration Profiles - Pre-built tournament setups

Provides pre-configured tournament profiles for common competitive scenarios,
making it easy to set up tournaments for different objectives and formats.
"""

import logging
from typing import Dict, List, Any
from datetime import timedelta

from .tournament_models import (
    TournamentConfig,
    TournamentParticipant,
    TournamentType,
    ParticipantType
)


class TournamentProfiles:
    """
    Pre-built tournament configuration profiles for different scenarios.

    Provides templates for:
    - Quick battles (fast testing)
    - Championship tournaments (comprehensive competition)
    - Research tournaments (strategy comparison)
    - Endurance challenges (long-form competition)
    """

    def __init__(self):
        self.logger = logging.getLogger("TournamentProfiles")

    def get_quick_battle_config(self, name: str = "Quick Battle") -> TournamentConfig:
        """
        Quick battle configuration for fast AI comparison.

        Ideal for:
        - Rapid strategy testing
        - Development iteration
        - Quick verification of changes
        """
        return TournamentConfig(
            name=name,
            description="Fast-paced tournament for quick AI strategy comparison",
            tournament_type=TournamentType.SINGLE_ELIMINATION,
            max_participants=8,
            min_participants=4,
            match_duration_minutes=10,
            max_actions_per_match=300,
            save_state_path="roms/pokemon_crystal.gbc.state",
            primary_metric="reward",
            secondary_metrics=["actions_per_second"],
            auto_advance=True,
            round_delay_minutes=1,
            use_experiment_framework=True,
            experiment_template="quick_tournament_match"
        )

    def get_championship_config(self, name: str = "Championship Tournament") -> TournamentConfig:
        """
        Comprehensive championship configuration.

        Ideal for:
        - Determining best overall strategy
        - Comprehensive competitive analysis
        - Showcase events
        """
        return TournamentConfig(
            name=name,
            description="Comprehensive championship tournament for determining the ultimate AI strategy",
            tournament_type=TournamentType.DOUBLE_ELIMINATION,
            max_participants=16,
            min_participants=8,
            match_duration_minutes=45,
            max_actions_per_match=2000,
            save_state_path="roms/pokemon_crystal.gbc.state",
            primary_metric="progress_rate",
            secondary_metrics=["reward", "battle_win_rate", "exploration_coverage"],
            auto_advance=True,
            round_delay_minutes=10,
            use_experiment_framework=True,
            experiment_template="championship_match"
        )

    def get_research_config(self, name: str = "Research Tournament") -> TournamentConfig:
        """
        Research-focused tournament configuration.

        Ideal for:
        - Strategy effectiveness research
        - Academic studies
        - Detailed performance analysis
        """
        return TournamentConfig(
            name=name,
            description="Round robin tournament for comprehensive strategy research and analysis",
            tournament_type=TournamentType.ROUND_ROBIN,
            max_participants=6,
            min_participants=4,
            match_duration_minutes=30,
            max_actions_per_match=1500,
            save_state_path="roms/pokemon_crystal.gbc.state",
            primary_metric="progress_rate",
            secondary_metrics=[
                "reward",
                "actions_per_second",
                "battle_win_rate",
                "exploration_coverage",
                "llm_decision_quality"
            ],
            auto_advance=True,
            round_delay_minutes=5,
            use_experiment_framework=True,
            experiment_template="research_match"
        )

    def get_endurance_config(self, name: str = "Endurance Challenge") -> TournamentConfig:
        """
        Long-form endurance tournament configuration.

        Ideal for:
        - Testing strategy stability
        - Long-term performance analysis
        - Identifying robust strategies
        """
        return TournamentConfig(
            name=name,
            description="Extended endurance tournament testing long-term strategy performance",
            tournament_type=TournamentType.SWISS_SYSTEM,
            max_participants=12,
            min_participants=6,
            match_duration_minutes=60,
            max_actions_per_match=3000,
            save_state_path="roms/pokemon_crystal.gbc.state",
            primary_metric="progress_rate",
            secondary_metrics=["reward", "battle_win_rate", "exploration_coverage"],
            auto_advance=True,
            round_delay_minutes=15,
            use_experiment_framework=True,
            experiment_template="endurance_match"
        )

    def get_speedrun_config(self, name: str = "Speedrun Tournament") -> TournamentConfig:
        """
        Speedrun-focused tournament configuration.

        Ideal for:
        - Testing speed optimization strategies
        - Fast completion challenges
        - Actions-per-second optimization
        """
        return TournamentConfig(
            name=name,
            description="Speedrun tournament focused on rapid game progression",
            tournament_type=TournamentType.SINGLE_ELIMINATION,
            max_participants=8,
            min_participants=4,
            match_duration_minutes=20,
            max_actions_per_match=800,
            save_state_path="roms/pokemon_crystal.gbc.state",
            primary_metric="actions_per_second",
            secondary_metrics=["progress_rate", "reward"],
            auto_advance=True,
            round_delay_minutes=2,
            use_experiment_framework=True,
            experiment_template="speedrun_match"
        )

    def get_battle_focused_config(self, name: str = "Battle Tournament") -> TournamentConfig:
        """
        Battle-focused tournament configuration.

        Ideal for:
        - Testing combat strategies
        - Battle AI optimization
        - Type effectiveness analysis
        """
        return TournamentConfig(
            name=name,
            description="Tournament focused on battle strategy and combat effectiveness",
            tournament_type=TournamentType.SINGLE_ELIMINATION,
            max_participants=8,
            min_participants=4,
            match_duration_minutes=25,
            max_actions_per_match=1000,
            save_state_path="roms/pokemon_crystal_battle.gbc.state",  # Battle-specific save state
            primary_metric="battle_win_rate",
            secondary_metrics=["reward", "progress_rate"],
            auto_advance=True,
            round_delay_minutes=3,
            use_experiment_framework=True,
            experiment_template="battle_match"
        )

    def create_sample_participants(self, participant_type: str = "mixed") -> List[TournamentParticipant]:
        """
        Create sample participants for tournaments.

        Args:
            participant_type: Type of participants to create
                - "mixed": Variety of different strategies
                - "battle": Battle-focused strategies
                - "exploration": Exploration-focused strategies
                - "speed": Speed-optimized strategies
                - "research": Research baseline strategies

        Returns:
            List of sample participants
        """
        participants = []

        if participant_type == "mixed":
            participants.extend(self._create_mixed_participants())
        elif participant_type == "battle":
            participants.extend(self._create_battle_participants())
        elif participant_type == "exploration":
            participants.extend(self._create_exploration_participants())
        elif participant_type == "speed":
            participants.extend(self._create_speed_participants())
        elif participant_type == "research":
            participants.extend(self._create_research_participants())
        else:
            self.logger.warning(f"Unknown participant type: {participant_type}, using mixed")
            participants.extend(self._create_mixed_participants())

        return participants

    def _create_mixed_participants(self) -> List[TournamentParticipant]:
        """Create diverse set of participants"""
        return [
            TournamentParticipant(
                name="Battle Master",
                description="AI optimized for battle performance",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["aggressive_battle_strategy"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 5,
                    "focus": "battle_optimization",
                    "battle_strategy": "aggressive"
                }
            ),
            TournamentParticipant(
                name="Explorer Pro",
                description="AI focused on map exploration and discovery",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["systematic_exploration"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 10,
                    "focus": "exploration_coverage",
                    "exploration_pattern": "systematic"
                }
            ),
            TournamentParticipant(
                name="Balanced Strategist",
                description="Well-rounded AI with balanced approach",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["balanced_strategy"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 8,
                    "focus": "balanced_performance",
                    "strategy_type": "adaptive"
                }
            ),
            TournamentParticipant(
                name="Speed Runner",
                description="AI optimized for rapid progression",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["speed_optimization"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 3,
                    "focus": "actions_per_second",
                    "optimization_target": "speed"
                }
            ),
            TournamentParticipant(
                name="Progression Expert",
                description="AI specialized in story progression",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["progression_focused"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 12,
                    "focus": "progress_rate",
                    "progression_strategy": "objective_based"
                }
            ),
            TournamentParticipant(
                name="Hybrid Intelligence",
                description="Advanced hybrid LLM-RL approach",
                participant_type=ParticipantType.HYBRID_SETUP,
                configuration={
                    "plugins": ["hybrid_strategy"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 15,
                    "focus": "adaptive_learning",
                    "hybrid_mode": "llm_rl_combined",
                    "rl_weight": 0.3,
                    "llm_weight": 0.7
                }
            )
        ]

    def _create_battle_participants(self) -> List[TournamentParticipant]:
        """Create battle-focused participants"""
        return [
            TournamentParticipant(
                name="Aggressive Battler",
                description="All-out aggressive battle strategy",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["aggressive_battle_strategy"],
                    "battle_strategy": "aggressive",
                    "focus": "battle_win_rate"
                }
            ),
            TournamentParticipant(
                name="Defensive Strategist",
                description="Defensive battle approach with longevity focus",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["defensive_battle_strategy"],
                    "battle_strategy": "defensive",
                    "focus": "battle_survival"
                }
            ),
            TournamentParticipant(
                name="Type Specialist",
                description="Strategy focused on type effectiveness",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["type_effectiveness_strategy"],
                    "battle_strategy": "type_focused",
                    "focus": "type_advantage"
                }
            ),
            TournamentParticipant(
                name="Adaptive Fighter",
                description="Adaptive battle strategy based on opponent",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["adaptive_battle_strategy"],
                    "battle_strategy": "adaptive",
                    "focus": "opponent_analysis"
                }
            )
        ]

    def _create_exploration_participants(self) -> List[TournamentParticipant]:
        """Create exploration-focused participants"""
        return [
            TournamentParticipant(
                name="Systematic Explorer",
                description="Methodical systematic exploration",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["systematic_exploration"],
                    "exploration_pattern": "systematic",
                    "focus": "exploration_coverage"
                }
            ),
            TournamentParticipant(
                name="Spiral Searcher",
                description="Spiral-based exploration pattern",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["spiral_exploration"],
                    "exploration_pattern": "spiral",
                    "focus": "map_discovery"
                }
            ),
            TournamentParticipant(
                name="Wall Follower",
                description="Wall-following exploration strategy",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["wall_following_exploration"],
                    "exploration_pattern": "wall_following",
                    "focus": "thorough_coverage"
                }
            ),
            TournamentParticipant(
                name="Random Walker",
                description="Random exploration with smart backtracking",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["random_exploration"],
                    "exploration_pattern": "random_smart",
                    "focus": "discovery_rate"
                }
            )
        ]

    def _create_speed_participants(self) -> List[TournamentParticipant]:
        """Create speed-optimized participants"""
        return [
            TournamentParticipant(
                name="Lightning Fast",
                description="Maximum speed optimization",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["speed_optimization"],
                    "llm_interval": 1,
                    "focus": "actions_per_second",
                    "speed_mode": "maximum"
                }
            ),
            TournamentParticipant(
                name="Efficient Mover",
                description="Efficient movement patterns",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["efficient_movement"],
                    "llm_interval": 2,
                    "focus": "movement_efficiency",
                    "movement_optimization": True
                }
            ),
            TournamentParticipant(
                name="Quick Decider",
                description="Fast decision making with minimal LLM",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["quick_decisions"],
                    "llm_interval": 20,
                    "focus": "decision_speed",
                    "quick_mode": True
                }
            ),
            TournamentParticipant(
                name="Speedrun Pro",
                description="Professional speedrun techniques",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["speedrun_techniques"],
                    "llm_interval": 5,
                    "focus": "completion_time",
                    "speedrun_optimizations": True
                }
            )
        ]

    def _create_research_participants(self) -> List[TournamentParticipant]:
        """Create research baseline participants"""
        return [
            TournamentParticipant(
                name="Control Baseline",
                description="Standard baseline configuration for comparison",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": [],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 10,
                    "focus": "baseline",
                    "research_baseline": True
                }
            ),
            TournamentParticipant(
                name="LLM Heavy",
                description="High reliance on LLM decision making",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": [],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 1,
                    "focus": "llm_decisions",
                    "llm_heavy": True
                }
            ),
            TournamentParticipant(
                name="Plugin Ensemble",
                description="Multiple plugins working together",
                participant_type=ParticipantType.PLUGIN_COMBINATION,
                configuration={
                    "plugins": ["balanced_strategy", "adaptive_exploration", "smart_battle"],
                    "llm_model": "smollm2:1.7b",
                    "llm_interval": 8,
                    "focus": "plugin_synergy",
                    "ensemble_mode": True
                }
            ),
            TournamentParticipant(
                name="Rule Based",
                description="Primarily rule-based with minimal LLM",
                participant_type=ParticipantType.AI_CONFIGURATION,
                configuration={
                    "plugins": ["rule_based_strategy"],
                    "llm_interval": 30,
                    "focus": "rule_based",
                    "rule_heavy": True
                }
            )
        ]

    def get_profile_list(self) -> List[Dict[str, Any]]:
        """Get list of available tournament profiles"""
        return [
            {
                "id": "quick_battle",
                "name": "Quick Battle",
                "description": "Fast-paced tournament for quick AI strategy comparison",
                "duration": "Short (10-15 minutes)",
                "participants": "4-8",
                "format": "Single Elimination",
                "use_case": "Development testing, rapid iteration"
            },
            {
                "id": "championship",
                "name": "Championship Tournament",
                "description": "Comprehensive championship for determining ultimate AI strategy",
                "duration": "Long (2-4 hours)",
                "participants": "8-16",
                "format": "Double Elimination",
                "use_case": "Showcase events, determining best strategy"
            },
            {
                "id": "research",
                "name": "Research Tournament",
                "description": "Round robin for comprehensive strategy research",
                "duration": "Medium (1-2 hours)",
                "participants": "4-6",
                "format": "Round Robin",
                "use_case": "Academic research, detailed analysis"
            },
            {
                "id": "endurance",
                "name": "Endurance Challenge",
                "description": "Extended tournament testing long-term performance",
                "duration": "Very Long (4-8 hours)",
                "participants": "6-12",
                "format": "Swiss System",
                "use_case": "Stability testing, robust strategy identification"
            },
            {
                "id": "speedrun",
                "name": "Speedrun Tournament",
                "description": "Tournament focused on rapid game progression",
                "duration": "Short (15-30 minutes)",
                "participants": "4-8",
                "format": "Single Elimination",
                "use_case": "Speed optimization, actions-per-second testing"
            },
            {
                "id": "battle_focused",
                "name": "Battle Tournament",
                "description": "Tournament focused on battle strategy and combat",
                "duration": "Medium (30-60 minutes)",
                "participants": "4-8",
                "format": "Single Elimination",
                "use_case": "Combat strategy testing, battle AI optimization"
            }
        ]