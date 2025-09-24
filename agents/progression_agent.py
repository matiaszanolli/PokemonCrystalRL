"""
ProgressionAgent - Specialist agent for story completion and quest management

This agent specializes in game progression, quest completion, story advancement,
and achieving major milestones. It leverages the quest tracking system.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent

try:
    from training.components.strategic_context_builder import StrategicContextBuilder, QuestTracker
except ImportError:
    print("⚠️  StrategicContextBuilder not available")
    StrategicContextBuilder = None
    QuestTracker = None


class ProgressionPhase(Enum):
    """Different phases of game progression"""
    TUTORIAL = "tutorial"                    # Getting first Pokemon
    EARLY_EXPLORATION = "early_exploration"  # Learning basics, first routes
    GYM_PREPARATION = "gym_preparation"      # Preparing for gym challenges
    GYM_CHALLENGE = "gym_challenge"          # Active gym progression
    STORY_ADVANCEMENT = "story_advancement"   # Main story quests
    ENDGAME = "endgame"                      # Elite Four and beyond
    POST_GAME = "post_game"                  # Kanto and post-game content


@dataclass
class ProgressionObjective:
    """Represents a progression objective with strategy"""
    objective_id: str
    name: str
    description: str
    phase: ProgressionPhase
    priority: int  # 1-10
    estimated_actions: int
    success_criteria: Dict[str, Any]
    strategy_notes: str
    dependencies: List[str]


@dataclass
class ProgressionDecision:
    """Represents a progression-focused decision"""
    action: int
    decision_type: str  # 'quest_action', 'navigation', 'preparation', 'interaction'
    confidence: float
    reasoning: str
    objective_focus: str
    estimated_progress: float  # Expected progress toward current objective


class ProgressionAgent(BaseAgent):
    """Specialist agent optimized for story progression and quest completion"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = logging.getLogger("ProgressionAgent")

        # Progression-specific configuration
        self.progression_config = config.get('progression_config', {}) if config else {}
        self.efficiency_focus = self.progression_config.get('efficiency', 0.8)  # 0.0-1.0
        self.completionist_mode = self.progression_config.get('completionist', False)
        self.story_priority = self.progression_config.get('story_priority', 0.9)  # 0.0-1.0

        # Initialize strategic context for quest intelligence
        self.context_builder = StrategicContextBuilder() if StrategicContextBuilder else None

        # Progression tracking
        self.current_phase = ProgressionPhase.TUTORIAL
        self.active_objectives = []
        self.completed_objectives = set()
        self.failed_objectives = set()
        self.progression_history = []

        # Story and quest state
        self.story_progress = {
            'starter_received': False,
            'rival_encountered': False,
            'first_gym_completed': False,
            'elite_four_access': False
        }

        # Milestone tracking
        self.milestones = self._initialize_milestones()
        self.achievement_log = []

        # Strategy and decision making
        self.decision_patterns = {}
        self.success_strategies = {}
        self.failed_strategies = {}

        self.logger.info(f"ProgressionAgent initialized with efficiency={self.efficiency_focus}, story_priority={self.story_priority}")

    def get_action(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Get progression-optimized action"""

        # Update progression state
        self._update_progression_tracking(game_state, info)

        # Analyze progression context
        progression_analysis = self._analyze_progression_context(game_state, info)

        # Make progression decision
        decision = self._make_progression_decision(game_state, progression_analysis)

        # Track decision for learning
        self._track_progression_decision(decision, game_state)

        return decision.action, {
            'source': 'progression_agent',
            'decision_type': decision.decision_type,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'objective_focus': decision.objective_focus,
            'estimated_progress': decision.estimated_progress,
            'current_phase': self.current_phase.value,
            'active_objectives': len(self.active_objectives)
        }

    def _analyze_progression_context(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive progression situation analysis"""

        # Update current phase
        self._update_current_phase(game_state)

        # Get quest objectives if available
        quest_objectives = []
        if self.context_builder:
            quest_objectives = self.context_builder.get_current_quest_objectives(game_state)

        analysis = {
            'current_phase': self.current_phase,
            'game_state_analysis': {
                'party_count': game_state.get('party_count', 0),
                'badges_total': game_state.get('badges_total', 0),
                'player_level': game_state.get('player_level', 0),
                'money': game_state.get('money', 0),
                'player_map': game_state.get('player_map', 0)
            },
            'progression_status': {
                'story_progress': self.story_progress,
                'active_objectives': self.active_objectives,
                'completed_count': len(self.completed_objectives),
                'phase_completion': self._calculate_phase_completion()
            },
            'quest_intelligence': {
                'available_quests': quest_objectives,
                'highest_priority_quest': self._get_highest_priority_quest(quest_objectives),
                'quest_recommendations': self._analyze_quest_recommendations(quest_objectives, game_state)
            },
            'strategic_position': {
                'readiness_for_next_phase': self._assess_readiness_for_next_phase(game_state),
                'bottlenecks': self._identify_progression_bottlenecks(game_state),
                'opportunities': self._identify_progression_opportunities(game_state)
            }
        }

        return analysis

    def _make_progression_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Make intelligent progression decision"""

        # Handle phase-specific decision making
        phase_decision = self._make_phase_specific_decision(game_state, analysis)
        if phase_decision.confidence > 0.7:
            return phase_decision

        # Quest-driven decision making
        quest_decision = self._make_quest_driven_decision(game_state, analysis)
        if quest_decision.confidence > 0.6:
            return quest_decision

        # Strategic progression decision
        strategic_decision = self._make_strategic_progression_decision(game_state, analysis)
        if strategic_decision.confidence > 0.5:
            return strategic_decision

        # Fallback to basic progression
        return self._fallback_progression_decision(game_state, analysis)

    def _make_phase_specific_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Make decision based on current progression phase"""

        if self.current_phase == ProgressionPhase.TUTORIAL:
            return self._tutorial_phase_decision(game_state, analysis)
        elif self.current_phase == ProgressionPhase.EARLY_EXPLORATION:
            return self._early_exploration_decision(game_state, analysis)
        elif self.current_phase == ProgressionPhase.GYM_PREPARATION:
            return self._gym_preparation_decision(game_state, analysis)
        elif self.current_phase == ProgressionPhase.GYM_CHALLENGE:
            return self._gym_challenge_decision(game_state, analysis)
        elif self.current_phase == ProgressionPhase.STORY_ADVANCEMENT:
            return self._story_advancement_decision(game_state, analysis)
        else:
            return self._general_progression_decision(game_state, analysis)

    def _tutorial_phase_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Handle tutorial phase decisions"""
        party_count = game_state.get('party_count', 0)

        if party_count == 0:
            return ProgressionDecision(
                action=5,  # A button - interact with Professor Elm
                decision_type='quest_action',
                confidence=0.9,
                reasoning='No Pokemon yet - priority is getting starter from Professor Elm',
                objective_focus='get_starter_pokemon',
                estimated_progress=0.8
            )
        else:
            # Tutorial complete, transition to exploration
            return ProgressionDecision(
                action=2,  # Down - head to Route 29
                decision_type='navigation',
                confidence=0.8,
                reasoning='Starter obtained - begin exploration to Route 29',
                objective_focus='begin_journey',
                estimated_progress=0.6
            )

    def _early_exploration_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Handle early exploration phase decisions"""
        player_level = game_state.get('player_level', 0)
        current_map = game_state.get('player_map', 0)

        # Focus on leveling up and reaching Violet City
        if player_level < 10:
            return ProgressionDecision(
                action=5,  # A button - battle wild Pokemon/trainers
                decision_type='preparation',
                confidence=0.8,
                reasoning=f'Level {player_level} too low - need to train before gym',
                objective_focus='level_grinding',
                estimated_progress=0.4
            )
        elif current_map != 8:  # Not in Violet City
            return ProgressionDecision(
                action=1,  # Up - navigate toward Violet City
                decision_type='navigation',
                confidence=0.7,
                reasoning='Ready for gym - navigate to Violet City',
                objective_focus='reach_violet_city',
                estimated_progress=0.6
            )
        else:
            return ProgressionDecision(
                action=5,  # A button - enter gym
                decision_type='quest_action',
                confidence=0.9,
                reasoning='In Violet City and ready - challenge gym',
                objective_focus='gym_challenge',
                estimated_progress=0.9
            )

    def _gym_preparation_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Handle gym preparation phase decisions"""
        player_hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
        badges = game_state.get('badges_total', 0)

        # Ensure readiness for gym battle
        if player_hp_ratio < 0.8:
            return ProgressionDecision(
                action=1,  # Up - find Pokemon Center
                decision_type='preparation',
                confidence=0.9,
                reasoning='Low HP - heal at Pokemon Center before gym',
                objective_focus='heal_before_gym',
                estimated_progress=0.3
            )
        else:
            return ProgressionDecision(
                action=5,  # A button - enter gym
                decision_type='quest_action',
                confidence=0.8,
                reasoning='Ready for gym challenge',
                objective_focus='enter_gym',
                estimated_progress=0.8
            )

    def _gym_challenge_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Handle active gym challenge decisions"""
        in_battle = game_state.get('in_battle', False)

        if in_battle:
            return ProgressionDecision(
                action=5,  # A button - battle action
                decision_type='quest_action',
                confidence=0.8,
                reasoning='In gym battle - focus on winning',
                objective_focus='win_gym_battle',
                estimated_progress=0.7
            )
        else:
            return ProgressionDecision(
                action=1,  # Up - navigate through gym
                decision_type='navigation',
                confidence=0.7,
                reasoning='Navigate through gym to reach leader',
                objective_focus='reach_gym_leader',
                estimated_progress=0.5
            )

    def _story_advancement_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Handle story advancement decisions"""
        badges = game_state.get('badges_total', 0)

        if badges < 8:
            return ProgressionDecision(
                action=1,  # Up - navigate to next gym
                decision_type='navigation',
                confidence=0.8,
                reasoning=f'Have {badges} badges - continue gym circuit',
                objective_focus='next_gym',
                estimated_progress=0.6
            )
        else:
            return ProgressionDecision(
                action=1,  # Up - navigate to Elite Four
                decision_type='navigation',
                confidence=0.9,
                reasoning='All gym badges obtained - head to Elite Four',
                objective_focus='elite_four',
                estimated_progress=0.8
            )

    def _general_progression_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """General progression decision when no specific phase applies"""
        return ProgressionDecision(
            action=5,  # A button - general interaction
            decision_type='quest_action',
            confidence=0.5,
            reasoning='General progression action',
            objective_focus='continue_journey',
            estimated_progress=0.4
        )

    def _make_quest_driven_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Make decision based on active quests"""
        highest_priority = analysis['quest_intelligence']['highest_priority_quest']

        if highest_priority:
            quest_action = self._determine_quest_action(highest_priority, game_state)
            return ProgressionDecision(
                action=quest_action,
                decision_type='quest_action',
                confidence=0.7,
                reasoning=f'Pursuing quest: {highest_priority.get("name", "unknown")}',
                objective_focus=highest_priority.get('quest_id', 'quest_progress'),
                estimated_progress=0.6
            )

        return ProgressionDecision(action=1, decision_type='navigation', confidence=0.3, reasoning='No clear quest guidance', objective_focus='explore', estimated_progress=0.2)

    def _make_strategic_progression_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Make strategic decision for long-term progression"""
        bottlenecks = analysis['strategic_position']['bottlenecks']
        opportunities = analysis['strategic_position']['opportunities']

        # Address bottlenecks first
        if bottlenecks:
            return ProgressionDecision(
                action=5,  # A button - address bottleneck
                decision_type='preparation',
                confidence=0.6,
                reasoning=f'Addressing progression bottleneck: {bottlenecks[0]}',
                objective_focus='remove_bottleneck',
                estimated_progress=0.5
            )

        # Pursue opportunities
        if opportunities:
            return ProgressionDecision(
                action=5,  # A button - pursue opportunity
                decision_type='quest_action',
                confidence=0.6,
                reasoning=f'Pursuing opportunity: {opportunities[0]}',
                objective_focus='seize_opportunity',
                estimated_progress=0.5
            )

        return ProgressionDecision(action=1, decision_type='navigation', confidence=0.4, reasoning='Strategic exploration', objective_focus='strategic_position', estimated_progress=0.3)

    def _fallback_progression_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> ProgressionDecision:
        """Fallback decision when no clear progression path"""
        return ProgressionDecision(
            action=5,  # A button - general forward progress
            decision_type='navigation',
            confidence=0.4,
            reasoning='No clear progression path - continue forward',
            objective_focus='general_progress',
            estimated_progress=0.3
        )

    def _update_current_phase(self, game_state: Dict[str, Any]):
        """Update current progression phase based on game state"""
        party_count = game_state.get('party_count', 0)
        badges = game_state.get('badges_total', 0)
        in_battle = game_state.get('in_battle', False)
        current_map = game_state.get('player_map', 0)

        if party_count == 0:
            self.current_phase = ProgressionPhase.TUTORIAL
        elif badges == 0 and party_count > 0:
            if current_map == 8 and in_battle:  # In Violet City gym battle
                self.current_phase = ProgressionPhase.GYM_CHALLENGE
            elif current_map == 8:  # In Violet City
                self.current_phase = ProgressionPhase.GYM_PREPARATION
            else:
                self.current_phase = ProgressionPhase.EARLY_EXPLORATION
        elif badges < 8:
            if in_battle and current_map in [8, 12, 16, 21]:  # In gym battle
                self.current_phase = ProgressionPhase.GYM_CHALLENGE
            else:
                self.current_phase = ProgressionPhase.STORY_ADVANCEMENT
        else:
            self.current_phase = ProgressionPhase.ENDGAME

    def _update_progression_tracking(self, game_state: Dict[str, Any], info: Dict[str, Any]):
        """Update progression state tracking"""
        party_count = game_state.get('party_count', 0)
        badges = game_state.get('badges_total', 0)

        # Update story progress flags
        if party_count > 0 and not self.story_progress['starter_received']:
            self.story_progress['starter_received'] = True
            self._log_achievement('starter_received', 'Received starter Pokemon')

        if badges >= 1 and not self.story_progress['first_gym_completed']:
            self.story_progress['first_gym_completed'] = True
            self._log_achievement('first_gym_completed', 'Completed first gym')

    def _log_achievement(self, achievement_id: str, description: str):
        """Log progression achievement"""
        achievement = {
            'id': achievement_id,
            'description': description,
            'timestamp': self.total_steps,
            'phase': self.current_phase.value
        }
        self.achievement_log.append(achievement)
        self.logger.info(f"Achievement unlocked: {description}")

    def _initialize_milestones(self) -> Dict[str, Dict[str, Any]]:
        """Initialize progression milestones"""
        return {
            'starter_obtained': {'completed': False, 'phase': ProgressionPhase.TUTORIAL},
            'first_route_explored': {'completed': False, 'phase': ProgressionPhase.EARLY_EXPLORATION},
            'first_gym_badge': {'completed': False, 'phase': ProgressionPhase.GYM_CHALLENGE},
            'half_gym_badges': {'completed': False, 'phase': ProgressionPhase.STORY_ADVANCEMENT},
            'all_gym_badges': {'completed': False, 'phase': ProgressionPhase.STORY_ADVANCEMENT},
            'elite_four_access': {'completed': False, 'phase': ProgressionPhase.ENDGAME}
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get progression agent statistics"""
        base_stats = super().get_stats()

        progression_stats = {
            'current_phase': self.current_phase.value,
            'completed_objectives': len(self.completed_objectives),
            'failed_objectives': len(self.failed_objectives),
            'achievements_unlocked': len(self.achievement_log),
            'efficiency_focus': self.efficiency_focus,
            'story_priority': self.story_priority,
            'phase_completion': self._calculate_phase_completion(),
            'story_progress': self.story_progress
        }

        return {**base_stats, **progression_stats}

    # Helper methods (simplified implementations)
    def _calculate_phase_completion(self) -> float:
        """Calculate completion percentage of current phase"""
        # Simplified implementation
        if self.current_phase == ProgressionPhase.TUTORIAL:
            return 1.0 if self.story_progress['starter_received'] else 0.0
        return 0.5  # Placeholder

    def _get_highest_priority_quest(self, quest_objectives: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the highest priority quest"""
        if not quest_objectives:
            return None
        return max(quest_objectives, key=lambda q: q.get('priority', 0))

    def _analyze_quest_recommendations(self, quest_objectives: List[Dict[str, Any]], game_state: Dict[str, Any]) -> List[str]:
        """Analyze and provide quest recommendations"""
        recommendations = []
        for quest in quest_objectives[:3]:  # Top 3 quests
            recommendations.append(f"Consider: {quest.get('name', 'Unknown quest')}")
        return recommendations

    def _assess_readiness_for_next_phase(self, game_state: Dict[str, Any]) -> bool:
        """Assess if ready to progress to next phase"""
        if self.current_phase == ProgressionPhase.TUTORIAL:
            return game_state.get('party_count', 0) > 0
        elif self.current_phase == ProgressionPhase.EARLY_EXPLORATION:
            return game_state.get('player_level', 0) >= 10
        return True

    def _identify_progression_bottlenecks(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify what's blocking progression"""
        bottlenecks = []
        if game_state.get('party_count', 0) == 0:
            bottlenecks.append('need_starter_pokemon')
        if game_state.get('player_level', 0) < 10 and game_state.get('badges_total', 0) == 0:
            bottlenecks.append('level_too_low')
        return bottlenecks

    def _identify_progression_opportunities(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify progression opportunities"""
        opportunities = []
        if game_state.get('player_level', 0) >= 10 and game_state.get('badges_total', 0) == 0:
            opportunities.append('ready_for_first_gym')
        return opportunities

    def _determine_quest_action(self, quest: Dict[str, Any], game_state: Dict[str, Any]) -> int:
        """Determine specific action for quest progression"""
        quest_id = quest.get('quest_id', '')

        if 'pokemon' in quest_id.lower():
            return 5  # A button - interact
        elif 'gym' in quest_id.lower():
            return 1  # Up - navigate to gym
        else:
            return 5  # A button - general interaction

    def _track_progression_decision(self, decision: ProgressionDecision, game_state: Dict[str, Any]):
        """Track decision for learning and improvement"""
        decision_record = {
            'action': decision.action,
            'decision_type': decision.decision_type,
            'confidence': decision.confidence,
            'objective_focus': decision.objective_focus,
            'phase': self.current_phase.value,
            'timestamp': self.total_steps
        }

        self.progression_history.append(decision_record)

        # Keep only recent history
        if len(self.progression_history) > 100:
            self.progression_history.pop(0)