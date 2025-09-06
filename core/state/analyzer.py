#!/usr/bin/env python3
"""
Game State Analyzer for Pokemon Crystal RL

This module provides comprehensive analysis of game state variables,
their relationships, and strategic implications for LLM decision-making.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class GamePhase(Enum):
    """Different phases of the Pokemon Crystal game"""
    INTRO = "intro"
    EARLY_GAME = "early_game"  # No Pokemon yet
    STARTER_PHASE = "starter_phase"  # Getting first Pokemon
    EXPLORATION = "exploration"  # Exploring with Pokemon
    GYM_BATTLES = "gym_battles"  # Fighting gym leaders
    LATE_GAME = "late_game"  # Elite Four, Champion
    POST_GAME = "post_game"  # Kanto region

class SituationCriticality(Enum):
    """How critical the current situation is"""
    EMERGENCY = "emergency"  # Immediate action needed
    URGENT = "urgent"  # Action needed soon
    MODERATE = "moderate"  # Normal decision making
    OPTIMAL = "optimal"  # Can focus on optimization

@dataclass
class AnalysisAnalysisStateVariable:
    """Definition of a game state variable for analysis purposes"""
    name: str
    type: str  # 'int', 'float', 'bool', 'tuple', 'bitfield'
    current_value: Any
    normal_range: Tuple[Any, Any]
    critical_thresholds: Dict[str, Any]
    impact_on_rewards: List[str]
    impact_on_survival: float  # 0.0 to 1.0
    description: str

@dataclass 
class GameStateAnalysis:
    """Comprehensive analysis of current game state"""
    # Basic state info
    phase: GamePhase
    criticality: SituationCriticality
    
    # Key metrics
    health_percentage: float
    progression_score: float  # 0-100 based on badges, levels, etc.
    exploration_score: float  # How much has been explored
    
    # Strategic insights
    immediate_threats: List[str]
    opportunities: List[str]
    recommended_priorities: List[str]
    
    # Context for LLM
    situation_summary: str
    strategic_context: str
    risk_assessment: str
    
    # Raw state variables
    state_variables: Dict[str, 'AnalysisStateVariable']

class GameStateAnalyzer:
    """Analyzes game state and provides strategic insights"""
    
    def __init__(self):
        self.state_variable_definitions = self._initialize_state_definitions()
        self.location_knowledge = self._initialize_location_knowledge()
        self.pokemon_knowledge = self._initialize_pokemon_knowledge()
        
    def analyze(self, raw_game_state: Dict) -> GameStateAnalysis:
        """
        Perform comprehensive analysis of the current game state
        
        Args:
            raw_game_state: Raw game state dictionary from memory reading
            
        Returns:
            GameStateAnalysis with strategic insights
        """
        # 1. Parse raw state into structured variables
        state_vars = self._parse_state_variables(raw_game_state)
        
        # 2. Determine game phase
        phase = self._determine_game_phase(state_vars)
        
        # 3. Assess criticality
        criticality = self._assess_criticality(state_vars, phase)
        
        # 4. Calculate key metrics
        health_pct = self._calculate_health_percentage(state_vars)
        progression = self._calculate_progression_score(state_vars)
        exploration = self._calculate_exploration_score(state_vars)
        
        # 5. Identify threats and opportunities
        threats = self._identify_threats(state_vars, phase)
        opportunities = self._identify_opportunities(state_vars, phase)
        
        # 6. Generate strategic recommendations
        priorities = self._recommend_priorities(state_vars, phase, criticality)
        
        # 7. Create context strings for LLM
        situation_summary = self._generate_situation_summary(state_vars, phase)
        strategic_context = self._generate_strategic_context(threats, opportunities, priorities)
        risk_assessment = self._generate_risk_assessment(state_vars, threats)
        
        return GameStateAnalysis(
            phase=phase,
            criticality=criticality,
            health_percentage=health_pct,
            progression_score=progression,
            exploration_score=exploration,
            immediate_threats=threats,
            opportunities=opportunities,
            recommended_priorities=priorities,
            situation_summary=situation_summary,
            strategic_context=strategic_context,
            risk_assessment=risk_assessment,
            state_variables=state_vars
        )
    
    def _initialize_state_definitions(self) -> Dict[str, Dict]:
        """Initialize definitions for all state variables"""
        return {
            'player_hp': {
                'type': 'int',
                'normal_range': (1, 999),
                'critical_thresholds': {'emergency': 0, 'low': 0.25, 'medium': 0.5},
                'impact_on_rewards': ['survival', 'battle_performance'],
                'impact_on_survival': 1.0,
                'description': 'Current HP of the active Pokemon'
            },
            'player_max_hp': {
                'type': 'int', 
                'normal_range': (1, 999),
                'critical_thresholds': {},
                'impact_on_rewards': ['progression'],
                'impact_on_survival': 0.3,
                'description': 'Maximum HP of the active Pokemon'
            },
            'player_level': {
                'type': 'int',
                'normal_range': (1, 100),
                'critical_thresholds': {'early': 10, 'mid': 30, 'late': 50},
                'impact_on_rewards': ['progression', 'battle_performance'],
                'impact_on_survival': 0.7,
                'description': 'Level of the active Pokemon'
            },
            'party_count': {
                'type': 'int',
                'normal_range': (0, 6),
                'critical_thresholds': {'none': 0, 'starter': 1, 'full': 6},
                'impact_on_rewards': ['progression', 'exploration'],
                'impact_on_survival': 0.8,
                'description': 'Number of Pokemon in party'
            },
            'badges': {
                'type': 'bitfield',
                'normal_range': (0, 16),
                'critical_thresholds': {'early': 2, 'mid': 8, 'complete': 16},
                'impact_on_rewards': ['major_progression'],
                'impact_on_survival': 0.2,
                'description': 'Number of badges earned'
            },
            'money': {
                'type': 'int',
                'normal_range': (0, 999999),
                'critical_thresholds': {'broke': 100, 'comfortable': 5000},
                'impact_on_rewards': ['resources'],
                'impact_on_survival': 0.1,
                'description': 'Amount of money available'
            },
            'player_x': {
                'type': 'int',
                'normal_range': (0, 255),
                'critical_thresholds': {},
                'impact_on_rewards': ['exploration', 'movement'],
                'impact_on_survival': 0.0,
                'description': 'Player X coordinate'
            },
            'player_y': {
                'type': 'int', 
                'normal_range': (0, 255),
                'critical_thresholds': {},
                'impact_on_rewards': ['exploration', 'movement'],
                'impact_on_survival': 0.0,
                'description': 'Player Y coordinate'
            },
            'player_map': {
                'type': 'int',
                'normal_range': (0, 255),
                'critical_thresholds': {},
                'impact_on_rewards': ['exploration', 'progression'],
                'impact_on_survival': 0.1,
                'description': 'Current map/location ID'
            },
            'in_battle': {
                'type': 'bool',
                'normal_range': (False, True),
                'critical_thresholds': {},
                'impact_on_rewards': ['battle_performance'],
                'impact_on_survival': 0.9,
                'description': 'Whether currently in battle'
            }
        }
    
    def _initialize_location_knowledge(self) -> Dict[int, Dict]:
        """Initialize knowledge about game locations"""
        return {
            0: {'name': 'Unknown', 'type': 'unknown', 'safety': 'unknown'},
            24: {'name': "Player's Bedroom", 'type': 'safe', 'safety': 'safe'},
            25: {'name': "Player's House", 'type': 'safe', 'safety': 'safe'},
            26: {'name': 'New Bark Town', 'type': 'town', 'safety': 'safe'},
            27: {'name': "Prof. Elm's Lab", 'type': 'important', 'safety': 'safe'},
            28: {'name': 'Route 29', 'type': 'route', 'safety': 'moderate'},
            29: {'name': 'Route 30', 'type': 'route', 'safety': 'moderate'},
            30: {'name': 'Cherrygrove City', 'type': 'town', 'safety': 'safe'},
        }
    
    def _initialize_pokemon_knowledge(self) -> Dict[int, Dict]:
        """Initialize knowledge about Pokemon species"""
        return {
            0: {'name': 'None', 'type': [], 'base_stats': {'hp': 0}},
            152: {'name': 'Chikorita', 'type': ['Grass'], 'base_stats': {'hp': 45}},
            155: {'name': 'Cyndaquil', 'type': ['Fire'], 'base_stats': {'hp': 39}},
            158: {'name': 'Totodile', 'type': ['Water'], 'base_stats': {'hp': 50}},
        }
    
    def _parse_state_variables(self, raw_state: Dict) -> Dict[str, 'AnalysisStateVariable']:
        """Parse raw state into structured AnalysisStateVariable objects"""
        state_vars = {}
        
        for var_name, definition in self.state_variable_definitions.items():
            current_value = raw_state.get(var_name, 0)
            
            state_vars[var_name] = AnalysisStateVariable(
                name=var_name,
                type=definition['type'],
                current_value=current_value,
                normal_range=definition['normal_range'],
                critical_thresholds=definition['critical_thresholds'],
                impact_on_rewards=definition['impact_on_rewards'],
                impact_on_survival=definition['impact_on_survival'],
                description=definition['description']
            )
        
        return state_vars
    
    def _determine_game_phase(self, state_vars: Dict[str, 'AnalysisStateVariable']) -> GamePhase:
        """Determine what phase of the game we're in"""
        party_count = state_vars['party_count'].current_value
        badges = state_vars['badges'].current_value
        player_level = state_vars.get('player_level', AnalysisStateVariable('', '', 0, (0, 0), {}, [], 0.0, '')).current_value
        
        # Count actual badges (number of set bits)
        if isinstance(badges, int):
            badge_count = bin(badges).count('1')
        else:
            badge_count = 0
        
        # Determine phase based on progression markers
        if party_count == 0:
            return GamePhase.EARLY_GAME
        elif party_count == 1 and player_level < 10:
            return GamePhase.STARTER_PHASE
        elif badge_count == 0:
            return GamePhase.EXPLORATION
        elif badge_count < 8:
            return GamePhase.GYM_BATTLES
        elif badge_count < 16:
            return GamePhase.LATE_GAME
        else:
            return GamePhase.POST_GAME
    
    def _assess_criticality(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> SituationCriticality:
        """Assess how critical the current situation is"""
        # Check for emergency conditions
        if self._has_emergency_conditions(state_vars):
            return SituationCriticality.EMERGENCY
        
        # Check for urgent conditions  
        if self._has_urgent_conditions(state_vars, phase):
            return SituationCriticality.URGENT
        
        # Check if we're in a generally good state
        if self._is_optimal_state(state_vars, phase):
            return SituationCriticality.OPTIMAL
        
        return SituationCriticality.MODERATE
    
    def _has_emergency_conditions(self, state_vars: Dict[str, 'AnalysisStateVariable']) -> bool:
        """Check for conditions requiring immediate action"""
        # Only check survival conditions if we have Pokemon
        party_count = state_vars.get('party_count')
        if not party_count or party_count.current_value == 0:
            return False
        
        # Pokemon at 0 HP
        hp = state_vars.get('player_hp')
        if hp and hp.current_value == 0:
            return True
        
        # In battle with critically low HP
        in_battle = state_vars.get('in_battle')
        if hp and in_battle and in_battle.current_value:
            max_hp = state_vars.get('player_max_hp')
            if max_hp and max_hp.current_value > 0:
                hp_percentage = hp.current_value / max_hp.current_value
                if hp_percentage < 0.1:  # Less than 10% HP in battle
                    return True
        
        return False
    
    def _has_urgent_conditions(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> bool:
        """Check for conditions requiring prompt attention"""
        party_count = state_vars.get('party_count')
        
        # No Pokemon but should have one (not during early game)
        if party_count and party_count.current_value == 0 and phase != GamePhase.EARLY_GAME:
            return True
        
        # Only check HP-related urgent conditions if we have Pokemon
        if not party_count or party_count.current_value == 0:
            return False
            
        # Low HP outside of battle
        hp = state_vars.get('player_hp')
        max_hp = state_vars.get('player_max_hp')
        in_battle = state_vars.get('in_battle')
        
        if hp and max_hp and not (in_battle and in_battle.current_value):
            if max_hp.current_value > 0:
                hp_percentage = hp.current_value / max_hp.current_value
                if hp_percentage < 0.25:  # Less than 25% HP
                    return True
        
        return False
    
    def _is_optimal_state(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> bool:
        """Check if we're in an optimal state for strategic planning"""
        party_count = state_vars.get('party_count')
        
        # Have Pokemon if we should (not during early game)
        if party_count and party_count.current_value == 0 and phase != GamePhase.EARLY_GAME:
            return False
        
        # If we're in early game with no Pokemon, that's actually optimal
        if phase == GamePhase.EARLY_GAME and (not party_count or party_count.current_value == 0):
            return True
        
        # Only check HP and battle conditions if we have Pokemon
        if party_count and party_count.current_value > 0:
            # Good HP
            hp = state_vars.get('player_hp')
            max_hp = state_vars.get('player_max_hp')
            if hp and max_hp and max_hp.current_value > 0:
                hp_percentage = hp.current_value / max_hp.current_value
                if hp_percentage < 0.8:  # Less than 80% HP
                    return False
            
            # Not in battle
            in_battle = state_vars.get('in_battle')
            if in_battle and in_battle.current_value:
                return False
        
        return True
    
    def _calculate_health_percentage(self, state_vars: Dict[str, 'AnalysisStateVariable']) -> float:
        """Calculate current health as a percentage"""
        hp = state_vars.get('player_hp')
        max_hp = state_vars.get('player_max_hp')
        
        if not hp or not max_hp or max_hp.current_value == 0:
            return 0.0
        
        return (hp.current_value / max_hp.current_value) * 100
    
    def _calculate_progression_score(self, state_vars: Dict[str, 'AnalysisStateVariable']) -> float:
        """Calculate overall progression score (0-100)"""
        score = 0.0
        
        # Badge progress (0-40 points)
        badges = state_vars.get('badges')
        if badges:
            badge_count = bin(badges.current_value).count('1') if isinstance(badges.current_value, int) else 0
            score += min(badge_count * 2.5, 40)  # Up to 40 points for 16 badges
        
        # Pokemon level progress (0-30 points)
        level = state_vars.get('player_level')
        if level:
            score += min(level.current_value * 0.3, 30)  # Up to 30 points for level 100
        
        # Party size (0-20 points)
        party_count = state_vars.get('party_count')
        if party_count:
            score += min(party_count.current_value * 3.33, 20)  # Up to 20 points for 6 Pokemon
        
        # Money (0-10 points)
        money = state_vars.get('money')
        if money:
            score += min(money.current_value / 10000, 10)  # Up to 10 points for 100k money
        
        return min(score, 100.0)
    
    def _calculate_exploration_score(self, state_vars: Dict[str, 'AnalysisStateVariable']) -> float:
        """Calculate exploration score based on current location"""
        # This would be enhanced with actual exploration tracking
        # For now, just use map ID as a proxy
        map_id = state_vars.get('player_map')
        if not map_id:
            return 0.0
        
        # Higher map IDs generally mean more exploration
        return min(map_id.current_value * 2.0, 100.0)
    
    def _identify_threats(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> List[str]:
        """Identify immediate threats requiring attention"""
        threats = []
        
        # Calculate health percentage once for use throughout the method
        hp_pct = self._calculate_health_percentage(state_vars)
        
        # Only check health-related threats if we actually have Pokemon
        party_count = state_vars.get('party_count')
        if party_count and party_count.current_value > 0:
            # Health-related threats
            if hp_pct == 0:
                threats.append("Pokemon has fainted - needs immediate healing")
            elif hp_pct < 10:
                threats.append("Critically low HP - emergency healing needed")
            elif hp_pct < 25:
                threats.append("Low HP - should heal soon")
        
        # Battle threats (only if we have Pokemon)
        if party_count and party_count.current_value > 0:
            in_battle = state_vars.get('in_battle')
            if in_battle and in_battle.current_value and hp_pct < 50:
                threats.append("In battle with low HP - consider fleeing or using items")
        
        # Progression threats
        if party_count and party_count.current_value == 0 and phase != GamePhase.EARLY_GAME:
            threats.append("No Pokemon in party - game cannot progress")
        
        return threats
    
    def _identify_opportunities(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> List[str]:
        """Identify opportunities for advancement"""
        opportunities = []
        
        # Progression opportunities
        if phase == GamePhase.EARLY_GAME:
            opportunities.append("Visit Prof. Elm's lab to get starter Pokemon")
        elif phase == GamePhase.STARTER_PHASE:
            opportunities.append("Train starter Pokemon and explore nearby routes")
        
        # Battle opportunities (only if we have Pokemon)
        party_count = state_vars.get('party_count')
        if party_count and party_count.current_value > 0:
            in_battle = state_vars.get('in_battle')
            hp_pct = self._calculate_health_percentage(state_vars)
            if in_battle and in_battle.current_value and hp_pct > 50:
                opportunities.append("In battle with good HP - opportunity for victory")
        
        # Exploration opportunities (only if we have Pokemon for training)
        map_id = state_vars.get('player_map')
        if map_id and map_id.current_value in self.location_knowledge:
            location = self.location_knowledge[map_id.current_value]
            if location['type'] == 'route' and party_count and party_count.current_value > 0:
                opportunities.append("On route - good for training and catching Pokemon")
        
        return opportunities
    
    def _recommend_priorities(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase, criticality: SituationCriticality) -> List[str]:
        """Recommend strategic priorities based on current state"""
        priorities = []
        
        if criticality == SituationCriticality.EMERGENCY:
            priorities.extend([
                "Immediate survival actions",
                "Heal Pokemon or flee from danger",
                "Use emergency items"
            ])
        elif criticality == SituationCriticality.URGENT:
            priorities.extend([
                "Address health concerns",
                "Stabilize situation",
                "Prepare for challenges"
            ])
        else:
            # Strategic priorities based on phase
            if phase == GamePhase.EARLY_GAME:
                priorities.extend([
                    "Navigate to Prof. Elm's lab",
                    "Get starter Pokemon",
                    "Begin adventure"
                ])
            elif phase == GamePhase.STARTER_PHASE:
                priorities.extend([
                    "Train starter Pokemon",
                    "Explore nearby areas",
                    "Learn battle basics"
                ])
            elif phase == GamePhase.EXPLORATION:
                priorities.extend([
                    "Build strong Pokemon team",
                    "Explore new areas",
                    "Prepare for gym battles"
                ])
            elif phase == GamePhase.GYM_BATTLES:
                priorities.extend([
                    "Challenge gym leaders",
                    "Strengthen Pokemon team",
                    "Progress through regions"
                ])
        
        return priorities
    
    def _generate_situation_summary(self, state_vars: Dict[str, 'AnalysisStateVariable'], phase: GamePhase) -> str:
        """Generate a human-readable summary of the current situation"""
        party_count = state_vars.get('party_count', AnalysisStateVariable('', '', 0, (0, 0), {}, [], 0.0, '')).current_value
        hp_pct = self._calculate_health_percentage(state_vars)
        in_battle = state_vars.get('in_battle', AnalysisStateVariable('', '', False, (False, True), {}, [], 0.0, '')).current_value
        map_id = state_vars.get('player_map', AnalysisStateVariable('', '', 0, (0, 0), {}, [], 0.0, '')).current_value
        
        location_name = self.location_knowledge.get(map_id, {}).get('name', f'Map {map_id}')
        
        summary = f"Currently in {location_name} during {phase.value.replace('_', ' ')} phase. "
        
        if party_count == 0:
            summary += "No Pokemon in party. "
        else:
            summary += f"Have {party_count} Pokemon. "
            if hp_pct > 80:
                summary += "Pokemon health is excellent. "
            elif hp_pct > 50:
                summary += "Pokemon health is good. "
            elif hp_pct > 25:
                summary += "Pokemon health is low. "
            else:
                summary += "Pokemon health is critically low. "
        
        if in_battle:
            summary += "Currently in battle. "
        
        return summary.strip()
    
    def _generate_strategic_context(self, threats: List[str], opportunities: List[str], priorities: List[str]) -> str:
        """Generate strategic context for LLM decision-making"""
        context = ""
        
        if threats:
            context += f"THREATS: {'; '.join(threats)}. "
        
        if opportunities:
            context += f"OPPORTUNITIES: {'; '.join(opportunities)}. "
        
        if priorities:
            context += f"PRIORITIES: {'; '.join(priorities[:3])}. "  # Top 3 priorities
        
        return context.strip()
    
    def _generate_risk_assessment(self, state_vars: Dict[str, 'AnalysisStateVariable'], threats: List[str]) -> str:
        """Generate risk assessment for current situation"""
        risk_level = "LOW"
        
        if any("fainted" in threat.lower() for threat in threats):
            risk_level = "CRITICAL"
        elif any("critically low" in threat.lower() for threat in threats):
            risk_level = "HIGH" 
        elif any("low hp" in threat.lower() for threat in threats):
            risk_level = "MODERATE"
        
        return f"Risk Level: {risk_level}"
