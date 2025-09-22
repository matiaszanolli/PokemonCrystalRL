#!/usr/bin/env python3
"""
State Variable Dictionary for Pokemon Crystal RL

Comprehensive mapping between raw memory values and game mechanics
as specified in ROADMAP_ENHANCED Phase 1.1
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

class VariableType(Enum):
    """Types of state variables"""
    INT = "int"
    FLOAT = "float" 
    TUPLE = "tuple"
    BITFIELD = "bitfield"
    BOOLEAN = "boolean"
    STRING = "string"
    ARRAY = "array"

class ImpactCategory(Enum):
    """Categories of impact on gameplay"""
    SURVIVAL = "survival"               # Directly affects survival (HP, status)
    EXPLORATION_REWARDS = "exploration_rewards"  # Affects exploration rewards
    MAJOR_PROGRESS = "major_progress"   # Major story/badge progress
    BATTLE_EFFECTIVENESS = "battle_effectiveness"  # Combat performance
    RESOURCE_MANAGEMENT = "resource_management"    # Items, money
    NAVIGATION = "navigation"           # Movement and location
    INFORMATION = "information"         # Non-critical information

@dataclass
class StateVariable:
    """Complete description of a game state variable"""
    name: str
    memory_address: int
    variable_type: VariableType
    impact_category: ImpactCategory
    
    # Value constraints
    valid_range: Tuple[Union[int, float], Union[int, float]]
    critical_threshold: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    
    # Semantic information
    description: str = ""
    interpretation_notes: str = ""
    
    # Reward function impact
    reward_weight: float = 1.0
    reward_function_role: str = ""
    
    # Relationship to other variables
    dependent_variables: List[str] = None
    affects_variables: List[str] = None
    
    # Danger indicators
    danger_conditions: List[str] = None
    opportunity_conditions: List[str] = None
    
    def __post_init__(self):
        if self.dependent_variables is None:
            self.dependent_variables = []
        if self.affects_variables is None:
            self.affects_variables = []
        if self.danger_conditions is None:
            self.danger_conditions = []
        if self.opportunity_conditions is None:
            self.opportunity_conditions = []

class StateVariableDictionary:
    """Comprehensive mapping of all Pokemon Crystal state variables"""
    
    def __init__(self):
        self.logger = logging.getLogger("pokemon_trainer.state_dict")
        self.variables: Dict[str, StateVariable] = {}
        self._initialize_state_variables()
    
    def _initialize_state_variables(self):
        """Initialize all state variable definitions"""
        
        # Player Position Variables
        self.variables['player_x'] = StateVariable(
            name='player_x',
            memory_address=0xDCB8,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.NAVIGATION,
            valid_range=(0, 255),
            description="Player X coordinate on current map",
            interpretation_notes="Screen coordinates, resets when changing maps",
            reward_weight=0.5,
            reward_function_role="Used for exploration rewards and stuck detection",
            affects_variables=['exploration_progress'],
            danger_conditions=["value unchanged for >20 steps indicates stuck"],
            opportunity_conditions=["new value indicates exploration progress"]
        )
        
        self.variables['player_y'] = StateVariable(
            name='player_y',
            memory_address=0xDCB9,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.NAVIGATION,
            valid_range=(0, 255),
            description="Player Y coordinate on current map",
            interpretation_notes="Screen coordinates, resets when changing maps",
            reward_weight=0.5,
            reward_function_role="Used for exploration rewards and stuck detection",
            affects_variables=['exploration_progress'],
            danger_conditions=["value unchanged for >20 steps indicates stuck"],
            opportunity_conditions=["new value indicates exploration progress"]
        )
        
        self.variables['player_map'] = StateVariable(
            name='player_map',
            memory_address=0xDCBA,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.MAJOR_PROGRESS,
            valid_range=(0, 255),
            description="Current map/location ID",
            interpretation_notes="Each location has unique ID, 0 often indicates transition/loading",
            reward_weight=3.0,
            reward_function_role="Major source of exploration and progression rewards",
            affects_variables=['exploration_progress', 'available_npcs', 'available_items'],
            opportunity_conditions=["new value indicates major exploration progress"]
        )
        
        # Player Health Variables - CRITICAL FOR SURVIVAL
        self.variables['player_hp'] = StateVariable(
            name='player_hp',
            memory_address=0xDCDA,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.SURVIVAL,
            valid_range=(0, 999),  # Theoretical max HP in Pokemon
            critical_threshold=0.25,  # Below 25% is critical
            description="Current HP of player's active Pokemon",
            interpretation_notes="2 bytes little endian, 0 = fainted",
            reward_weight=5.0,
            reward_function_role="Primary survival metric, heavily penalized when low",
            dependent_variables=['player_max_hp'],
            affects_variables=['battle_effectiveness', 'survival_status'],
            danger_conditions=[
                "value < 25% of max_hp indicates critical danger",
                "value == 0 indicates Pokemon fainted"
            ],
            opportunity_conditions=["value == max_hp indicates full health opportunity"]
        )
        
        self.variables['player_max_hp'] = StateVariable(
            name='player_max_hp',
            memory_address=0xDCDB,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.BATTLE_EFFECTIVENESS,
            valid_range=(1, 999),
            description="Maximum HP of player's active Pokemon",
            interpretation_notes="Increases with level and stats, used to calculate HP percentage",
            reward_weight=1.0,
            reward_function_role="Used to normalize HP for percentage calculations",
            affects_variables=['hp_percentage', 'battle_effectiveness']
        )
        
        # Player Level and Experience
        self.variables['player_level'] = StateVariable(
            name='player_level',
            memory_address=0xDCD3,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.BATTLE_EFFECTIVENESS,
            valid_range=(1, 100),
            max_value=100,
            description="Current level of player's active Pokemon",
            interpretation_notes="Level 1-100, directly affects battle capability",
            reward_weight=2.0,
            reward_function_role="Moderate reward for leveling up, indicates training progress",
            affects_variables=['player_max_hp', 'battle_effectiveness', 'available_moves'],
            opportunity_conditions=["increase indicates successful training"]
        )
        
        # Party Information
        self.variables['party_count'] = StateVariable(
            name='party_count',
            memory_address=0xDCD7,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.MAJOR_PROGRESS,
            valid_range=(0, 6),
            max_value=6,
            description="Number of Pokemon in player's party",
            interpretation_notes="0-6, 0 at start of game before getting starter",
            reward_weight=4.0,
            reward_function_role="Major reward for obtaining first Pokemon, moderate for expanding team",
            affects_variables=['battle_options', 'team_strength'],
            danger_conditions=["value == 0 indicates no Pokemon available"],
            opportunity_conditions=[
                "increase from 0 to 1 indicates starter Pokemon obtained",
                "increase indicates team expansion"
            ]
        )
        
        # Game Progress - Badges (CRITICAL PROGRESSION METRIC)
        self.variables['badges'] = StateVariable(
            name='badges',
            memory_address=0xD855,
            variable_type=VariableType.BITFIELD,
            impact_category=ImpactCategory.MAJOR_PROGRESS,
            valid_range=(0, 255),
            max_value=8,  # 8 badges max for Johto
            description="Johto gym badges earned (bitfield)",
            interpretation_notes="Each bit represents a badge: bit 0=Zephyr, bit 1=Hive, etc.",
            reward_weight=10.0,
            reward_function_role="Highest reward weight - major progression milestones",
            affects_variables=['progression_score', 'available_areas', 'story_progress'],
            opportunity_conditions=[
                "new bit set indicates gym victory - major achievement",
                "multiple badges indicate significant progress"
            ]
        )
        
        self.variables['kanto_badges'] = StateVariable(
            name='kanto_badges',
            memory_address=0xD856,
            variable_type=VariableType.BITFIELD,
            impact_category=ImpactCategory.MAJOR_PROGRESS,
            valid_range=(0, 255),
            max_value=8,
            description="Kanto gym badges earned (bitfield)", 
            interpretation_notes="Post-game content, each bit represents a Kanto badge",
            reward_weight=15.0,
            reward_function_role="Very high reward - end-game progression",
            affects_variables=['end_game_progress', 'champion_accessibility'],
            opportunity_conditions=["any bit set indicates post-game achievement"]
        )
        
        # Battle State Variables
        self.variables['in_battle'] = StateVariable(
            name='in_battle',
            memory_address=0xD057,
            variable_type=VariableType.BOOLEAN,
            impact_category=ImpactCategory.BATTLE_EFFECTIVENESS,
            valid_range=(0, 1),
            description="Whether player is currently in battle",
            interpretation_notes="1 = in battle, 0 = not in battle",
            reward_weight=0.0,
            reward_function_role="State indicator, affects action interpretation",
            affects_variables=['available_actions', 'action_consequences'],
            danger_conditions=["value == 1 AND player_hp < 25% indicates battle danger"],
            opportunity_conditions=["value == 1 AND player_hp > 75% indicates battle opportunity"]
        )
        
        self.variables['enemy_hp'] = StateVariable(
            name='enemy_hp',
            memory_address=0xCFE6,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.BATTLE_EFFECTIVENESS,
            valid_range=(0, 999),
            description="Current HP of enemy Pokemon",
            interpretation_notes="Only valid when in_battle == 1",
            reward_weight=1.0,
            reward_function_role="Used to assess battle progress and victory likelihood",
            dependent_variables=['in_battle', 'enemy_max_hp'],
            opportunity_conditions=[
                "low value relative to enemy_max_hp indicates near victory",
                "value == 0 indicates battle victory"
            ]
        )
        
        self.variables['enemy_level'] = StateVariable(
            name='enemy_level',
            memory_address=0xCFE3,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.BATTLE_EFFECTIVENESS,
            valid_range=(1, 100),
            description="Level of enemy Pokemon",
            interpretation_notes="Used to assess battle difficulty relative to player level",
            reward_weight=0.0,
            reward_function_role="Risk assessment for battle decisions",
            dependent_variables=['in_battle'],
            affects_variables=['battle_risk_level'],
            danger_conditions=["value >> player_level indicates dangerous battle"],
            opportunity_conditions=["value << player_level indicates easy battle"]
        )
        
        # Money and Resources
        self.variables['money'] = StateVariable(
            name='money',
            memory_address=0xD84E,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.RESOURCE_MANAGEMENT,
            valid_range=(0, 999999),
            description="Player's current money amount",
            interpretation_notes="3 bytes BCD format, used for purchasing items",
            reward_weight=1.0,
            reward_function_role="Minor reward for money gained from battles/items",
            affects_variables=['purchasing_power', 'item_accessibility'],
            opportunity_conditions=["increase indicates successful battle or item sale"]
        )
        
        # Menu and UI State
        self.variables['menu_state'] = StateVariable(
            name='menu_state',
            memory_address=0xD0A0,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.INFORMATION,
            valid_range=(0, 255),
            description="Current menu or UI state",
            interpretation_notes="Indicates which menu is open, affects available actions",
            reward_weight=0.0,
            reward_function_role="State information for action planning",
            affects_variables=['available_actions']
        )
        
        # Movement and Navigation States
        self.variables['can_move'] = StateVariable(
            name='can_move',
            memory_address=0xD0B0,
            variable_type=VariableType.BOOLEAN,
            impact_category=ImpactCategory.NAVIGATION,
            valid_range=(0, 1),
            description="Whether player can currently move",
            interpretation_notes="0 during cutscenes, dialogues, or forced events",
            reward_weight=0.0,
            reward_function_role="Determines if movement actions are valid",
            affects_variables=['available_actions'],
            danger_conditions=["value == 0 for extended periods may indicate stuck state"]
        )
        
        # Time and Context
        self.variables['time_of_day'] = StateVariable(
            name='time_of_day',
            memory_address=0xD269,
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.INFORMATION,
            valid_range=(1, 4),
            description="Current time of day",
            interpretation_notes="1=morning, 2=day, 3=evening, 4=night - affects Pokemon encounters",
            reward_weight=0.0,
            reward_function_role="Context information for encounter predictions",
            affects_variables=['available_pokemon', 'encounter_rates']
        )
        
        # Additional important variables
        self.variables['cutscene_flag'] = StateVariable(
            name='cutscene_flag',
            memory_address=0xD0C5,
            variable_type=VariableType.BOOLEAN,
            impact_category=ImpactCategory.INFORMATION,
            valid_range=(0, 1),
            description="Whether a cutscene or special event is active",
            interpretation_notes="1 when in cutscene, affects available actions",
            reward_weight=0.0,
            reward_function_role="State information for action planning",
            affects_variables=['available_actions', 'can_move']
        )
        
        # Add derived/calculated variables
        self._add_derived_variables()
    
    def _add_derived_variables(self):
        """Add calculated/derived state variables"""
        
        self.variables['hp_percentage'] = StateVariable(
            name='hp_percentage',
            memory_address=-1,  # Calculated value
            variable_type=VariableType.FLOAT,
            impact_category=ImpactCategory.SURVIVAL,
            valid_range=(0.0, 1.0),
            critical_threshold=0.25,
            description="Current HP as percentage of max HP",
            interpretation_notes="Calculated: player_hp / player_max_hp",
            reward_weight=3.0,
            reward_function_role="Primary health metric for reward calculations",
            dependent_variables=['player_hp', 'player_max_hp'],
            affects_variables=['survival_status', 'healing_urgency'],
            danger_conditions=[
                "value < 0.25 indicates critical health",
                "value < 0.10 indicates emergency health"
            ],
            opportunity_conditions=["value == 1.0 indicates full health"]
        )
        
        self.variables['badges_total'] = StateVariable(
            name='badges_total',
            memory_address=-1,  # Calculated value
            variable_type=VariableType.INT,
            impact_category=ImpactCategory.MAJOR_PROGRESS,
            valid_range=(0, 16),
            max_value=16,
            description="Total badges earned across both regions",
            interpretation_notes="Calculated: count_bits(badges) + count_bits(kanto_badges)",
            reward_weight=8.0,
            reward_function_role="Ultimate progression metric",
            dependent_variables=['badges', 'kanto_badges'],
            affects_variables=['game_completion_percentage'],
            opportunity_conditions=[
                "value >= 8 indicates Johto completion",
                "value == 16 indicates full badge completion"
            ]
        )
        
        self.variables['exploration_progress'] = StateVariable(
            name='exploration_progress',
            memory_address=-1,  # Calculated value
            variable_type=VariableType.FLOAT,
            impact_category=ImpactCategory.EXPLORATION_REWARDS,
            valid_range=(0.0, 1.0),
            description="Overall exploration progress metric",
            interpretation_notes="Based on unique locations visited and map coverage",
            reward_weight=2.0,
            reward_function_role="Rewards for discovering new areas",
            dependent_variables=['player_map', 'player_x', 'player_y'],
            opportunity_conditions=["increase indicates successful exploration"]
        )
    
    def get_variable(self, name: str) -> Optional[StateVariable]:
        """Get state variable definition by name"""
        return self.variables.get(name)
    
    def get_variables_by_impact(self, impact_category: ImpactCategory) -> List[StateVariable]:
        """Get all variables in a specific impact category"""
        return [var for var in self.variables.values() 
                if var.impact_category == impact_category]
    
    def get_critical_variables(self) -> List[StateVariable]:
        """Get variables that have critical thresholds defined"""
        return [var for var in self.variables.values() 
                if var.critical_threshold is not None]
    
    def get_high_impact_variables(self, min_weight: float = 3.0) -> List[StateVariable]:
        """Get variables with high reward weights"""
        return [var for var in self.variables.values() 
                if var.reward_weight >= min_weight]
    
    def analyze_variable_relationships(self) -> Dict[str, List[str]]:
        """Analyze dependencies between variables"""
        relationships = {}
        for var in self.variables.values():
            if var.dependent_variables or var.affects_variables:
                relationships[var.name] = {
                    'depends_on': var.dependent_variables,
                    'affects': var.affects_variables
                }
        return relationships
    
    def evaluate_danger_conditions(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate danger conditions across all variables"""
        dangers = []
        
        for var_name, var_def in self.variables.items():
            if var_name not in current_state:
                continue
                
            current_value = current_state[var_name]
            
            for condition in var_def.danger_conditions:
                if self._evaluate_condition(condition, var_name, current_value, current_state):
                    dangers.append({
                        'variable': var_name,
                        'condition': condition,
                        'current_value': current_value,
                        'impact': var_def.impact_category.value,
                        'severity': var_def.reward_weight
                    })
        
        return sorted(dangers, key=lambda x: x['severity'], reverse=True)
    
    def evaluate_opportunities(self, current_state: Dict[str, Any], 
                             previous_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Evaluate opportunity conditions across all variables"""
        opportunities = []
        
        for var_name, var_def in self.variables.items():
            if var_name not in current_state:
                continue
                
            current_value = current_state[var_name]
            previous_value = previous_state.get(var_name) if previous_state else None
            
            for condition in var_def.opportunity_conditions:
                if self._evaluate_opportunity_condition(condition, var_name, current_value, 
                                                      previous_value, current_state):
                    opportunities.append({
                        'variable': var_name,
                        'condition': condition,
                        'current_value': current_value,
                        'previous_value': previous_value,
                        'impact': var_def.impact_category.value,
                        'reward_potential': var_def.reward_weight
                    })
        
        return sorted(opportunities, key=lambda x: x['reward_potential'], reverse=True)
    
    def _evaluate_condition(self, condition: str, var_name: str, current_value: Any, 
                           current_state: Dict[str, Any]) -> bool:
        """Evaluate a danger condition string"""
        # Simple condition evaluation - can be expanded
        try:
            if "< 25%" in condition and var_name == "player_hp":
                max_hp = current_state.get('player_max_hp', 1)
                return current_value < (max_hp * 0.25)
            elif "== 0" in condition:
                return current_value == 0
            elif "unchanged for" in condition:
                # Would need history tracking for this
                return False
            # Add more condition patterns as needed
        except Exception:
            pass
        return False
    
    def _evaluate_opportunity_condition(self, condition: str, var_name: str, 
                                      current_value: Any, previous_value: Any,
                                      current_state: Dict[str, Any]) -> bool:
        """Evaluate an opportunity condition string"""
        try:
            if "increase" in condition:
                return previous_value is not None and current_value > previous_value
            elif "new value" in condition:
                return previous_value is not None and current_value != previous_value
            elif "== max_hp" in condition and var_name == "player_hp":
                max_hp = current_state.get('player_max_hp', current_value)
                return current_value == max_hp
            # Add more opportunity patterns as needed
        except Exception:
            pass
        return False
    
    def get_variable_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the variable dictionary"""
        total_vars = len(self.variables)
        
        impact_counts = {}
        for category in ImpactCategory:
            impact_counts[category.value] = len(self.get_variables_by_impact(category))
        
        type_counts = {}
        for var_type in VariableType:
            type_counts[var_type.value] = len([v for v in self.variables.values() 
                                              if v.variable_type == var_type])
        
        return {
            'total_variables': total_vars,
            'impact_distribution': impact_counts,
            'type_distribution': type_counts,
            'critical_variables': len(self.get_critical_variables()),
            'high_impact_variables': len(self.get_high_impact_variables()),
            'memory_addresses': len([v for v in self.variables.values() if v.memory_address > 0])
        }

# Global instance for easy access
STATE_VARIABLES = StateVariableDictionary()

def get_state_variable_info(variable_name: str) -> Optional[StateVariable]:
    """Convenience function to get variable information"""
    return STATE_VARIABLES.get_variable(variable_name)

def analyze_game_state_comprehensive(current_state: Dict[str, Any], 
                                   previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Comprehensive analysis of game state using variable dictionary"""
    
    dangers = STATE_VARIABLES.evaluate_danger_conditions(current_state)
    opportunities = STATE_VARIABLES.evaluate_opportunities(current_state, previous_state)
    
    # Calculate overall risk and opportunity scores
    total_danger_score = sum(d['severity'] for d in dangers)
    total_opportunity_score = sum(o['reward_potential'] for o in opportunities)
    
    return {
        'dangers': dangers,
        'opportunities': opportunities,
        'total_danger_score': total_danger_score,
        'total_opportunity_score': total_opportunity_score,
        'critical_variables': [(var.name, current_state.get(var.name, 'N/A')) 
                              for var in STATE_VARIABLES.get_critical_variables()],
        'high_impact_changes': [o for o in opportunities if o['reward_potential'] >= 5.0]
    }

if __name__ == "__main__":
    # Example usage and validation
    print("Pokemon Crystal RL State Variable Dictionary")
    print("=" * 50)
    
    summary = STATE_VARIABLES.get_variable_summary()
    print(f"Total Variables: {summary['total_variables']}")
    print(f"Memory Addresses: {summary['memory_addresses']}")
    print(f"Critical Variables: {summary['critical_variables']}")
    print(f"High Impact Variables: {summary['high_impact_variables']}")
    
    print("\nImpact Category Distribution:")
    for category, count in summary['impact_distribution'].items():
        print(f"  {category}: {count}")
    
    print("\nHigh Impact Variables:")
    for var in STATE_VARIABLES.get_high_impact_variables():
        print(f"  {var.name}: {var.reward_weight} ({var.impact_category.value})")