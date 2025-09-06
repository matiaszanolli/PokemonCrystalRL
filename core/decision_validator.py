#!/usr/bin/env python3
"""
Decision Validation Layer for Pokemon Crystal RL

Pre-validates LLM decisions against game state to prevent harmful actions
and provides override system for critical situations as specified in 
ROADMAP_ENHANCED Phase 2.1
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality
from .state.variables import STATE_VARIABLES, ImpactCategory

class ValidationResult(Enum):
    """Result of action validation"""
    APPROVED = "approved"           # Action is safe and beneficial
    APPROVED_WITH_WARNING = "approved_with_warning"  # Safe but suboptimal
    REJECTED_HARMFUL = "rejected_harmful"            # Action would cause harm
    REJECTED_INEFFECTIVE = "rejected_ineffective"    # Action won't help current situation
    OVERRIDE_REQUIRED = "override_required"          # Needs manual approval

class ActionRisk(Enum):
    """Risk levels for actions"""
    BENEFICIAL = "beneficial"       # Actively helpful
    SAFE = "safe"                  # No harm, neutral
    RISKY = "risky"               # Potential for harm
    DANGEROUS = "dangerous"        # Likely to cause harm
    CRITICAL = "critical"          # Immediate danger/harm

@dataclass
class ValidationDecision:
    """Complete validation decision with reasoning"""
    original_action: int
    approved_action: Optional[int]
    result: ValidationResult
    risk_level: ActionRisk
    reasoning: str
    confidence: float  # 0.0 to 1.0
    
    # Alternative suggestions
    suggested_alternatives: List[Tuple[int, str, float]] = None  # (action, reason, confidence)
    
    # Override information
    can_override: bool = False
    override_conditions: List[str] = None
    
    def __post_init__(self):
        if self.suggested_alternatives is None:
            self.suggested_alternatives = []
        if self.override_conditions is None:
            self.override_conditions = []

class DecisionValidator:
    """Validates LLM decisions against game state to prevent harmful actions"""
    
    def __init__(self):
        self.logger = logging.getLogger("pokemon_trainer.validator")
        
        # Action mappings
        self.action_names = {
            1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "A", 6: "B", 7: "START", 8: "SELECT"
        }
        
        # Emergency override rules
        self.emergency_overrides = {
            "critical_health": {"priority": 1, "actions": [7, 6]},  # START for menu, B to flee
            "pokemon_fainted": {"priority": 1, "actions": [7]},     # START for menu/items
            "stuck_loop": {"priority": 2, "actions": [1, 2, 3, 4]}, # Movement to break loops
        }
        
        # Action validation rules
        self._initialize_validation_rules()
    
    def _initialize_validation_rules(self):
        """Initialize action validation rules"""
        
        # Critical health rules
        self.critical_health_rules = {
            "forbidden_actions": [],  # No actions absolutely forbidden
            "preferred_actions": [7, 6],  # START for items, B to flee
            "risk_threshold": 0.15,   # Below 15% HP is critical
            "reasoning": "Pokemon health is critically low - prioritize healing or escape"
        }
        
        # Battle situation rules
        self.battle_rules = {
            "safe_actions": [5, 6, 7],  # A (attack), B (run), START (menu)
            "risky_actions": [1, 2, 3, 4],  # Movement actions in battle are risky
            "reasoning": "In battle - focus on combat actions rather than movement"
        }
        
        # Menu navigation rules
        self.menu_rules = {
            "effective_actions": [1, 2, 5, 6],  # UP, DOWN, A, B for menu navigation
            "ineffective_actions": [3, 4],      # LEFT, RIGHT often don't work in menus
            "reasoning": "In menu - use appropriate navigation actions"
        }
        
        # Stuck prevention rules
        self.stuck_prevention_rules = {
            "max_repeated_actions": 5,
            "alternative_actions": [1, 2, 3, 4, 5],  # Try different movements and interaction
            "reasoning": "Breaking stuck pattern with alternative actions"
        }
    
    def validate_action(self, proposed_action: int, analysis: GameStateAnalysis, 
                       action_history: Optional[List[int]] = None) -> ValidationDecision:
        """
        Validate a proposed action against current game state
        
        Args:
            proposed_action: Action number (1-8) proposed by LLM
            analysis: Current game state analysis
            action_history: Recent action history for pattern detection
            
        Returns:
            ValidationDecision with approval/rejection and reasoning
        """
        
        if action_history is None:
            action_history = []
        
        # 1. Emergency situation checks (highest priority)
        emergency_result = self._check_emergency_situations(proposed_action, analysis)
        if emergency_result:
            return emergency_result
        
        # 2. Critical health checks
        health_result = self._check_critical_health(proposed_action, analysis)
        if health_result:
            return health_result
        
        # 3. Battle situation checks
        battle_result = self._check_battle_situation(proposed_action, analysis)
        if battle_result:
            return battle_result
        
        # 4. Stuck pattern detection
        stuck_result = self._check_stuck_patterns(proposed_action, analysis, action_history)
        if stuck_result:
            return stuck_result
        
        # 5. General safety and effectiveness checks
        safety_result = self._check_general_safety(proposed_action, analysis)
        if safety_result:
            return safety_result
        
        # 6. Default approval if no issues detected
        return ValidationDecision(
            original_action=proposed_action,
            approved_action=proposed_action,
            result=ValidationResult.APPROVED,
            risk_level=ActionRisk.SAFE,
            reasoning="No issues detected with proposed action",
            confidence=0.8
        )
    
    def _check_emergency_situations(self, action: int, analysis: GameStateAnalysis) -> Optional[ValidationDecision]:
        """Check for emergency situations requiring immediate override"""
        
        # Critical situation priority override
        if analysis.criticality == SituationCriticality.EMERGENCY:
            
            # Pokemon fainted emergency
            if analysis.health_percentage == 0:
                if action not in [7]:  # Only START is safe when Pokemon fainted
                    return ValidationDecision(
                        original_action=action,
                        approved_action=7,  # Force START to access menu
                        result=ValidationResult.OVERRIDE_REQUIRED,
                        risk_level=ActionRisk.CRITICAL,
                        reasoning="Pokemon fainted - must access menu for revival items or Pokemon switch",
                        confidence=0.95,
                        suggested_alternatives=[(7, "Access menu for revival items", 0.95)],
                        can_override=True,
                        override_conditions=["User explicitly confirms they want to risk game over"]
                    )
            
            # Critically low health in battle
            elif analysis.health_percentage < 0.1 and self._is_in_battle(analysis):
                if action not in [6, 7]:  # B (flee) or START (menu)
                    return ValidationDecision(
                        original_action=action,
                        approved_action=6,  # Force flee attempt
                        result=ValidationResult.OVERRIDE_REQUIRED,
                        risk_level=ActionRisk.CRITICAL,
                        reasoning="Critically low HP in battle - must flee or use items to avoid fainting",
                        confidence=0.9,
                        suggested_alternatives=[
                            (6, "Attempt to flee from battle", 0.9),
                            (7, "Access menu for healing items", 0.8)
                        ],
                        can_override=True
                    )
        
        return None
    
    def _check_critical_health(self, action: int, analysis: GameStateAnalysis) -> Optional[ValidationDecision]:
        """Check for critical health situations"""
        
        if analysis.health_percentage < self.critical_health_rules["risk_threshold"]:
            
            # In battle with critical health
            if self._is_in_battle(analysis):
                if action not in [6, 7]:  # Not fleeing or accessing menu
                    return ValidationDecision(
                        original_action=action,
                        approved_action=action,  # Allow but warn
                        result=ValidationResult.APPROVED_WITH_WARNING,
                        risk_level=ActionRisk.DANGEROUS,
                        reasoning="Critical health in battle - fleeing or using items recommended",
                        confidence=0.7,
                        suggested_alternatives=[
                            (6, "Flee from battle to preserve Pokemon", 0.85),
                            (7, "Access menu for healing items", 0.8)
                        ]
                    )
            
            # In overworld with critical health
            else:
                if action in [1, 2, 3, 4] and not self._action_leads_to_healing(action, analysis):
                    return ValidationDecision(
                        original_action=action,
                        approved_action=action,  # Allow movement but warn
                        result=ValidationResult.APPROVED_WITH_WARNING,
                        risk_level=ActionRisk.RISKY,
                        reasoning="Critical health - consider finding Pokemon Center or using items",
                        confidence=0.6,
                        suggested_alternatives=[
                            (7, "Access menu for healing items", 0.8)
                        ]
                    )
        
        return None
    
    def _check_battle_situation(self, action: int, analysis: GameStateAnalysis) -> Optional[ValidationDecision]:
        """Check battle-specific action validity"""
        
        if not self._is_in_battle(analysis):
            return None
        
        # Movement actions in battle are usually ineffective
        if action in [1, 2, 3, 4]:  # Movement actions
            return ValidationDecision(
                original_action=action,
                approved_action=5,  # Default to A button (attack)
                result=ValidationResult.REJECTED_INEFFECTIVE,
                risk_level=ActionRisk.SAFE,
                reasoning="Movement actions are not effective in battle - using attack instead",
                confidence=0.8,
                suggested_alternatives=[
                    (5, "Attack the enemy Pokemon", 0.9),
                    (6, "Attempt to flee if battle is unfavorable", 0.7),
                    (7, "Access battle menu for items/Pokemon", 0.6)
                ]
            )
        
        # Attack when very low health might be risky
        if action == 5 and analysis.health_percentage < 0.2:
            return ValidationDecision(
                original_action=action,
                approved_action=action,  # Allow attack but warn
                result=ValidationResult.APPROVED_WITH_WARNING,
                risk_level=ActionRisk.RISKY,
                reasoning="Attacking with low health is risky - consider fleeing or using items",
                confidence=0.6,
                suggested_alternatives=[
                    (6, "Flee to preserve Pokemon", 0.8),
                    (7, "Use healing items", 0.9)
                ]
            )
        
        return None
    
    def _check_stuck_patterns(self, action: int, analysis: GameStateAnalysis, 
                            action_history: List[int]) -> Optional[ValidationDecision]:
        """Check for stuck patterns and repetitive actions"""
        
        if len(action_history) < 5:
            return None
        
        recent_actions = action_history[-5:]
        
        # Detect repetitive action patterns
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions in last 5
            
            # If proposing the same action that's been repeated
            if action in recent_actions[-3:]:
                
                # Find alternative actions to suggest
                alternative_actions = []
                for alt_action in [1, 2, 3, 4, 5]:  # Basic movement + interact
                    if alt_action not in recent_actions[-3:]:
                        action_name = self.action_names.get(alt_action, str(alt_action))
                        alternative_actions.append((
                            alt_action, 
                            f"Try {action_name} to break stuck pattern", 
                            0.7
                        ))
                
                return ValidationDecision(
                    original_action=action,
                    approved_action=alternative_actions[0][0] if alternative_actions else action,
                    result=ValidationResult.REJECTED_INEFFECTIVE,
                    risk_level=ActionRisk.SAFE,
                    reasoning=f"Detected stuck pattern - trying alternative to break loop",
                    confidence=0.8,
                    suggested_alternatives=alternative_actions[:3]
                )
        
        # Detect oscillating patterns (A->B->A->B)
        if len(recent_actions) >= 4:
            if (recent_actions[-4] == recent_actions[-2] and 
                recent_actions[-3] == recent_actions[-1] and
                recent_actions[-4] != recent_actions[-3]):
                
                # Suggest a third option to break oscillation
                return ValidationDecision(
                    original_action=action,
                    approved_action=5,  # Default to interaction
                    result=ValidationResult.REJECTED_INEFFECTIVE,
                    risk_level=ActionRisk.SAFE,
                    reasoning="Detected oscillating pattern - breaking with interaction",
                    confidence=0.85,
                    suggested_alternatives=[
                        (5, "Try interaction to break oscillation", 0.8),
                        (7, "Open menu to change context", 0.6)
                    ]
                )
        
        return None
    
    def _check_general_safety(self, action: int, analysis: GameStateAnalysis) -> Optional[ValidationDecision]:
        """Check general action safety and effectiveness"""
        
        # Invalid action numbers
        if action not in range(1, 9):
            return ValidationDecision(
                original_action=action,
                approved_action=5,  # Default to A button
                result=ValidationResult.REJECTED_HARMFUL,
                risk_level=ActionRisk.DANGEROUS,
                reasoning=f"Invalid action number {action} - using safe default",
                confidence=0.95,
                suggested_alternatives=[
                    (5, "Safe interaction action", 0.9),
                    (1, "Try movement up", 0.7)
                ]
            )
        
        # Check if action makes sense in current context
        context_result = self._check_contextual_appropriateness(action, analysis)
        if context_result:
            return context_result
        
        return None
    
    def _check_contextual_appropriateness(self, action: int, analysis: GameStateAnalysis) -> Optional[ValidationDecision]:
        """Check if action is appropriate for current game context"""
        
        # Early game - prioritize progression actions
        if analysis.phase == GamePhase.EARLY_GAME:
            if analysis.progression_score < 0.1:  # Very early in game
                # Movement and interaction are most valuable
                if action in [1, 2, 3, 4, 5]:  # Movement + interaction
                    return None  # Action is appropriate
                else:
                    return ValidationDecision(
                        original_action=action,
                        approved_action=action,
                        result=ValidationResult.APPROVED_WITH_WARNING,
                        risk_level=ActionRisk.SAFE,
                        reasoning="Early game - movement and interaction are most valuable",
                        confidence=0.6,
                        suggested_alternatives=[
                            (5, "Interact with objects/NPCs", 0.8),
                            (2, "Move down to explore", 0.7),
                            (4, "Move right to explore", 0.7)
                        ]
                    )
        
        return None
    
    def _is_in_battle(self, analysis: GameStateAnalysis) -> bool:
        """Check if currently in battle"""
        in_battle_var = analysis.state_variables.get('in_battle')
        return in_battle_var is not None and in_battle_var.current_value == 1
    
    def _action_leads_to_healing(self, action: int, analysis: GameStateAnalysis) -> bool:
        """Check if action might lead to healing (e.g., movement toward Pokemon Center)"""
        # This is a simplified heuristic - could be enhanced with location knowledge
        current_map = analysis.state_variables.get('player_map')
        
        # If in a Pokemon Center area, movement might lead to nurse
        if current_map and 'center' in str(current_map.current_value).lower():
            return action in [1, 2, 3, 4, 5]  # Movement or interaction
        
        # START menu always has potential for healing items
        if action == 7:
            return True
        
        return False
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about validation decisions (would need tracking)"""
        # This would be enhanced with actual tracking in a full implementation
        return {
            "total_validations": 0,
            "approvals": 0,
            "warnings": 0,
            "rejections": 0,
            "overrides": 0,
            "common_override_reasons": []
        }
    
    def update_validation_rules(self, feedback: Dict[str, Any]):
        """Update validation rules based on performance feedback"""
        # This could learn from outcomes to improve validation accuracy
        # For example, if an action marked as risky actually worked well,
        # adjust the risk assessment for similar situations
        pass

class EnhancedLLMManager:
    """Enhanced LLM Manager with Decision Validation"""
    
    def __init__(self, base_llm_manager, validator: Optional[DecisionValidator] = None):
        self.base_manager = base_llm_manager
        self.validator = validator or DecisionValidator()
        self.validation_enabled = True
        
        # Statistics
        self.validation_stats = {
            'total_actions': 0,
            'overridden_actions': 0,
            'warnings_issued': 0,
            'rejections': 0
        }
    
    def get_validated_action(self, screenshot=None, game_state="overworld", 
                           step=0, stuck_counter=0, analysis: Optional[GameStateAnalysis] = None,
                           action_history: Optional[List[int]] = None) -> Tuple[int, Optional[ValidationDecision]]:
        """
        Get LLM action with validation layer
        
        Returns:
            Tuple of (final_action, validation_decision)
        """
        
        self.validation_stats['total_actions'] += 1
        
        # Get original LLM action
        proposed_action = self.base_manager.get_action(
            screenshot=screenshot, 
            game_state=game_state, 
            step=step, 
            stuck_counter=stuck_counter
        )
        
        if proposed_action is None or not self.validation_enabled:
            return proposed_action, None
        
        # Validate the action if we have analysis
        if analysis:
            validation = self.validator.validate_action(
                proposed_action, 
                analysis, 
                action_history or []
            )
            
            # Update statistics
            if validation.result == ValidationResult.OVERRIDE_REQUIRED:
                self.validation_stats['overridden_actions'] += 1
            elif validation.result == ValidationResult.APPROVED_WITH_WARNING:
                self.validation_stats['warnings_issued'] += 1
            elif validation.result in [ValidationResult.REJECTED_HARMFUL, ValidationResult.REJECTED_INEFFECTIVE]:
                self.validation_stats['rejections'] += 1
            
            return validation.approved_action, validation
        
        # No validation possible without analysis
        return proposed_action, None
    
    def set_validation_enabled(self, enabled: bool):
        """Enable or disable action validation"""
        self.validation_enabled = enabled
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.validation_stats['total_actions']
        if total == 0:
            return self.validation_stats
        
        return {
            **self.validation_stats,
            'override_rate': self.validation_stats['overridden_actions'] / total,
            'warning_rate': self.validation_stats['warnings_issued'] / total,
            'rejection_rate': self.validation_stats['rejections'] / total
        }

if __name__ == "__main__":
    # Example usage and testing
    validator = DecisionValidator()
    
    print("Pokemon Crystal RL Decision Validator")
    print("=" * 40)
    
    # Test validation of different actions
    from .game_state_analyzer import GameStateAnalysis, GamePhase, SituationCriticality
    
    # Mock analysis for testing
    mock_analysis = GameStateAnalysis(
        phase=GamePhase.EARLY_GAME,
        criticality=SituationCriticality.NORMAL,
        health_percentage=15.0,  # Critical health
        progression_score=10.0,
        situation_summary="Low health situation",
        risk_assessment="High risk due to low health",
        strategic_context="Need healing urgently",
        immediate_threats=["Low health"],
        opportunities=[],
        state_variables={}
    )
    
    # Test critical health validation
    result = validator.validate_action(5, mock_analysis)  # A button with critical health
    print(f"Action 5 (A) with critical health: {result.result.value}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Alternatives: {[alt[1] for alt in result.suggested_alternatives]}")