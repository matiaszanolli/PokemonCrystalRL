#!/usr/bin/env python3
"""
Strategic Context Builder for Pokemon Crystal RL

This module builds rich, strategic context for LLM decision-making,
including historical patterns, action consequences, and strategic recommendations.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
from datetime import datetime

from .state.analyzer import GameStateAnalyzer, GameStateAnalysis, GamePhase, SituationCriticality
from .goal_oriented_planner import GoalOrientedPlanner
from .decision_history_analyzer import DecisionHistoryAnalyzer
from .adaptive_strategy_system import AdaptiveStrategySystem

@dataclass
class ActionConsequence:
    """Prediction of what an action might lead to"""
    action: str
    likely_outcome: str
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    reward_potential: str  # LOW, MODERATE, HIGH
    strategic_value: str  # Description of strategic importance

@dataclass 
class DecisionContext:
    """Complete context for LLM decision-making"""
    # Current state analysis
    current_analysis: GameStateAnalysis
    
    # Historical context
    recent_actions: List[str]
    recent_rewards: List[float]
    stuck_patterns: List[str]
    successful_patterns: List[str]
    
    # Strategic recommendations
    action_consequences: Dict[str, ActionConsequence]
    emergency_actions: List[str]
    strategic_goals: List[str]
    
    # LLM prompt components
    situation_prompt: str
    context_prompt: str
    guidance_prompt: str
    complete_prompt: str

class StrategicContextBuilder:
    """Builds comprehensive strategic context for LLM decision-making"""
    
    def __init__(self, max_history=20):
        self.game_state_analyzer = GameStateAnalyzer()
        self.goal_planner = GoalOrientedPlanner()
        self.history_analyzer = DecisionHistoryAnalyzer()
        self.adaptive_strategy = AdaptiveStrategySystem(self.history_analyzer, self.goal_planner)
        self.max_history = max_history
        
        # History tracking
        self.action_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        self.state_history = deque(maxlen=max_history)
        
        # Pattern recognition
        self.stuck_patterns = []
        self.successful_patterns = []
        self.location_memory = {}  # Track what we've learned about locations
        
        # Action definitions
        self.action_definitions = self._initialize_action_definitions()
        
    def build_context(self, raw_game_state: Dict, last_action: Optional[str] = None, 
                     last_reward: Optional[float] = None) -> DecisionContext:
        """
        Build comprehensive decision context for LLM
        
        Args:
            raw_game_state: Current game state from memory reading
            last_action: Last action taken (if any)
            last_reward: Reward received from last action
            
        Returns:
            DecisionContext with all strategic information
        """
        # Update history
        if last_action is not None:
            self.action_history.append(last_action)
        if last_reward is not None:
            self.reward_history.append(last_reward)
        
        # 1. Analyze current game state
        current_analysis = self.game_state_analyzer.analyze(raw_game_state)
        self.state_history.append(current_analysis)
        
        # 2. Analyze patterns
        self._update_patterns(current_analysis)
        
        # 3. Build action consequences
        action_consequences = self._build_action_consequences(current_analysis)
        
        # 4. Identify emergency actions
        emergency_actions = self._identify_emergency_actions(current_analysis)
        
        # 5. Set strategic goals using goal planner
        strategic_goals = self._determine_strategic_goals(current_analysis)
        
        # 6. Build prompts
        prompts = self._build_prompts(current_analysis, action_consequences, 
                                    emergency_actions, strategic_goals)
        
        return DecisionContext(
            current_analysis=current_analysis,
            recent_actions=list(self.action_history)[-10:],  # Last 10 actions
            recent_rewards=list(self.reward_history)[-10:],  # Last 10 rewards
            stuck_patterns=self.stuck_patterns[-5:],  # Recent stuck patterns
            successful_patterns=self.successful_patterns[-5:],  # Recent successes
            action_consequences=action_consequences,
            emergency_actions=emergency_actions,
            strategic_goals=strategic_goals,
            situation_prompt=prompts['situation'],
            context_prompt=prompts['context'],
            guidance_prompt=prompts['guidance'], 
            complete_prompt=prompts['complete']
        )
    
    def _initialize_action_definitions(self) -> Dict[str, Dict]:
        """Initialize definitions for all possible actions"""
        return {
            'up': {
                'description': 'Move up/north',
                'typical_outcome': 'Change position, possibly enter new area',
                'risks': ['May hit wall/obstacle', 'Could enter dangerous area'],
                'benefits': ['Exploration', 'Reaching new locations']
            },
            'down': {
                'description': 'Move down/south', 
                'typical_outcome': 'Change position, possibly enter new area',
                'risks': ['May hit wall/obstacle', 'Could enter dangerous area'],
                'benefits': ['Exploration', 'Reaching new locations']
            },
            'left': {
                'description': 'Move left/west',
                'typical_outcome': 'Change position, possibly enter new area',
                'risks': ['May hit wall/obstacle', 'Could enter dangerous area'], 
                'benefits': ['Exploration', 'Reaching new locations']
            },
            'right': {
                'description': 'Move right/east',
                'typical_outcome': 'Change position, possibly enter new area',
                'risks': ['May hit wall/obstacle', 'Could enter dangerous area'],
                'benefits': ['Exploration', 'Reaching new locations']
            },
            'a': {
                'description': 'Interact/Confirm button',
                'typical_outcome': 'Interact with objects, confirm menu choices, attack in battle',
                'risks': ['May confirm unwanted choice', 'Could waste turn in battle'],
                'benefits': ['Progress dialogue', 'Interact with important objects', 'Attack enemies']
            },
            'b': {
                'description': 'Cancel/Back button', 
                'typical_outcome': 'Cancel actions, go back in menus, run from battle',
                'risks': ['May cancel important actions', 'Could flee from winnable battles'],
                'benefits': ['Escape danger', 'Navigate menus', 'Undo mistakes']
            },
            'start': {
                'description': 'Open pause menu',
                'typical_outcome': 'Open game menu for items, Pokemon, save, etc.',
                'risks': ['Time wasted in menus', 'May interrupt important sequences'],
                'benefits': ['Access healing items', 'Check Pokemon status', 'Save game']
            },
            'select': {
                'description': 'Select button (context-dependent)',
                'typical_outcome': 'Various context-dependent functions',
                'risks': ['Unpredictable effects', 'May cause glitches'],
                'benefits': ['Context-specific shortcuts', 'Special functions']
            }
        }
    
    def _update_patterns(self, analysis: GameStateAnalysis):
        """Update pattern recognition based on recent actions and outcomes"""
        if len(self.action_history) < 5:
            return
        
        # Detect stuck patterns (repeated actions with no reward improvement)
        recent_actions = list(self.action_history)[-5:]
        recent_rewards = list(self.reward_history)[-5:] if len(self.reward_history) >= 5 else []
        
        # Check for repeated action sequences
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions in last 5
            if recent_rewards and all(r <= 0 for r in recent_rewards):  # All negative/zero rewards
                stuck_pattern = f"Repeating {recent_actions[-1]} with no progress"
                if stuck_pattern not in self.stuck_patterns:
                    self.stuck_patterns.append(stuck_pattern)
        
        # Detect successful patterns (good rewards)
        if recent_rewards and any(r > 1.0 for r in recent_rewards):  # Some good rewards
            action_sequence = " -> ".join(recent_actions[-3:])  # Last 3 actions
            success_pattern = f"Successful sequence: {action_sequence}"
            if success_pattern not in self.successful_patterns:
                self.successful_patterns.append(success_pattern)
    
    def _build_action_consequences(self, analysis: GameStateAnalysis) -> Dict[str, ActionConsequence]:
        """Build predictions for what each action might lead to"""
        consequences = {}
        
        for action, definition in self.action_definitions.items():
            # Determine risk level based on current state
            risk_level = self._assess_action_risk(action, analysis)
            
            # Determine reward potential
            reward_potential = self._assess_reward_potential(action, analysis)
            
            # Generate likely outcome description
            likely_outcome = self._predict_action_outcome(action, analysis)
            
            # Assess strategic value
            strategic_value = self._assess_strategic_value(action, analysis)
            
            consequences[action] = ActionConsequence(
                action=action,
                likely_outcome=likely_outcome,
                risk_level=risk_level,
                reward_potential=reward_potential,
                strategic_value=strategic_value
            )
        
        return consequences
    
    def _assess_action_risk(self, action: str, analysis: GameStateAnalysis) -> str:
        """Assess risk level of taking a specific action"""
        if analysis.criticality == SituationCriticality.EMERGENCY:
            # In emergency, most actions except healing/fleeing are risky
            if action in ['b', 'start']:  # Actions that might help escape/heal
                return "MODERATE"
            else:
                return "HIGH"
        
        # Movement actions in unknown areas
        if action in ['up', 'down', 'left', 'right']:
            current_location = analysis.state_variables.get('player_map')
            if current_location and current_location.current_value == 0:
                return "MODERATE"  # Unknown location = moderate risk
        
        # Battle context
        if analysis.state_variables.get('in_battle', None) and analysis.state_variables['in_battle'].current_value:
            if analysis.health_percentage < 25:
                if action == 'b':  # Fleeing when low HP
                    return "MODERATE"
                elif action == 'a':  # Attacking when low HP
                    return "HIGH"
        
        return "LOW"
    
    def _assess_reward_potential(self, action: str, analysis: GameStateAnalysis) -> str:
        """Assess potential rewards from taking a specific action"""
        # Emergency situations - survival actions have high reward potential
        if analysis.criticality == SituationCriticality.EMERGENCY:
            if action in ['b', 'start']:  # Escape/healing actions
                return "HIGH"
            else:
                return "LOW"
        
        # Phase-specific reward potentials
        if analysis.phase == GamePhase.EARLY_GAME:
            if action in ['up', 'down', 'left', 'right', 'a']:  # Movement and interaction
                return "HIGH"  # High potential to progress story
        
        # Battle context
        if analysis.state_variables.get('in_battle', None) and analysis.state_variables['in_battle'].current_value:
            if analysis.health_percentage > 50:
                if action == 'a':  # Attacking with good HP
                    return "HIGH"
            else:
                if action == 'b':  # Fleeing with low HP
                    return "MODERATE"
        
        return "MODERATE"
    
    def _predict_action_outcome(self, action: str, analysis: GameStateAnalysis) -> str:
        """Predict what will likely happen if we take this action"""
        base_outcome = self.action_definitions[action]['typical_outcome']
        
        # Modify based on current context
        if analysis.criticality == SituationCriticality.EMERGENCY:
            if action == 'b':
                return "Attempt to flee from danger or cancel harmful action"
            elif action == 'start':
                return "Open menu to access healing items or save"
        
        # Battle context
        if analysis.state_variables.get('in_battle', None) and analysis.state_variables['in_battle'].current_value:
            if action == 'a':
                return "Attack the enemy Pokemon"
            elif action == 'b':
                return "Attempt to run from battle"
            elif action == 'start':
                return "Open battle menu (items/Pokemon)"
        
        # Movement in specific locations
        current_map = analysis.state_variables.get('player_map')
        if current_map and action in ['up', 'down', 'left', 'right']:
            if current_map.current_value == 0:
                return f"Move {action} in unknown location - might hit obstacle or find exit"
        
        return base_outcome
    
    def _assess_strategic_value(self, action: str, analysis: GameStateAnalysis) -> str:
        """Assess the strategic importance of this action"""
        # Emergency actions
        if analysis.criticality == SituationCriticality.EMERGENCY:
            if action in ['b', 'start']:
                return "Critical for survival - highest priority"
            else:
                return "Dangerous in emergency situation - avoid"
        
        # Phase-specific strategic value
        if analysis.phase == GamePhase.EARLY_GAME:
            if action == 'a':
                return "Essential for progression - interact with Prof. Elm"
            elif action in ['up', 'down', 'left', 'right']:
                return "Important for navigation to key locations"
        
        # Movement when stuck
        recent_actions = list(self.action_history)[-5:]
        if len(recent_actions) >= 3 and len(set(recent_actions)) <= 2:
            if action not in recent_actions[-3:]:
                return "High value - breaks stuck pattern"
            else:
                return "Low value - continues stuck pattern"
        
        return "Standard strategic value"
    
    def _identify_emergency_actions(self, analysis: GameStateAnalysis) -> List[str]:
        """Identify actions that should be taken in emergency situations"""
        emergency_actions = []
        
        if analysis.criticality == SituationCriticality.EMERGENCY:
            # Health emergency
            if any("fainted" in threat.lower() for threat in analysis.immediate_threats):
                emergency_actions.extend(["start", "b"])  # Menu for items, or flee
            
            # Battle emergency with low HP
            if analysis.state_variables.get('in_battle', None) and analysis.state_variables['in_battle'].current_value:
                if analysis.health_percentage < 10:
                    emergency_actions.append("b")  # Flee from battle
        
        return emergency_actions
    
    def _determine_strategic_goals(self, analysis: GameStateAnalysis) -> List[str]:
        """Determine current strategic goals using the goal-oriented planner"""
        # Get strategic goals from the planner
        active_goals = self.goal_planner.evaluate_goals(analysis)
        
        # Convert to string descriptions for compatibility
        goal_descriptions = []
        for goal in active_goals:
            progress_str = f"({goal.progress_percentage:.0f}%)" if goal.progress_percentage > 0 else ""
            goal_descriptions.append(f"{goal.name} {progress_str}")
        
        # Add immediate tactical goals if needed
        if analysis.health_percentage < 25:
            goal_descriptions.insert(0, "URGENT: Restore Pokemon health")
        
        if analysis.immediate_threats:
            goal_descriptions.insert(0, "CRITICAL: Address survival threats")
        
        return goal_descriptions[:3]  # Top 3 goals
    
    def _build_prompts(self, analysis: GameStateAnalysis, consequences: Dict[str, ActionConsequence],
                      emergency_actions: List[str], strategic_goals: List[str]) -> Dict[str, str]:
        """Build LLM prompts with comprehensive context"""
        
        # Situation prompt
        situation_prompt = f"""
CURRENT SITUATION:
{analysis.situation_summary}

GAME PHASE: {analysis.phase.value.replace('_', ' ').title()}
CRITICALITY: {analysis.criticality.value.upper()}
HEALTH: {analysis.health_percentage:.1f}%
PROGRESSION: {analysis.progression_score:.1f}/100

{analysis.risk_assessment}
"""
        
        # Context prompt
        recent_actions_list = list(self.action_history)
        recent_rewards_list = list(self.reward_history)
        
        context_prompt = f"""
STRATEGIC CONTEXT:
{analysis.strategic_context}

RECENT ACTIONS: {' → '.join(recent_actions_list[-5:])}
RECENT REWARDS: {[f'{r:.2f}' for r in recent_rewards_list[-5:]]}
"""
        
        if self.stuck_patterns:
            context_prompt += f"\nSTUCK PATTERNS TO AVOID: {'; '.join(self.stuck_patterns[-3:])}"
        
        if self.successful_patterns:
            context_prompt += f"\nSUCCESSFUL PATTERNS: {'; '.join(self.successful_patterns[-3:])}"
        
        # Guidance prompt
        guidance_prompt = f"""
STRATEGIC GOALS: {'; '.join(strategic_goals)}

ACTION ANALYSIS:
"""
        
        # Get goal-oriented action recommendations
        goal_recommendations = self.goal_planner.get_recommended_actions(analysis)
        
        # Combine with consequence-based recommendations
        action_priorities = []
        
        # Add goal-oriented recommendations with high priority
        for action, reason, weight in goal_recommendations:
            if action in consequences:
                consequence = consequences[action]
                # Boost priority for goal-aligned actions
                boosted_consequence = ActionConsequence(
                    action=consequence.action,
                    likely_outcome=f"{consequence.likely_outcome} (Goal-aligned: {reason})",
                    risk_level=consequence.risk_level,
                    reward_potential="HIGH",  # Goal-aligned actions get high reward potential
                    strategic_value=f"High strategic value: {reason}"
                )
                action_priorities.append((action, boosted_consequence))
        
        # Add high-reward potential actions that aren't goal-aligned
        for action, consequence in consequences.items():
            if consequence.reward_potential == "HIGH" and not any(action == rec[0] for rec in goal_recommendations):
                action_priorities.append((action, consequence))
        
        # Sort by strategic value and reward potential  
        action_priorities.sort(key=lambda x: (
            "goal-aligned" in x[1].strategic_value.lower(),  # Goal-aligned first
            x[1].reward_potential == "HIGH",
            x[1].risk_level != "HIGH", 
            "breaks stuck pattern" in x[1].strategic_value.lower()
        ), reverse=True)
        
        for action, consequence in action_priorities[:3]:  # Top 3 actions
            guidance_prompt += f"• {action.upper()}: {consequence.strategic_value} (Risk: {consequence.risk_level}, Reward: {consequence.reward_potential})\n"
        
        if emergency_actions:
            guidance_prompt += f"\nEMERGENCY ACTIONS RECOMMENDED: {', '.join(emergency_actions).upper()}"
        
        # Complete prompt
        complete_prompt = f"""You are playing Pokemon Crystal and need to make strategic decisions. Here's your current situation:

{situation_prompt.strip()}

{context_prompt.strip()}

{guidance_prompt.strip()}

Available actions: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

Based on this comprehensive analysis, what action should you take and why? Choose the action that best addresses your current priorities while avoiding known stuck patterns.

Action: """
        
        return {
            'situation': situation_prompt.strip(),
            'context': context_prompt.strip(),
            'guidance': guidance_prompt.strip(),
            'complete': complete_prompt.strip()
        }
    
    def get_strategic_summary(self, analysis: GameStateAnalysis) -> str:
        """Get a comprehensive strategic summary including goals"""
        return self.goal_planner.get_current_strategy_summary()
    
    def get_goal_statistics(self) -> Dict[str, Any]:
        """Get goal completion statistics"""
        return self.goal_planner.get_goal_stats()
    
    def get_adaptive_action(self, analysis: GameStateAnalysis, llm_manager=None) -> Tuple[int, str, str]:
        """
        Get action using the adaptive strategy system
        
        Returns:
            Tuple of (action, decision_source, reasoning)
        """
        recent_actions = list(self.action_history)[-10:] if self.action_history else []
        action, source, reasoning = self.adaptive_strategy.get_next_action(
            analysis, llm_manager, recent_actions
        )
        
        # Record the action for history
        self.action_history.append(action)
        
        return action, source.value, reasoning
    
    def record_action_outcome(self, reward: float, led_to_progress: bool = False, was_effective: bool = True):
        """Record the outcome of the last action for learning"""
        self.reward_history.append(reward)
        
        # Update adaptive strategy performance
        self.adaptive_strategy.record_outcome(reward, led_to_progress, was_effective)
        
        # If we have enough history, also record in decision history analyzer
        if len(self.action_history) > 0 and hasattr(self, '_last_game_state'):
            last_action = self.action_history[-1]
            self.history_analyzer.record_decision(
                game_state=self._last_game_state,
                action_taken=last_action,
                llm_reasoning=getattr(self, '_last_reasoning', None),
                reward_received=reward,
                led_to_progress=led_to_progress,
                was_effective=was_effective
            )
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get insights about strategy performance and learning"""
        return {
            "adaptive_strategy": self.adaptive_strategy.get_strategy_stats(),
            "learned_patterns": self.history_analyzer.get_patterns_summary(),
            "goal_progress": self.goal_planner.get_goal_stats()
        }
    
    def set_strategy(self, strategy_name: str):
        """Force a specific strategy (for testing/debugging)"""
        from .adaptive_strategy_system import StrategyType
        
        try:
            strategy = StrategyType(strategy_name)
            self.adaptive_strategy.force_strategy(strategy)
        except ValueError:
            available = [s.value for s in StrategyType]
            raise ValueError(f"Invalid strategy '{strategy_name}'. Available: {available}")
