#!/usr/bin/env python3
"""
Goal-Oriented Planning System for Pokemon Crystal RL

This module implements hierarchical goal planning and tracking
as specified in ROADMAP_ENHANCED Phase 2.1
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

from environments.state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality

class GoalPriority(Enum):
    """Goal priority levels"""
    EMERGENCY = 1      # Immediate survival needs
    CRITICAL = 2       # Essential progression blockers  
    HIGH = 3           # Important objectives
    MEDIUM = 4         # Standard progression
    LOW = 5            # Optional/exploratory

class GoalStatus(Enum):
    """Goal completion status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

@dataclass
class Goal:
    """Represents a strategic goal"""
    id: str
    name: str
    description: str
    priority: GoalPriority
    status: GoalStatus = GoalStatus.PENDING
    
    # Goal parameters
    target_conditions: Dict[str, Any] = None  # Conditions that must be met
    prerequisites: List[str] = None           # Other goal IDs that must complete first
    estimated_steps: int = 100               # Rough estimate of steps needed
    
    # Tracking
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    
    # Strategic context
    actions_suggested: List[str] = None      # Recommended actions for this goal
    abort_conditions: List[str] = None       # When to abandon this goal
    
    def __post_init__(self):
        if self.target_conditions is None:
            self.target_conditions = {}
        if self.prerequisites is None:
            self.prerequisites = []
        if self.actions_suggested is None:
            self.actions_suggested = []
        if self.abort_conditions is None:
            self.abort_conditions = []
        if self.created_at is None:
            self.created_at = datetime.now()

class GoalOrientedPlanner:
    """Implements strategic goal planning and execution tracking"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []  # Currently pursuing goals (by ID)
        self.goal_history: List[Tuple[str, GoalStatus, datetime]] = []
        
        # Initialize core goal templates
        self._initialize_goal_templates()
    
    def _initialize_goal_templates(self):
        """Initialize predefined goal templates for Pokemon Crystal"""
        
        # Emergency goals (highest priority)
        self.register_goal(Goal(
            id="survive_critical_hp",
            name="Survive Critical Health",
            description="Pokemon health is critically low - need immediate healing",
            priority=GoalPriority.EMERGENCY,
            target_conditions={"player_hp_percentage": "> 25"},
            actions_suggested=["start", "b"],  # Menu for items or flee
            abort_conditions=["player_hp == 0"],
            estimated_steps=5
        ))
        
        # Early game progression goals
        self.register_goal(Goal(
            id="obtain_starter_pokemon", 
            name="Obtain Starter Pokemon",
            description="Get first Pokemon from Professor Elm",
            priority=GoalPriority.CRITICAL,
            target_conditions={"party_size": ">= 1", "game_phase": "starter_phase"},
            actions_suggested=["a", "up", "down"],  # Interact and navigate
            estimated_steps=50
        ))
        
        self.register_goal(Goal(
            id="reach_professor_elm",
            name="Reach Professor Elm's Lab", 
            description="Navigate to Professor Elm to begin Pokemon journey",
            priority=GoalPriority.CRITICAL,
            target_conditions={"current_map": "elm_lab"},
            prerequisites=["obtain_starter_pokemon"],
            actions_suggested=["up", "down", "left", "right", "a"],
            estimated_steps=30
        ))
        
        # Gym progression goals
        for i in range(1, 9):  # 8 gym badges
            self.register_goal(Goal(
                id=f"defeat_gym_{i}",
                name=f"Defeat Gym Leader {i}",
                description=f"Challenge and defeat gym leader #{i}",
                priority=GoalPriority.HIGH if i <= 2 else GoalPriority.MEDIUM,
                target_conditions={"badges": f">= {i}"},
                prerequisites=[f"defeat_gym_{i-1}"] if i > 1 else ["obtain_starter_pokemon"],
                actions_suggested=["a", "b", "up", "down"],  # Battle actions
                estimated_steps=200
            ))
        
        # Training goals
        self.register_goal(Goal(
            id="level_up_team",
            name="Level Up Pokemon Team", 
            description="Train Pokemon to competitive levels",
            priority=GoalPriority.MEDIUM,
            target_conditions={"average_pokemon_level": ">= 15"},
            actions_suggested=["up", "down", "left", "right", "a"],  # Movement and battles
            estimated_steps=500
        ))
        
        # Exploration goals  
        self.register_goal(Goal(
            id="explore_new_areas",
            name="Explore New Areas",
            description="Discover new locations and routes", 
            priority=GoalPriority.LOW,
            target_conditions={"locations_visited": "> 5"},
            actions_suggested=["up", "down", "left", "right"],
            estimated_steps=300
        ))
    
    def register_goal(self, goal: Goal):
        """Register a new goal in the system"""
        self.goals[goal.id] = goal
        
    def evaluate_goals(self, analysis: GameStateAnalysis) -> List[Goal]:
        """Evaluate which goals should be active based on current game state"""
        
        # First, update progress on existing goals
        self._update_goal_progress(analysis)
        
        # Determine which goals should be active
        candidate_goals = []
        
        # Emergency goals always take priority
        for goal in self.goals.values():
            if goal.priority == GoalPriority.EMERGENCY and self._goal_is_applicable(goal, analysis):
                candidate_goals.append(goal)
        
        # If no emergency goals, consider other priorities
        if not candidate_goals:
            for goal in self.goals.values():
                if (goal.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS] and 
                    self._goal_is_applicable(goal, analysis) and
                    self._prerequisites_met(goal, analysis)):
                    candidate_goals.append(goal)
        
        # Sort by priority and return top goals
        candidate_goals.sort(key=lambda g: (g.priority.value, g.estimated_steps))
        
        # Update active goals
        self.active_goals = [g.id for g in candidate_goals[:3]]  # Top 3 active goals
        
        return candidate_goals[:3]
    
    def _goal_is_applicable(self, goal: Goal, analysis: GameStateAnalysis) -> bool:
        """Check if a goal is applicable in the current game state"""
        
        # Check if goal is already completed
        if goal.status == GoalStatus.COMPLETED:
            return False
        
        # Check abort conditions
        for abort_condition in goal.abort_conditions:
            if self._evaluate_condition(abort_condition, analysis):
                goal.status = GoalStatus.ABANDONED
                return False
        
        # Emergency goals
        if goal.id == "survive_critical_hp":
            return analysis.health_percentage < 25
        
        # Early game goals 
        if goal.id == "obtain_starter_pokemon":
            party_size = analysis.state_variables.get('party_size', None)
            if party_size and party_size.current_value == 0:
                return True
        
        # Phase-specific goals
        if analysis.phase == GamePhase.EARLY_GAME:
            return goal.id in ["obtain_starter_pokemon", "reach_professor_elm"]
        elif analysis.phase == GamePhase.STARTER_PHASE:
            return goal.id in ["level_up_team", "explore_new_areas"]
        elif analysis.phase == GamePhase.EXPLORATION:
            return goal.id.startswith("defeat_gym_") or goal.id in ["level_up_team", "explore_new_areas"]
        
        return True
    
    def _prerequisites_met(self, goal: Goal, analysis: GameStateAnalysis) -> bool:
        """Check if all prerequisites for a goal are met"""
        for prereq_id in goal.prerequisites:
            if prereq_id in self.goals:
                prereq_goal = self.goals[prereq_id]
                if prereq_goal.status != GoalStatus.COMPLETED:
                    return False
        return True
    
    def _update_goal_progress(self, analysis: GameStateAnalysis):
        """Update progress on all active goals"""
        for goal_id in self.active_goals:
            if goal_id not in self.goals:
                continue
                
            goal = self.goals[goal_id]
            
            # Check if goal is completed
            if self._goal_is_completed(goal, analysis):
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
                goal.progress_percentage = 100.0
                self.goal_history.append((goal_id, GoalStatus.COMPLETED, datetime.now()))
                continue
            
            # Update progress percentage
            goal.progress_percentage = self._calculate_goal_progress(goal, analysis)
            
            # Update status
            if goal.status == GoalStatus.PENDING and goal.progress_percentage > 0:
                goal.status = GoalStatus.IN_PROGRESS
                goal.started_at = datetime.now()
    
    def _goal_is_completed(self, goal: Goal, analysis: GameStateAnalysis) -> bool:
        """Check if a goal's completion conditions are met"""
        for condition_key, condition_value in goal.target_conditions.items():
            if not self._evaluate_condition(f"{condition_key} {condition_value}", analysis):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, analysis: GameStateAnalysis) -> bool:
        """Evaluate a string condition against current game state"""
        # Simple condition parser - can be expanded
        parts = condition.split()
        if len(parts) < 3:
            return False
            
        var_name = parts[0]
        operator = parts[1] 
        target_value = " ".join(parts[2:])
        
        # Get current value from analysis
        current_value = None
        
        if var_name == "player_hp_percentage":
            current_value = analysis.health_percentage
        elif var_name == "party_size":
            party_var = analysis.state_variables.get('party_size')
            current_value = party_var.current_value if party_var else 0
        elif var_name == "badges":
            badges_var = analysis.state_variables.get('badges')
            current_value = badges_var.current_value if badges_var else 0
        elif var_name == "game_phase":
            current_value = analysis.phase.value
        # Add more variable mappings as needed
        
        if current_value is None:
            return False
        
        # Evaluate the condition
        try:
            if operator == ">":
                return float(current_value) > float(target_value)
            elif operator == ">=":
                return float(current_value) >= float(target_value)
            elif operator == "<":
                return float(current_value) < float(target_value)
            elif operator == "<=":
                return float(current_value) <= float(target_value)
            elif operator == "==":
                return str(current_value) == target_value.strip('"')
            elif operator == "!=":
                return str(current_value) != target_value.strip('"')
        except (ValueError, TypeError):
            # For string comparisons
            if operator == "==":
                return str(current_value) == target_value.strip('"')
            elif operator == "!=":
                return str(current_value) != target_value.strip('"')
        
        return False
    
    def _calculate_goal_progress(self, goal: Goal, analysis: GameStateAnalysis) -> float:
        """Calculate progress percentage for a goal"""
        
        # Simple heuristic-based progress calculation
        if goal.id == "obtain_starter_pokemon":
            party_var = analysis.state_variables.get('party_size')
            if party_var and party_var.current_value > 0:
                return 100.0
            # Progress based on location - if we're in lab area, we're making progress
            current_map = analysis.state_variables.get('current_map')
            if current_map and "elm" in str(current_map.current_value).lower():
                return 75.0
            return goal.progress_percentage  # Keep current progress
        
        elif goal.id.startswith("defeat_gym_"):
            gym_num = int(goal.id.split("_")[2])
            badges_var = analysis.state_variables.get('badges')
            current_badges = badges_var.current_value if badges_var else 0
            if current_badges >= gym_num:
                return 100.0
            elif current_badges == gym_num - 1:
                return 50.0  # Close to the goal
        
        elif goal.id == "level_up_team":
            # Progress based on average level (simplified)
            level_var = analysis.state_variables.get('player_level')
            if level_var:
                current_level = level_var.current_value
                target_level = 15
                return min(100.0, (current_level / target_level) * 100)
        
        return goal.progress_percentage
    
    def get_recommended_actions(self, analysis: GameStateAnalysis) -> List[Tuple[str, str, float]]:
        """Get recommended actions based on active goals"""
        active_goals = self.evaluate_goals(analysis)
        recommendations = []
        
        for goal in active_goals:
            # Weight recommendations by priority
            priority_weight = (6 - goal.priority.value) / 5.0  # Emergency=1.0, Low=0.2
            
            for action in goal.actions_suggested:
                recommendations.append((
                    action,
                    f"For goal: {goal.name}",
                    priority_weight
                ))
        
        # Sort by weight and return top recommendations
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:5]  # Top 5 recommendations
    
    def get_current_strategy_summary(self) -> str:
        """Get a summary of current strategic goals for LLM context"""
        active_goals = [self.goals[goal_id] for goal_id in self.active_goals if goal_id in self.goals]
        
        if not active_goals:
            return "No active strategic goals currently."
        
        summary_parts = ["Current Strategic Goals:"]
        for i, goal in enumerate(active_goals, 1):
            progress = f"{goal.progress_percentage:.1f}%"
            summary_parts.append(f"{i}. {goal.name} ({progress}) - {goal.description}")
        
        return "\n".join(summary_parts)
    
    def get_goal_stats(self) -> Dict[str, Any]:
        """Get statistics about goal completion and performance"""
        total_goals = len(self.goals)
        completed_goals = len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED])
        active_goals = len(self.active_goals)
        
        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "active_goals": active_goals,
            "completion_rate": completed_goals / total_goals if total_goals > 0 else 0.0,
            "recent_completions": len([h for h in self.goal_history if h[2] > datetime.now() - timedelta(minutes=10)])
        }