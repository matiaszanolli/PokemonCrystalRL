"""
Game Intelligence Orchestrator

Main coordinator that combines all intelligence modules for comprehensive game analysis.
"""

from typing import Dict, List
import logging

from environments.state.analyzer import GamePhase
from .location import LocationType, IntelligenceGameContext, GameContext, ActionPlan, LocationAnalyzer
from .progression import ProgressTracker
from .battle import BattleStrategy
from .inventory import InventoryManager


class GameIntelligence:
    """Main game intelligence coordinator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.location_analyzer = LocationAnalyzer()
        self.progress_tracker = ProgressTracker()
        self.battle_strategy = BattleStrategy()
        self.inventory_manager = InventoryManager()

    def analyze_game_context(self, game_state: Dict, screen_analysis: Dict) -> GameContext:
        """Perform comprehensive game analysis"""

        # Location analysis
        location_type, location_name = self.location_analyzer.analyze_location(game_state)

        # Game phase
        phase = self.progress_tracker.get_game_phase(game_state)

        # Goals
        immediate_goals = self.progress_tracker.get_immediate_goals(game_state, location_type)
        strategic_goals = self.progress_tracker.get_strategic_goals(game_state)

        # Health and party status
        party_count = game_state.get('party_count', 0)

        if party_count > 0:
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
            health_status = "Critical" if hp_ratio < 0.3 else "Low" if hp_ratio < 0.6 else "Good"
        else:
            hp_ratio = 1.0  # No Pokemon = no health concerns
            health_status = "No Pokemon"

        party_status = f"{party_count} Pokemon" + (" (need more!)" if party_count < 2 else "")

        # Recommended actions based on context
        recommended_actions = self.location_analyzer.get_location_strategy(location_type, game_state)

        # Battle strategy if in battle
        if game_state.get('in_battle', 0):
            battle_advice = self.battle_strategy.get_battle_strategy(game_state)
            recommended_actions.insert(0, f"Battle: {battle_advice}")

        # Inventory and item recommendations
        inventory_analysis = self.inventory_manager.analyze_inventory_needs(game_state, screen_analysis)
        if inventory_analysis['immediate_needs']:
            for need in inventory_analysis['immediate_needs']:
                recommended_actions.insert(0 if need == 'emergency_healing' else 1, f"Item: {need}")

        # Add item usage advice to recommendations
        item_recommendation = self.inventory_manager.recommend_item_usage(game_state)
        if item_recommendation['should_use_item']:
            urgency_text = f"({item_recommendation['urgency']})" if item_recommendation['urgency'] != 'normal' else ''
            recommended_actions.insert(0, f"Use {item_recommendation['item_type']} item {urgency_text}")

        # Urgency level
        urgency = 1
        if party_count == 0:
            urgency = 4  # High priority: get first Pokemon
        elif party_count > 0 and hp_ratio < 0.2:
            urgency = 5  # Critical: heal immediately
        elif party_count > 0 and hp_ratio < 0.4:
            urgency = 3  # Medium: consider healing

        return GameContext(
            phase=phase,
            location_type=location_type,
            location_name=location_name,
            immediate_goals=immediate_goals,
            strategic_goals=strategic_goals,
            health_status=health_status,
            party_status=party_status,
            recommended_actions=recommended_actions,
            urgency_level=urgency
        )

    def get_action_plan(self, game_context: GameContext, game_state: Dict) -> List[ActionPlan]:
        """Generate multi-step action plans"""
        plans = []

        # Emergency healing plan (only if we have Pokemon)
        party_count = game_state.get('party_count', 0)
        if game_context.urgency_level >= 4 and party_count > 0:
            if game_context.location_type == LocationType.POKEMON_CENTER:
                plans.append(ActionPlan(
                    goal="Emergency heal at Pokemon Center",
                    steps=["Walk to counter", "Interact with nurse", "Confirm healing"],
                    priority=10,
                    estimated_actions=5
                ))
            else:
                plans.append(ActionPlan(
                    goal="Find Pokemon Center for emergency healing",
                    steps=["Open map", "Navigate to nearest town", "Find Pokemon Center"],
                    priority=9,
                    estimated_actions=20
                ))

        # Tutorial plan
        if game_context.phase == GamePhase.TUTORIAL:
            plans.append(ActionPlan(
                goal="Get starter Pokemon from Professor Elm",
                steps=["Navigate to lab", "Talk to Professor Elm", "Choose starter"],
                priority=8,
                estimated_actions=15
            ))

        # Gym challenge plan
        elif game_context.phase == GamePhase.GYM_BATTLES:
            if game_context.location_type == LocationType.GYM:
                plans.append(ActionPlan(
                    goal="Challenge gym leader",
                    steps=["Navigate through gym", "Battle gym trainers", "Challenge leader"],
                    priority=7,
                    estimated_actions=30
                ))

        return sorted(plans, key=lambda x: x.priority, reverse=True)

    def get_contextual_advice(self, game_context: GameContext, recent_actions: List[str]) -> str:
        """Get human-readable advice for current situation"""
        advice_parts = []

        # Phase context
        advice_parts.append(f"Phase: {game_context.phase.name}")
        advice_parts.append(f"Location: {game_context.location_name} ({game_context.location_type.name})")

        # Urgent matters
        if game_context.urgency_level >= 4:
            advice_parts.append(f"⚠️ URGENT ({game_context.urgency_level}/5): {game_context.immediate_goals[0]}")

        # Immediate goals
        if game_context.immediate_goals:
            advice_parts.append(f"Next: {game_context.immediate_goals[0]}")

        # Strategic context
        if game_context.strategic_goals:
            advice_parts.append(f"Goal: {game_context.strategic_goals[0]}")

        # Recent action analysis
        if recent_actions:
            recent_str = " → ".join(recent_actions[-3:])
            if "START" in recent_actions[-2:] and "b" not in recent_actions[-1:]:
                advice_parts.append("⚠️ Recently opened menu - consider exiting with 'b'")
            elif recent_actions[-1:] == recent_actions[-2:-1]:
                advice_parts.append("⚠️ Repeated action detected - may be stuck")

        return " | ".join(advice_parts)
