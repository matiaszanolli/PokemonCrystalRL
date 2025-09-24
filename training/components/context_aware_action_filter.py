"""
Context-Aware Action Filter - Intelligent action filtering for Pokemon Crystal RL

Filters and prioritizes actions based on current game context to make LLM decisions
more appropriate and effective.
"""

import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GameContext(Enum):
    """Different game contexts that require different action sets."""
    OVERWORLD = "overworld"
    BATTLE = "battle"
    DIALOGUE = "dialogue"
    MENU = "menu"
    TITLE_SCREEN = "title_screen"
    SHOP = "shop"
    POKEMON_CENTER = "pokemon_center"
    GYM = "gym"
    UNKNOWN = "unknown"


@dataclass
class ActionPriority:
    """Represents action priorities for different contexts."""
    primary: List[int]      # Most important actions
    secondary: List[int]    # Useful actions
    discouraged: List[int]  # Actions to avoid
    forbidden: List[int]    # Actions that should never be used


class ContextAwareActionFilter:
    """Filters actions based on current game context and situation."""

    def __init__(self):
        self.logger = logging.getLogger("ContextAwareActionFilter")

        # Action mappings (PyBoy action codes 1-8)
        self.action_names = {
            1: "up",
            2: "down",
            3: "left",
            4: "right",
            5: "a",
            6: "b",
            7: "start",
            8: "select"
        }

        # Define action priorities for different contexts
        self.context_priorities = self._initialize_context_priorities()

        # Situation-specific modifiers
        self.situation_modifiers = self._initialize_situation_modifiers()

        self.logger.info("Context-aware action filter initialized")

    def _initialize_context_priorities(self) -> Dict[GameContext, ActionPriority]:
        """Initialize action priorities for different game contexts."""
        return {
            GameContext.OVERWORLD: ActionPriority(
                primary=[1, 2, 3, 4, 5],  # Movement + A for interaction
                secondary=[6],             # B for running/canceling
                discouraged=[7, 8],       # Start/Select less useful in overworld
                forbidden=[]
            ),

            GameContext.BATTLE: ActionPriority(
                primary=[5, 2, 1],        # A for attack, DOWN/UP for menu navigation
                secondary=[6, 3, 4],      # B for back, LEFT/RIGHT for move selection
                discouraged=[7, 8],       # Start/Select rarely needed in battle
                forbidden=[]
            ),

            GameContext.DIALOGUE: ActionPriority(
                primary=[5],              # A to advance dialogue
                secondary=[6],            # B sometimes works to skip
                discouraged=[1, 2, 3, 4, 7, 8],  # Movement usually doesn't work
                forbidden=[]
            ),

            GameContext.MENU: ActionPriority(
                primary=[1, 2, 5, 6],     # UP/DOWN navigation, A to select, B to back
                secondary=[3, 4],         # LEFT/RIGHT for some menus
                discouraged=[7, 8],       # Start/Select usually not needed
                forbidden=[]
            ),

            GameContext.TITLE_SCREEN: ActionPriority(
                primary=[7, 5],           # START to begin, A to select
                secondary=[1, 2],         # UP/DOWN for menu
                discouraged=[3, 4, 6, 8], # Other actions less useful
                forbidden=[]
            ),

            GameContext.SHOP: ActionPriority(
                primary=[1, 2, 5, 6],     # Navigation and selection
                secondary=[3, 4],         # Sometimes needed for quantity
                discouraged=[7, 8],       # Start/Select rarely needed
                forbidden=[]
            ),

            GameContext.POKEMON_CENTER: ActionPriority(
                primary=[1, 2, 5, 6],     # Navigation and interaction
                secondary=[3, 4],         # Movement to heal machine
                discouraged=[7, 8],       # Start/Select not needed
                forbidden=[]
            ),

            GameContext.GYM: ActionPriority(
                primary=[1, 2, 3, 4, 5],  # Movement and interaction crucial
                secondary=[6],            # B for backing out
                discouraged=[7, 8],       # Start/Select less important
                forbidden=[]
            ),

            GameContext.UNKNOWN: ActionPriority(
                primary=[1, 2, 3, 4, 5, 6],  # Conservative: allow basic actions
                secondary=[7, 8],             # Allow menu actions as secondary
                discouraged=[],
                forbidden=[]
            )
        }

    def _initialize_situation_modifiers(self) -> Dict[str, Dict[str, List[int]]]:
        """Initialize situation-specific action modifiers."""
        return {
            # Health-based modifiers
            "low_health": {
                "prioritize": [5],        # A button to use items/heal
                "discourage": [1, 2, 3, 4]  # Avoid movement when low health
            },

            "no_pokemon": {
                "prioritize": [1, 2, 3, 4, 5],  # Need to explore and interact
                "discourage": []
            },

            # Stuck situation modifiers
            "stuck": {
                "prioritize": [6, 7, 8],     # Try menu actions when stuck
                "discourage": []              # Don't discourage any movement
            },

            "repeated_location": {
                "prioritize": [3, 4],        # Try different horizontal movement
                "discourage": [1, 2]         # Discourage vertical if stuck
            },

            # Progress-based modifiers
            "early_game": {
                "prioritize": [1, 2, 3, 4, 5],  # Exploration is key
                "discourage": [7, 8]             # Menus less important early
            },

            "has_pokemon": {
                "prioritize": [5],           # A for battles and interactions
                "discourage": []
            },

            # Map-specific modifiers
            "new_location": {
                "prioritize": [1, 2, 3, 4, 5],  # Explore thoroughly
                "discourage": [6]                # Don't run away immediately
            },

            "familiar_location": {
                "prioritize": [5, 6],       # Quick interactions, B to run
                "discourage": []
            }
        }

    def filter_actions(self,
                      game_state: Dict[str, Any],
                      context: Dict[str, Any],
                      available_actions: List[int] = None) -> Dict[str, Any]:
        """Filter and prioritize actions based on current context.

        Args:
            game_state: Current game state
            context: Enhanced context from strategic builder
            available_actions: List of available actions (default: all 1-8)

        Returns:
            Dict containing filtered actions and reasoning
        """
        if available_actions is None:
            available_actions = list(range(1, 9))  # All PyBoy actions

        try:
            # Determine current game context
            game_context = self._determine_game_context(game_state, context)

            # Get base priorities for this context
            base_priorities = self.context_priorities.get(game_context,
                                                         self.context_priorities[GameContext.UNKNOWN])

            # Apply situation modifiers
            modified_priorities = self._apply_situation_modifiers(
                base_priorities, game_state, context
            )

            # Filter actions based on priorities
            filtered_actions = self._apply_action_filtering(
                available_actions, modified_priorities
            )

            # Generate recommendation reasoning
            reasoning = self._generate_action_reasoning(
                game_context, modified_priorities, filtered_actions
            )

            return {
                'context': game_context.value,
                'primary_actions': filtered_actions['primary'],
                'secondary_actions': filtered_actions['secondary'],
                'discouraged_actions': filtered_actions['discouraged'],
                'forbidden_actions': filtered_actions['forbidden'],
                'recommended_action': self._get_recommended_action(filtered_actions),
                'reasoning': reasoning,
                'confidence': self._calculate_confidence(game_context, filtered_actions)
            }

        except Exception as e:
            self.logger.error(f"Action filtering failed: {e}")
            return self._get_fallback_filtering(available_actions)

    def _determine_game_context(self,
                               game_state: Dict[str, Any],
                               context: Dict[str, Any]) -> GameContext:
        """Determine current game context from state and screen analysis."""
        # Check for battle first
        if game_state.get('in_battle', False):
            return GameContext.BATTLE

        # Check detected state from context
        detected_state = context.get('detected_state', context.get('current_state', 'unknown'))

        if detected_state == 'battle':
            return GameContext.BATTLE
        elif detected_state == 'dialogue':
            return GameContext.DIALOGUE
        elif detected_state == 'menu':
            return GameContext.MENU
        elif detected_state == 'title_screen':
            return GameContext.TITLE_SCREEN
        elif detected_state in ['overworld', 'early_game', 'beginning_journey', 'progressing']:
            # Check for special locations
            map_id = game_state.get('player_map', 0)
            if self._is_pokemon_center(map_id):
                return GameContext.POKEMON_CENTER
            elif self._is_shop(map_id):
                return GameContext.SHOP
            elif self._is_gym(map_id):
                return GameContext.GYM
            else:
                return GameContext.OVERWORLD
        else:
            return GameContext.UNKNOWN

    def _is_pokemon_center(self, map_id: int) -> bool:
        """Check if current map is a Pokemon Center."""
        # Pokemon Center map IDs (these would need to be researched)
        pokemon_center_maps = {10, 11, 20, 21}  # Example IDs
        return map_id in pokemon_center_maps

    def _is_shop(self, map_id: int) -> bool:
        """Check if current map is a shop."""
        # Shop map IDs (these would need to be researched)
        shop_maps = {15, 16, 25, 26}  # Example IDs
        return map_id in shop_maps

    def _is_gym(self, map_id: int) -> bool:
        """Check if current map is a gym."""
        # Gym map IDs (these would need to be researched)
        gym_maps = {32, 33, 34, 35}  # Example IDs - Sprout Tower, etc.
        return map_id in gym_maps

    def _apply_situation_modifiers(self,
                                  base_priorities: ActionPriority,
                                  game_state: Dict[str, Any],
                                  context: Dict[str, Any]) -> ActionPriority:
        """Apply situation-specific modifiers to base priorities."""
        modified = ActionPriority(
            primary=base_priorities.primary.copy(),
            secondary=base_priorities.secondary.copy(),
            discouraged=base_priorities.discouraged.copy(),
            forbidden=base_priorities.forbidden.copy()
        )

        # Check various situations and apply modifiers
        situations = self._detect_situations(game_state, context)

        for situation in situations:
            if situation in self.situation_modifiers:
                modifiers = self.situation_modifiers[situation]

                # Add prioritized actions
                for action in modifiers.get('prioritize', []):
                    if action not in modified.primary:
                        modified.primary.append(action)
                    # Remove from discouraged if it was there
                    if action in modified.discouraged:
                        modified.discouraged.remove(action)

                # Add discouraged actions
                for action in modifiers.get('discourage', []):
                    if action not in modified.discouraged and action not in modified.primary:
                        modified.discouraged.append(action)

        return modified

    def _detect_situations(self,
                          game_state: Dict[str, Any],
                          context: Dict[str, Any]) -> List[str]:
        """Detect current situations that should modify action priorities."""
        situations = []

        # Health-based situations
        if context.get('progress', {}).get('health_status') in ['critical', 'low']:
            situations.append('low_health')

        # Pokemon availability
        if game_state.get('party_count', 0) == 0:
            situations.append('no_pokemon')
        else:
            situations.append('has_pokemon')

        # Stuck detection
        if context.get('stuck_analysis', {}).get('is_stuck', False):
            situations.append('stuck')

        # Progress-based situations
        if game_state.get('badges', 0) == 0 and game_state.get('party_count', 0) == 0:
            situations.append('early_game')

        # Location familiarity
        if context.get('position', {}).get('is_new_location', False):
            situations.append('new_location')
        else:
            situations.append('familiar_location')

        # Movement patterns
        if context.get('performance', {}).get('exploration_rate', 0) < 0.3:
            situations.append('repeated_location')

        return situations

    def _apply_action_filtering(self,
                               available_actions: List[int],
                               priorities: ActionPriority) -> Dict[str, List[int]]:
        """Apply priority filtering to available actions."""
        filtered = {
            'primary': [],
            'secondary': [],
            'discouraged': [],
            'forbidden': []
        }

        for action in available_actions:
            if action in priorities.forbidden:
                filtered['forbidden'].append(action)
            elif action in priorities.primary:
                filtered['primary'].append(action)
            elif action in priorities.secondary:
                filtered['secondary'].append(action)
            elif action in priorities.discouraged:
                filtered['discouraged'].append(action)
            else:
                # Default to secondary if not categorized
                filtered['secondary'].append(action)

        return filtered

    def _get_recommended_action(self, filtered_actions: Dict[str, List[int]]) -> Optional[int]:
        """Get the single most recommended action."""
        # Prefer primary actions
        if filtered_actions['primary']:
            return filtered_actions['primary'][0]

        # Fall back to secondary
        if filtered_actions['secondary']:
            return filtered_actions['secondary'][0]

        # If everything is discouraged, pick least discouraged
        if filtered_actions['discouraged']:
            return filtered_actions['discouraged'][0]

        return None

    def _generate_action_reasoning(self,
                                  context: GameContext,
                                  priorities: ActionPriority,
                                  filtered_actions: Dict[str, List[int]]) -> str:
        """Generate human-readable reasoning for action choices."""
        reasoning_parts = [f"Context: {context.value}"]

        if filtered_actions['primary']:
            actions_str = ", ".join([self.action_names[a] for a in filtered_actions['primary']])
            reasoning_parts.append(f"Primary actions: {actions_str}")

        if filtered_actions['discouraged']:
            actions_str = ", ".join([self.action_names[a] for a in filtered_actions['discouraged']])
            reasoning_parts.append(f"Discouraged: {actions_str}")

        if filtered_actions['forbidden']:
            actions_str = ", ".join([self.action_names[a] for a in filtered_actions['forbidden']])
            reasoning_parts.append(f"Forbidden: {actions_str}")

        return " | ".join(reasoning_parts)

    def _calculate_confidence(self,
                             context: GameContext,
                             filtered_actions: Dict[str, List[int]]) -> float:
        """Calculate confidence in the action filtering."""
        confidence = 0.5  # Base confidence

        # Higher confidence for well-defined contexts
        if context in [GameContext.BATTLE, GameContext.DIALOGUE, GameContext.MENU]:
            confidence += 0.3
        elif context == GameContext.OVERWORLD:
            confidence += 0.2
        elif context == GameContext.UNKNOWN:
            confidence -= 0.2

        # Higher confidence when we have clear primary actions
        if len(filtered_actions['primary']) > 0:
            confidence += 0.2

        # Lower confidence when many actions are forbidden
        if len(filtered_actions['forbidden']) > 3:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _get_fallback_filtering(self, available_actions: List[int]) -> Dict[str, Any]:
        """Provide fallback filtering when normal filtering fails."""
        return {
            'context': 'fallback',
            'primary_actions': [1, 2, 3, 4, 5],  # Basic movement + A
            'secondary_actions': [6],              # B button
            'discouraged_actions': [7, 8],         # Menu buttons
            'forbidden_actions': [],
            'recommended_action': 5,               # A button
            'reasoning': 'Fallback filtering due to error',
            'confidence': 0.3
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about action filtering."""
        return {
            'contexts_supported': len(self.context_priorities),
            'situation_modifiers': len(self.situation_modifiers),
            'action_types': len(self.action_names)
        }


class NPCInteractionPatterns:
    """Enhanced NPC interaction pattern recognition and dialogue management"""

    def __init__(self):
        self.logger = logging.getLogger("NPCInteractionPatterns")

        # NPC types and their interaction patterns
        self.npc_types = {
            'nurse_joy': {
                'location': 'pokemon_center',
                'purpose': 'healing',
                'dialogue_pattern': 'service_standard',
                'actions': ['approach', 'interact', 'confirm_healing', 'wait', 'finish'],
                'key_phrases': ['heal', 'pokemon', 'center'],
                'expected_outcome': 'pokemon_healed'
            },
            'gym_leader': {
                'location': 'gym',
                'purpose': 'battle',
                'dialogue_pattern': 'challenge_introduction',
                'actions': ['approach', 'interact', 'accept_challenge', 'battle'],
                'key_phrases': ['challenge', 'battle', 'badge', 'gym'],
                'expected_outcome': 'battle_initiated'
            },
            'professor': {
                'location': 'lab',
                'purpose': 'story_progression',
                'dialogue_pattern': 'tutorial_exposition',
                'actions': ['approach', 'listen', 'acknowledge', 'receive_item'],
                'key_phrases': ['professor', 'research', 'pokemon', 'help'],
                'expected_outcome': 'quest_advancement'
            },
            'shopkeeper': {
                'location': 'shop',
                'purpose': 'commerce',
                'dialogue_pattern': 'transaction_focused',
                'actions': ['approach', 'browse', 'select', 'purchase', 'leave'],
                'key_phrases': ['buy', 'sell', 'shop', 'item', 'money'],
                'expected_outcome': 'transaction_completed'
            },
            'trainer': {
                'location': 'route',
                'purpose': 'battle',
                'dialogue_pattern': 'challenge_direct',
                'actions': ['approach', 'engage', 'battle'],
                'key_phrases': ['trainer', 'battle', 'pokemon', 'fight'],
                'expected_outcome': 'battle_initiated'
            },
            'quest_npc': {
                'location': 'various',
                'purpose': 'quest_management',
                'dialogue_pattern': 'information_exchange',
                'actions': ['approach', 'listen', 'inquire', 'accept_quest', 'complete_quest'],
                'key_phrases': ['help', 'quest', 'task', 'reward', 'find'],
                'expected_outcome': 'quest_update'
            }
        }

        # Dialogue state patterns
        self.dialogue_states = {
            'greeting': {
                'indicators': ['hello', 'hi', 'welcome', 'greetings'],
                'responses': ['A to continue', 'Listen to greeting'],
                'next_expected': 'service_offer'
            },
            'service_offer': {
                'indicators': ['help', 'heal', 'buy', 'challenge'],
                'responses': ['A to accept', 'A to proceed'],
                'next_expected': 'confirmation'
            },
            'confirmation': {
                'indicators': ['yes', 'no', 'confirm', 'sure'],
                'responses': ['A for yes', 'B for no'],
                'next_expected': 'service_execution'
            },
            'service_execution': {
                'indicators': ['processing', 'working', 'battling'],
                'responses': ['Wait', 'Let action complete'],
                'next_expected': 'completion'
            },
            'completion': {
                'indicators': ['done', 'finished', 'thank you', 'goodbye'],
                'responses': ['A to acknowledge', 'Continue interaction'],
                'next_expected': 'exit'
            },
            'information_request': {
                'indicators': ['what', 'where', 'how', 'tell me'],
                'responses': ['A to continue', 'Listen carefully'],
                'next_expected': 'information_delivery'
            },
            'information_delivery': {
                'indicators': ['here is', 'you need', 'go to', 'find'],
                'responses': ['A to acknowledge', 'Remember information'],
                'next_expected': 'completion'
            }
        }

    def identify_npc_type(self,
                         game_state: Dict[str, Any],
                         context: Dict[str, Any]) -> Optional[str]:
        """Identify the type of NPC being interacted with"""
        location = context.get('detected_state', 'unknown')
        player_map = game_state.get('player_map', 0)

        # Location-based NPC identification
        if location == 'dialogue' or context.get('dialogue_active', False):
            # Analyze location context
            if 'pokemon_center' in str(context).lower() or player_map in [1, 5, 8]:  # Common Pokemon Center maps
                return 'nurse_joy'
            elif 'gym' in str(context).lower() or player_map in [6, 12, 16, 21]:  # Gym locations
                return 'gym_leader'
            elif 'shop' in str(context).lower() or 'mart' in str(context).lower():
                return 'shopkeeper'
            elif 'lab' in str(context).lower() or player_map == 1:  # Professor Elm's lab
                return 'professor'
            else:
                # Default to quest NPC for unknown dialogue situations
                return 'quest_npc'

        return None

    def get_dialogue_strategy(self,
                            npc_type: str,
                            game_state: Dict[str, Any],
                            dialogue_history: List[str] = None) -> Dict[str, Any]:
        """Get intelligent dialogue strategy for specific NPC type"""
        if npc_type not in self.npc_types:
            return self._get_generic_dialogue_strategy()

        npc_info = self.npc_types[npc_type]

        strategy = {
            'npc_type': npc_type,
            'primary_purpose': npc_info['purpose'],
            'expected_actions': npc_info['actions'],
            'dialogue_approach': self._determine_dialogue_approach(npc_type, game_state),
            'recommended_responses': self._get_recommended_responses(npc_type),
            'interaction_goal': npc_info['expected_outcome'],
            'patience_level': self._calculate_patience_level(npc_type, game_state)
        }

        return strategy

    def _determine_dialogue_approach(self, npc_type: str, game_state: Dict[str, Any]) -> str:
        """Determine the best approach for dialogue with this NPC type"""
        party_count = game_state.get('party_count', 0)
        badges_count = game_state.get('badges_total', 0)
        player_hp = game_state.get('player_hp', 0)
        player_max_hp = game_state.get('player_max_hp', 1)
        hp_ratio = player_hp / max(player_max_hp, 1) if party_count > 0 else 1.0

        if npc_type == 'nurse_joy':
            if hp_ratio < 0.5:
                return 'urgent_healing'
            else:
                return 'standard_service'

        elif npc_type == 'gym_leader':
            if hp_ratio > 0.8 and party_count > 0:
                return 'ready_for_challenge'
            else:
                return 'prepare_first'

        elif npc_type == 'shopkeeper':
            money = game_state.get('money', 0)
            if money > 1000:
                return 'purchase_ready'
            else:
                return 'browse_only'

        elif npc_type == 'professor':
            if party_count == 0:
                return 'first_pokemon'
            else:
                return 'progress_check'

        return 'standard_interaction'

    def _get_recommended_responses(self, npc_type: str) -> List[str]:
        """Get context-appropriate response recommendations"""
        responses = {
            'nurse_joy': [
                'A to request healing',
                'A to confirm healing',
                'A to acknowledge completion'
            ],
            'gym_leader': [
                'A to accept challenge',
                'A to continue dialogue',
                'B if not ready to battle'
            ],
            'shopkeeper': [
                'A to select items',
                'A to confirm purchase',
                'B to exit shop'
            ],
            'professor': [
                'A to continue conversation',
                'A to accept Pokemon/items',
                'A to acknowledge advice'
            ],
            'trainer': [
                'A to engage in battle',
                'A to continue dialogue'
            ],
            'quest_npc': [
                'A to continue listening',
                'A to accept quest',
                'A to complete quest'
            ]
        }

        return responses.get(npc_type, ['A to continue', 'B to exit'])

    def _calculate_patience_level(self, npc_type: str, game_state: Dict[str, Any]) -> str:
        """Calculate how patient to be with this NPC interaction"""
        # Some NPCs require more patience than others
        patience_map = {
            'nurse_joy': 'high',      # Healing is important
            'professor': 'high',      # Story progression is crucial
            'gym_leader': 'medium',   # Challenge acceptance needs care
            'shopkeeper': 'medium',   # Transaction efficiency
            'trainer': 'low',         # Battle engagement is quick
            'quest_npc': 'high'      # Quest information is valuable
        }

        return patience_map.get(npc_type, 'medium')

    def _get_generic_dialogue_strategy(self) -> Dict[str, Any]:
        """Fallback strategy for unknown NPC types"""
        return {
            'npc_type': 'unknown',
            'primary_purpose': 'information_gathering',
            'expected_actions': ['approach', 'listen', 'respond'],
            'dialogue_approach': 'cautious_exploration',
            'recommended_responses': ['A to continue', 'B to exit'],
            'interaction_goal': 'gather_information',
            'patience_level': 'medium'
        }

    def analyze_dialogue_progress(self,
                                dialogue_history: List[str],
                                current_state: str) -> Dict[str, Any]:
        """Analyze dialogue progression and suggest next actions"""
        if not dialogue_history:
            return {
                'stage': 'initial',
                'suggestion': 'Begin interaction with A button',
                'confidence': 0.8
            }

        # Analyze recent dialogue for patterns
        recent_text = ' '.join(dialogue_history[-3:]).lower()

        for state, info in self.dialogue_states.items():
            for indicator in info['indicators']:
                if indicator in recent_text:
                    return {
                        'stage': state,
                        'suggestion': info['responses'][0],
                        'next_expected': info['next_expected'],
                        'confidence': 0.9
                    }

        # Default if no pattern matches
        return {
            'stage': 'unknown',
            'suggestion': 'Continue with A button',
            'confidence': 0.5
        }