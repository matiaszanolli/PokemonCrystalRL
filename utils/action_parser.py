"""
Pokemon Crystal Action Parser Utilities

Functions for parsing, validating, and translating game actions.
"""

from typing import Dict, List, Tuple, Optional
import logging
from config.constants import (
    SCREEN_STATES,
    AVAILABLE_ACTIONS,
    TRAINING_PARAMS,
)

logger = logging.getLogger(__name__)

def parse_action_response(response: str) -> str:
    """Parse LLM response to extract action with enhanced synonym recognition."""
    try:
        response_lower = response.lower().strip()
        
        # Enhanced action word mapping with synonyms
        action_mappings = {
            # Basic directions
            'up': ['up', 'north', 'forward'],
            'down': ['down', 'south', 'backward'],
            'left': ['left', 'west'],
            'right': ['right', 'east'],
            # Buttons
            'a': ['a', 'interact', 'confirm', 'attack', 'select_pokemon', 'use'],
            'b': ['b', 'cancel', 'back', 'flee', 'run', 'escape'],
            'start': ['start', 'menu', 'pause'],
            'select': ['select']
        }
        
        # Look for ACTION: pattern first
        if "action:" in response_lower:
            action_part = response_lower.split("action:")[1].split("\\n")[0].strip()
            words = action_part.split()
            for word in words:
                clean_word = word.strip('.,!?()[]{}').lower()
                for action, synonyms in action_mappings.items():
                    if clean_word in synonyms:
                        return action
        
        # Look for any action word or synonym in the response
        for action, synonyms in action_mappings.items():
            for synonym in synonyms:
                if synonym in response_lower:
                    return action
        
        # Fallback to 'a' if nothing found
        return 'a'
        
    except Exception as e:
        logger.error(f"Error parsing action response: {str(e)}")
        return 'a'  # Safe fallback

def get_allowed_action(
    action: str, 
    game_state: Dict, 
    screen_state: str,
    recent_actions: Optional[List[str]] = None
) -> str:
    """Get an allowed alternative when an action is forbidden."""
    try:
        # Check if action is currently allowed
        if is_action_allowed(action, game_state):
            return action
            
        # Context-aware alternatives based on screen state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Always attack in battle
        elif screen_state == SCREEN_STATES['DIALOGUE']:
            return 'a'  # Progress dialogue
        elif screen_state == SCREEN_STATES['MENU']:
            return 'b'  # Exit menus when we can't use START
        elif screen_state == SCREEN_STATES['LOADING']:
            return 'a'  # Wait during loading
        else:
            # In overworld - focus on exploration and interaction
            return get_exploration_fallback(recent_actions)
            
    except Exception as e:
        logger.error(f"Error getting allowed action: {str(e)}")
        return 'a'  # Safe fallback

def is_action_allowed(action: str, game_state: Dict) -> bool:
    """Check if an action is currently allowed based on game state."""
    try:
        # START and SELECT are forbidden until first Pokemon
        if action in AVAILABLE_ACTIONS['FORBIDDEN_INITIAL']:
            if game_state.get('party_count', 0) == 0:
                return False
                
        # All other actions are allowed
        return True
        
    except Exception as e:
        logger.error(f"Error checking action allowance: {str(e)}")
        return True  # Allow by default

def get_exploration_fallback(recent_actions: Optional[List[str]] = None) -> str:
    """Get smart exploration fallback action avoiding repetition."""
    try:
        recent = recent_actions[-3:] if recent_actions else []
        
        # Priority order: interact with objects/NPCs, then explore
        exploration_priority = ['a', 'up', 'down', 'left', 'right']
        
        # Try to avoid recently used actions
        for action in exploration_priority:
            if action not in recent:
                return action
                
        # If all actions recently used, pick next in sequence
        return exploration_priority[len(recent) % len(exploration_priority)]
        
    except Exception as e:
        logger.error(f"Error getting exploration fallback: {str(e)}")
        return 'a'  # Safe fallback

def get_context_specific_action(
    screen_state: str,
    game_state: Dict,
    recent_actions: Optional[List[str]] = None
) -> Tuple[str, str]:
    """Get context-appropriate action based on game state."""
    try:
        recent = recent_actions[-3:] if recent_actions else []
        
        if game_state.get('in_battle', 0) == 1:
            return 'a', "Attack in battle"
        elif screen_state == SCREEN_STATES['DIALOGUE']:
            return 'a', "Progress dialogue"
        elif screen_state == SCREEN_STATES['SETTINGS_MENU']:
            return 'b', "Exit settings menu"
        elif screen_state == SCREEN_STATES['MENU']:
            # Smart menu handling - check recent actions
            if 'START' in ' '.join(recent).upper():
                return 'b', "Exit recently opened menu"
            else:
                return 'b', "Exit menu"
        elif screen_state == SCREEN_STATES['LOADING']:
            return 'a', "Wait during loading"
        else:
            # Smart exploration avoiding START button spam
            action = get_exploration_pattern_action(recent)
            return action, "Explore overworld"
            
    except Exception as e:
        logger.error(f"Error getting context action: {str(e)}")
        return 'a', "Error fallback"

def get_exploration_pattern_action(recent_actions: Optional[List[str]] = None) -> str:
    """Get next action in exploration pattern avoiding repetition."""
    try:
        # Exploration pattern with interaction
        exploration_actions = [
            'up', 'up', 'a',
            'right', 'right', 'a',
            'down', 'down', 'a',
            'left', 'left', 'a'
        ]
        
        if not recent_actions:
            return exploration_actions[0]
            
        # Find last action in pattern
        last_action = recent_actions[-1]
        try:
            last_idx = exploration_actions.index(last_action)
            next_idx = (last_idx + 1) % len(exploration_actions)
            return exploration_actions[next_idx]
        except ValueError:
            # Last action not in pattern, start fresh
            return exploration_actions[0]
            
    except Exception as e:
        logger.error(f"Error getting exploration pattern: {str(e)}")
        return 'a'  # Safe fallback

def is_action_safe(
    action: str,
    game_state: Dict,
    screen_state: str,
    stuck_action_count: Optional[Dict[str, int]] = None
) -> bool:
    """Check if an action is safe in current context to prevent loops."""
    try:
        # Never safe to use forbidden actions
        if not is_action_allowed(action, game_state):
            return False
            
        # Check for stuck action patterns
        if stuck_action_count and action in stuck_action_count:
            if stuck_action_count[action] >= 5:  # Too many repeats
                return False
                
        # Context-specific checks
        if screen_state == SCREEN_STATES['SETTINGS_MENU']:
            return action == 'b'  # Only safe to exit
        elif screen_state == SCREEN_STATES['MENU']:
            if 'START' in action.upper():
                return False  # Don't open nested menus
                
        return True
        
    except Exception as e:
        logger.error(f"Error checking action safety: {str(e)}")
        return True  # Allow by default
