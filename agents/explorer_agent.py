"""
ExplorerAgent - Specialist agent for map discovery and navigation

This agent specializes in exploration, discovery of new areas, item collection,
and efficient navigation. It leverages the enhanced location mapping system.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque

from .base_agent import BaseAgent
from core.event_system import EventType, Event, EventSubscriber, get_event_bus

try:
    from training.components.strategic_context_builder import StrategicContextBuilder
except ImportError:
    print("⚠️  StrategicContextBuilder not available")
    StrategicContextBuilder = None


@dataclass
class ExplorationTarget:
    """Represents an exploration objective"""
    location_id: int
    location_name: str
    priority: int  # 1-10
    exploration_type: str  # 'new_area', 'item_search', 'pokemon_hunt', 'connection_mapping'
    estimated_distance: int
    expected_rewards: List[str]
    prerequisites: List[str]


@dataclass
class NavigationDecision:
    """Represents a navigation decision with exploration logic"""
    action: int
    direction: str  # 'up', 'down', 'left', 'right', 'interact', 'menu'
    confidence: float
    reasoning: str
    exploration_goal: str
    discovery_potential: float  # 0.0-1.0


class ExplorerAgent(BaseAgent, EventSubscriber):
    """Specialist agent optimized for exploration and map discovery"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = logging.getLogger("ExplorerAgent")

        # Exploration-specific configuration
        self.exploration_config = config.get('exploration_config', {}) if config else {}
        self.curiosity_level = self.exploration_config.get('curiosity', 0.8)  # 0.0-1.0
        self.risk_taking = self.exploration_config.get('risk_taking', 0.6)  # 0.0-1.0
        self.thoroughness = self.exploration_config.get('thoroughness', 0.7)  # 0.0-1.0

        # Initialize strategic context for location intelligence
        self.context_builder = StrategicContextBuilder() if StrategicContextBuilder else None

        # Exploration tracking
        self.visited_locations = set()
        self.location_visit_counts = defaultdict(int)
        self.discovered_connections = defaultdict(set)
        self.discovered_items = []
        self.discovered_npcs = []
        self.discovered_pokemon = []

        # Navigation and pathfinding
        self.movement_history = deque(maxlen=50)
        self.stuck_positions = set()
        self.exploration_targets = []
        self.current_exploration_goal = None

        # Mapping and discovery
        self.map_coverage = {}  # map_id -> coverage_percentage
        self.area_priorities = self._initialize_area_priorities()
        self.discovery_log = []

        # Efficient exploration patterns
        self.exploration_patterns = {
            'systematic_sweep': self._systematic_sweep_pattern,
            'spiral_search': self._spiral_search_pattern,
            'wall_following': self._wall_following_pattern,
            'random_walk': self._random_walk_pattern
        }
        self.current_pattern = 'systematic_sweep'

        # Event system integration
        self.event_bus = get_event_bus()
        self.event_bus.subscribe(self)

        self.logger.info(f"ExplorerAgent initialized with curiosity={self.curiosity_level}, thoroughness={self.thoroughness}")

    def get_action(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Get exploration-optimized action"""

        # Update exploration state
        self._update_exploration_tracking(game_state, info)

        # Analyze exploration context
        exploration_analysis = self._analyze_exploration_context(game_state, info)

        # Make exploration decision
        decision = self._make_exploration_decision(game_state, exploration_analysis)

        # Update movement tracking
        self._track_movement(decision, game_state)

        return decision.action, {
            'source': 'explorer_agent',
            'direction': decision.direction,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'exploration_goal': decision.exploration_goal,
            'discovery_potential': decision.discovery_potential,
            'current_pattern': self.current_pattern,
            'coverage': self._calculate_area_coverage(game_state.get('player_map', 0))
        }

    def _analyze_exploration_context(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive exploration situation analysis"""
        current_map = game_state.get('player_map', 0)
        current_x = game_state.get('player_x', 0)
        current_y = game_state.get('player_y', 0)
        current_position = (current_x, current_y)

        analysis = {
            'current_location': {
                'map_id': current_map,
                'position': current_position,
                'visit_count': self.location_visit_counts[current_map],
                'is_new_area': current_map not in self.visited_locations
            },
            'exploration_status': {
                'total_areas_discovered': len(self.visited_locations),
                'current_area_coverage': self._calculate_area_coverage(current_map),
                'discovery_rate': self._calculate_discovery_rate(),
                'stuck_indicator': self._check_if_stuck(current_position)
            },
            'navigation_context': {
                'recent_movements': list(self.movement_history)[-5:],
                'available_directions': self._detect_available_directions(game_state, info),
                'unexplored_directions': self._find_unexplored_directions(current_position),
                'backtrack_recommendation': self._should_backtrack()
            },
            'discovery_opportunities': {
                'items_nearby': self._detect_nearby_items(info),
                'npcs_nearby': self._detect_nearby_npcs(info),
                'new_areas_accessible': self._find_accessible_new_areas(current_map),
                'hidden_areas_potential': self._assess_hidden_area_potential(current_position)
            }
        }

        return analysis

    def _make_exploration_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Make intelligent exploration decision"""

        # Handle special exploration situations first
        if analysis['exploration_status']['stuck_indicator']:
            return self._handle_stuck_situation(game_state, analysis)

        if analysis['current_location']['is_new_area']:
            return self._explore_new_area_decision(game_state, analysis)

        # Priority-based decision making
        decision_priorities = [
            ('discovery_opportunities', self._discovery_decision),
            ('unexplored_directions', self._unexplored_direction_decision),
            ('systematic_exploration', self._systematic_exploration_decision),
            ('connection_mapping', self._connection_mapping_decision)
        ]

        for priority_type, decision_func in decision_priorities:
            decision = decision_func(game_state, analysis)
            if decision.confidence > 0.6:  # High confidence decision found
                return decision

        # Fallback to exploration pattern
        return self._pattern_based_decision(game_state, analysis)

    def _discovery_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Make decision based on discovery opportunities"""
        opportunities = analysis['discovery_opportunities']

        # Prioritize item discovery
        if opportunities['items_nearby']:
            return NavigationDecision(
                action=5,  # A button - interact
                direction='interact',
                confidence=0.9,
                reasoning='Item discovered nearby - investigate',
                exploration_goal='item_collection',
                discovery_potential=0.8
            )

        # Prioritize NPC interactions
        if opportunities['npcs_nearby']:
            return NavigationDecision(
                action=5,  # A button - interact
                direction='interact',
                confidence=0.8,
                reasoning='NPC discovered - gather information',
                exploration_goal='npc_interaction',
                discovery_potential=0.7
            )

        # Check for new area access
        if opportunities['new_areas_accessible']:
            return self._navigate_to_new_area(game_state, analysis)

        return NavigationDecision(action=1, direction='up', confidence=0.3, reasoning='Low discovery opportunity', exploration_goal='continue_search', discovery_potential=0.2)

    def _unexplored_direction_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Make decision based on unexplored directions"""
        unexplored = analysis['navigation_context']['unexplored_directions']
        available = analysis['navigation_context']['available_directions']

        # Find best unexplored direction
        for direction in unexplored:
            if direction in available:
                action = self._direction_to_action(direction)
                return NavigationDecision(
                    action=action,
                    direction=direction,
                    confidence=0.8,
                    reasoning=f'Unexplored {direction} direction available',
                    exploration_goal='area_mapping',
                    discovery_potential=0.6
                )

        return NavigationDecision(action=1, direction='up', confidence=0.4, reasoning='No clear unexplored directions', exploration_goal='systematic_search', discovery_potential=0.3)

    def _systematic_exploration_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Use systematic exploration pattern"""
        pattern_func = self.exploration_patterns[self.current_pattern]
        action, direction = pattern_func(game_state, analysis)

        return NavigationDecision(
            action=action,
            direction=direction,
            confidence=0.7,
            reasoning=f'Following {self.current_pattern} exploration pattern',
            exploration_goal='systematic_coverage',
            discovery_potential=0.5
        )

    def _systematic_sweep_pattern(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[int, str]:
        """Systematic left-to-right, top-to-bottom sweep"""
        # Simple implementation: prioritize right and down movement
        available = analysis['navigation_context']['available_directions']

        if 'right' in available:
            return 4, 'right'  # Right
        elif 'down' in available:
            return 2, 'down'   # Down
        elif 'left' in available:
            return 3, 'left'   # Left
        else:
            return 1, 'up'     # Up

    def _spiral_search_pattern(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[int, str]:
        """Spiral outward from center point"""
        # Simplified spiral: alternate between right, down, left, up
        recent_moves = analysis['navigation_context']['recent_movements']
        available = analysis['navigation_context']['available_directions']

        spiral_sequence = ['right', 'down', 'left', 'up']

        if recent_moves:
            last_direction = recent_moves[-1] if recent_moves else 'right'
            try:
                current_index = spiral_sequence.index(last_direction)
                next_direction = spiral_sequence[(current_index + 1) % 4]
            except ValueError:
                next_direction = 'right'
        else:
            next_direction = 'right'

        if next_direction in available:
            return self._direction_to_action(next_direction), next_direction
        else:
            # Find next available direction in spiral
            for direction in spiral_sequence:
                if direction in available:
                    return self._direction_to_action(direction), direction
            return 1, 'up'

    def _wall_following_pattern(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[int, str]:
        """Follow walls for systematic coverage"""
        available = analysis['navigation_context']['available_directions']

        # Simple wall following: prefer right wall
        wall_priority = ['right', 'down', 'left', 'up']

        for direction in wall_priority:
            if direction in available:
                return self._direction_to_action(direction), direction

        return 1, 'up'

    def _random_walk_pattern(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[int, str]:
        """Controlled random exploration"""
        import random
        available = analysis['navigation_context']['available_directions']

        if available:
            direction = random.choice(available)
            return self._direction_to_action(direction), direction

        return 1, 'up'

    def _direction_to_action(self, direction: str) -> int:
        """Convert direction string to action number"""
        direction_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4,
            'interact': 5
        }
        return direction_map.get(direction, 1)

    def _handle_stuck_situation(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Handle when agent appears stuck"""
        current_position = analysis['current_location']['position']

        # Add to stuck positions
        self.stuck_positions.add(current_position)

        # Try different exploration pattern
        if self.current_pattern != 'random_walk':
            self.current_pattern = 'random_walk'

        return NavigationDecision(
            action=6,  # B button - might back out of stuck state
            direction='back',
            confidence=0.6,
            reasoning='Stuck position detected - attempting recovery',
            exploration_goal='unstuck',
            discovery_potential=0.2
        )

    def _explore_new_area_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        """Special handling for newly discovered areas"""
        return NavigationDecision(
            action=5,  # A button - thorough investigation
            direction='interact',
            confidence=0.9,
            reasoning='New area discovered - thorough exploration',
            exploration_goal='new_area_mapping',
            discovery_potential=0.9
        )

    def _update_exploration_tracking(self, game_state: Dict[str, Any], info: Dict[str, Any]):
        """Update exploration state tracking"""
        current_map = game_state.get('player_map', 0)
        current_x = game_state.get('player_x', 0)
        current_y = game_state.get('player_y', 0)

        # Update visited locations
        self.visited_locations.add(current_map)
        self.location_visit_counts[current_map] += 1

        # Track discoveries
        self._track_discoveries(game_state, info)

    def _track_discoveries(self, game_state: Dict[str, Any], info: Dict[str, Any]):
        """Track new discoveries"""
        current_map = game_state.get('player_map', 0)

        # This would be enhanced with actual item/NPC detection
        # For now, log visits to new areas
        if current_map not in self.visited_locations:
            self.discovery_log.append({
                'type': 'new_area',
                'map_id': current_map,
                'timestamp': self.total_steps
            })

    def _calculate_area_coverage(self, map_id: int) -> float:
        """Calculate exploration coverage for an area"""
        visit_count = self.location_visit_counts[map_id]

        # Simple coverage estimation based on visit frequency
        # More visits generally indicate better coverage
        if visit_count == 0:
            return 0.0
        elif visit_count < 5:
            return 0.3
        elif visit_count < 15:
            return 0.6
        elif visit_count < 30:
            return 0.8
        else:
            return 0.95

    def _calculate_discovery_rate(self) -> float:
        """Calculate rate of new discoveries"""
        if self.total_steps == 0:
            return 0.0

        return len(self.discovery_log) / max(self.total_steps, 1)

    def _detect_available_directions(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
        """Detect which directions are available for movement"""
        # This would analyze screen state to detect walls/obstacles
        # For now, assume all directions are available
        return ['up', 'down', 'left', 'right']

    def _find_unexplored_directions(self, current_position: Tuple[int, int]) -> List[str]:
        """Find directions that haven't been explored from current position"""
        # This would track which directions have been tried from each position
        # For now, return random subset
        return ['up', 'right']  # Simplified

    def _initialize_area_priorities(self) -> Dict[int, int]:
        """Initialize exploration priorities for different areas"""
        return {
            1: 10,   # New Bark Town - high priority starting area
            2: 9,    # Route 29 - important route
            5: 8,    # Cherrygrove City - key town
            8: 9,    # Violet City - gym town
            32: 7,   # Sprout Tower - special area
            # Add more as needed
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get exploration agent statistics"""
        base_stats = super().get_stats()

        exploration_stats = {
            'areas_discovered': len(self.visited_locations),
            'total_discoveries': len(self.discovery_log),
            'discovery_rate': self._calculate_discovery_rate(),
            'curiosity_level': self.curiosity_level,
            'thoroughness': self.thoroughness,
            'current_pattern': self.current_pattern,
            'stuck_positions': len(self.stuck_positions),
            'movement_history_size': len(self.movement_history)
        }

        return {**base_stats, **exploration_stats}

    # Additional helper methods (simplified implementations)
    def _should_backtrack(self) -> bool:
        return False

    def _detect_nearby_items(self, info: Dict[str, Any]) -> bool:
        return False

    def _detect_nearby_npcs(self, info: Dict[str, Any]) -> bool:
        return False

    def _find_accessible_new_areas(self, current_map: int) -> List[int]:
        return []

    def _assess_hidden_area_potential(self, position: Tuple[int, int]) -> float:
        return 0.3

    def _navigate_to_new_area(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        return NavigationDecision(action=1, direction='up', confidence=0.5, reasoning='Navigate to new area', exploration_goal='area_transition', discovery_potential=0.7)

    def _connection_mapping_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        return NavigationDecision(action=1, direction='up', confidence=0.4, reasoning='Connection mapping', exploration_goal='map_connections', discovery_potential=0.4)

    def _pattern_based_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> NavigationDecision:
        return self._systematic_exploration_decision(game_state, analysis)

    def _check_if_stuck(self, current_position: Tuple[int, int]) -> bool:
        return current_position in self.stuck_positions

    def _track_movement(self, decision: NavigationDecision, game_state: Dict[str, Any]):
        """Track movement for pattern analysis"""
        self.movement_history.append(decision.direction)

    def get_subscribed_events(self) -> set:
        """Return set of event types this subscriber is interested in"""
        return {
            EventType.LOCATION_CHANGED,
            EventType.NEW_AREA_DISCOVERED,
            EventType.ITEM_DISCOVERED,
            EventType.NPC_ENCOUNTERED,
            EventType.GAME_STATE_CHANGED
        }

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the event system"""
        try:
            if event.event_type == EventType.LOCATION_CHANGED:
                self._handle_location_changed_event(event)
            elif event.event_type == EventType.NEW_AREA_DISCOVERED:
                self._handle_new_area_discovered_event(event)
            elif event.event_type == EventType.ITEM_DISCOVERED:
                self._handle_item_discovered_event(event)
            elif event.event_type == EventType.NPC_ENCOUNTERED:
                self._handle_npc_encountered_event(event)
            elif event.event_type == EventType.GAME_STATE_CHANGED:
                self._handle_game_state_change_event(event)
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type.value}: {e}")

    def _handle_location_changed_event(self, event: Event) -> None:
        """Handle location change event"""
        old_map = event.data.get('old_map', 0)
        new_map = event.data.get('new_map', 0)

        self.logger.info(f"Location changed from {old_map} to {new_map}")

        # Update exploration tracking
        self.visited_locations.add(new_map)
        self.location_visit_counts[new_map] += 1

        # Update discovered connections
        if old_map != 0:
            self.discovered_connections[old_map].add(new_map)
            self.discovered_connections[new_map].add(old_map)

        # Publish exploration progress event
        self._publish_exploration_progress_event(new_map, old_map)

    def _handle_new_area_discovered_event(self, event: Event) -> None:
        """Handle new area discovery event"""
        area_id = event.data.get('area_id', 0)
        area_name = event.data.get('area_name', 'Unknown')

        self.logger.info(f"New area discovered: {area_name} (ID: {area_id})")

        # Add to discovery log
        discovery_record = {
            'type': 'area',
            'timestamp': event.timestamp,
            'area_id': area_id,
            'area_name': area_name
        }
        self.discovery_log.append(discovery_record)

        # Update area priorities
        self.area_priorities[area_id] = 0.9  # High priority for new areas

    def _handle_item_discovered_event(self, event: Event) -> None:
        """Handle item discovery event"""
        item_name = event.data.get('item_name', 'Unknown Item')
        location = event.data.get('location', 'Unknown')

        self.logger.info(f"Item discovered: {item_name} at {location}")

        # Add to discovered items
        self.discovered_items.append({
            'name': item_name,
            'location': location,
            'timestamp': event.timestamp
        })

    def _handle_npc_encountered_event(self, event: Event) -> None:
        """Handle NPC encounter event"""
        npc_type = event.data.get('npc_type', 'Unknown')
        location = event.data.get('location', 'Unknown')

        self.logger.info(f"NPC encountered: {npc_type} at {location}")

        # Add to discovered NPCs
        self.discovered_npcs.append({
            'type': npc_type,
            'location': location,
            'timestamp': event.timestamp
        })

    def _handle_game_state_change_event(self, event: Event) -> None:
        """Handle general game state changes"""
        changes = event.data.get('changes', {})

        # Track exploration-relevant changes
        if 'location_changed' in changes:
            location_data = changes['location_changed']
            # Additional location change processing
            pass

    def _publish_exploration_progress_event(self, new_map: int, old_map: int) -> None:
        """Publish exploration progress update event"""
        import time

        event = Event(
            event_type=EventType.AGENT_ACTION_TAKEN,
            timestamp=time.time(),
            source="explorer_agent",
            data={
                'agent_type': 'explorer',
                'action_type': 'location_change',
                'from_map': old_map,
                'to_map': new_map,
                'total_locations_visited': len(self.visited_locations),
                'exploration_progress': len(self.visited_locations) / 100.0,  # Estimate
                'discoveries_made': len(self.discovery_log)
            },
            priority=5
        )

        self.event_bus.publish(event)