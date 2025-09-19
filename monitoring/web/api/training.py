"""Training API for Pokemon Crystal RL.

This API provides endpoints for training-related data:
- Training statistics and metrics
- LLM decisions and history
- Action tracking and analysis
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingMetrics:
    """Training metrics data."""
    total_actions: int = 0
    actions_per_second: float = 0.0
    llm_calls: int = 0
    total_reward: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create metrics from dictionary."""
        return cls(
            total_actions=data.get('total_actions', 0),
            actions_per_second=data.get('actions_per_second', 0.0),
            llm_calls=data.get('llm_calls', 0),
            total_reward=data.get('total_reward', 0.0)
        )


class TrainingAPI:
    """Training statistics and control API."""
    
    # Action name mapping
    ACTION_NAMES = {
        0: "NONE",       # No action/invalid
        1: "UP",        # D-pad UP
        2: "DOWN",      # D-pad DOWN
        3: "LEFT",      # D-pad LEFT
        4: "RIGHT",     # D-pad RIGHT
        5: "A",         # A button
        6: "B",         # B button
        7: "START",     # START button
        8: "SELECT"     # SELECT button
    }
    
    def __init__(self, trainer=None):
        """Initialize training API.
        
        Args:
            trainer: The Pokemon trainer instance
        """
        self.trainer = trainer
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics including game state and memory debug.

        Returns:
            Dictionary with training metrics, game state, and memory data
        """
        # Get metrics from trainer
        if self.trainer and hasattr(self.trainer, 'stats'):
            metrics = TrainingMetrics.from_dict(self.trainer.stats)
        elif self.trainer and hasattr(self.trainer, 'get_current_stats'):
            stats = self.trainer.get_current_stats()
            metrics = TrainingMetrics.from_dict(stats)
        else:
            metrics = TrainingMetrics()

        # Base training stats
        result = {
            'total_actions': metrics.total_actions,
            'actions_per_second': metrics.actions_per_second,
            'llm_calls': metrics.llm_calls,
            'total_reward': metrics.total_reward
        }

        # Add game state information
        game_state = self._get_game_state()
        if game_state:
            # Transform field names to match dashboard expectations
            result['current_map'] = game_state.get('player_map', 0)
            result['player_position'] = {
                'x': game_state.get('player_x', 0),
                'y': game_state.get('player_y', 0)
            }
            result['badges_earned'] = game_state.get('badges', 0)
            result['money'] = game_state.get('money', 0)

        # Add memory debug information
        memory_data = self._get_memory_debug()
        if memory_data and 'error' not in memory_data:
            result['memory_data'] = memory_data

        return result
    
    def get_llm_decisions(self) -> Dict[str, Any]:
        """Get LLM decisions with enhanced information.
        
        Returns:
            Dictionary with decision data and statistics
        """
        current_time = time.time()
        decisions_data = {
            'recent_decisions': [],
            'total_decisions': 0,
            'decision_rate': 0.0,
            'average_response_time_ms': 0.0,
            'last_decision_age_seconds': None,
            'timestamp': current_time
        }
        
        # Get raw decisions
        all_decisions = self._collect_decisions()
        
        # Process decisions
        enhanced_decisions, total_time, time_count = self._process_decisions(
            all_decisions,
            current_time
        )
        
        # Calculate statistics
        decisions_data['recent_decisions'] = enhanced_decisions[:20]
        decisions_data['total_decisions'] = len(enhanced_decisions)
        
        if enhanced_decisions:
            decisions_data['last_decision_age_seconds'] = enhanced_decisions[0].get('age_seconds')
            decisions_data['decision_rate'] = self._calculate_decision_rate(
                enhanced_decisions,
                current_time
            )
        
        if time_count > 0:
            decisions_data['average_response_time_ms'] = total_time / time_count
        
        return decisions_data
    
    def _collect_decisions(self) -> List[Dict[str, Any]]:
        """Collect decisions from all sources."""
        all_decisions = []
        
        # Primary source: trainer.llm_decisions deque
        if self.trainer and hasattr(self.trainer, 'llm_decisions'):
            all_decisions.extend(list(self.trainer.llm_decisions))
        
        # Secondary source: trainer.stats['recent_llm_decisions']
        if (self.trainer and hasattr(self.trainer, 'stats') and
            'recent_llm_decisions' in self.trainer.stats):
            # Merge unique decisions
            stats_decisions = self.trainer.stats['recent_llm_decisions']
            existing_timestamps = {d.get('timestamp') for d in all_decisions}
            for decision in stats_decisions:
                if decision.get('timestamp') not in existing_timestamps:
                    all_decisions.append(decision)
        
        return all_decisions
    
    def _process_decisions(
        self,
        decisions: List[Dict[str, Any]],
        current_time: float
    ) -> tuple[List[Dict[str, Any]], float, int]:
        """Process and enhance decision data."""
        enhanced_decisions = []
        total_response_time = 0
        response_time_count = 0
        
        for decision in decisions:
            enhanced = decision.copy()
            
            # Add computed fields
            if 'timestamp' in decision:
                enhanced['age_seconds'] = current_time - decision['timestamp']
                enhanced['timestamp_readable'] = time.strftime(
                    '%H:%M:%S',
                    time.localtime(decision['timestamp'])
                )
            
            # Add action name
            if 'action' in decision and 'action_name' not in decision:
                enhanced['action_name'] = self._get_action_name(decision['action'])
            
            # Process response time
            if 'response_time_ms' in decision:
                total_response_time += decision['response_time_ms']
                response_time_count += 1
            elif 'response_time' in decision:
                # Convert seconds to milliseconds
                enhanced['response_time_ms'] = decision['response_time'] * 1000
                total_response_time += enhanced['response_time_ms']
                response_time_count += 1
            
            # Format reasoning
            if 'reasoning' in decision and len(decision['reasoning']) > 200:
                enhanced['reasoning_truncated'] = decision['reasoning'][:200] + "..."
                enhanced['reasoning_full'] = decision['reasoning']
            else:
                enhanced['reasoning_truncated'] = decision.get('reasoning', '')
                enhanced['reasoning_full'] = decision.get('reasoning', '')
            
            enhanced_decisions.append(enhanced)
        
        # Sort by timestamp (most recent first)
        enhanced_decisions.sort(
            key=lambda x: x.get('timestamp', 0),
            reverse=True
        )
        
        return enhanced_decisions, total_response_time, response_time_count
    
    def _calculate_decision_rate(
        self,
        decisions: List[Dict[str, Any]],
        current_time: float
    ) -> float:
        """Calculate decisions per minute over the last hour."""
        # Get decisions from last hour
        recent_decisions = [
            d for d in decisions
            if d.get('timestamp', 0) > current_time - 3600
        ]
        
        if recent_decisions:
            time_span = current_time - min(
                d.get('timestamp', current_time)
                for d in recent_decisions
            )
            if time_span > 0:
                return len(recent_decisions) * 60.0 / time_span
        
        return 0.0
    
    def _get_action_name(self, action) -> str:
        """Convert action number to readable name."""
        return self.ACTION_NAMES.get(action, f"ACTION_{action}")
    
    def _get_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current game state from trainer."""
        if not self.trainer:
            return None

        try:
            # Try unified trainer first
            if hasattr(self.trainer, '_get_game_state'):
                try:
                    return self.trainer._get_game_state()
                except Exception:
                    pass

            # Try getting from stats
            if hasattr(self.trainer, 'get_current_stats'):
                try:
                    stats = self.trainer.get_current_stats()
                    if 'player_map' in stats or 'player_x' in stats:
                        return stats
                except Exception:
                    pass
        except Exception:
            # Gracefully handle any errors (e.g., mock objects in tests)
            pass

        return None

    def _get_memory_debug(self) -> Optional[Dict[str, Any]]:
        """Get memory debug information."""
        if not self.trainer:
            return None

        try:
            # Import memory reader for game state debugging
            try:
                from trainer.memory_reader import PokemonCrystalMemoryReader
            except ImportError:
                return None

            # Initialize memory reader if needed
            if not hasattr(self.trainer, 'memory_reader') or self.trainer.memory_reader is None:
                # Try unified trainer structure first
                pyboy_instance = None
                if hasattr(self.trainer, 'emulation_manager') and self.trainer.emulation_manager:
                    pyboy_instance = self.trainer.emulation_manager.get_instance()
                elif hasattr(self.trainer, 'pyboy') and self.trainer.pyboy is not None:
                    pyboy_instance = self.trainer.pyboy

                if pyboy_instance is not None:
                    self.trainer.memory_reader = PokemonCrystalMemoryReader(pyboy_instance)
                else:
                    return None

            # Get memory state
            if hasattr(self.trainer, 'memory_reader') and self.trainer.memory_reader is not None:
                try:
                    memory_state = self.trainer.memory_reader.read_game_state()
                    # Add debug info
                    memory_state['debug_info'] = self.trainer.memory_reader.get_debug_info()
                    return memory_state
                except Exception:
                    return None
        except Exception:
            # Gracefully handle any errors (e.g., mock objects in tests)
            pass

        return None

    def update_trainer(self, trainer) -> None:
        """Update trainer reference."""
        self.trainer = trainer
