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
            total_actions=data.get('actions_taken', 0),
            actions_per_second=data.get('actions_per_second', 0.0),
            llm_calls=data.get('llm_decision_count', 0),
            total_reward=data.get('total_reward', 0.0)
        )


class TrainingAPI:
    """Training statistics and control API."""
    
    # Action name mapping
    ACTION_NAMES = {
        0: "RIGHT",
        1: "LEFT",
        2: "UP",
        3: "DOWN",
        4: "A",
        5: "B",
        6: "SELECT",
        7: "START"
    }
    
    def __init__(self, trainer=None):
        """Initialize training API.
        
        Args:
            trainer: The Pokemon trainer instance
        """
        self.trainer = trainer
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        # Get metrics from trainer
        if self.trainer and hasattr(self.trainer, 'stats'):
            metrics = TrainingMetrics.from_dict(self.trainer.stats)
        elif self.trainer and hasattr(self.trainer, 'get_current_stats'):
            stats = self.trainer.get_current_stats()
            metrics = TrainingMetrics.from_dict(stats)
        else:
            metrics = TrainingMetrics()
        
        return {
            'total_actions': metrics.total_actions,
            'actions_per_second': metrics.actions_per_second,
            'llm_calls': metrics.llm_calls,
            'total_reward': metrics.total_reward
        }
    
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
    
    def update_trainer(self, trainer) -> None:
        """Update trainer reference."""
        self.trainer = trainer
