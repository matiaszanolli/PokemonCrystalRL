#!/usr/bin/env python3
"""
Experience Memory System for Pokemon Crystal AI

This system allows the AI to remember successful action sequences and patterns,
building up knowledge over multiple training sessions.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

@dataclass
class ExperienceEntry:
    """Single experience entry"""
    situation_hash: str          # Hash of game state + screen state
    action_sequence: List[str]   # Actions taken
    outcome_reward: float        # Total reward achieved
    success_rate: float         # Success rate when this pattern was used
    usage_count: int            # How many times this pattern was tried
    last_used: float           # Timestamp of last usage
    context: Dict              # Additional context (game phase, location, etc.)

@dataclass
class ActionPattern:
    """Pattern of actions that led to success"""
    pattern: List[str]         # Sequence of actions
    success_situations: List[str]  # Situation hashes where this worked
    average_reward: float      # Average reward when this pattern succeeded
    confidence: float          # How confident we are in this pattern (0-1)

class ExperienceMemory:
    """Manages AI experience and learning across sessions"""
    
    def __init__(self, memory_file: str = "logs/experience_memory.json"):
        self.memory_file = memory_file
        self.experiences: Dict[str, ExperienceEntry] = {}
        self.successful_patterns: List[ActionPattern] = []
        self.situation_to_actions: Dict[str, List[str]] = defaultdict(list)
        
        # Learning parameters
        self.min_success_rate = 0.6  # Minimum success rate to recommend a pattern
        self.max_memory_entries = 10000  # Prevent memory from growing too large
        self.pattern_length = 5  # Length of action patterns to track
        
        self.load_memory()
    
    def get_situation_hash(self, game_state: Dict, screen_analysis: Dict, context: Dict = None) -> str:
        """Create a hash representing the current situation"""
        # Create a simplified representation of the current situation
        key_elements = {
            'phase': context.get('phase', 'unknown') if context else 'unknown',
            'location_type': context.get('location_type', 'unknown') if context else 'unknown',
            'screen_state': screen_analysis.get('state', 'unknown'),
            'party_count': game_state.get('party_count', 0),
            'player_level_range': game_state.get('player_level', 0) // 10,  # Group by 10s
            'badges_total': game_state.get('badges_total', 0),
            'in_battle': game_state.get('in_battle', 0),
            # Add more relevant state elements as needed
        }
        
        # Create a stable hash
        return str(sorted(key_elements.items())).__hash__().__str__()
    
    def record_experience(self, situation_hash: str, actions: List[str], 
                         reward: float, context: Dict = None):
        """Record a new experience or update existing one"""
        
        if situation_hash in self.experiences:
            # Update existing experience
            exp = self.experiences[situation_hash]
            exp.usage_count += 1
            
            # Update success rate based on reward
            success = 1.0 if reward > 0.1 else 0.0
            exp.success_rate = (exp.success_rate * (exp.usage_count - 1) + success) / exp.usage_count
            exp.outcome_reward = (exp.outcome_reward * (exp.usage_count - 1) + reward) / exp.usage_count
            exp.last_used = time.time()
            exp.action_sequence = actions  # Update with latest successful sequence
        else:
            # Create new experience
            success = 1.0 if reward > 0.1 else 0.0
            self.experiences[situation_hash] = ExperienceEntry(
                situation_hash=situation_hash,
                action_sequence=actions,
                outcome_reward=reward,
                success_rate=success,
                usage_count=1,
                last_used=time.time(),
                context=context or {}
            )
        
        # Update pattern tracking
        self._update_patterns(situation_hash, actions, reward > 0.1)
        
        # Cleanup old memories if needed
        self._cleanup_memory()
    
    def get_recommended_actions(self, situation_hash: str, 
                               current_context: Dict = None) -> Optional[List[str]]:
        """Get recommended actions for current situation"""
        
        # Direct experience lookup
        if situation_hash in self.experiences:
            exp = self.experiences[situation_hash]
            if exp.success_rate >= self.min_success_rate and exp.usage_count >= 2:
                return exp.action_sequence
        
        # Look for similar situations
        similar_actions = self._find_similar_situation_actions(situation_hash)
        if similar_actions:
            return similar_actions
        
        # Look for successful patterns
        pattern_actions = self._find_pattern_match(current_context)
        if pattern_actions:
            return pattern_actions
        
        return None
    
    def _update_patterns(self, situation_hash: str, actions: List[str], success: bool):
        """Update action patterns based on new experience"""
        if len(actions) >= self.pattern_length:
            pattern = actions[-self.pattern_length:]  # Last N actions
            
            # Find or create pattern
            existing_pattern = None
            for p in self.successful_patterns:
                if p.pattern == pattern:
                    existing_pattern = p
                    break
            
            if existing_pattern:
                if success:
                    existing_pattern.success_situations.append(situation_hash)
                    # Recalculate confidence based on success rate
                    total_uses = len(existing_pattern.success_situations)
                    existing_pattern.confidence = min(1.0, total_uses / 10.0)  # Max confidence at 10 uses
            elif success:
                # Create new successful pattern
                self.successful_patterns.append(ActionPattern(
                    pattern=pattern,
                    success_situations=[situation_hash],
                    average_reward=0.0,  # Will be calculated later
                    confidence=0.1
                ))
        
        # Keep only top patterns
        self.successful_patterns = sorted(
            self.successful_patterns, 
            key=lambda x: x.confidence, 
            reverse=True
        )[:100]
    
    def _find_similar_situation_actions(self, target_hash: str) -> Optional[List[str]]:
        """Find actions that worked in similar situations"""
        # Simple similarity: look for exact matches first
        # Could be enhanced with more sophisticated similarity metrics
        
        best_exp = None
        best_score = 0
        
        for hash_key, exp in self.experiences.items():
            if exp.success_rate >= self.min_success_rate and exp.usage_count >= 2:
                # Simple scoring based on success rate and usage
                score = exp.success_rate * min(exp.usage_count / 10.0, 1.0)
                if score > best_score:
                    best_score = score
                    best_exp = exp
        
        return best_exp.action_sequence if best_exp else None
    
    def _find_pattern_match(self, context: Dict = None) -> Optional[List[str]]:
        """Find a successful pattern that might apply"""
        if not self.successful_patterns:
            return None
        
        # Return the most confident pattern
        best_pattern = max(self.successful_patterns, key=lambda x: x.confidence)
        if best_pattern.confidence >= 0.3:
            return best_pattern.pattern
        
        return None
    
    def _cleanup_memory(self):
        """Remove old/unused memories to keep size manageable"""
        if len(self.experiences) <= self.max_memory_entries:
            return
        
        # Sort by combination of success rate and recency
        experiences_list = list(self.experiences.items())
        experiences_list.sort(key=lambda x: (
            x[1].success_rate * 0.7 + 
            (time.time() - x[1].last_used) / (86400 * 30) * -0.3  # Favor recent
        ), reverse=True)
        
        # Keep top experiences
        keep_experiences = dict(experiences_list[:self.max_memory_entries])
        self.experiences = keep_experiences
    
    def save_memory(self):
        """Save experience memory to disk"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # Convert to serializable format
        save_data = {
            'experiences': {k: asdict(v) for k, v in self.experiences.items()},
            'successful_patterns': [asdict(p) for p in self.successful_patterns],
            'metadata': {
                'save_time': time.time(),
                'total_experiences': len(self.experiences),
                'total_patterns': len(self.successful_patterns)
            }
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_memory(self):
        """Load experience memory from disk"""
        if not os.path.exists(self.memory_file):
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
            
            # Load experiences
            self.experiences = {}
            for k, v in data.get('experiences', {}).items():
                self.experiences[k] = ExperienceEntry(**v)
            
            # Load patterns
            self.successful_patterns = []
            for p in data.get('successful_patterns', []):
                self.successful_patterns.append(ActionPattern(**p))
            
            print(f"📚 Loaded {len(self.experiences)} experiences and {len(self.successful_patterns)} patterns")
            
        except Exception as e:
            print(f"⚠️ Failed to load experience memory: {e}")
            self.experiences = {}
            self.successful_patterns = []
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory"""
        if not self.experiences:
            return {"total_experiences": 0, "patterns": 0}
        
        avg_success_rate = sum(exp.success_rate for exp in self.experiences.values()) / len(self.experiences)
        successful_experiences = sum(1 for exp in self.experiences.values() if exp.success_rate > 0.5)
        
        return {
            "total_experiences": len(self.experiences),
            "successful_experiences": successful_experiences,
            "average_success_rate": avg_success_rate,
            "total_patterns": len(self.successful_patterns),
            "high_confidence_patterns": sum(1 for p in self.successful_patterns if p.confidence > 0.5)
        }
