"""
Experience Memory System - Learning from past experiences for Pokemon Crystal RL

This system stores, analyzes, and retrieves past experiences to improve decision
making over time. It learns from both successes and failures to make the AI
progressively smarter.
"""

import time
import json
import logging
import sqlite3
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Represents a single experience/decision point."""
    timestamp: float
    game_state: Dict[str, Any]
    context: Dict[str, Any]
    action_taken: int
    outcome_reward: float
    success_level: float  # 0.0-1.0, how successful this decision was
    situation_type: str   # battle, exploration, dialogue, etc.
    consequences: Dict[str, Any]  # What happened after this action


@dataclass
class SituationPattern:
    """Represents a learned pattern for a specific situation."""
    situation_id: str
    successful_actions: Counter  # Actions that worked well
    failed_actions: Counter      # Actions that failed
    context_factors: Dict[str, Any]  # Key context factors
    confidence: float            # How confident we are in this pattern
    sample_count: int           # Number of experiences this is based on


class ExperienceMemorySystem:
    """Memory system for storing and learning from past experiences."""

    def __init__(self, memory_file: str = "data/experience_memory.db"):
        self.logger = logging.getLogger("ExperienceMemorySystem")

        # Database setup
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(exist_ok=True)
        self._init_database()

        # In-memory caches for fast access
        self.recent_experiences = []  # Last 1000 experiences
        self.situation_patterns = {}  # Learned patterns by situation type
        self.action_success_rates = defaultdict(list)  # Success rates by action

        # Configuration
        self.max_memory_size = 10000  # Maximum experiences to keep
        self.pattern_confidence_threshold = 0.6  # Minimum confidence for recommendations
        self.experience_relevance_window = 24 * 3600  # 24 hours for relevance

        # Load existing patterns
        self._load_existing_patterns()

        self.logger.info(f"Experience memory system initialized with {len(self.recent_experiences)} experiences")

    def _init_database(self):
        """Initialize SQLite database for persistent memory storage."""
        self.conn = sqlite3.connect(str(self.memory_file), check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            situation_hash TEXT,
            game_state TEXT,
            context TEXT,
            action_taken INTEGER,
            outcome_reward REAL,
            success_level REAL,
            situation_type TEXT,
            consequences TEXT
        )''')

        self.conn.execute('''CREATE TABLE IF NOT EXISTS situation_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            situation_id TEXT UNIQUE,
            pattern_data TEXT,
            confidence REAL,
            sample_count INTEGER,
            last_updated REAL
        )''')

        self.conn.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp)''')
        self.conn.execute('''CREATE INDEX IF NOT EXISTS idx_situation_type ON experiences(situation_type)''')
        self.conn.execute('''CREATE INDEX IF NOT EXISTS idx_situation_hash ON experiences(situation_hash)''')

        self.conn.commit()

    def _load_existing_patterns(self):
        """Load existing patterns from database."""
        try:
            cursor = self.conn.execute('''
                SELECT situation_id, pattern_data, confidence, sample_count
                FROM situation_patterns
                WHERE confidence >= ?
            ''', (self.pattern_confidence_threshold,))

            for row in cursor.fetchall():
                situation_id, pattern_data, confidence, sample_count = row
                try:
                    pattern_dict = json.loads(pattern_data)
                    pattern = SituationPattern(
                        situation_id=situation_id,
                        successful_actions=Counter(pattern_dict.get('successful_actions', {})),
                        failed_actions=Counter(pattern_dict.get('failed_actions', {})),
                        context_factors=pattern_dict.get('context_factors', {}),
                        confidence=confidence,
                        sample_count=sample_count
                    )
                    self.situation_patterns[situation_id] = pattern
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to load pattern {situation_id}: {e}")

            # Load recent experiences for in-memory cache
            cursor = self.conn.execute('''
                SELECT timestamp, game_state, context, action_taken, outcome_reward,
                       success_level, situation_type, consequences
                FROM experiences
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')

            for row in cursor.fetchall():
                try:
                    timestamp, game_state, context, action_taken, outcome_reward, success_level, situation_type, consequences = row
                    experience = Experience(
                        timestamp=timestamp,
                        game_state=json.loads(game_state),
                        context=json.loads(context),
                        action_taken=action_taken,
                        outcome_reward=outcome_reward,
                        success_level=success_level,
                        situation_type=situation_type,
                        consequences=json.loads(consequences)
                    )
                    self.recent_experiences.append(experience)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to load experience: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load existing patterns: {e}")

    def record_experience(self,
                         game_state: Dict[str, Any],
                         context: Dict[str, Any],
                         action_taken: int,
                         outcome_reward: float,
                         situation_type: str = None,
                         consequences: Dict[str, Any] = None) -> None:
        """Record a new experience in memory.

        Args:
            game_state: Game state when decision was made
            context: Context information available
            action_taken: Action that was taken (1-8)
            outcome_reward: Reward received from this action
            situation_type: Type of situation (battle, exploration, etc.)
            consequences: What happened as a result
        """
        try:
            timestamp = time.time()

            # Determine situation type if not provided
            if situation_type is None:
                situation_type = self._classify_situation(game_state, context)

            # Calculate success level based on reward and context
            success_level = self._calculate_success_level(outcome_reward, context)

            # Create experience
            experience = Experience(
                timestamp=timestamp,
                game_state=game_state.copy(),
                context=context.copy(),
                action_taken=action_taken,
                outcome_reward=outcome_reward,
                success_level=success_level,
                situation_type=situation_type,
                consequences=consequences or {}
            )

            # Add to in-memory cache
            self.recent_experiences.append(experience)
            if len(self.recent_experiences) > 1000:
                self.recent_experiences.pop(0)

            # Store in database
            self._store_experience_db(experience)

            # Update patterns periodically
            if len(self.recent_experiences) % 50 == 0:
                self._update_situation_patterns()

        except Exception as e:
            self.logger.error(f"Failed to record experience: {e}")

    def _classify_situation(self, game_state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Classify the type of situation based on game state and context."""
        if game_state.get('in_battle', False):
            return 'battle'
        elif context.get('detected_state') == 'dialogue':
            return 'dialogue'
        elif context.get('detected_state') == 'menu':
            return 'menu'
        elif context.get('stuck_analysis', {}).get('is_stuck', False):
            return 'stuck_recovery'
        elif game_state.get('party_count', 0) == 0:
            return 'early_game'
        else:
            return 'exploration'

    def _calculate_success_level(self, reward: float, context: Dict[str, Any]) -> float:
        """Calculate how successful an action was (0.0-1.0)."""
        # Base success on reward
        success = 0.5  # Neutral baseline

        if reward > 0:
            success += min(0.4, reward / 100.0)  # Positive reward increases success
        elif reward < 0:
            success -= min(0.4, abs(reward) / 100.0)  # Negative reward decreases success

        # Context-based adjustments
        if context.get('progress', {}).get('recent_changes'):
            success += 0.1  # Progress is good

        if context.get('stuck_analysis', {}).get('is_stuck', False):
            success -= 0.2  # Being stuck is bad

        return max(0.0, min(1.0, success))

    def _store_experience_db(self, experience: Experience):
        """Store experience in database."""
        try:
            situation_hash = self._generate_situation_hash(experience.game_state, experience.context)

            self.conn.execute('''
                INSERT INTO experiences
                (timestamp, situation_hash, game_state, context, action_taken,
                 outcome_reward, success_level, situation_type, consequences)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience.timestamp,
                situation_hash,
                json.dumps(experience.game_state),
                json.dumps(experience.context),
                experience.action_taken,
                experience.outcome_reward,
                experience.success_level,
                experience.situation_type,
                json.dumps(experience.consequences)
            ))
            self.conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store experience in database: {e}")

    def _generate_situation_hash(self, game_state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a hash representing similar situations."""
        # Use key features that define similar situations
        key_features = {
            'map': game_state.get('player_map', 0),
            'in_battle': game_state.get('in_battle', False),
            'party_count': game_state.get('party_count', 0),
            'badges': game_state.get('badges', 0),
            'detected_state': context.get('detected_state', 'unknown'),
            'situation_type': context.get('current_objective', {}).get('primary', 'explore')
        }

        # Create hash from key features
        feature_str = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]

    def get_recommendation(self,
                          game_state: Dict[str, Any],
                          context: Dict[str, Any],
                          available_actions: List[int] = None) -> Dict[str, Any]:
        """Get action recommendations based on past experiences.

        Args:
            game_state: Current game state
            context: Current context
            available_actions: Available actions to choose from

        Returns:
            Dict with recommendations and confidence
        """
        try:
            if available_actions is None:
                available_actions = list(range(1, 9))

            # Classify current situation
            situation_type = self._classify_situation(game_state, context)

            # Get similar experiences
            similar_experiences = self._find_similar_experiences(game_state, context, situation_type)

            # Get pattern-based recommendations
            pattern_recommendations = self._get_pattern_recommendations(situation_type, available_actions)

            # Get experience-based recommendations
            experience_recommendations = self._analyze_similar_experiences(similar_experiences, available_actions)

            # Combine recommendations
            combined_recommendations = self._combine_recommendations(
                pattern_recommendations,
                experience_recommendations,
                available_actions
            )

            return {
                'recommended_actions': combined_recommendations['actions'],
                'confidence': combined_recommendations['confidence'],
                'reasoning': combined_recommendations['reasoning'],
                'similar_experiences_count': len(similar_experiences),
                'pattern_match': pattern_recommendations.get('pattern_id'),
                'experience_source': 'memory_system'
            }

        except Exception as e:
            self.logger.error(f"Failed to get recommendation: {e}")
            return self._get_fallback_recommendation(available_actions)

    def _find_similar_experiences(self,
                                 game_state: Dict[str, Any],
                                 context: Dict[str, Any],
                                 situation_type: str,
                                 limit: int = 20) -> List[Experience]:
        """Find similar past experiences."""
        situation_hash = self._generate_situation_hash(game_state, context)

        # First try exact situation hash matches
        similar = []
        for exp in reversed(self.recent_experiences):
            if exp.situation_type == situation_type:
                exp_hash = self._generate_situation_hash(exp.game_state, exp.context)
                if exp_hash == situation_hash:
                    similar.append(exp)
                    if len(similar) >= limit // 2:
                        break

        # If not enough exact matches, find similar situations
        if len(similar) < limit // 2:
            for exp in reversed(self.recent_experiences):
                if exp.situation_type == situation_type and exp not in similar:
                    similarity = self._calculate_similarity(game_state, context, exp.game_state, exp.context)
                    if similarity > 0.7:  # 70% similarity threshold
                        similar.append(exp)
                        if len(similar) >= limit:
                            break

        return similar

    def _calculate_similarity(self,
                            state1: Dict[str, Any], context1: Dict[str, Any],
                            state2: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two situations (0.0-1.0)."""
        similarity_factors = []

        # Map similarity
        if state1.get('player_map') == state2.get('player_map'):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Battle state similarity
        if state1.get('in_battle') == state2.get('in_battle'):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Progress similarity (badges, party count)
        badge_diff = abs(state1.get('badges', 0) - state2.get('badges', 0))
        party_diff = abs(state1.get('party_count', 0) - state2.get('party_count', 0))

        similarity_factors.append(max(0.0, 1.0 - badge_diff / 8.0))  # 8 badges max
        similarity_factors.append(max(0.0, 1.0 - party_diff / 6.0))  # 6 party max

        # Context similarity
        if context1.get('detected_state') == context2.get('detected_state'):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.5)

        return sum(similarity_factors) / len(similarity_factors)

    def _get_pattern_recommendations(self, situation_type: str, available_actions: List[int]) -> Dict[str, Any]:
        """Get recommendations from learned patterns."""
        if situation_type not in self.situation_patterns:
            return {'actions': [], 'confidence': 0.0, 'pattern_id': None}

        pattern = self.situation_patterns[situation_type]

        # Score actions based on pattern
        action_scores = {}
        for action in available_actions:
            success_count = pattern.successful_actions.get(action, 0)
            failure_count = pattern.failed_actions.get(action, 0)
            total_count = success_count + failure_count

            if total_count > 0:
                success_rate = success_count / total_count
                # Weight by sample size and pattern confidence
                score = success_rate * pattern.confidence * min(1.0, total_count / 10.0)
                action_scores[action] = score

        # Sort actions by score
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_actions = [action for action, score in sorted_actions if score > 0.3]

        avg_confidence = sum(score for _, score in sorted_actions) / len(sorted_actions) if sorted_actions else 0.0

        return {
            'actions': recommended_actions[:3],  # Top 3 recommendations
            'confidence': avg_confidence,
            'pattern_id': situation_type,
            'reasoning': f"Based on {pattern.sample_count} similar experiences"
        }

    def _analyze_similar_experiences(self, similar_experiences: List[Experience], available_actions: List[int]) -> Dict[str, Any]:
        """Analyze similar experiences to get recommendations."""
        if not similar_experiences:
            return {'actions': [], 'confidence': 0.0, 'reasoning': 'No similar experiences'}

        # Group by action and calculate success rates
        action_outcomes = defaultdict(list)
        for exp in similar_experiences:
            if exp.action_taken in available_actions:
                action_outcomes[exp.action_taken].append(exp.success_level)

        # Calculate success rates and confidence
        action_scores = {}
        for action, outcomes in action_outcomes.items():
            avg_success = sum(outcomes) / len(outcomes)
            confidence = min(1.0, len(outcomes) / 5.0)  # More samples = higher confidence
            action_scores[action] = avg_success * confidence

        # Sort and recommend
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_actions = [action for action, score in sorted_actions if score > 0.4]

        avg_confidence = sum(score for _, score in sorted_actions) / len(sorted_actions) if sorted_actions else 0.0

        return {
            'actions': recommended_actions[:3],
            'confidence': avg_confidence,
            'reasoning': f"Based on {len(similar_experiences)} similar experiences"
        }

    def _combine_recommendations(self,
                               pattern_rec: Dict[str, Any],
                               experience_rec: Dict[str, Any],
                               available_actions: List[int]) -> Dict[str, Any]:
        """Combine pattern and experience recommendations."""
        # Weighted combination based on confidence
        pattern_weight = pattern_rec.get('confidence', 0.0)
        experience_weight = experience_rec.get('confidence', 0.0)
        total_weight = pattern_weight + experience_weight

        if total_weight == 0:
            return self._get_fallback_recommendation(available_actions)

        # Combine action lists with preference for higher confidence source
        combined_actions = []

        if pattern_weight >= experience_weight:
            # Prefer pattern recommendations
            combined_actions.extend(pattern_rec.get('actions', []))
            for action in experience_rec.get('actions', []):
                if action not in combined_actions:
                    combined_actions.append(action)
        else:
            # Prefer experience recommendations
            combined_actions.extend(experience_rec.get('actions', []))
            for action in pattern_rec.get('actions', []):
                if action not in combined_actions:
                    combined_actions.append(action)

        # Limit to top 3 actions
        combined_actions = combined_actions[:3]

        # Combined confidence
        combined_confidence = (pattern_weight + experience_weight) / 2.0

        # Combined reasoning
        reasoning_parts = []
        if pattern_rec.get('reasoning'):
            reasoning_parts.append(f"Pattern: {pattern_rec['reasoning']}")
        if experience_rec.get('reasoning'):
            reasoning_parts.append(f"Experience: {experience_rec['reasoning']}")

        return {
            'actions': combined_actions,
            'confidence': combined_confidence,
            'reasoning': " | ".join(reasoning_parts)
        }

    def _update_situation_patterns(self):
        """Update learned patterns based on recent experiences."""
        try:
            # Group experiences by situation type
            situation_groups = defaultdict(list)
            for exp in self.recent_experiences[-200:]:  # Last 200 experiences
                situation_groups[exp.situation_type].append(exp)

            # Update patterns for each situation type
            for situation_type, experiences in situation_groups.items():
                if len(experiences) >= 10:  # Need minimum sample size
                    self._update_pattern(situation_type, experiences)

        except Exception as e:
            self.logger.error(f"Failed to update situation patterns: {e}")

    def _update_pattern(self, situation_type: str, experiences: List[Experience]):
        """Update pattern for a specific situation type."""
        successful_actions = Counter()
        failed_actions = Counter()
        context_factors = defaultdict(list)

        # Analyze experiences
        for exp in experiences:
            if exp.success_level >= 0.6:  # Successful
                successful_actions[exp.action_taken] += 1
            elif exp.success_level <= 0.4:  # Failed
                failed_actions[exp.action_taken] += 1

            # Collect context factors
            for key, value in exp.context.items():
                if isinstance(value, (int, float, bool, str)):
                    context_factors[key].append(value)

        # Calculate confidence based on sample size and consistency
        total_samples = len(experiences)
        confidence = min(0.95, total_samples / 100.0)  # Max 95% confidence

        # Create or update pattern
        pattern = SituationPattern(
            situation_id=situation_type,
            successful_actions=successful_actions,
            failed_actions=failed_actions,
            context_factors=dict(context_factors),
            confidence=confidence,
            sample_count=total_samples
        )

        self.situation_patterns[situation_type] = pattern

        # Store in database
        self._store_pattern_db(pattern)

    def _store_pattern_db(self, pattern: SituationPattern):
        """Store pattern in database."""
        try:
            pattern_data = {
                'successful_actions': dict(pattern.successful_actions),
                'failed_actions': dict(pattern.failed_actions),
                'context_factors': pattern.context_factors
            }

            self.conn.execute('''
                INSERT OR REPLACE INTO situation_patterns
                (situation_id, pattern_data, confidence, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                pattern.situation_id,
                json.dumps(pattern_data),
                pattern.confidence,
                pattern.sample_count,
                time.time()
            ))
            self.conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store pattern in database: {e}")

    def _get_fallback_recommendation(self, available_actions: List[int]) -> Dict[str, Any]:
        """Provide fallback recommendation when no patterns are available."""
        return {
            'actions': available_actions[:3] if len(available_actions) >= 3 else available_actions,
            'confidence': 0.2,
            'reasoning': 'No experience patterns available, using fallback',
            'experience_source': 'fallback'
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience memory system."""
        return {
            'total_experiences': len(self.recent_experiences),
            'learned_patterns': len(self.situation_patterns),
            'pattern_confidence_avg': sum(p.confidence for p in self.situation_patterns.values()) / len(self.situation_patterns) if self.situation_patterns else 0.0,
            'situation_types': list(set(exp.situation_type for exp in self.recent_experiences)),
            'memory_file_size': self.memory_file.stat().st_size if self.memory_file.exists() else 0,
            'database_connected': self.conn is not None
        }

    def cleanup_old_experiences(self, max_age_days: int = 30):
        """Clean up old experiences to prevent database from growing too large."""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)

            self.conn.execute('DELETE FROM experiences WHERE timestamp < ?', (cutoff_time,))
            self.conn.execute('VACUUM')  # Reclaim space
            self.conn.commit()

            # Also clean in-memory cache
            self.recent_experiences = [exp for exp in self.recent_experiences if exp.timestamp >= cutoff_time]

            self.logger.info(f"Cleaned up experiences older than {max_age_days} days")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old experiences: {e}")

    def shutdown(self):
        """Shutdown the memory system and close database connection."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            self.logger.info("Experience memory system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")