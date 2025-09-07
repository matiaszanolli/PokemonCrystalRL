"""
Decision Analysis Database Operations

Handles all database persistence for decision records and patterns.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from .models import DecisionRecord, DecisionPattern, PatternType, OutcomeType
from environments.state.analyzer import GamePhase, SituationCriticality

logger = logging.getLogger(__name__)


class DecisionDatabase:
    """Handles database operations for decision analysis"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path("decision_history.db")
        self.logger = logging.getLogger("pokemon_trainer.decision_db")
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Decision history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    game_state TEXT,
                    action_taken INTEGER,
                    llm_reasoning TEXT,
                    reward_received REAL,
                    outcome_type TEXT,
                    game_phase TEXT,
                    criticality TEXT,
                    health_percentage REAL,
                    progression_score REAL,
                    led_to_progress BOOLEAN,
                    was_effective BOOLEAN,
                    time_to_next_decision REAL
                )
            ''')
            
            # Pattern discovery table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    action_sequence TEXT,
                    context_conditions TEXT,
                    times_observed INTEGER,
                    success_rate REAL,
                    average_reward REAL,
                    effectiveness_score REAL,
                    confidence REAL,
                    last_seen DATETIME,
                    examples TEXT,
                    recommended_contexts TEXT,
                    avoid_contexts TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT
                )
            ''')
            
            conn.commit()
    
    def store_decision(self, record: DecisionRecord) -> str:
        """Store decision record in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO decision_history 
                (timestamp, game_state, action_taken, llm_reasoning, reward_received,
                 outcome_type, game_phase, criticality, health_percentage, progression_score,
                 led_to_progress, was_effective, time_to_next_decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp,
                json.dumps(record.game_state),
                record.action_taken,
                record.llm_reasoning,
                record.reward_received,
                record.outcome_type.value,
                record.game_phase.value,
                record.criticality.value,
                record.health_percentage,
                record.progression_score,
                record.led_to_progress,
                record.was_effective,
                record.time_to_next_decision
            ))
            
            return str(cursor.lastrowid)
    
    def load_patterns(self) -> Dict[str, DecisionPattern]:
        """Load existing patterns from database"""
        patterns = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM decision_patterns')
                
                for row in cursor.fetchall():
                    (pattern_id, pattern_type, action_sequence, context_conditions,
                     times_observed, success_rate, average_reward, effectiveness_score,
                     confidence, last_seen, examples, recommended_contexts, avoid_contexts) = row
                    
                    pattern = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType(pattern_type),
                        action_sequence=json.loads(action_sequence),
                        context_conditions=json.loads(context_conditions),
                        times_observed=times_observed,
                        success_rate=success_rate,
                        average_reward=average_reward,
                        effectiveness_score=effectiveness_score,
                        confidence=confidence,
                        last_seen=datetime.fromisoformat(last_seen),
                        examples=json.loads(examples) if examples else [],
                        recommended_contexts=json.loads(recommended_contexts) if recommended_contexts else [],
                        avoid_contexts=json.loads(avoid_contexts) if avoid_contexts else []
                    )
                    
                    patterns[pattern_id] = pattern
                    
        except (sqlite3.Error, json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Could not load patterns from database: {e}")
        
        return patterns
    
    def save_patterns(self, patterns: Dict[str, DecisionPattern]):
        """Save patterns to database"""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for pattern in patterns.values():
                    cursor.execute('''
                        INSERT OR REPLACE INTO decision_patterns VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    json.dumps(pattern.action_sequence),
                    json.dumps(pattern.context_conditions),
                    pattern.times_observed,
                    pattern.success_rate,
                    pattern.average_reward,
                    pattern.effectiveness_score,
                    pattern.confidence,
                    pattern.last_seen.isoformat(),
                    json.dumps(pattern.examples),
                    json.dumps(pattern.recommended_contexts),
                    json.dumps(pattern.avoid_contexts)
                ))
            
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            self.logger.warning(f"Could not save patterns to database: {e}")
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent decisions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM decision_history 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    (id, timestamp, game_state, action_taken, llm_reasoning, reward_received,
                     outcome_type, game_phase, criticality, health_percentage, progression_score,
                     led_to_progress, was_effective, time_to_next_decision) = row
                    
                    results.append({
                        'id': id,
                        'timestamp': timestamp,
                        'game_state': json.loads(game_state) if game_state else {},
                        'action_taken': action_taken,
                        'llm_reasoning': llm_reasoning,
                        'reward_received': reward_received,
                        'outcome_type': outcome_type,
                        'game_phase': game_phase,
                        'criticality': criticality,
                        'health_percentage': health_percentage,
                        'progression_score': progression_score,
                        'led_to_progress': led_to_progress,
                        'was_effective': was_effective,
                        'time_to_next_decision': time_to_next_decision
                    })
                
                return results
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not fetch recent decisions: {e}")
            return []
    
    def close(self):
        """Close database connection (placeholder for future connection pooling)"""
        pass