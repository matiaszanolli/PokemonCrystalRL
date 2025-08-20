"""
text_logger.py - Text Transcription Logger for Pokemon Crystal

This module logs all detected text from the game for analysis,
debugging, and creating a transcript of gameplay sessions.
"""

import json
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

try:
    from vision.vision_processor import DetectedText, VisualContext
except ImportError:
    # Create stub classes if vision processor is not available
    class DetectedText:
        def __init__(self, text, confidence, bbox, location):
            self.text = text
            self.confidence = confidence
            self.bbox = bbox
            self.location = location
    
    class VisualContext:
        def __init__(self, screen_type, detected_text, ui_elements, dominant_colors, game_phase, visual_summary):
            self.screen_type = screen_type
            self.detected_text = detected_text or []
            self.ui_elements = ui_elements or []
            self.dominant_colors = dominant_colors or []
            self.game_phase = game_phase
            self.visual_summary = visual_summary


@dataclass
class TextLogEntry:
    """Single text detection log entry"""
    timestamp: str
    frame_number: int
    screen_type: str
    text_content: str
    text_location: str  # 'dialogue', 'menu', 'ui', 'world'
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    session_id: str
    text_hash: str  # For deduplication


@dataclass
class GameplaySession:
    """Represents a single gameplay session"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_frames: int
    total_text_detections: int
    unique_text_count: int
    session_summary: str


class PokemonTextLogger:
    """
    Logs and manages all text transcriptions from Pokemon Crystal gameplay
    """
    
    def __init__(self, log_dir: str = "text_logs"):
        """Initialize the text logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Database for structured storage
        self.db_path = self.log_dir / "text_transcriptions.db"
        self.session_id = self._generate_session_id()
        self.frame_number = 0
        
        # In-memory caches
        self.recent_text = []  # Last 10 text detections
        self.text_frequency = {}  # Count of each unique text
        self.dialogue_history = []  # Chronological dialogue
        
        # Initialize database
        self._init_database()
        
        # Start new session
        self._start_session()
        
        print(f"📝 Text logger initialized - Session: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def _init_database(self):
        """Initialize SQLite database for text logging"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Text entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    screen_type TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    text_location TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    text_hash TEXT NOT NULL,
                    UNIQUE(session_id, frame_number, text_hash)
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_frames INTEGER DEFAULT 0,
                    total_text_detections INTEGER DEFAULT 0,
                    unique_text_count INTEGER DEFAULT 0,
                    session_summary TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_frame ON text_entries(session_id, frame_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_location ON text_entries(text_location)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screen_type ON text_entries(screen_type)")
            
            conn.commit()
    
    def _start_session(self):
        """Start a new logging session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (session_id, start_time)
                VALUES (?, ?)
            """, (self.session_id, datetime.now().isoformat()))
            conn.commit()
    
    def log_visual_context(self, context: VisualContext):
        """Log all text from a visual context"""
        self.frame_number += 1
        
        for detected_text in context.detected_text:
            self._log_text_entry(
                text_content=detected_text.text,
                text_location=detected_text.location,
                confidence=detected_text.confidence,
                bbox=detected_text.bbox,
                screen_type=context.screen_type
            )
    
    def _log_text_entry(self, text_content: str, text_location: str, 
                       confidence: float, bbox: tuple, screen_type: str):
        """Log a single text detection entry"""
        
        # Generate hash for deduplication
        text_hash = hashlib.md5(
            f"{text_content}_{text_location}_{screen_type}".encode()
        ).hexdigest()[:12]
        
        # Create log entry
        entry = TextLogEntry(
            timestamp=datetime.now().isoformat(),
            frame_number=self.frame_number,
            screen_type=screen_type,
            text_content=text_content,
            text_location=text_location,
            confidence=confidence,
            bbox=bbox,
            session_id=self.session_id,
            text_hash=text_hash
        )
        
        # Store in database
        self._store_entry(entry)
        
        # Update in-memory caches
        self._update_caches(entry)
    
    def _store_entry(self, entry: TextLogEntry):
        """Store entry in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO text_entries 
                    (timestamp, frame_number, session_id, screen_type, text_content,
                     text_location, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, text_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.timestamp, entry.frame_number, entry.session_id,
                    entry.screen_type, entry.text_content, entry.text_location,
                    entry.confidence, entry.bbox[0], entry.bbox[1], 
                    entry.bbox[2], entry.bbox[3], entry.text_hash
                ))
                conn.commit()
        except sqlite3.Error as e:
            print(f"⚠️ Database error logging text: {e}")
    
    def _update_caches(self, entry: TextLogEntry):
        """Update in-memory caches"""
        # Recent text (keep last 10)
        self.recent_text.append(entry)
        if len(self.recent_text) > 10:
            self.recent_text.pop(0)
        
        # Text frequency
        self.text_frequency[entry.text_content] = self.text_frequency.get(entry.text_content, 0) + 1
        
        # Dialogue history (only dialogue location)
        if entry.text_location == 'dialogue':
            self.dialogue_history.append({
                'timestamp': entry.timestamp,
                'frame': entry.frame_number,
                'text': entry.text_content,
                'screen_type': entry.screen_type
            })
    
    def get_recent_text(self, count: int = 5) -> List[Dict]:
        """Get most recent text detections"""
        return [asdict(entry) for entry in self.recent_text[-count:]]
    
    def get_dialogue_history(self, count: int = 10) -> List[Dict]:
        """Get recent dialogue entries"""
        return self.dialogue_history[-count:]
    
    def get_text_frequency(self, min_count: int = 2) -> Dict[str, int]:
        """Get frequently appearing text"""
        return {text: count for text, count in self.text_frequency.items() 
                if count >= min_count}
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute("""
                SELECT COUNT(*) FROM text_entries WHERE session_id = ?
            """, (self.session_id,))
            total_detections = cursor.fetchone()[0]
            
            # Unique text count
            cursor.execute("""
                SELECT COUNT(DISTINCT text_hash) FROM text_entries WHERE session_id = ?
            """, (self.session_id,))
            unique_text = cursor.fetchone()[0]
            
            # By location
            cursor.execute("""
                SELECT text_location, COUNT(*) FROM text_entries 
                WHERE session_id = ? GROUP BY text_location
            """, (self.session_id,))
            by_location = dict(cursor.fetchall())
            
            # By screen type
            cursor.execute("""
                SELECT screen_type, COUNT(*) FROM text_entries 
                WHERE session_id = ? GROUP BY screen_type
            """, (self.session_id,))
            by_screen_type = dict(cursor.fetchall())
        
        return {
            'session_id': self.session_id,
            'total_frames': self.frame_number,
            'total_detections': total_detections,
            'unique_text': unique_text,
            'by_location': by_location,
            'by_screen_type': by_screen_type,
            'dialogue_count': len(self.dialogue_history)
        }
    
    def export_session_transcript(self, output_file: Optional[str] = None) -> str:
        """Export session as readable transcript"""
        if output_file is None:
            output_file = self.log_dir / f"{self.session_id}_transcript.txt"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, frame_number, screen_type, text_location, text_content
                FROM text_entries 
                WHERE session_id = ?
                ORDER BY frame_number
            """, (self.session_id,))
            
            entries = cursor.fetchall()
        
        # Generate readable transcript
        transcript_lines = [
            f"Pokemon Crystal Gameplay Transcript",
            f"Session: {self.session_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Frames: {self.frame_number}",
            f"Total Text Detections: {len(entries)}",
            "=" * 60,
            ""
        ]
        
        current_screen = None
        current_frame = None
        
        for timestamp, frame, screen_type, location, text in entries:
            # Add frame/screen separator
            if frame != current_frame:
                if current_frame is not None:
                    transcript_lines.append("")
                transcript_lines.append(f"Frame {frame} ({screen_type}):")
                current_frame = frame
                current_screen = screen_type
            
            # Format text by location
            location_prefix = {
                'dialogue': '💬',
                'menu': '📋',
                'ui': '🔧',
                'world': '🌍'
            }.get(location, '📝')
            
            transcript_lines.append(f"  {location_prefix} {location.upper()}: {text}")
        
        # Write to file
        transcript_content = '\n'.join(transcript_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        print(f"📄 Transcript exported: {output_file}")
        return str(output_file)
    
    def search_text(self, query: str, location: Optional[str] = None) -> List[Dict]:
        """Search for specific text content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            sql = """
                SELECT timestamp, frame_number, screen_type, text_location, text_content
                FROM text_entries 
                WHERE session_id = ? AND text_content LIKE ?
            """
            params = [self.session_id, f"%{query}%"]
            
            if location:
                sql += " AND text_location = ?"
                params.append(location)
            
            sql += " ORDER BY frame_number"
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
        
        return [
            {
                'timestamp': row[0],
                'frame': row[1],
                'screen_type': row[2],
                'location': row[3],
                'text': row[4]
            }
            for row in results
        ]
    
    def close_session(self):
        """Close current session and update statistics"""
        stats = self.get_session_stats()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET end_time = ?, total_frames = ?, total_text_detections = ?, 
                    unique_text_count = ?, session_summary = ?
                WHERE session_id = ?
            """, (
                datetime.now().isoformat(),
                stats['total_frames'],
                stats['total_detections'],
                stats['unique_text'],
                f"Frames: {stats['total_frames']}, Detections: {stats['total_detections']}, Dialogue: {stats['dialogue_count']}",
                self.session_id
            ))
            conn.commit()
        
        # Export final transcript
        transcript_file = self.export_session_transcript()
        
        print(f"📝 Session {self.session_id} closed")
        print(f"📊 Final stats: {stats}")
        return transcript_file


def test_text_logger():
    """Test the text logger with sample data"""
    from vision.vision_processor import DetectedText, VisualContext
    
    # Create logger
    logger = PokemonTextLogger("test_text_logs")
    
    # Create sample visual context
    sample_texts = [
        DetectedText("DIALOGUE", 0.8, (10, 100, 150, 130), "dialogue"),
        DetectedText("MENU", 0.7, (120, 20, 160, 40), "menu"),
        DetectedText("HP", 0.9, (5, 5, 25, 15), "ui")
    ]
    
    context = VisualContext(
        screen_type="dialogue",
        detected_text=sample_texts,
        ui_elements=[],
        dominant_colors=[(255, 255, 255)],
        game_phase="dialogue_interaction",
        visual_summary="Test context"
    )
    
    # Log several frames
    for i in range(5):
        logger.log_visual_context(context)
        time.sleep(0.1)  # Simulate frame delay
    
    # Print stats
    stats = logger.get_session_stats()
    print("📊 Test session stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export transcript
    transcript_file = logger.close_session()
    print(f"📄 Test transcript: {transcript_file}")


if __name__ == "__main__":
    test_text_logger()
