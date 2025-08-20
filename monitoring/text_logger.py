"""
Text logger module for Pokemon Crystal RL.

This module provides structured logging capabilities for game events, training progress,
and system status. It includes features like:
- Categorized logging with custom formatters
- Event history tracking
- Log event filtering and search
- Data bus integration for real-time log streaming
- Log file rotation and management
"""

import logging
import sys
import os
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from logging.handlers import RotatingFileHandler
import time
from enum import Enum, auto
from pathlib import Path
import re
from queue import Queue

from .data_bus import DataType, get_data_bus


class LogLevel(Enum):
    """Custom log levels for game-specific events."""
    GAME_ACTION = auto()      # In-game actions (moves, items, etc.)
    GAME_STATE = auto()       # Game state changes
    GAME_PROGRESS = auto()    # Progress markers (badges, story events)
    TRAINING = auto()         # Training-related events
    REWARD = auto()           # Reward signals
    PERFORMANCE = auto()      # Performance metrics
    SYSTEM = auto()           # System events
    DEBUG = auto()            # Debug information


@dataclass
class LogEvent:
    """Structured log event with metadata."""
    timestamp: float
    level: Union[LogLevel, str]
    category: str
    message: str
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    event_id: Optional[str] = None


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for different log levels."""
    
    COLORS = {
        'GAME_ACTION': '\033[36m',    # Cyan
        'GAME_STATE': '\033[34m',     # Blue
        'GAME_PROGRESS': '\033[32m',  # Green
        'TRAINING': '\033[35m',       # Magenta
        'REWARD': '\033[33m',         # Yellow
        'PERFORMANCE': '\033[36m',    # Cyan
        'SYSTEM': '\033[37m',         # White
        'DEBUG': '\033[90m',          # Gray
        'INFO': '\033[37m',           # White
        'WARNING': '\033[33m',        # Yellow
        'ERROR': '\033[31m',          # Red
        'CRITICAL': '\033[41m',       # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color if terminal supports it and it's not being redirected
        if sys.stdout.isatty():
            level_name = record.levelname
            if level_name in self.COLORS:
                record.levelname = f"{self.COLORS[level_name]}{level_name}{self.RESET}"
                record.msg = f"{self.COLORS[level_name]}{record.msg}{self.RESET}"
        return super().format(record)


class TextLogger:
    """Structured logger for game and training events."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_history: int = 1000,
                 file_max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 file_backup_count: int = 5):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("pokemon_crystal_rl")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(console_handler)
        
        # File handlers for different log types
        self._setup_file_handlers(file_max_bytes, file_backup_count)
        
        # Event history
        self.max_history = max_history
        self.event_history: List[LogEvent] = []
        self._history_lock = threading.Lock()
        
        # Event queue for real-time streaming
        self.event_queue = Queue(maxsize=1000)
        self.is_streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        
        # Data bus connection
        self.data_bus = get_data_bus()
        
        # Register custom log levels
        for level in LogLevel:
            logging.addLevelName(level.value, level.name)
        
        self.logger.info("📝 Text logger initialized")
    
    def _setup_file_handlers(self, max_bytes: int, backup_count: int) -> None:
        """Setup rotating file handlers for different log categories."""
        handlers = {
            'game': RotatingFileHandler(
                self.log_dir / 'game.log',
                maxBytes=max_bytes,
                backupCount=backup_count
            ),
            'training': RotatingFileHandler(
                self.log_dir / 'training.log',
                maxBytes=max_bytes,
                backupCount=backup_count
            ),
            'system': RotatingFileHandler(
                self.log_dir / 'system.log',
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        }
        
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for handler in handlers.values():
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_event(self, 
                  level: Union[LogLevel, str],
                  message: str,
                  category: str = "general",
                  data: Optional[Dict[str, Any]] = None,
                  source: Optional[str] = None) -> None:
        """Log a structured event with metadata."""
        try:
            # Create event object
            event = LogEvent(
                timestamp=time.time(),
                level=level,
                category=category,
                message=message,
                data=data or {},
                source=source,
                event_id=f"{int(time.time() * 1000)}_{threading.get_ident()}"
            )
            
            # Add to history
            with self._history_lock:
                self.event_history.append(event)
                if len(self.event_history) > self.max_history:
                    self.event_history.pop(0)
            
            # Add to streaming queue
            if self.is_streaming:
                try:
                    self.event_queue.put_nowait(event)
                except:
                    pass  # Queue full
            
            # Log using standard logging
            log_level = (
                level.value if isinstance(level, LogLevel)
                else getattr(logging, level.upper(), logging.INFO)
            )
            self.logger.log(log_level, message)
            
            # Publish to data bus
            if self.data_bus:
                self.data_bus.publish(
                    DataType.LOG_EVENT,
                    {
                        'timestamp': event.timestamp,
                        'level': event.level.name if isinstance(event.level, LogLevel) else event.level,
                        'category': event.category,
                        'message': event.message,
                        'data': event.data,
                        'source': event.source,
                        'event_id': event.event_id
                    },
                    'text_logger'
                )
                
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")
    
    def start_streaming(self) -> None:
        """Start real-time event streaming."""
        if self.is_streaming:
            return
            
        self.is_streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            daemon=True
        )
        self._stream_thread.start()
        
        self.logger.info("📡 Log streaming started")
    
    def stop_streaming(self) -> None:
        """Stop real-time event streaming."""
        self.is_streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5.0)
        
        self.logger.info("⏹️ Log streaming stopped")
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self.is_streaming:
            try:
                event = self.event_queue.get(timeout=1.0)
                if event and self.data_bus:
                    self.data_bus.publish(
                        DataType.LOG_STREAM,
                        {
                            'timestamp': event.timestamp,
                            'level': event.level.name if isinstance(event.level, LogLevel) else event.level,
                            'category': event.category,
                            'message': event.message,
                            'data': event.data,
                            'source': event.source,
                            'event_id': event.event_id
                        },
                        'text_logger'
                    )
            except:
                continue
    
    def get_events(self,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   levels: Optional[List[Union[LogLevel, str]]] = None,
                   categories: Optional[List[str]] = None,
                   search_text: Optional[str] = None) -> List[LogEvent]:
        """Get filtered log events from history."""
        with self._history_lock:
            events = self.event_history.copy()
        
        # Apply filters
        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]
        if levels:
            events = [e for e in events if e.level in levels]
        if categories:
            events = [e for e in events if e.category in categories]
        if search_text:
            pattern = re.compile(search_text, re.IGNORECASE)
            events = [e for e in events if pattern.search(e.message)]
        
        return events
    
    def save_events(self, filepath: str,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> None:
        """Save log events to a JSON file."""
        try:
            events = self.get_events(start_time, end_time)
            event_data = [
                {
                    'timestamp': e.timestamp,
                    'level': e.level.name if isinstance(e.level, LogLevel) else e.level,
                    'category': e.category,
                    'message': e.message,
                    'data': e.data,
                    'source': e.source,
                    'event_id': e.event_id
                }
                for e in events
            ]
            
            with open(filepath, 'w') as f:
                json.dump(event_data, f, indent=2)
            
            self.logger.info(f"💾 Events saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save events: {e}")
    
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._history_lock:
            self.event_history.clear()
        
        self.logger.info("🧹 Event history cleared")
    
    # Convenience methods for common log levels
    def game_action(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.GAME_ACTION, message, "game", **kwargs)
    
    def game_state(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.GAME_STATE, message, "game", **kwargs)
    
    def game_progress(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.GAME_PROGRESS, message, "game", **kwargs)
    
    def training(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.TRAINING, message, "training", **kwargs)
    
    def reward(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.REWARD, message, "training", **kwargs)
    
    def performance(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.PERFORMANCE, message, "system", **kwargs)
    
    def system(self, message: str, **kwargs) -> None:
        self.log_event(LogLevel.SYSTEM, message, "system", **kwargs)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_streaming()
