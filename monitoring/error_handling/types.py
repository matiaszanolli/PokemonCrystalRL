"""
Error handling types, enums, and data structures.

This module defines the core data types used throughout the error handling system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # System-breaking errors
    HIGH = "high"             # High priority errors
    ERROR = "error"           # Serious errors that need immediate attention
    MEDIUM = "medium"         # Medium priority errors
    WARNING = "warning"       # Issues that need monitoring
    INFO = "info"            # Informational errors


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    SYSTEM = "system"         # System-level errors
    NETWORK = "network"       # Network-related issues
    DATABASE = "database"     # Database errors
    GAME = "game"            # Game-specific errors
    TRAINING = "training"     # ML training errors
    MEMORY = "memory"        # Memory-related issues
    PERFORMANCE = "performance"  # Performance problems
    UNKNOWN = "unknown"       # Uncategorized errors


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    NONE = "none"  # No recovery attempted
    RETRY = "retry"  # Retry the operation
    RESTART = "restart"  # Restart the component
    RESTART_COMPONENT = "restart_component"  # Restart specific component
    RESET = "reset"  # Full system reset
    GRACEFUL_SHUTDOWN = "graceful_shutdown"  # Graceful shutdown
    FALLBACK = "fallback"  # Use fallback mechanism


@dataclass
class ErrorContext:
    """Context information for an error."""
    timestamp: float
    error_type: str
    error_message: str
    traceback: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    additional_data: Optional[Dict[str, Any]] = None
    error_id: Optional[str] = None
    handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ErrorEvent:
    """Simplified error event for testing and basic error handling."""
    timestamp: float
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: str = ""
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE
