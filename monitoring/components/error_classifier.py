"""
Error Classifier - Error categorization, severity assessment, and recovery strategies

Extracted from ErrorHandler to handle error analysis, classification,
and recovery strategy determination.
"""

import traceback
import logging
import re
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


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
    error: Exception
    component: str
    operation: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    recovery_strategy: RecoveryStrategy
    traceback_str: Optional[str] = None
    metadata: Dict[str, Any] = None
    is_duplicate: bool = False
    duplicate_count: int = 1


@dataclass
class ErrorEvent:
    """Error event for data bus publication."""
    timestamp: float
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    recovery_strategy: RecoveryStrategy
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = None


class ErrorClassifier:
    """Classifies errors by severity, category, and determines recovery strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger("ErrorClassifier")
        
        # Error patterns for classification
        self._severity_patterns = {
            ErrorSeverity.CRITICAL: [
                r"system.*critical|critical.*system",
                r"memory.*error|out of memory|segmentation fault",
                r"corruption|corrupt|damaged",
                r"fatal.*error|fatal.*exception",
                r"unable to start|failed to initialize|startup.*failed"
            ],
            ErrorSeverity.HIGH: [
                r"connection.*lost|connection.*failed|network.*error",
                r"database.*error|sql.*error|query.*failed",
                r"authentication.*failed|permission.*denied",
                r"timeout.*error|request.*timeout",
                r"service.*unavailable|server.*error"
            ],
            ErrorSeverity.MEDIUM: [
                r"validation.*error|invalid.*input|bad.*request",
                r"configuration.*error|config.*invalid",
                r"file.*not.*found|missing.*file",
                r"rate.*limit|quota.*exceeded"
            ],
            ErrorSeverity.WARNING: [
                r"deprecated|warning|caution",
                r"retry.*attempt|attempting.*retry",
                r"fallback|using.*default"
            ]
        }
        
        self._category_patterns = {
            ErrorCategory.SYSTEM: [
                r"system|os|platform|environment",
                r"file.*system|disk|storage",
                r"process|thread|signal"
            ],
            ErrorCategory.NETWORK: [
                r"network|connection|socket|http|tcp|udp",
                r"request|response|url|api",
                r"timeout|unreachable|refused"
            ],
            ErrorCategory.DATABASE: [
                r"database|sql|query|table|column",
                r"transaction|commit|rollback",
                r"constraint|foreign.*key|primary.*key"
            ],
            ErrorCategory.GAME: [
                r"game|pyboy|pokemon|emulation|rom",
                r"screen|capture|trainer|action",
                r"state|memory.*address|battle"
            ],
            ErrorCategory.TRAINING: [
                r"training|model|llm|dqn|agent",
                r"reward|action|decision|policy",
                r"neural|network|tensor|gradient"
            ],
            ErrorCategory.MEMORY: [
                r"memory|ram|heap|stack|allocation",
                r"garbage.*collection|gc|leak",
                r"out.*of.*memory|memory.*error"
            ],
            ErrorCategory.PERFORMANCE: [
                r"performance|slow|timeout|bottleneck",
                r"cpu|gpu|resource|utilization",
                r"optimization|efficiency"
            ]
        }
        
        # Recovery strategies by error type and category
        self._recovery_strategies = {
            # By exception type
            ConnectionError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            MemoryError: RecoveryStrategy.GRACEFUL_SHUTDOWN,
            PermissionError: RecoveryStrategy.FALLBACK,
            FileNotFoundError: RecoveryStrategy.FALLBACK,
            KeyboardInterrupt: RecoveryStrategy.GRACEFUL_SHUTDOWN,
            SystemExit: RecoveryStrategy.GRACEFUL_SHUTDOWN,
            
            # By category
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.DATABASE: RecoveryStrategy.RETRY,
            ErrorCategory.MEMORY: RecoveryStrategy.RESTART_COMPONENT,
            ErrorCategory.TRAINING: RecoveryStrategy.FALLBACK,
            ErrorCategory.GAME: RecoveryStrategy.RESTART_COMPONENT,
            
            # By severity
            ErrorSeverity.CRITICAL: RecoveryStrategy.GRACEFUL_SHUTDOWN,
            ErrorSeverity.HIGH: RecoveryStrategy.RESTART_COMPONENT,
            ErrorSeverity.MEDIUM: RecoveryStrategy.RETRY,
            ErrorSeverity.WARNING: RecoveryStrategy.NONE
        }
        
        self.logger.info("Error classifier initialized with pattern-based classification")
    
    def classify_error(self, 
                      error: Exception, 
                      component: str = "unknown",
                      operation: str = "unknown",
                      metadata: Dict[str, Any] = None) -> ErrorContext:
        """Classify an error and determine recovery strategy.
        
        Args:
            error: Exception to classify
            component: Component where error occurred
            operation: Operation being performed
            metadata: Additional context metadata
            
        Returns:
            ErrorContext with classification and recovery strategy
        """
        # Determine severity and category
        severity = self._determine_severity(error)
        category = self._determine_category(error, component)
        recovery_strategy = self._determine_recovery_strategy(error, severity, category)
        
        # Get traceback
        traceback_str = self._extract_traceback(error)
        
        # Create error context
        context = ErrorContext(
            error=error,
            component=component,
            operation=operation,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            recovery_strategy=recovery_strategy,
            traceback_str=traceback_str,
            metadata=metadata or {}
        )
        
        self.logger.debug(
            f"Classified error: {type(error).__name__} -> "
            f"Severity: {severity.value}, Category: {category.value}, "
            f"Recovery: {recovery_strategy.value}"
        )
        
        return context
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type and message."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        combined_text = f"{error_type} {error_message}"
        
        # Check patterns in order of severity (highest to lowest)
        for severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH, 
                        ErrorSeverity.MEDIUM, ErrorSeverity.WARNING]:
            patterns = self._severity_patterns.get(severity, [])
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return severity
        
        # Check specific exception types
        if isinstance(error, (MemoryError, SystemError, OSError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (Warning, UserWarning)):
            return ErrorSeverity.WARNING
        
        # Default severity
        return ErrorSeverity.ERROR
    
    def _determine_category(self, error: Exception, component: str) -> ErrorCategory:
        """Determine error category based on exception and component context."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        component_lower = component.lower()
        combined_text = f"{error_type} {error_message} {component_lower}"
        
        # Check patterns for each category
        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return category
        
        # Component-based classification
        if any(keyword in component_lower for keyword in ['trainer', 'agent', 'llm', 'dqn']):
            return ErrorCategory.TRAINING
        elif any(keyword in component_lower for keyword in ['game', 'pyboy', 'emulation']):
            return ErrorCategory.GAME
        elif any(keyword in component_lower for keyword in ['network', 'api', 'web']):
            return ErrorCategory.NETWORK
        elif any(keyword in component_lower for keyword in ['database', 'db', 'sql']):
            return ErrorCategory.DATABASE
        elif any(keyword in component_lower for keyword in ['memory', 'monitor']):
            return ErrorCategory.MEMORY
        
        # Exception type-based classification
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(error, (OSError, SystemError)):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _determine_recovery_strategy(self, 
                                   error: Exception,
                                   severity: ErrorSeverity,
                                   category: ErrorCategory) -> RecoveryStrategy:
        """Determine appropriate recovery strategy."""
        # Check specific exception type first
        error_type = type(error)
        if error_type in self._recovery_strategies:
            return self._recovery_strategies[error_type]
        
        # Check by category
        if category in self._recovery_strategies:
            return self._recovery_strategies[category]
        
        # Check by severity
        if severity in self._recovery_strategies:
            return self._recovery_strategies[severity]
        
        # Default strategy based on severity
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.GRACEFUL_SHUTDOWN
        elif severity in [ErrorSeverity.HIGH, ErrorSeverity.ERROR]:
            return RecoveryStrategy.RESTART_COMPONENT
        elif severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.NONE
    
    def _extract_traceback(self, error: Exception) -> Optional[str]:
        """Extract formatted traceback from exception."""
        try:
            return traceback.format_exc()
        except Exception:
            return f"Traceback unavailable for {type(error).__name__}: {error}"
    
    def create_error_signature(self, context: ErrorContext) -> str:
        """Create unique signature for error deduplication.
        
        Args:
            context: Error context
            
        Returns:
            str: Unique signature for the error
        """
        error_type = type(context.error).__name__
        error_message = str(context.error)
        
        # Create signature from key components
        signature_parts = [
            context.component,
            context.operation,
            error_type,
            # Use first 100 chars of message to avoid too specific signatures
            error_message[:100] if error_message else ""
        ]
        
        signature = "|".join(signature_parts)
        
        # Hash for consistent length
        import hashlib
        return hashlib.md5(signature.encode()).hexdigest()
    
    def create_error_event(self, context: ErrorContext) -> ErrorEvent:
        """Create error event for data bus publication.
        
        Args:
            context: Error context
            
        Returns:
            ErrorEvent: Event ready for publication
        """
        return ErrorEvent(
            timestamp=context.timestamp.timestamp(),
            component=context.component,
            error_type=type(context.error).__name__,
            message=str(context.error),
            severity=context.severity,
            category=context.category,
            recovery_strategy=context.recovery_strategy,
            traceback=context.traceback_str,
            metadata=context.metadata
        )
    
    def add_custom_pattern(self, 
                          severity: ErrorSeverity, 
                          pattern: str) -> None:
        """Add custom pattern for severity classification.
        
        Args:
            severity: Severity level to associate with pattern
            pattern: Regular expression pattern
        """
        if severity not in self._severity_patterns:
            self._severity_patterns[severity] = []
        
        self._severity_patterns[severity].append(pattern)
        self.logger.info(f"Added custom pattern for {severity.value}: {pattern}")
    
    def add_custom_recovery_strategy(self,
                                   key: Union[Type[Exception], ErrorCategory, ErrorSeverity],
                                   strategy: RecoveryStrategy) -> None:
        """Add custom recovery strategy for specific error types/categories.
        
        Args:
            key: Exception type, category, or severity level
            strategy: Recovery strategy to use
        """
        self._recovery_strategies[key] = strategy
        self.logger.info(f"Added custom recovery strategy: {key} -> {strategy.value}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about error classification patterns.
        
        Returns:
            Dict: Classification statistics
        """
        return {
            'severity_patterns': {k.value: len(v) for k, v in self._severity_patterns.items()},
            'category_patterns': {k.value: len(v) for k, v in self._category_patterns.items()},
            'recovery_strategies': len(self._recovery_strategies),
            'supported_severities': [s.value for s in ErrorSeverity],
            'supported_categories': [c.value for c in ErrorCategory],
            'supported_strategies': [s.value for s in RecoveryStrategy]
        }