"""
Notification System - Error notification and alert management

Extracted from ErrorHandler to handle error notifications, batching,
rate limiting, and alert delivery mechanisms.
"""

import time
import queue
import logging
import threading
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta


class NotificationLevel(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DATA_BUS = "data_bus"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class NotificationConfig:
    """Configuration for notification system."""
    batch_size: int = 10
    batch_interval: float = 5.0  # seconds
    max_queue_size: int = 1000
    rate_limit_window: float = 60.0  # seconds
    max_notifications_per_window: int = 50
    enable_deduplication: bool = True
    deduplication_window: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class NotificationMessage:
    """A notification message."""
    id: str
    level: NotificationLevel
    title: str
    message: str
    timestamp: float
    component: str
    channels: List[NotificationChannel]
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    next_retry_time: Optional[float] = None


class NotificationHandler:
    """Base class for notification handlers."""
    
    def __init__(self, channel: NotificationChannel):
        self.channel = channel
        self.logger = logging.getLogger(f"NotificationHandler.{channel.value}")
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification message.
        
        Args:
            message: Notification to send
            
        Returns:
            bool: True if sent successfully
        """
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if handler is available.
        
        Returns:
            bool: True if handler can send notifications
        """
        return True


class LogNotificationHandler(NotificationHandler):
    """Log-based notification handler."""
    
    def __init__(self):
        super().__init__(NotificationChannel.LOG)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification to log."""
        try:
            log_method = getattr(self.logger, message.level.value.lower(), self.logger.info)
            log_method(f"[{message.component}] {message.title}: {message.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log notification: {e}")
            return False


class ConsoleNotificationHandler(NotificationHandler):
    """Console-based notification handler."""
    
    def __init__(self):
        super().__init__(NotificationChannel.CONSOLE)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification to console."""
        try:
            timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
            level_emoji = {
                NotificationLevel.LOW: "ℹ️",
                NotificationLevel.MEDIUM: "⚠️", 
                NotificationLevel.HIGH: "🚨",
                NotificationLevel.CRITICAL: "🔥"
            }
            
            emoji = level_emoji.get(message.level, "📢")
            print(f"{emoji} [{timestamp}] {message.component}: {message.title}")
            if len(message.message) <= 100:
                print(f"   {message.message}")
            else:
                print(f"   {message.message[:97]}...")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send console notification: {e}")
            return False


class FileNotificationHandler(NotificationHandler):
    """File-based notification handler."""
    
    def __init__(self, file_path: str):
        super().__init__(NotificationChannel.FILE)
        self.file_path = file_path
        self._lock = threading.Lock()
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification to file."""
        try:
            notification_data = {
                'timestamp': datetime.fromtimestamp(message.timestamp).isoformat(),
                'level': message.level.value,
                'component': message.component,
                'title': message.title,
                'message': message.message,
                'metadata': message.metadata or {}
            }
            
            with self._lock:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(notification_data) + '\n')
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to write notification to file: {e}")
            return False


class DataBusNotificationHandler(NotificationHandler):
    """Data bus notification handler."""
    
    def __init__(self, data_bus=None):
        super().__init__(NotificationChannel.DATA_BUS)
        self.data_bus = data_bus
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification to data bus."""
        if not self.data_bus:
            return False
        
        try:
            from ..data_bus import DataType
            
            notification_data = {
                'id': message.id,
                'level': message.level.value,
                'title': message.title,
                'message': message.message,
                'timestamp': message.timestamp,
                'component': message.component,
                'metadata': message.metadata or {}
            }
            
            self.data_bus.publish(
                DataType.NOTIFICATION,
                notification_data,
                "notification_system"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish notification to data bus: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if data bus is available."""
        return self.data_bus is not None


class WebhookNotificationHandler(NotificationHandler):
    """Webhook notification handler."""
    
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        super().__init__(NotificationChannel.WEBHOOK)
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def send(self, message: NotificationMessage) -> bool:
        """Send notification via webhook."""
        try:
            import requests
            
            payload = {
                'id': message.id,
                'level': message.level.value,
                'title': message.title,
                'message': message.message,
                'timestamp': message.timestamp,
                'component': message.component,
                'metadata': message.metadata or {}
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if webhook is available."""
        try:
            import requests
            return True
        except ImportError:
            return False


class NotificationSystem:
    """Manages error notifications with batching, rate limiting, and deduplication."""
    
    def __init__(self, config: NotificationConfig = None, data_bus=None):
        self.config = config or NotificationConfig()
        self.logger = logging.getLogger("NotificationSystem")
        
        # Notification queue
        self.notification_queue = queue.Queue(maxsize=self.config.max_queue_size)
        
        # Processing state
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Rate limiting
        self.rate_limit_window = deque()
        self.rate_limit_lock = threading.Lock()
        
        # Deduplication
        self.notification_signatures: Dict[str, float] = {}
        self.dedup_lock = threading.Lock()
        
        # Handlers
        self.handlers: Dict[NotificationChannel, NotificationHandler] = {
            NotificationChannel.LOG: LogNotificationHandler(),
            NotificationChannel.CONSOLE: ConsoleNotificationHandler()
        }
        
        # Add data bus handler if available
        if data_bus:
            self.handlers[NotificationChannel.DATA_BUS] = DataBusNotificationHandler(data_bus)
        
        # Statistics
        self.stats = {
            'total_notifications': 0,
            'sent_notifications': 0,
            'failed_notifications': 0,
            'rate_limited_notifications': 0,
            'deduplicated_notifications': 0,
            'retries_attempted': 0
        }
        
        self.logger.info("Notification system initialized")
    
    def add_handler(self, handler: NotificationHandler) -> None:
        """Add notification handler.
        
        Args:
            handler: Notification handler to add
        """
        self.handlers[handler.channel] = handler
        self.logger.info(f"Added notification handler: {handler.channel.value}")
    
    def add_file_handler(self, file_path: str) -> None:
        """Add file notification handler.
        
        Args:
            file_path: Path to notification log file
        """
        self.add_handler(FileNotificationHandler(file_path))
    
    def add_webhook_handler(self, webhook_url: str, timeout: float = 10.0) -> None:
        """Add webhook notification handler.
        
        Args:
            webhook_url: Webhook URL
            timeout: Request timeout
        """
        self.add_handler(WebhookNotificationHandler(webhook_url, timeout))
    
    def start_processing(self) -> None:
        """Start notification processing thread."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self._shutdown_event.clear()
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="NotificationProcessor"
        )
        self.processing_thread.start()
        
        self.logger.info("📢 Notification processing started")
    
    def stop_processing(self) -> None:
        """Stop notification processing thread."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self._shutdown_event.set()
        
        # Process remaining notifications
        self._process_remaining_notifications()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("📢 Notification processing stopped")
    
    def send_notification(self,
                         level: NotificationLevel,
                         title: str,
                         message: str,
                         component: str = "system",
                         channels: List[NotificationChannel] = None,
                         metadata: Dict[str, Any] = None) -> bool:
        """Send a notification.
        
        Args:
            level: Notification level
            title: Notification title
            message: Notification message
            component: Component that generated notification
            channels: Channels to send to (default: LOG and CONSOLE)
            metadata: Additional metadata
            
        Returns:
            bool: True if notification was queued successfully
        """
        if channels is None:
            channels = [NotificationChannel.LOG, NotificationChannel.CONSOLE]
        
        # Check rate limiting
        if not self._check_rate_limit():
            self.stats['rate_limited_notifications'] += 1
            self.logger.debug("Notification rate limited")
            return False
        
        # Create notification
        notification = NotificationMessage(
            id=self._generate_notification_id(),
            level=level,
            title=title,
            message=message,
            timestamp=time.time(),
            component=component,
            channels=channels,
            metadata=metadata or {}
        )
        
        # Check deduplication
        if self.config.enable_deduplication:
            signature = self._create_notification_signature(notification)
            if self._is_duplicate(signature):
                self.stats['deduplicated_notifications'] += 1
                self.logger.debug("Notification deduplicated")
                return False
            self._record_notification_signature(signature)
        
        # Queue notification
        try:
            self.notification_queue.put(notification, block=False)
            self.stats['total_notifications'] += 1
            return True
        except queue.Full:
            self.logger.warning("Notification queue full, dropping notification")
            return False
    
    def _processing_loop(self) -> None:
        """Main notification processing loop."""
        batch = []
        last_batch_time = time.time()
        
        while self.is_processing or not self.notification_queue.empty():
            try:
                # Get notification with timeout
                try:
                    notification = self.notification_queue.get(timeout=1.0)
                    batch.append(notification)
                except queue.Empty:
                    notification = None
                
                current_time = time.time()
                
                # Process batch if full or timeout reached
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_batch_time >= self.config.batch_interval) or
                    self._shutdown_event.is_set()
                )
                
                if should_process and batch:
                    self._process_notification_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}")
                time.sleep(1.0)
    
    def _process_notification_batch(self, notifications: List[NotificationMessage]) -> None:
        """Process a batch of notifications.
        
        Args:
            notifications: List of notifications to process
        """
        for notification in notifications:
            self._process_single_notification(notification)
    
    def _process_single_notification(self, notification: NotificationMessage) -> None:
        """Process a single notification.
        
        Args:
            notification: Notification to process
        """
        success_count = 0
        
        for channel in notification.channels:
            handler = self.handlers.get(channel)
            
            if not handler:
                self.logger.debug(f"No handler available for channel: {channel.value}")
                continue
            
            if not handler.is_available():
                self.logger.debug(f"Handler not available: {channel.value}")
                continue
            
            try:
                if handler.send(notification):
                    success_count += 1
                else:
                    self._handle_send_failure(notification, channel)
            except Exception as e:
                self.logger.error(f"Handler error for {channel.value}: {e}")
                self._handle_send_failure(notification, channel)
        
        # Update statistics
        if success_count > 0:
            self.stats['sent_notifications'] += 1
        else:
            self.stats['failed_notifications'] += 1
            self._handle_notification_retry(notification)
    
    def _handle_send_failure(self, notification: NotificationMessage, channel: NotificationChannel):
        """Handle notification send failure.
        
        Args:
            notification: Failed notification
            channel: Channel that failed
        """
        self.logger.debug(f"Failed to send notification via {channel.value}")
    
    def _handle_notification_retry(self, notification: NotificationMessage) -> None:
        """Handle notification retry logic.
        
        Args:
            notification: Notification to retry
        """
        if notification.retry_count < self.config.retry_attempts:
            notification.retry_count += 1
            notification.next_retry_time = time.time() + (self.config.retry_delay * notification.retry_count)
            
            # Re-queue for retry
            try:
                self.notification_queue.put(notification, block=False)
                self.stats['retries_attempted'] += 1
            except queue.Full:
                self.logger.warning("Queue full, cannot retry notification")
    
    def _check_rate_limit(self) -> bool:
        """Check if notification is within rate limit.
        
        Returns:
            bool: True if within rate limit
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove old entries
            cutoff_time = current_time - self.config.rate_limit_window
            while self.rate_limit_window and self.rate_limit_window[0] < cutoff_time:
                self.rate_limit_window.popleft()
            
            # Check limit
            if len(self.rate_limit_window) >= self.config.max_notifications_per_window:
                return False
            
            # Record this notification
            self.rate_limit_window.append(current_time)
            return True
    
    def _create_notification_signature(self, notification: NotificationMessage) -> str:
        """Create signature for notification deduplication.
        
        Args:
            notification: Notification to create signature for
            
        Returns:
            str: Signature string
        """
        signature_parts = [
            notification.level.value,
            notification.component,
            notification.title,
            notification.message[:100]  # First 100 chars
        ]
        
        signature = "|".join(signature_parts)
        
        # Hash for consistent length
        import hashlib
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _is_duplicate(self, signature: str) -> bool:
        """Check if notification is duplicate.
        
        Args:
            signature: Notification signature
            
        Returns:
            bool: True if duplicate
        """
        with self.dedup_lock:
            current_time = time.time()
            
            if signature in self.notification_signatures:
                last_time = self.notification_signatures[signature]
                if current_time - last_time < self.config.deduplication_window:
                    return True
            
            return False
    
    def _record_notification_signature(self, signature: str) -> None:
        """Record notification signature for deduplication.
        
        Args:
            signature: Notification signature
        """
        with self.dedup_lock:
            current_time = time.time()
            
            # Clean old signatures
            expired_signatures = [
                sig for sig, timestamp in self.notification_signatures.items()
                if current_time - timestamp >= self.config.deduplication_window
            ]
            
            for sig in expired_signatures:
                del self.notification_signatures[sig]
            
            # Record new signature
            self.notification_signatures[signature] = current_time
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID.
        
        Returns:
            str: Unique notification ID
        """
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _process_remaining_notifications(self) -> None:
        """Process any remaining notifications in the queue."""
        remaining = []
        
        try:
            while True:
                notification = self.notification_queue.get_nowait()
                remaining.append(notification)
        except queue.Empty:
            pass
        
        if remaining:
            self.logger.info(f"Processing {len(remaining)} remaining notifications")
            self._process_notification_batch(remaining)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics.
        
        Returns:
            Dict: Statistics about notification system
        """
        stats = self.stats.copy()
        stats.update({
            'queue_size': self.notification_queue.qsize(),
            'is_processing': self.is_processing,
            'available_handlers': [
                channel.value for channel, handler in self.handlers.items()
                if handler.is_available()
            ],
            'rate_limit_tokens': len(self.rate_limit_window),
            'deduplication_cache_size': len(self.notification_signatures),
            'success_rate': (
                self.stats['sent_notifications'] / 
                max(self.stats['total_notifications'], 1)
            )
        })
        return stats
    
    def clear_statistics(self) -> None:
        """Clear notification statistics."""
        self.stats = {
            'total_notifications': 0,
            'sent_notifications': 0,
            'failed_notifications': 0,
            'rate_limited_notifications': 0,
            'deduplicated_notifications': 0,
            'retries_attempted': 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_processing()