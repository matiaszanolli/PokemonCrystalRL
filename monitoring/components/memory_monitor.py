"""
Memory Monitor - System memory monitoring and management

Extracted from ErrorHandler to handle memory usage tracking,
leak detection, and automatic memory management.
"""

import gc
import time
import threading
import logging
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum


class MemoryThresholdLevel(Enum):
    """Memory threshold levels."""
    NORMAL = "normal"        # Under normal threshold
    WARNING = "warning"      # Approaching limit
    CRITICAL = "critical"    # Over critical threshold
    EMERGENCY = "emergency"  # Emergency cleanup needed


@dataclass
class MemoryConfig:
    """Configuration for memory monitoring."""
    threshold_mb: float = 1024.0              # Memory threshold in MB
    critical_threshold_mb: float = 2048.0     # Critical threshold in MB
    check_interval: float = 60.0              # Check interval in seconds
    history_size: int = 100                   # Number of readings to keep
    gc_on_threshold: bool = True              # Auto garbage collect
    detailed_tracking: bool = False           # Track detailed memory info
    alert_on_leak: bool = True                # Alert on potential leaks
    leak_detection_window: int = 10           # Readings for leak detection


@dataclass
class MemoryReading:
    """Single memory reading."""
    timestamp: float
    rss_mb: float          # Resident Set Size
    vms_mb: float          # Virtual Memory Size
    shared_mb: float       # Shared Memory
    data_mb: float         # Data Segment Size
    available_mb: float    # Available system memory
    percent_used: float    # Memory usage percentage
    gc_objects: int        # Number of tracked objects
    threshold_level: MemoryThresholdLevel


class MemoryMonitor:
    """Memory monitoring system with automatic management and leak detection."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger("MemoryMonitor")
        
        # Process handle
        try:
            self._process = psutil.Process()
        except Exception as e:
            self.logger.warning(f"Could not get process handle: {e}")
            self._process = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Memory tracking
        self.memory_history: deque = deque(maxlen=self.config.history_size)
        self.last_reading: Optional[MemoryReading] = None
        self._lock = threading.RLock()
        
        # Callbacks
        self.on_threshold_exceeded: Optional[Callable[[MemoryReading], None]] = None
        self.on_critical_threshold: Optional[Callable[[MemoryReading], None]] = None
        self.on_memory_leak_detected: Optional[Callable[[List[MemoryReading]], None]] = None
        
        # Statistics
        self.stats = {
            'readings_taken': 0,
            'gc_collections_triggered': 0,
            'threshold_violations': 0,
            'critical_violations': 0,
            'potential_leaks_detected': 0,
            'peak_memory_mb': 0.0,
            'average_memory_mb': 0.0
        }
        
        self.logger.info("Memory monitor initialized")
    
    def start_monitoring(self) -> bool:
        """Start memory monitoring.
        
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            self.logger.warning("Memory monitoring already active")
            return True
        
        if not self._process:
            self.logger.error("Cannot start monitoring without process handle")
            return False
        
        try:
            self.is_monitoring = True
            self._shutdown_event.clear()
            
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="MemoryMonitor"
            )
            self.monitoring_thread.start()
            
            self.logger.info("📊 Memory monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start memory monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping memory monitoring...")
        self.is_monitoring = False
        self._shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("📊 Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring and not self._shutdown_event.is_set():
            try:
                # Take memory reading
                reading = self.get_memory_reading()
                
                if reading:
                    self._process_reading(reading)
                
                # Sleep until next check
                self._shutdown_event.wait(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def get_memory_reading(self) -> Optional[MemoryReading]:
        """Get current memory reading.
        
        Returns:
            Optional[MemoryReading]: Current memory reading or None if failed
        """
        if not self._process:
            return None
        
        try:
            # Get process memory info
            mem_info = self._process.memory_info()
            
            # Get system memory info
            system_mem = psutil.virtual_memory()
            
            # Convert to MB
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
            shared_mb = getattr(mem_info, 'shared', 0) / (1024 * 1024)
            data_mb = getattr(mem_info, 'data', 0) / (1024 * 1024)
            available_mb = system_mem.available / (1024 * 1024)
            
            # Get GC info if detailed tracking enabled
            gc_objects = len(gc.get_objects()) if self.config.detailed_tracking else 0
            
            # Determine threshold level
            threshold_level = self._determine_threshold_level(rss_mb)
            
            reading = MemoryReading(
                timestamp=time.time(),
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                shared_mb=shared_mb,
                data_mb=data_mb,
                available_mb=available_mb,
                percent_used=system_mem.percent,
                gc_objects=gc_objects,
                threshold_level=threshold_level
            )
            
            return reading
            
        except Exception as e:
            self.logger.error(f"Failed to get memory reading: {e}")
            return None
    
    def _determine_threshold_level(self, rss_mb: float) -> MemoryThresholdLevel:
        """Determine memory threshold level.
        
        Args:
            rss_mb: Current RSS memory in MB
            
        Returns:
            MemoryThresholdLevel: Current threshold level
        """
        if rss_mb >= self.config.critical_threshold_mb:
            return MemoryThresholdLevel.CRITICAL
        elif rss_mb >= self.config.threshold_mb * 0.8:  # 80% of threshold
            return MemoryThresholdLevel.WARNING
        else:
            return MemoryThresholdLevel.NORMAL
    
    def _process_reading(self, reading: MemoryReading) -> None:
        """Process a memory reading.
        
        Args:
            reading: Memory reading to process
        """
        with self._lock:
            # Store reading
            self.memory_history.append(reading)
            self.last_reading = reading
            
            # Update statistics
            self._update_statistics(reading)
            
            # Handle threshold violations
            self._handle_thresholds(reading)
            
            # Check for memory leaks
            if self.config.alert_on_leak:
                self._check_for_memory_leaks()
    
    def _update_statistics(self, reading: MemoryReading) -> None:
        """Update monitoring statistics.
        
        Args:
            reading: Current memory reading
        """
        self.stats['readings_taken'] += 1
        
        # Update peak memory
        if reading.rss_mb > self.stats['peak_memory_mb']:
            self.stats['peak_memory_mb'] = reading.rss_mb
        
        # Update average memory (rolling average)
        if len(self.memory_history) > 1:
            total_memory = sum(r.rss_mb for r in self.memory_history)
            self.stats['average_memory_mb'] = total_memory / len(self.memory_history)
    
    def _handle_thresholds(self, reading: MemoryReading) -> None:
        """Handle memory threshold violations.
        
        Args:
            reading: Current memory reading
        """
        if reading.threshold_level == MemoryThresholdLevel.CRITICAL:
            self.stats['critical_violations'] += 1
            self.logger.warning(
                f"🚨 Critical memory threshold exceeded: {reading.rss_mb:.1f}MB "
                f"(limit: {self.config.critical_threshold_mb}MB)"
            )
            
            # Trigger critical callback
            if self.on_critical_threshold:
                try:
                    self.on_critical_threshold(reading)
                except Exception as e:
                    self.logger.error(f"Error in critical threshold callback: {e}")
            
            # Emergency garbage collection
            if self.config.gc_on_threshold:
                self._trigger_garbage_collection("critical threshold")
                
        elif reading.threshold_level == MemoryThresholdLevel.WARNING:
            if (not hasattr(self, '_last_warning_time') or 
                time.time() - self._last_warning_time > 300):  # Every 5 minutes
                
                self.stats['threshold_violations'] += 1
                self.logger.warning(
                    f"⚠️ Memory threshold warning: {reading.rss_mb:.1f}MB "
                    f"(threshold: {self.config.threshold_mb}MB)"
                )
                self._last_warning_time = time.time()
                
                # Trigger threshold callback
                if self.on_threshold_exceeded:
                    try:
                        self.on_threshold_exceeded(reading)
                    except Exception as e:
                        self.logger.error(f"Error in threshold callback: {e}")
                
                # Preventive garbage collection
                if self.config.gc_on_threshold:
                    self._trigger_garbage_collection("warning threshold")
    
    def _check_for_memory_leaks(self) -> None:
        """Check for potential memory leaks."""
        if len(self.memory_history) < self.config.leak_detection_window:
            return
        
        # Get recent readings
        recent_readings = list(self.memory_history)[-self.config.leak_detection_window:]
        
        # Calculate trend
        start_memory = recent_readings[0].rss_mb
        end_memory = recent_readings[-1].rss_mb
        memory_growth = end_memory - start_memory
        
        # Check if memory is consistently growing
        consistent_growth = True
        for i in range(1, len(recent_readings)):
            if recent_readings[i].rss_mb < recent_readings[i-1].rss_mb:
                consistent_growth = False
                break
        
        # Detect potential leak
        growth_threshold = self.config.threshold_mb * 0.1  # 10% of threshold
        if consistent_growth and memory_growth > growth_threshold:
            self.stats['potential_leaks_detected'] += 1
            
            self.logger.warning(
                f"🔍 Potential memory leak detected: "
                f"{memory_growth:.1f}MB growth over {len(recent_readings)} readings"
            )
            
            # Trigger leak detection callback
            if self.on_memory_leak_detected:
                try:
                    self.on_memory_leak_detected(recent_readings)
                except Exception as e:
                    self.logger.error(f"Error in leak detection callback: {e}")
    
    def _trigger_garbage_collection(self, reason: str = "manual") -> Dict[str, int]:
        """Force garbage collection and return collection statistics.
        
        Args:
            reason: Reason for garbage collection
            
        Returns:
            Dict: Collection statistics
        """
        self.logger.info(f"🗑️ Triggering garbage collection: {reason}")
        
        collected = 0
        collections = 0
        
        try:
            # Run collection for all generations
            for generation in range(3):
                before_count = len(gc.get_objects())
                collected_gen = gc.collect(generation)
                after_count = len(gc.get_objects())
                
                collected += collected_gen
                collections += 1
                
                self.logger.debug(
                    f"GC generation {generation}: {collected_gen} objects collected, "
                    f"{before_count} -> {after_count} objects"
                )
            
            self.stats['gc_collections_triggered'] += 1
            
            result = {
                'objects_collected': collected,
                'collections': collections,
                'reason': reason
            }
            
            self.logger.info(f"GC completed: {collected} objects collected")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")
            return {'error': str(e)}
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Manually trigger garbage collection.
        
        Returns:
            Dict: Collection statistics
        """
        return self._trigger_garbage_collection("manual request")
    
    def get_current_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information.
        
        Returns:
            Dict: Memory usage statistics in MB
        """
        reading = self.get_memory_reading()
        
        if not reading:
            return {'error': 'Unable to get memory info'}
        
        return {
            'rss_mb': reading.rss_mb,
            'vms_mb': reading.vms_mb,
            'shared_mb': reading.shared_mb,
            'data_mb': reading.data_mb,
            'available_mb': reading.available_mb,
            'percent_used': reading.percent_used,
            'gc_objects': reading.gc_objects,
            'threshold_level': reading.threshold_level.value,
            'timestamp': reading.timestamp
        }
    
    def get_memory_trend(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get memory usage trend over specified window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dict: Trend analysis
        """
        with self._lock:
            if not self.memory_history:
                return {'error': 'No memory history available'}
            
            cutoff_time = time.time() - (window_minutes * 60)
            recent_readings = [
                r for r in self.memory_history
                if r.timestamp >= cutoff_time
            ]
            
            if len(recent_readings) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Calculate trend
            start_memory = recent_readings[0].rss_mb
            end_memory = recent_readings[-1].rss_mb
            memory_change = end_memory - start_memory
            
            # Calculate average and peak
            memories = [r.rss_mb for r in recent_readings]
            avg_memory = sum(memories) / len(memories)
            peak_memory = max(memories)
            min_memory = min(memories)
            
            return {
                'window_minutes': window_minutes,
                'readings_count': len(recent_readings),
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'memory_change_mb': memory_change,
                'average_memory_mb': avg_memory,
                'peak_memory_mb': peak_memory,
                'min_memory_mb': min_memory,
                'trend': 'increasing' if memory_change > 0 else 'decreasing' if memory_change < 0 else 'stable',
                'change_rate_mb_per_min': memory_change / max(window_minutes, 1)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory monitoring statistics.
        
        Returns:
            Dict: Monitoring statistics
        """
        stats = self.stats.copy()
        
        with self._lock:
            stats.update({
                'is_monitoring': self.is_monitoring,
                'history_size': len(self.memory_history),
                'current_memory_mb': self.last_reading.rss_mb if self.last_reading else 0,
                'threshold_mb': self.config.threshold_mb,
                'critical_threshold_mb': self.config.critical_threshold_mb,
                'current_threshold_level': (
                    self.last_reading.threshold_level.value 
                    if self.last_reading else 'unknown'
                ),
                'monitoring_uptime_seconds': (
                    time.time() - self.memory_history[0].timestamp
                    if self.memory_history else 0
                )
            })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset monitoring statistics."""
        self.stats = {
            'readings_taken': 0,
            'gc_collections_triggered': 0,
            'threshold_violations': 0,
            'critical_violations': 0,
            'potential_leaks_detected': 0,
            'peak_memory_mb': 0.0,
            'average_memory_mb': 0.0
        }
        
        self.logger.info("Memory monitor statistics reset")
    
    def set_thresholds(self, threshold_mb: float, critical_threshold_mb: float = None) -> None:
        """Update memory thresholds.
        
        Args:
            threshold_mb: Warning threshold in MB
            critical_threshold_mb: Critical threshold in MB (optional)
        """
        self.config.threshold_mb = threshold_mb
        
        if critical_threshold_mb:
            self.config.critical_threshold_mb = critical_threshold_mb
        else:
            self.config.critical_threshold_mb = threshold_mb * 2
        
        self.logger.info(
            f"Memory thresholds updated: {threshold_mb}MB warning, "
            f"{self.config.critical_threshold_mb}MB critical"
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()