"""
Configuration for the monitoring system.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MonitorConfig:
    """Configuration for the monitoring system."""
    
    db_path: str = "monitoring.db"
    data_dir: str = "monitor_data"
    static_dir: str = "static"
    web_port: int = 8080
    update_interval: float = 1.0
    snapshot_interval: float = 300.0
    max_events: int = 1000
    max_snapshots: int = 100
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.web_port < 0 or self.web_port > 65535:
            raise ValueError("web_port must be between 0 and 65535")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.max_events < 0:
            raise ValueError("max_events must be non-negative")
        if self.max_snapshots < 0:
            raise ValueError("max_snapshots must be non-negative")
