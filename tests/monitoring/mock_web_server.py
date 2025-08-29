"""Mock web server for testing."""

import threading
from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass
class MockServerConfig:
    """Mock server configuration."""
    port: int = 8080
    host: str = "localhost"
    update_interval: float = 0.1
    enable_debug: bool = False

    @classmethod
    def from_training_config(cls, config: Any) -> 'MockServerConfig':
        """Create from training config."""
        return cls(
            port=getattr(config, 'web_port', 8080),
            host=getattr(config, 'web_host', 'localhost'),
            enable_debug=getattr(config, 'debug_mode', False)
        )

class MockWebServer:
    """Mock web server for testing.
    
    Provides the same interface as TrainingWebServer but without actual network/threads.
    """
    
    ServerConfig = MockServerConfig
    
    def __init__(self, config: MockServerConfig, trainer: Any):
        """Initialize mock server."""
        self.config = config
        self.trainer = trainer
        self._running = False
        self.port = config.port
        self.host = config.host
        
        # State tracking for tests
        self.stats_updates: list = []
        self.screen_updates: list = []
        self._error_count = 0
        
    def start(self) -> bool:
        """Start the mock server."""
        self._running = True
        return True
        
    def stop(self) -> None:
        """Stop the mock server."""
        self._running = False
        
    def shutdown(self) -> None:
        """Shutdown the mock server."""
        self.stop()
        
    def update_stats(self, stats: Dict[str, Any]) -> None:
        """Record stats update."""
        if not self._running:
            return
        self.stats_updates.append(stats)
        
    def update_screen(self, screen_data: Dict[str, Any]) -> None:
        """Record screen update."""
        if not self._running:
            return
        self.screen_updates.append(screen_data)
        
    def get_last_stats(self) -> Optional[Dict[str, Any]]:
        """Get most recent stats update."""
        return self.stats_updates[-1] if self.stats_updates else None
        
    def get_last_screen(self) -> Optional[Dict[str, Any]]:
        """Get most recent screen update."""
        return self.screen_updates[-1] if self.screen_updates else None
        
    def clear_history(self) -> None:
        """Clear update history."""
        self.stats_updates.clear()
        self.screen_updates.clear()
        
    def run_in_thread(self) -> None:
        """Mock thread runner that does nothing."""
        pass
