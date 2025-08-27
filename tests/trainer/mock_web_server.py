"""Mock web server for tests."""

import dataclasses
import threading


@dataclasses.dataclass
class ServerConfig:
    """Mock server config"""
    host: str = "localhost"
    port: int = 8080

    @classmethod
    def from_training_config(cls, training_config):
        """Create mock config from training config"""
        return cls(
            host=training_config.web_host,
            port=training_config.web_port
        )


class MockWebServer:
    """Mock web server for tests with proper port handling"""
    ServerConfig = ServerConfig
    _used_ports = set()  # Track ports in use across instances
    
    def __init__(self, config=None, trainer=None):
        self.config = config or ServerConfig()
        self.trainer = trainer
        self.port = self.config.port
        self._running = False
        
    def start(self):
        """Start mock server with port conflict resolution"""
        # Check if port is available
        if self.port in self._used_ports:
            # Try to find next available port
            for new_port in range(self.port + 1, self.port + 10):
                if new_port not in self._used_ports:
                    self.port = new_port
                    self._used_ports.add(new_port)
                    self._running = True
                    return True
            # No ports available
            return False
            
        # Port is available
        self._used_ports.add(self.port)
        self._running = True
        return True
        
    def stop(self):
        """Stop mock server"""
        if self._running:
            self._used_ports.discard(self.port)
            self._running = False
        
    def run_in_thread(self):
        """Run mock server thread"""
        pass
        
    def shutdown(self):
        """Stop mock server and cleanup"""
        self.stop()
