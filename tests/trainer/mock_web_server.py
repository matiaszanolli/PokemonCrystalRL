"""Mock web server for tests."""

import dataclasses
import threading
import logging

from monitoring.web_server import ServerConfig


class MockWebServer:
    """Mock web server for tests with proper port handling"""
    ServerConfig = ServerConfig
    
    _used_ports = set()  # Track ports in use across instances
    
    def __init__(self, config=None, trainer=None):
        self.config = config or ServerConfig()
        self.trainer = trainer
        self._trainer = trainer  # For backward compatibility
        self.port = self.config.port
        self._running = False
        self.logger = logging.getLogger("mock_web_server")
        
    def start(self):
        """Start mock server with port conflict resolution"""
        try:
            # Check if port is available
            if self.port in self._used_ports:
                # Try to find next available port
                for new_port in range(self.port + 1, self.port + 10):
                    if new_port not in self._used_ports:
                        self.port = new_port
                        self._used_ports.add(new_port)
                        self._running = True
                        return self
                # No ports available
                return None

            # Port is available
            self._used_ports.add(self.port)
            self._running = True
            return self
        except Exception:
            return None
        
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
