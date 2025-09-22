"""Test helpers for socket server management."""
import socket
import time
import threading
from contextlib import contextmanager
import signal
import os
import logging
from typing import Optional, List, Tuple


def find_free_port(start_port: int = 8000, max_tries: int = 100) -> Optional[int]:
    """Find a free port starting from start_port.
    
    Args:
        start_port: Port number to start searching from
        max_tries: Maximum number of ports to try
        
    Returns:
        Free port number or None if no port found
    """
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port.
    
    Args:
        port: Port number to free up
    """
    import psutil
    for proc in psutil.process_iter(['pid', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    time.sleep(0.1)  # Give process time to terminate
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Process already terminated
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class SocketTestManager:
    """Manages socket servers for testing.
    
    This class helps manage socket servers in tests by:
    - Tracking active servers and ports
    - Cleaning up servers between tests 
    - Ensuring ports are freed
    """
    
    def __init__(self):
        self.active_ports: List[Tuple[int, threading.Thread]] = []
        self._lock = threading.Lock()
        
    def start_server(self, port: Optional[int] = None) -> int:
        """Start a test server.
        
        Args:
            port: Optional specific port to use
            
        Returns:
            Port number server is running on
        
        Raises:
            RuntimeError if no ports available
        """
        with self._lock:
            if port is None:
                port = find_free_port()
            if port is None:
                raise RuntimeError("No ports available")
                
            # Kill anything using the port
            kill_process_on_port(port)
            
            def dummy_server():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.listen(1)
                    while True:
                        try:
                            c, _ = s.accept()
                            c.close()
                        except:
                            break
                            
            server_thread = threading.Thread(target=dummy_server)
            server_thread.daemon = True
            server_thread.start()
            
            self.active_ports.append((port, server_thread))
            return port
            
    def stop_server(self, port: int) -> None:
        """Stop a test server.
        
        Args:
            port: Port number of server to stop
        """
        with self._lock:
            for i, (p, thread) in enumerate(self.active_ports):
                if p == port:
                    kill_process_on_port(port)
                    thread.join(timeout=0.5)
                    del self.active_ports[i]
                    break
                    
    def cleanup(self) -> None:
        """Stop all active servers."""
        with self._lock:
            for port, _ in self.active_ports:
                kill_process_on_port(port)
            self.active_ports.clear()


@contextmanager 
def temp_server(port: Optional[int] = None) -> int:
    """Context manager for temporary test server.
    
    Args:
        port: Optional specific port to use
        
    Yields:
        Port number server is running on
    """
    manager = SocketTestManager()
    try:
        p = manager.start_server(port)
        yield p
    finally:
        manager.cleanup()
