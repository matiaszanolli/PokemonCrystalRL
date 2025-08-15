#!/usr/bin/env python3
"""
test_socket_fix.py - Demonstrate the fixed Socket.IO fallback behavior

This script shows how the dashboard now gracefully handles both:
1. Servers with Socket.IO support (like advanced_web_monitor.py)  
2. Servers without Socket.IO support (like the unified trainer)
"""

import time
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# Optional imports - gracefully handle missing dependencies
try:
    from monitoring.advanced_web_monitor import AdvancedWebMonitor
    ADVANCED_MONITOR_AVAILABLE = True
except ImportError:
    ADVANCED_MONITOR_AVAILABLE = False
    print("⚠️ Advanced monitor not available (missing dependencies)")


def test_http_only_server():
    """Test server without Socket.IO (like unified trainer)"""
    
    class HTTPOnlyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                # Serve the fixed dashboard
                try:
                    with open('templates/dashboard.html', 'r') as f:
                        html = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode())
                except FileNotFoundError:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Dashboard template not found')
            
            elif self.path.startswith('/socket.io/'):
                # This is the key fix - graceful Socket.IO fallback
                response = {
                    'error': 'WebSocket/Socket.IO not implemented',
                    'message': 'This trainer uses HTTP polling instead of WebSockets',
                    'use_polling': True,
                    'polling_endpoints': {
                        'status': '/api/status',
                        'system': '/api/system',
                        'screenshot': '/api/screenshot'
                    }
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            elif self.path == '/api/status':
                status = {
                    'is_training': True,
                    'total_actions': 1234,
                    'actions_per_second': 12.5,
                    'start_time': int(time.time() - 300),  # Started 5 minutes ago
                    'llm_calls': 123
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            
            elif self.path == '/api/system':
                system = {
                    'cpu_percent': 45.2,
                    'memory_percent': 62.1
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(system).encode())
            
            elif self.path == '/api/screenshot':
                # Return a simple placeholder image
                import base64
                # 1x1 pixel transparent PNG
                pixel_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==')
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(pixel_png)
            
            else:
                self.send_response(404)
                self.end_headers()
    
    return HTTPServer(('127.0.0.1', 8080), HTTPOnlyHandler)


def test_socketio_server():
    """Test server with Socket.IO support"""
    return AdvancedWebMonitor(host='127.0.0.1', port=5000)


def main():
    print("🧪 Testing Socket.IO Fallback Fix")
    print("=" * 50)
    
    print("\n📋 Test Options:")
    print("1. Test HTTP-only server (like unified trainer)")
    print("2. Test Socket.IO server (like advanced monitor)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🌐 Starting HTTP-only server on http://127.0.0.1:8080")
        print("📊 This simulates the unified trainer behavior")
        print("🎯 The dashboard should:")
        print("   - Show 'Connecting...' initially")
        print("   - Try Socket.IO connection first")
        print("   - Detect connection failure/error immediately")
        print("   - Fall back to HTTP polling gracefully")
        print("   - Show 'HTTP Polling' connection status")
        print("   - Update stats via REST API calls every 1s")
        print("   - Update screenshots via REST API every 2s")
        print("   - Handle connection errors gracefully")
        print()
        
        server = test_http_only_server()
        print("✅ Server started! Open http://127.0.0.1:8080 in your browser")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 HTTP-only server stopped")
    
    elif choice == "2":
        if not ADVANCED_MONITOR_AVAILABLE:
            print("\n❌ Socket.IO server test not available")
            print("📊 Missing dependencies for advanced monitor")
            print("\n💡 To test Socket.IO functionality:")
            print("   1. Install missing dependencies (Flask-SocketIO, etc.)")
            print("   2. Or use option 1 to test HTTP polling fallback")
            return
        
        print("\n🌐 Starting Socket.IO server on http://127.0.0.1:5000")
        print("📊 This simulates the advanced monitor behavior")
        print("🎯 The dashboard should:")
        print("   - Connect to Socket.IO successfully")
        print("   - Show 'Connected' status")
        print("   - Receive real-time updates via WebSocket")
        print()
        
        try:
            monitor = test_socketio_server()
            print("✅ Server started! Open http://127.0.0.1:5000 in your browser")
            print("🛑 Press Ctrl+C to stop")
            monitor.run(debug=False)
        except KeyboardInterrupt:
            print("\n🛑 Socket.IO server stopped")
        except Exception as e:
            print(f"\n❌ Error starting Socket.IO server: {e}")
            print("💡 Try option 1 to test HTTP polling instead")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print(f"❌ Invalid choice: {choice}")


if __name__ == "__main__":
    main()
