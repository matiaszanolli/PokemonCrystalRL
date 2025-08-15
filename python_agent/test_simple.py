#!/usr/bin/env python3
"""
test_simple.py - Simple standalone test for Socket.IO fallback fix

This test demonstrates the HTTP polling fallback without any external dependencies.
Perfect for testing the unified trainer scenario.
"""

import time
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class UnifiedTrainerSimulator(BaseHTTPRequestHandler):
    """Simulates the unified trainer HTTP server"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve the dashboard
            self._serve_dashboard()
        
        elif self.path.startswith('/socket.io/'):
            # Handle Socket.IO fallback (the key fix!)
            self._handle_socketio_fallback()
        
        elif self.path == '/api/status':
            # Training status endpoint
            self._serve_status()
        
        elif self.path == '/api/system':
            # System stats endpoint
            self._serve_system()
        
        elif self.path.startswith('/api/screenshot'):
            # Screenshot endpoint
            self._serve_screenshot()
        
        else:
            self.send_error(404)
    
    def _serve_dashboard(self):
        """Serve the fixed dashboard template"""
        try:
            with open('templates/dashboard.html', 'r') as f:
                html = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            print("📄 Dashboard served successfully")
            
        except FileNotFoundError:
            self.send_error(404)
            print("❌ Dashboard template not found!")
    
    def _handle_socketio_fallback(self):
        """Handle Socket.IO requests gracefully (the fix!)"""
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
        
        self.send_response(200)  # Changed from 404 to 200
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
        print("🔄 Socket.IO fallback response sent (no error!)")
    
    def _serve_status(self):
        """Serve training status (simulated data)"""
        # Simulate dynamic training data
        uptime = int(time.time()) % 3600  # Reset every hour for demo
        actions = 1000 + (uptime * 2)  # Increasing actions
        
        status = {
            'is_training': True,
            'total_actions': actions,
            'actions_per_second': 2.5 + (time.time() % 10) / 10,  # Varying speed
            'start_time': int(time.time() - uptime),
            'llm_calls': actions // 10,
            'mode': 'fast_monitored',
            'model': 'smollm2:1.7b'
        }
        
        self._send_json(status)
        print(f"📊 Status sent: {actions} actions, {status['actions_per_second']:.1f} a/s")
    
    def _serve_system(self):
        """Serve system stats (simulated data)"""
        # Simulate realistic system stats
        cpu = 45 + (time.time() % 20) - 10  # 35-65% range
        memory = 60 + (time.time() % 15) - 7  # 53-67% range
        
        system = {
            'cpu_percent': max(0, min(100, cpu)),
            'memory_percent': max(0, min(100, memory)),
            'disk_usage': 45.2,
            'gpu_available': False
        }
        
        self._send_json(system)
        print(f"💻 System stats sent: CPU {system['cpu_percent']:.1f}%, RAM {system['memory_percent']:.1f}%")
    
    def _serve_screenshot(self):
        """Serve a placeholder screenshot"""
        # Create a Game Boy sized screenshot (160x144 pixels)
        # This represents what the unified trainer would send
        
        width, height = 160, 144  # Actual Game Boy screen dimensions
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a Game Boy-like screen
            img = Image.new('RGB', (width, height), color='#9bbc0f')  # Game Boy green
            draw = ImageDraw.Draw(img)
            
            # Add some Game Boy-like graphics
            color_cycle = int(time.time()) % 6
            colors = [
                '#0f380f',  # Dark green
                '#306230',  # Medium green  
                '#8bac0f',  # Light green
                '#9bbc0f',  # Lightest green
                '#8b956d',  # Brown-ish
                '#4e4a4e'   # Gray
            ]
            
            # Draw background pattern (simulates Pokemon world)
            for y in range(0, height, 16):
                for x in range(0, width, 16):
                    color_idx = ((x // 16) + (y // 16) + color_cycle) % len(colors)
                    draw.rectangle([x, y, x+16, y+16], fill=colors[color_idx])
            
            # Draw a "player character" in the center
            player_x, player_y = width // 2 - 8, height // 2 - 8
            draw.rectangle([player_x, player_y, player_x+16, player_y+16], fill='#ff0000')  # Red player
            
            # Draw some "UI elements" at the bottom
            ui_y = height - 30
            draw.rectangle([0, ui_y, width, height], fill='#0f380f')  # Dark green UI bar
            
            # Add some text (simulates Pokemon text boxes)
            text_lines = [
                "POKEMON CRYSTAL RL",
                f"Training Active: {int(time.time()) % 9999} steps",
                "HTTP Polling Mode"
            ]
            
            try:
                # Try to use a simple font
                font = ImageFont.load_default()
                for i, line in enumerate(text_lines[:2]):  # Only show first 2 lines
                    draw.text((5, ui_y + 5 + i*10), line[:20], fill='#9bbc0f', font=font)
            except:
                # Fallback: just draw rectangles as "text"
                for i in range(3):
                    draw.rectangle([5, ui_y + 5 + i*8, 100, ui_y + 10 + i*8], fill='#9bbc0f')
            
            # Convert to PNG
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(buffer.getvalue())
            print(f"🖼️ Game Boy screenshot sent ({width}x{height}, color cycle: {color_cycle})")
            
        except ImportError:
            # Fallback: Create a simple pattern using basic graphics
            # Generate a proper sized image without PIL
            
            # Create a simple PNG programmatically (very basic)
            # For now, send a larger transparent image
            import struct
            import zlib
            
            # Create raw image data (RGB, 160x144)
            raw_data = []
            for y in range(144):
                for x in range(160):
                    # Create a simple pattern
                    if (x // 20 + y // 18) % 2 == 0:
                        raw_data.extend([0x9b, 0xbc, 0x0f])  # Game Boy green
                    else:
                        raw_data.extend([0x0f, 0x38, 0x0f])  # Dark green
            
            # Add PNG header and format (simplified)
            png_data = self._create_simple_png(160, 144, raw_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(png_data)
            print("🖼️ Fallback Game Boy screenshot sent (160x144, no PIL)")
    
    def _create_simple_png(self, width, height, rgb_data):
        """Create a simple PNG without PIL (fallback method)"""
        try:
            import struct
            import zlib
            
            # Create PNG data manually (simplified)
            # This is a very basic PNG implementation
            def write_chunk(chunk_type, data):
                chunk = struct.pack('>I', len(data))
                chunk += chunk_type
                chunk += data
                chunk += struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
                return chunk
            
            # PNG signature
            png_data = b'\x89PNG\r\n\x1a\n'
            
            # IHDR chunk
            ihdr = struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0)
            png_data += write_chunk(b'IHDR', ihdr)
            
            # Convert RGB data to bytes
            row_data = b''
            for y in range(height):
                row_data += b'\x00'  # No filter
                for x in range(width):
                    idx = (y * width + x) * 3
                    row_data += bytes(rgb_data[idx:idx+3])
            
            # IDAT chunk
            compressor = zlib.compressobj()
            compressed_data = compressor.compress(row_data)
            compressed_data += compressor.flush()
            png_data += write_chunk(b'IDAT', compressed_data)
            
            # IEND chunk
            png_data += write_chunk(b'IEND', b'')
            
            return png_data
            
        except Exception as e:
            print(f"PNG generation failed: {e}")
            # Ultimate fallback: 1x1 transparent pixel
            return base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==')
    
    def _send_json(self, data):
        """Helper to send JSON responses"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main():
    """Main test function"""
    print("🧪 Simple Socket.IO Fallback Test")
    print("=" * 50)
    print()
    print("This test simulates the unified trainer (HTTP-only server)")
    print("and demonstrates that the Socket.IO error has been fixed.")
    print()
    print("🎯 Expected behavior:")
    print("   ✅ Dashboard loads without errors")
    print("   ✅ Shows 'Connecting...' → 'HTTP Polling' status")
    print("   ✅ Updates stats every 1 second")
    print("   ✅ Updates screenshots every 2 seconds")
    print("   ✅ No browser console errors")
    print("   ✅ Memory usage stays stable")
    print()
    
    # Create and start server
    server_address = ('127.0.0.1', 8080)
    server = HTTPServer(server_address, UnifiedTrainerSimulator)
    
    print(f"🚀 Starting test server on http://{server_address[0]}:{server_address[1]}")
    print()
    print("📋 Test Instructions:")
    print("   1. Open http://127.0.0.1:8080 in your browser")
    print("   2. Watch the connection status in the top-right corner")
    print("   3. Observe that it changes from 'Connecting...' to 'HTTP Polling'")
    print("   4. Check that stats update every second")
    print("   5. Verify no errors appear in browser console")
    print("   6. Leave it running to test memory stability")
    print()
    print("🛑 Press Ctrl+C to stop the test server")
    print("=" * 50)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Test server stopped")
        print("\n📊 Test Results:")
        print("   If you saw 'HTTP Polling' status and updating stats,")
        print("   then the Socket.IO fallback fix is working correctly!")


if __name__ == "__main__":
    main()
