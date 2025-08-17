#!/usr/bin/env python3
"""
Debug script for socket connection and emulator output issues.

This script will help identify:
1. Socket connection stability
2. Screenshot capture consistency
3. Data transfer reliability
4. PyBoy emulator state
"""

import time
import threading
import numpy as np
import base64
import io
import json
from datetime import datetime
from PIL import Image
import cv2

# Import system components
from monitoring.web_monitor import PokemonRLWebMonitor, create_dashboard_templates
from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge
from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend


class SocketEmulatorDebugger:
    """Comprehensive debugger for socket and emulator issues"""
    
    def __init__(self):
        self.debug_stats = {
            'socket_connections': 0,
            'socket_disconnections': 0,
            'screenshots_sent': 0,
            'screenshots_received': 0,
            'blank_screenshots': 0,
            'valid_screenshots': 0,
            'bridge_errors': 0,
            'emulator_errors': 0
        }
        
        self.connected_clients = set()
        self.screenshot_history = []
        self.error_log = []
        
    def log_event(self, event_type: str, message: str, data=None):
        """Log debugging events with timestamp"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'type': event_type,
            'message': message,
            'data': data
        }
        self.error_log.append(log_entry)
        print(f"[{timestamp}] {event_type.upper()}: {message}")
        
        if data and isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
    
    def analyze_screenshot(self, screenshot_data) -> dict:
        """Analyze screenshot data for issues"""
        analysis = {
            'is_valid': False,
            'is_blank': False,
            'size': 0,
            'dimensions': None,
            'color_variance': 0,
            'issues': []
        }
        
        try:
            if isinstance(screenshot_data, dict):
                # Web monitor format
                if 'image' in screenshot_data:
                    img_data = screenshot_data['image']
                    if img_data.startswith('data:image/png;base64,'):
                        b64_data = img_data[22:]  # Remove data URL prefix
                        img_bytes = base64.b64decode(b64_data)
                        analysis['size'] = len(img_bytes)
                        
                        # Decode image to analyze content
                        img = Image.open(io.BytesIO(img_bytes))
                        img_array = np.array(img)
                        analysis['dimensions'] = img_array.shape
                        analysis['color_variance'] = np.var(img_array)
                        analysis['is_valid'] = True
                        
                        # Check if blank (low variance)
                        if analysis['color_variance'] < 10:
                            analysis['is_blank'] = True
                            analysis['issues'].append('Low color variance - possibly blank')
                    else:
                        analysis['issues'].append('Invalid base64 data URL format')
                else:
                    analysis['issues'].append('No image data in screenshot object')
                    
            elif isinstance(screenshot_data, np.ndarray):
                # Direct numpy array
                analysis['dimensions'] = screenshot_data.shape
                analysis['color_variance'] = np.var(screenshot_data)
                analysis['is_valid'] = True
                analysis['size'] = screenshot_data.nbytes
                
                if analysis['color_variance'] < 10:
                    analysis['is_blank'] = True
                    analysis['issues'].append('Low color variance - possibly blank')
                    
            else:
                analysis['issues'].append(f'Unknown screenshot data type: {type(screenshot_data)}')
                
        except Exception as e:
            analysis['issues'].append(f'Analysis error: {str(e)}')
            
        return analysis
    
    def test_emulator_screenshot_capture(self, config: TrainingConfig, num_frames: int = 20):
        """Test emulator screenshot capture directly"""
        self.log_event('test', f'Testing emulator screenshot capture for {num_frames} frames')
        
        try:
            # Create trainer
            trainer = UnifiedPokemonTrainer(config)
            
            # Test screenshot capture loop
            valid_frames = 0
            blank_frames = 0
            
            for i in range(num_frames):
                self.log_event('capture', f'Capturing frame {i+1}/{num_frames}')
                
                try:
                    # Try to get screenshot from trainer
                    if hasattr(trainer, 'latest_screen') and trainer.latest_screen:
                        screenshot = trainer.latest_screen
                        analysis = self.analyze_screenshot(screenshot)
                        
                        if analysis['is_valid']:
                            valid_frames += 1
                            if analysis['is_blank']:
                                blank_frames += 1
                                self.log_event('warning', f'Frame {i+1} is blank', analysis)
                            else:
                                self.log_event('success', f'Frame {i+1} valid', {
                                    'size': analysis['size'],
                                    'variance': analysis['color_variance']
                                })
                        else:
                            self.log_event('error', f'Frame {i+1} invalid', {'issues': analysis['issues']})
                    else:
                        self.log_event('error', f'Frame {i+1}: No screenshot available from trainer')
                    
                    # Wait for next frame
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.log_event('error', f'Frame {i+1} capture failed: {str(e)}')
            
            self.log_event('summary', f'Emulator test complete', {
                'total_frames': num_frames,
                'valid_frames': valid_frames,
                'blank_frames': blank_frames,
                'error_frames': num_frames - valid_frames
            })
            
            return {
                'total_frames': num_frames,
                'valid_frames': valid_frames,
                'blank_frames': blank_frames,
                'success_rate': valid_frames / num_frames if num_frames > 0 else 0
            }
            
        except Exception as e:
            self.log_event('error', f'Emulator test setup failed: {str(e)}')
            return None
    
    def test_socket_streaming(self, config: TrainingConfig, duration: int = 30):
        """Test socket streaming with web monitor"""
        self.log_event('test', f'Testing socket streaming for {duration} seconds')
        
        try:
            # Create trainer and web monitor
            trainer = UnifiedPokemonTrainer(config)
            web_monitor = PokemonRLWebMonitor(training_session=None)
            
            # Create bridge
            bridge = TrainerWebMonitorBridge(trainer, web_monitor)
            
            # Hook into socket events
            original_emit = web_monitor.socketio.emit
            
            def debug_emit(event, data, *args, **kwargs):
                if event == 'screenshot':
                    self.debug_stats['screenshots_sent'] += 1
                    analysis = self.analyze_screenshot(data)
                    
                    if analysis['is_valid']:
                        self.debug_stats['valid_screenshots'] += 1
                        if analysis['is_blank']:
                            self.debug_stats['blank_screenshots'] += 1
                            self.log_event('warning', 'Blank screenshot sent via socket', analysis)
                        else:
                            self.log_event('debug', f'Valid screenshot sent', {
                                'size': analysis['size'],
                                'variance': analysis['color_variance']
                            })
                    else:
                        self.log_event('error', 'Invalid screenshot sent via socket', analysis)
                
                return original_emit(event, data, *args, **kwargs)
            
            web_monitor.socketio.emit = debug_emit
            
            # Start monitoring components
            web_monitor.start_monitoring()
            bridge.start_bridge()
            
            # Monitor for specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(1)
                
                # Log current stats
                if int(time.time() - start_time) % 10 == 0:  # Every 10 seconds
                    self.log_event('status', 'Streaming status', self.debug_stats.copy())
            
            # Stop components
            bridge.stop_bridge()
            web_monitor.stop_monitoring()
            
            self.log_event('summary', 'Socket streaming test complete', self.debug_stats)
            
            return self.debug_stats.copy()
            
        except Exception as e:
            self.log_event('error', f'Socket streaming test failed: {str(e)}')
            return None
    
    def test_pyboy_screen_methods(self, rom_path: str):
        """Test PyBoy screen access methods"""
        self.log_event('test', 'Testing PyBoy screen methods')
        
        try:
            from pyboy import PyBoy
            
            # Initialize PyBoy
            pyboy = PyBoy(rom_path, window="null", debug=False)
            
            # Let game boot
            for _ in range(10):
                pyboy.tick()
            
            # Test different screen access methods
            methods_tested = []
            
            # Method 1: screen.ndarray
            try:
                if hasattr(pyboy.screen, 'ndarray'):
                    screen_data = pyboy.screen.ndarray
                    analysis = self.analyze_screenshot(screen_data)
                    methods_tested.append({
                        'method': 'screen.ndarray',
                        'success': True,
                        'analysis': analysis
                    })
                    self.log_event('success', 'screen.ndarray works', analysis)
                else:
                    methods_tested.append({
                        'method': 'screen.ndarray',
                        'success': False,
                        'error': 'Method not available'
                    })
                    self.log_event('error', 'screen.ndarray not available')
            except Exception as e:
                methods_tested.append({
                    'method': 'screen.ndarray',
                    'success': False,
                    'error': str(e)
                })
                self.log_event('error', f'screen.ndarray failed: {str(e)}')
            
            # Method 2: screen.image
            try:
                if hasattr(pyboy.screen, 'image'):
                    screen_image = pyboy.screen.image()
                    screen_array = np.array(screen_image)
                    analysis = self.analyze_screenshot(screen_array)
                    methods_tested.append({
                        'method': 'screen.image',
                        'success': True,
                        'analysis': analysis
                    })
                    self.log_event('success', 'screen.image works', analysis)
                else:
                    methods_tested.append({
                        'method': 'screen.image',
                        'success': False,
                        'error': 'Method not available'
                    })
                    self.log_event('error', 'screen.image not available')
            except Exception as e:
                methods_tested.append({
                    'method': 'screen.image',
                    'success': False,
                    'error': str(e)
                })
                self.log_event('error', f'screen.image failed: {str(e)}')
            
            # Test continuous capture
            self.log_event('test', 'Testing continuous capture')
            successful_frames = 0
            blank_frames = 0
            
            for i in range(10):
                pyboy.tick()
                
                try:
                    if hasattr(pyboy.screen, 'ndarray'):
                        screen_data = pyboy.screen.ndarray.copy()
                        analysis = self.analyze_screenshot(screen_data)
                        
                        if analysis['is_valid']:
                            successful_frames += 1
                            if analysis['is_blank']:
                                blank_frames += 1
                        
                        if i % 3 == 0:  # Log every 3rd frame
                            self.log_event('debug', f'Continuous frame {i}', analysis)
                    
                except Exception as e:
                    self.log_event('error', f'Continuous frame {i} failed: {str(e)}')
                
                time.sleep(0.1)
            
            pyboy.stop()
            
            results = {
                'methods_tested': methods_tested,
                'continuous_test': {
                    'total_frames': 10,
                    'successful_frames': successful_frames,
                    'blank_frames': blank_frames
                }
            }
            
            self.log_event('summary', 'PyBoy screen methods test complete', results)
            return results
            
        except Exception as e:
            self.log_event('error', f'PyBoy screen methods test failed: {str(e)}')
            return None
    
    def run_comprehensive_debug(self, rom_path: str):
        """Run comprehensive debugging session"""
        self.log_event('start', 'Starting comprehensive socket/emulator debugging')
        
        # Create test configuration
        config = TrainingConfig(
            rom_path=rom_path,
            mode=TrainingMode.FAST_MONITORED,
            llm_backend=LLMBackend.NONE,
            max_actions=100,
            capture_screens=True,
            capture_fps=2,
            enable_web=False,
            headless=True,
            debug_mode=True
        )
        
        results = {}
        
        # Test 1: PyBoy screen methods
        self.log_event('phase', 'Phase 1: Testing PyBoy screen access methods')
        results['pyboy_test'] = self.test_pyboy_screen_methods(rom_path)
        time.sleep(2)
        
        # Test 2: Emulator screenshot capture
        self.log_event('phase', 'Phase 2: Testing emulator screenshot capture')
        results['emulator_test'] = self.test_emulator_screenshot_capture(config, num_frames=10)
        time.sleep(2)
        
        # Test 3: Socket streaming
        self.log_event('phase', 'Phase 3: Testing socket streaming')
        results['socket_test'] = self.test_socket_streaming(config, duration=20)
        
        # Generate report
        self.log_event('complete', 'Debugging complete - generating report')
        self.generate_debug_report(results)
        
        return results
    
    def generate_debug_report(self, results: dict):
        """Generate comprehensive debug report"""
        print("\n" + "="*80)
        print("🔍 SOCKET & EMULATOR DEBUG REPORT")
        print("="*80)
        
        # PyBoy Test Results
        if results.get('pyboy_test'):
            pyboy_results = results['pyboy_test']
            print("\n📱 PYBOY SCREEN ACCESS:")
            print("-" * 40)
            
            for method in pyboy_results.get('methods_tested', []):
                status = "✅ WORKS" if method['success'] else "❌ FAILED"
                print(f"  {method['method']}: {status}")
                if not method['success']:
                    print(f"    Error: {method['error']}")
                elif 'analysis' in method:
                    analysis = method['analysis']
                    print(f"    Dimensions: {analysis['dimensions']}")
                    print(f"    Color variance: {analysis['color_variance']:.2f}")
                    print(f"    Is blank: {analysis['is_blank']}")
        
        # Emulator Test Results
        if results.get('emulator_test'):
            emu_results = results['emulator_test']
            print("\n🎮 EMULATOR SCREENSHOT CAPTURE:")
            print("-" * 40)
            print(f"  Success rate: {emu_results['success_rate']*100:.1f}%")
            print(f"  Valid frames: {emu_results['valid_frames']}/{emu_results['total_frames']}")
            print(f"  Blank frames: {emu_results['blank_frames']}")
            
            if emu_results['success_rate'] < 0.5:
                print("  ⚠️  LOW SUCCESS RATE - Emulator may have issues")
        
        # Socket Test Results
        if results.get('socket_test'):
            socket_results = results['socket_test']
            print("\n🔌 SOCKET STREAMING:")
            print("-" * 40)
            print(f"  Screenshots sent: {socket_results['screenshots_sent']}")
            print(f"  Valid screenshots: {socket_results['valid_screenshots']}")
            print(f"  Blank screenshots: {socket_results['blank_screenshots']}")
            print(f"  Bridge errors: {socket_results['bridge_errors']}")
            
            if socket_results['blank_screenshots'] > socket_results['valid_screenshots'] / 2:
                print("  ⚠️  HIGH BLANK RATE - Check emulator output")
        
        # Issue Analysis
        print("\n🔍 ISSUE ANALYSIS:")
        print("-" * 40)
        
        issues_found = []
        
        # Check for common issues
        if results.get('socket_test', {}).get('blank_screenshots', 0) > 3:
            issues_found.append("High blank screenshot rate - emulator may not be generating valid frames")
        
        if results.get('emulator_test', {}).get('success_rate', 0) < 0.8:
            issues_found.append("Low emulator capture success rate - check PyBoy initialization")
        
        if results.get('socket_test', {}).get('screenshots_sent', 0) < 5:
            issues_found.append("Few screenshots sent - bridge may not be transferring data")
        
        if not issues_found:
            print("  ✅ No major issues detected")
        else:
            for issue in issues_found:
                print(f"  ❌ {issue}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        recommendations = []
        
        if results.get('socket_test', {}).get('blank_screenshots', 0) > 2:
            recommendations.extend([
                "1. Check if ROM file is valid and loads properly",
                "2. Increase bridge update interval to give emulator more time",
                "3. Add frame validation before sending to socket"
            ])
        
        if results.get('emulator_test', {}).get('success_rate', 0) < 0.5:
            recommendations.extend([
                "4. Verify PyBoy version compatibility",
                "5. Try different PyBoy screen access methods",
                "6. Check if emulator is properly initialized before capture"
            ])
        
        if not recommendations:
            recommendations.append("System appears to be working correctly")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        # Error Log Summary
        error_count = len([log for log in self.error_log if log['type'] == 'error'])
        warning_count = len([log for log in self.error_log if log['type'] == 'warning'])
        
        print(f"\n📋 DEBUG SESSION SUMMARY:")
        print("-" * 40)
        print(f"  Total events logged: {len(self.error_log)}")
        print(f"  Errors: {error_count}")
        print(f"  Warnings: {warning_count}")
        
        print("\n" + "="*80)


def main():
    """Main debug function"""
    print("🔍 Socket Connection & Emulator Output Debugger")
    print("=" * 60)
    print()
    
    # Find ROM file
    import os
    from pathlib import Path
    
    rom_locations = [
        "../roms/pokemon_crystal.gbc",
        "roms/pokemon_crystal.gbc", 
        "pokemon_crystal.gbc",
        "test.gbc"
    ]
    
    rom_path = None
    for location in rom_locations:
        if Path(location).exists():
            rom_path = location
            break
    
    if not rom_path:
        print("❌ No ROM file found. Please place pokemon_crystal.gbc in one of these locations:")
        for location in rom_locations:
            print(f"   - {location}")
        return
    
    print(f"🎮 Using ROM: {rom_path}")
    print()
    
    # Create dashboard templates
    print("📄 Creating dashboard templates...")
    create_dashboard_templates()
    
    # Run debugging
    debugger = SocketEmulatorDebugger()
    results = debugger.run_comprehensive_debug(rom_path)
    
    # Save debug log
    debug_log_file = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(debug_log_file, 'w') as f:
        json.dump({
            'debug_stats': debugger.debug_stats,
            'error_log': debugger.error_log,
            'test_results': results
        }, f, indent=2)
    
    print(f"\n📝 Debug log saved to: {debug_log_file}")
    print("\n🎯 Debug session complete!")


if __name__ == "__main__":
    main()
