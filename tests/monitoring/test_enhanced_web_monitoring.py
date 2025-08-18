#!/usr/bin/env python3
"""
test_enhanced_web_monitoring.py - Tests for Enhanced Web Monitoring Dashboard

Tests the improved web monitoring system including:
- Real-time OCR text display
- Performance charts and metrics
- Live screenshot streaming
- API endpoint enhancements
- Memory-efficient data streaming
- Multi-client support
"""

import pytest
import json
import time
import threading
import queue
import base64
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import numpy as np
from PIL import Image
import http.server
import urllib.request
import urllib.error

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the enhanced trainer system
try:
    from pokemon_crystal_rl.trainer.trainer import (
        UnifiedPokemonTrainer,
        TrainingConfig,
        TrainingMode,
        LLMBackend
    )
except ImportError:
    # Fallback import path
    from scripts.pokemon_trainer import (
        UnifiedPokemonTrainer,
        TrainingConfig,
        TrainingMode,
        LLMBackend
    )


@pytest.mark.web_monitoring
@pytest.mark.integration
class TestWebServerIntegration:
    """Test web server integration and API endpoints"""
    
    @pytest.fixture
    def web_config(self):
        """Configuration for web monitoring tests"""
        import socket
        # Find an available port
        sock = socket.socket()
        sock.bind(('', 0))
        available_port = sock.getsockname()[1]
        sock.close()
        
        return TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=available_port,
            capture_screens=True,
            headless=True,
            debug_mode=True
        )
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('trainer.web_server.HTTPServer')
    def trainer_with_web(self, mock_http_server, mock_pyboy_class, web_config):
        """Create trainer with web server enabled"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock HTTP server
        mock_server_instance = Mock()
        mock_http_server.return_value = mock_server_instance
        
        trainer = UnifiedPokemonTrainer(web_config)
        trainer.mock_server = mock_server_instance
        
        return trainer
    
    def test_web_server_initialization(self, trainer_with_web):
        """Test web server initializes correctly"""
        trainer = trainer_with_web
        
        # Web server should be initialized
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        assert trainer.screen_queue is not None
        assert trainer.capture_active is False  # Initially inactive
        
        # Verify that HTTPServer was called during initialization
        # The mock should have been called once when the web server started
        assert hasattr(trainer, 'mock_server'), "Mock server should be attached to trainer"
        
        # Alternative verification: check that the web server's server attribute is set
        assert trainer.web_server.server is not None, "Web server should have server instance"
        
        # Verify the mock server is the one we created
        assert trainer.web_server.server is trainer.mock_server, "Web server should use the mocked HTTPServer"
        
    def test_api_status_endpoint_structure(self, trainer_with_web):
        """Test API status endpoint returns correct data structure"""
        trainer = trainer_with_web
        trainer._training_active = True
        trainer.stats.update({
            'total_actions': 150,
            'llm_calls': 25,
            'start_time': time.time() - 60,  # 1 minute ago
            'mode': 'fast_monitored',
            'model': 'smollm2:1.7b'
        })
        
        # Simulate building status response (this would be called by web handler)
        expected_fields = [
            'is_training', 'mode', 'model', 'start_time', 
            'total_actions', 'llm_calls', 'actions_per_second',
            'uptime_seconds', 'current_state'
        ]
        
        # Test that trainer has the necessary data
        assert trainer._training_active is True
        assert trainer.stats['total_actions'] == 150
        assert trainer.stats['llm_calls'] == 25
        assert trainer.stats['mode'] == 'fast_monitored'
        assert trainer.stats['start_time'] > 0
        
    def test_screen_capture_queue_management(self, trainer_with_web):
        """Test screen capture queue management"""
        trainer = trainer_with_web
        
        # Test queue initialization
        assert isinstance(trainer.screen_queue, queue.Queue)
        assert trainer.screen_queue.maxsize == 30
        
        # Test screen capture and queuing
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            # Fill queue partially
            for i in range(15):
                trainer._capture_and_queue_screen()
            
            # Queue should contain screens
            assert not trainer.screen_queue.empty()
            assert trainer.screen_queue.qsize() <= 30
            
            # Test queue overflow handling
            for i in range(20):  # Add more than max size
                trainer._capture_and_queue_screen()
            
            # Queue should not exceed maximum size
            assert trainer.screen_queue.qsize() <= 30
    
    def test_screenshot_encoding_format(self, trainer_with_web):
        """Test screenshot encoding for web transfer"""
        trainer = trainer_with_web
        
        # Create test screen
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            trainer._capture_and_queue_screen()
            
            # Get queued screen data
            if not trainer.screen_queue.empty():
                screen_data = trainer.screen_queue.get()
                
                # Should contain required fields
                assert 'image_b64' in screen_data
                assert 'timestamp' in screen_data
                
                # Base64 should be valid
                image_b64 = screen_data['image_b64']
                assert isinstance(image_b64, str)
                assert len(image_b64) > 0
                
                # Should be decodable
                try:
                    image_data = base64.b64decode(image_b64)
                    assert len(image_data) > 0
                except Exception as e:
                    pytest.fail(f"Base64 decode failed: {e}")


@pytest.mark.web_monitoring
@pytest.mark.unit
class TestOCRDisplaySystem:
    """Test OCR text display in web interface"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Use random port to avoid conflicts
        import random
        port = random.randint(50000, 65000)
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=port,
            capture_screens=True,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_ocr_data_structure(self, trainer):
        """Test OCR data structure for web display"""
        # Mock OCR detection
        mock_ocr_data = {
            'detected_texts': [
                {
                    'text': 'Hello! I\'m Professor Elm!',
                    'confidence': 0.95,
                    'coordinates': [10, 20, 200, 40],
                    'text_type': 'dialogue'
                },
                {
                    'text': 'Yes',
                    'confidence': 0.98,
                    'coordinates': [20, 90, 50, 110],
                    'text_type': 'choice'
                }
            ],
            'screen_type': 'dialogue',
            'game_phase': 'dialogue_interaction',
            'timestamp': time.time()
        }
        
        # Verify OCR data structure
        assert 'detected_texts' in mock_ocr_data
        assert 'screen_type' in mock_ocr_data
        assert 'timestamp' in mock_ocr_data
        
        for text_data in mock_ocr_data['detected_texts']:
            assert 'text' in text_data
            assert 'confidence' in text_data
            assert 'coordinates' in text_data
            assert 'text_type' in text_data
            
            # Confidence should be valid
            assert 0.0 <= text_data['confidence'] <= 1.0
    
    def test_ocr_integration_with_capture(self, trainer):
        """Test OCR integration with screen capture"""
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Mock vision processor if available
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            # Mock OCR processing
            mock_ocr_result = {
                'detected_texts': [
                    {'text': 'Test dialogue', 'confidence': 0.9}
                ],
                'screen_type': 'dialogue'
            }
            
            # This would be integration point for OCR
            with patch.object(trainer, '_process_vision_ocr', return_value=mock_ocr_result) as mock_ocr:
                trainer._capture_and_queue_screen()
                
                # Verify OCR processing would be called
                # (In actual implementation, OCR would be integrated here)
                assert mock_ocr_result['detected_texts'][0]['text'] == 'Test dialogue'
    
    def test_ocr_text_filtering_and_formatting(self, trainer):
        """Test OCR text filtering and formatting for display"""
        raw_ocr_texts = [
            {'text': 'Hello! I\'m Professor Elm!', 'confidence': 0.95, 'type': 'dialogue'},
            {'text': 'asdf', 'confidence': 0.1, 'type': 'noise'},  # Low confidence
            {'text': 'Yes', 'confidence': 0.98, 'type': 'choice'},
            {'text': '', 'confidence': 0.5, 'type': 'empty'},  # Empty text
            {'text': 'No', 'confidence': 0.97, 'type': 'choice'},
        ]
        
        # Filter high-confidence, non-empty texts
        filtered_texts = [
            text for text in raw_ocr_texts 
            if text['confidence'] >= 0.5 and text['text'].strip()
        ]
        
        assert len(filtered_texts) == 3  # Should exclude low confidence and empty
        assert all(t['text'].strip() for t in filtered_texts)
        assert all(t['confidence'] >= 0.5 for t in filtered_texts)


@pytest.mark.web_monitoring
@pytest.mark.performance
class TestRealTimePerformanceCharts:
    """Test real-time performance charting system"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_with_stats(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Use random port to avoid conflicts
        import random
        random_port = random.randint(50000, 65000)
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            capture_screens=True,
            headless=True,
            web_port=random_port
        )
        
        trainer = UnifiedPokemonTrainer(config)
        
        # Initialize stats tracking
        trainer.stats.update({
            'start_time': time.time(),
            'total_actions': 0,
            'llm_calls': 0,
            'actions_per_second': 0.0,
            'mode': 'fast_monitored',
            'model': 'smollm2:1.7b'
        })
        
        return trainer
    
    def test_performance_metrics_calculation(self, trainer_with_stats):
        """Test calculation of performance metrics"""
        trainer = trainer_with_stats
        
        # Simulate training progress
        start_time = time.time() - 10  # 10 seconds ago
        trainer.stats['start_time'] = start_time
        trainer.stats['total_actions'] = 25
        trainer.stats['llm_calls'] = 8
        
        # Mock LLM manager to preserve the expected stats
        mock_llm_manager = Mock()
        mock_llm_manager.stats = {
            'llm_calls': 8,
            'llm_total_time': 0.0,
            'llm_avg_time': 0.0,
            'llm_success_rate': 0.0
        }
        trainer.llm_manager = mock_llm_manager
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        expected_aps = trainer.stats['total_actions'] / elapsed_time
        
        # Update stats (simulating what the real system does)
        trainer._update_stats()
        
        # Verify calculations
        assert trainer.stats['total_actions'] == 25
        assert trainer.stats['llm_calls'] == 8
        assert abs(trainer.stats['actions_per_second'] - expected_aps) < 0.5  # Allow some variance
    
    def test_metrics_data_structure(self, trainer_with_stats):
        """Test metrics data structure for charting"""
        trainer = trainer_with_stats
        
        # Simulate metrics over time
        metrics_history = []
        
        for i in range(10):
            trainer.stats['total_actions'] = i * 10
            trainer.stats['llm_calls'] = i * 2
            trainer._update_stats()
            
            # Simulate collecting metrics for charts
            metric_point = {
                'timestamp': time.time(),
                'actions_per_second': trainer.stats['actions_per_second'],
                'total_actions': trainer.stats['total_actions'],
                'llm_calls': trainer.stats['llm_calls'],
                'memory_usage': 150.5  # Mock memory usage in MB
            }
            
            metrics_history.append(metric_point)
        
        # Verify metrics structure
        assert len(metrics_history) == 10
        
        for metric in metrics_history:
            assert 'timestamp' in metric
            assert 'actions_per_second' in metric
            assert 'total_actions' in metric
            assert 'llm_calls' in metric
            assert isinstance(metric['actions_per_second'], (int, float))
            assert isinstance(metric['total_actions'], int)
            assert isinstance(metric['llm_calls'], int)
    
    def test_metrics_data_windowing(self, trainer_with_stats):
        """Test metrics data windowing for efficient storage"""
        trainer = trainer_with_stats
        
        # Simulate extended metrics collection
        metrics_buffer = []
        max_buffer_size = 100
        
        for i in range(150):
            metric_point = {
                'timestamp': time.time() + i,
                'actions_per_second': i * 0.1,
                'total_actions': i * 5
            }
            
            metrics_buffer.append(metric_point)
            
            # Implement windowing (keep only recent metrics)
            if len(metrics_buffer) > max_buffer_size:
                metrics_buffer = metrics_buffer[-max_buffer_size:]
        
        # Verify windowing worked
        assert len(metrics_buffer) == max_buffer_size
        assert metrics_buffer[-1]['total_actions'] == (150 - 1) * 5  # Latest data
        assert metrics_buffer[0]['total_actions'] == (150 - max_buffer_size) * 5  # Oldest in window


@pytest.mark.web_monitoring
@pytest.mark.performance
class TestMemoryEfficientStreaming:
    """Test memory-efficient data streaming for web interface"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def streaming_trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Use random port to avoid conflicts
        import random
        random_port = random.randint(50000, 65000)
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            capture_screens=True,
            headless=True,
            web_port=random_port
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_screenshot_compression_efficiency(self, streaming_trainer):
        """Test screenshot compression for efficient streaming"""
        trainer = streaming_trainer
        
        # Create test screenshot
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Test PNG compression
        image = Image.fromarray(test_screen)
        
        # Test different compression levels
        compression_results = {}
        
        for quality in [50, 75, 90]:
            buffer = BytesIO()
            if quality < 100:
                # Use JPEG for lossy compression
                image.save(buffer, format='JPEG', quality=quality, optimize=True)
            else:
                # Use PNG for lossless
                image.save(buffer, format='PNG', optimize=True)
            
            compressed_size = len(buffer.getvalue())
            compression_results[quality] = compressed_size
        
        # Lower quality should result in smaller file sizes
        assert compression_results[50] < compression_results[75]
        assert compression_results[75] < compression_results[90]
        
        # All compressed sizes should be reasonable (under 50KB)
        for size in compression_results.values():
            assert size < 50000, f"Compressed size {size} bytes too large"
    
    def test_streaming_queue_memory_bounds(self, streaming_trainer):
        """Test that streaming queues respect memory bounds"""
        trainer = streaming_trainer
        
        # Mock heavy screenshot generation
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            # Generate many screenshots
            for i in range(100):
                trainer._capture_and_queue_screen()
                
                # Queue should not grow unbounded
                assert trainer.screen_queue.qsize() <= 30
                
                # Memory usage should be bounded
                # (In real implementation, this would track actual memory usage)
    
    def test_data_streaming_rate_limiting(self, streaming_trainer):
        """Test rate limiting for data streaming"""
        trainer = streaming_trainer
        
        # Simulate rate limiting (5 FPS = 200ms intervals)
        target_fps = 5
        target_interval = 1.0 / target_fps
        
        timestamps = []
        
        # Simulate capturing at intervals
        for i in range(10):
            start_time = time.time()
            
            # Simulate capture and processing time
            time.sleep(0.01)  # 10ms processing time
            
            timestamps.append(time.time())
            
            # Rate limiting delay
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        
        # Calculate actual intervals
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        
        # Should be close to target interval
        assert abs(avg_interval - target_interval) < 0.05, f"Average interval {avg_interval:.3f}s, target {target_interval:.3f}s"


@pytest.mark.web_monitoring
@pytest.mark.integration
class TestMultiClientSupport:
    """Test multi-client support for web monitoring"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def multi_client_trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Use random port to avoid conflicts
        import random
        random_port = random.randint(50000, 65000)
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=random_port,
            capture_screens=True,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_concurrent_client_data_access(self, multi_client_trainer):
        """Test concurrent client access to monitoring data"""
        trainer = multi_client_trainer
        
        # Simulate concurrent data access
        results = {"reads": 0, "errors": 0}
        
        def client_reader(client_id):
            """Simulate a client reading monitoring data"""
            try:
                for i in range(20):
                    # Simulate reading different types of data
                    stats_data = trainer.stats.copy()
                    
                    # Simulate screen data access
                    if not trainer.screen_queue.empty():
                        try:
                            screen_data = trainer.screen_queue.get_nowait()
                            # Put it back for other clients
                            trainer.screen_queue.put_nowait(screen_data)
                        except queue.Empty:
                            pass
                    
                    results["reads"] += 1
                    time.sleep(0.001)  # Brief pause
                    
            except Exception:
                results["errors"] += 1
        
        # Run multiple concurrent clients
        clients = []
        for i in range(3):
            client = threading.Thread(target=client_reader, args=(i,))
            clients.append(client)
        
        # Start all clients
        for client in clients:
            client.start()
        
        # Wait for completion
        for client in clients:
            client.join()
        
        # Verify concurrent access worked
        assert results["errors"] == 0, f"Concurrent access had {results['errors']} errors"
        assert results["reads"] > 50, "Should have completed multiple reads per client"
    
    def test_data_consistency_across_clients(self, multi_client_trainer):
        """Test data consistency when multiple clients access the same data"""
        trainer = multi_client_trainer
        
        # Set known data
        trainer.stats.update({
            'total_actions': 100,
            'llm_calls': 20,
            'mode': 'fast_monitored'
        })
        
        # Simulate multiple clients reading the same data
        client_results = []
        
        def client_data_reader(client_id):
            """Read data and store result"""
            data = trainer.stats.copy()
            client_results.append(data)
        
        # Run clients concurrently
        clients = []
        for i in range(5):
            client = threading.Thread(target=client_data_reader, args=(i,))
            clients.append(client)
            client.start()
        
        for client in clients:
            client.join()
        
        # All clients should see the same data
        assert len(client_results) == 5
        
        first_result = client_results[0]
        for result in client_results[1:]:
            assert result['total_actions'] == first_result['total_actions']
            assert result['llm_calls'] == first_result['llm_calls']
            assert result['mode'] == first_result['mode']
    
    def test_resource_cleanup_on_client_disconnect(self, multi_client_trainer):
        """Test resource cleanup when clients disconnect"""
        trainer = multi_client_trainer
        
        # This would test proper cleanup of client resources
        # In a real implementation, this might track client connections
        
        # Simulate client connections and disconnections
        active_connections = []
        
        def simulate_client_connection():
            connection_id = f"client_{len(active_connections)}"
            active_connections.append(connection_id)
            return connection_id
        
        def simulate_client_disconnect(connection_id):
            if connection_id in active_connections:
                active_connections.remove(connection_id)
                # Cleanup resources associated with this client
                return True
            return False
        
        # Test connection management
        client1 = simulate_client_connection()
        client2 = simulate_client_connection()
        assert len(active_connections) == 2
        
        # Test disconnection
        assert simulate_client_disconnect(client1) is True
        assert len(active_connections) == 1
        assert client2 in active_connections
        
        # Test cleanup of non-existent client
        assert simulate_client_disconnect("non_existent") is False


@pytest.mark.web_monitoring
@pytest.mark.slow
class TestWebMonitoringEndToEnd:
    """End-to-end web monitoring system tests"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def e2e_trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=port,
            capture_screens=True,
            headless=True,
            max_actions=50
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_complete_monitoring_workflow(self, e2e_trainer):
        """Test complete monitoring workflow from training to web display"""
        trainer = e2e_trainer
        
        # Mock training components
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_capture.return_value = test_screen
            
            # Simulate training with monitoring
            training_data = {
                'actions_executed': 0,
                'screenshots_captured': 0,
                'stats_updates': 0
            }
            
            for step in range(20):
                # Execute action
                action = trainer._get_rule_based_action(step)
                if action:
                    trainer._execute_action(action)
                    training_data['actions_executed'] += 1
                
                # Capture screenshot periodically
                if step % 3 == 0:
                    trainer._capture_and_queue_screen()
                    training_data['screenshots_captured'] += 1
                
                # Update stats
                trainer._update_stats()
                training_data['stats_updates'] += 1
            
            # Verify training data was generated
            assert training_data['actions_executed'] > 15
            assert training_data['screenshots_captured'] >= 6
            assert training_data['stats_updates'] == 20
            
            # Verify monitoring data is available
            assert trainer.stats['total_actions'] > 0
            assert not trainer.screen_queue.empty()
            
            print(f"✅ Complete Workflow: {training_data['actions_executed']} actions, "
                  f"{training_data['screenshots_captured']} screenshots captured")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "web_monitoring"])
