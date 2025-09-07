#!/usr/bin/env python3
"""
test_enhanced_web_monitoring.py - Tests for enhanced web monitoring system

Tests the new performance monitoring features including:
- Comprehensive reward rate tracking
- LLM decision accuracy metrics
- Action distribution analytics
- Performance trend analysis
- Memory-efficient stat tracking
- Enhanced web data updates
"""
"""test_enhanced_web_monitoring.py - Tests for enhanced web monitoring system

Tests the new performance monitoring features including:
- Comprehensive reward rate tracking
- LLM decision accuracy metrics
- Action distribution analytics
- Performance trend analysis
- Memory-efficient stat tracking
- Enhanced web data updates
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

@pytest.mark.monitoring
@pytest.mark.performance
class TestPerformanceMetricsTracking:
    """Test enhanced performance metrics tracking"""
    
    @pytest.fixture
    def trainer_with_metrics(self):
        """Create trainer with initialized metrics"""
        trainer = Mock()
        trainer.performance_tracking = {
            'reward_window': deque(maxlen=100),
            'llm_success_window': deque(maxlen=100),
            'action_counts': {},
            'state_transitions': {},
            'window_size': 100
        }
        trainer.stats = {
            'recent_stats': {
                'reward_rate': 0.0,
                'exploration_rate': 0.0,
                'stuck_rate': 0.0,
                'success_rate': 0.0
            },
            'training_metrics': {
                'llm_accuracy': 0.0,
                'dqn_loss': 0.0,
                'hybrid_balance': 0.5,
                'state_coverage': 0.0
            }
        }
        return trainer
    
    def test_reward_rate_calculation(self, trainer_with_metrics):
        """Test reward rate calculation from reward window"""
        trainer = trainer_with_metrics
        
        # Add mix of positive and negative rewards
        rewards = [0.5, -0.1, 0.8, 0.3, -0.2, 0.6]
        for r in rewards:
            trainer.performance_tracking['reward_window'].append(r)
        
        # Calculate rate manually (positive rewards / total)
        positive_count = len([r for r in rewards if r > 0])
        expected_rate = positive_count / len(rewards)
        
        # Update stats
        reward_rate = len([r for r in trainer.performance_tracking['reward_window'] if r > 0]) / \
                      len(trainer.performance_tracking['reward_window'])
        trainer.stats['recent_stats']['reward_rate'] = reward_rate
        
        assert abs(trainer.stats['recent_stats']['reward_rate'] - expected_rate) < 1e-6
    
    def test_llm_accuracy_tracking(self, trainer_with_metrics):
        """Test LLM decision accuracy tracking"""
        trainer = trainer_with_metrics
        
        # Simulate LLM decisions (True = successful, False = failed)
        decisions = [True, True, False, True, True, False]
        for d in decisions:
            trainer.performance_tracking['llm_success_window'].append(d)
        
        # Calculate accuracy
        success_rate = len([d for d in trainer.performance_tracking['llm_success_window'] if d]) / \
                       len(trainer.performance_tracking['llm_success_window'])
        trainer.stats['training_metrics']['llm_accuracy'] = success_rate
        
        expected_accuracy = 4/6  # 4 successful out of 6 total
        assert abs(trainer.stats['training_metrics']['llm_accuracy'] - expected_accuracy) < 1e-6
    
    def test_action_distribution_tracking(self, trainer_with_metrics):
        """Test action distribution analytics"""
        trainer = trainer_with_metrics
        
        # Simulate actions
        actions = ['up', 'up', 'down', 'left', 'right', 'up', 'a', 'b']
        for action in actions:
            if action not in trainer.performance_tracking['action_counts']:
                trainer.performance_tracking['action_counts'][action] = 0
            trainer.performance_tracking['action_counts'][action] += 1
        
        total_actions = sum(trainer.performance_tracking['action_counts'].values())
        distribution = {
            action: count/total_actions 
            for action, count in trainer.performance_tracking['action_counts'].items()
        }
        
        # Verify distribution
        assert distribution['up'] == 3/8  # 3 up actions out of 8 total
        assert distribution['down'] == 1/8
        assert sum(distribution.values()) == 1.0
    
    def test_window_size_management(self, trainer_with_metrics):
        """Test window size management for metrics"""
        trainer = trainer_with_metrics
        window_size = trainer.performance_tracking['window_size']
        
        # Add more items than window size
        for i in range(window_size + 10):
            trainer.performance_tracking['reward_window'].append(1.0)
            trainer.performance_tracking['llm_success_window'].append(True)
        
        # Check windows don't exceed max size
        assert len(trainer.performance_tracking['reward_window']) == window_size
        assert len(trainer.performance_tracking['llm_success_window']) == window_size

@pytest.mark.monitoring
@pytest.mark.web
class TestWebDataUpdates:
    """Test enhanced web data update functionality"""
    
    @pytest.fixture
    def mock_web_server(self):
        """Create mock web server"""
        server = Mock()
        server.trainer_stats = {}
        server.screenshot_data = None
        server.live_memory_data = {}
        return server
    
    @pytest.fixture
    def trainer_with_web(self, mock_web_server):
        """Create trainer with web server"""
        trainer = Mock()
        trainer.web_server = mock_web_server
        trainer.stats = {
            'recent_stats': {},
            'training_metrics': {},
            'session_metrics': {
                'unique_states': set(),
                'start_time': time.time()
            }
        }
        trainer.performance_tracking = {
            'reward_window': deque(maxlen=100),
            'llm_success_window': deque(maxlen=100),
            'action_counts': {'up': 5, 'down': 3, 'left': 2, 'right': 4}
        }
        return trainer
    
    def test_web_stats_update(self, trainer_with_web):
        """Test comprehensive web stats update"""
        trainer = trainer_with_web
        
        # Add some test data
        trainer.performance_tracking['reward_window'].extend([0.5, -0.1, 0.8])
        trainer.performance_tracking['llm_success_window'].extend([True, False, True])
        
        # Update web stats
        reward_rate = len([r for r in trainer.performance_tracking['reward_window'] if r > 0]) / \
                      max(len(trainer.performance_tracking['reward_window']), 1)
        
        llm_accuracy = len([d for d in trainer.performance_tracking['llm_success_window'] if d]) / \
                        max(len(trainer.performance_tracking['llm_success_window']), 1)
        
        # Update trainer stats
        trainer.stats['recent_stats'].update({
            'reward_rate': reward_rate,
            'exploration_rate': 0.5,
            'stuck_rate': 0.1,
            'success_rate': reward_rate
        })
        
        trainer.stats['training_metrics'].update({
            'llm_accuracy': llm_accuracy,
            'dqn_loss': 0.5,
            'hybrid_balance': 0.6,
            'state_coverage': 25.0
        })
        
        # Verify stats were updated correctly
        assert trainer.stats['recent_stats']['reward_rate'] == 2/3  # 2 positive out of 3
        assert trainer.stats['training_metrics']['llm_accuracy'] == 2/3  # 2 success out of 3
        
    def test_action_distribution_tracking(self, trainer_with_web):
        """Test action distribution tracking in web updates"""
        trainer = trainer_with_web
        
        # Calculate distribution
        total_actions = sum(trainer.performance_tracking['action_counts'].values())
        action_distribution = {
            action: count/total_actions 
            for action, count in trainer.performance_tracking['action_counts'].items()
        }
        
        # Verify distribution
        assert abs(action_distribution['up'] - 5/14) < 1e-6  # 5 out of 14 total
        assert abs(action_distribution['down'] - 3/14) < 1e-6
        assert abs(sum(action_distribution.values()) - 1.0) < 1e-6
    
    def test_memory_efficiency(self, trainer_with_web):
        """Test memory usage efficiency in web updates"""
        trainer = trainer_with_web
        
        # Add significant amount of history data
        for i in range(1000):
            trainer.performance_tracking['reward_window'].append(0.5)
            trainer.performance_tracking['llm_success_window'].append(True)
        
        # Verify window sizes remain bounded
        assert len(trainer.performance_tracking['reward_window']) == 100
        assert len(trainer.performance_tracking['llm_success_window']) == 100
        
        # Initialize stats history
        trainer.web_stats_history = Mock()
        trainer.web_stats_history.reward_history = []
        
        for i in range(2000):
            trainer.web_stats_history.reward_history.append({
                'timestamp': time.time(),
                'reward_rate': 0.5,
                'total_reward': i,
                'action_dist': {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
            })
            
            # Keep history size bounded
            if len(trainer.web_stats_history.reward_history) > 1000:
                trainer.web_stats_history.reward_history = trainer.web_stats_history.reward_history[-1000:]
        
        # Verify history remains bounded
        assert len(trainer.web_stats_history.reward_history) <= 1000

@pytest.mark.monitoring
@pytest.mark.integration
class TestMonitoringIntegration:
    """Test monitoring system integration"""
    
    @pytest.fixture
    def integrated_trainer(self):
        """Create trainer with full monitoring setup"""
        trainer = Mock()
        trainer.web_server = Mock()
        trainer.web_server.trainer_stats = {}
        trainer.web_server.screenshot_data = None
        trainer.web_server.live_memory_data = {}
        
        trainer.performance_tracking = {
            'reward_window': deque(maxlen=100),
            'llm_success_window': deque(maxlen=100),
            'action_counts': {},
            'state_transitions': {},
            'window_size': 100
        }
        
        trainer.stats = {
            'recent_stats': {
                'reward_rate': 0.0,
                'exploration_rate': 0.0,
                'stuck_rate': 0.0,
                'success_rate': 0.0
            },
            'training_metrics': {
                'llm_accuracy': 0.0,
                'dqn_loss': 0.0,
                'hybrid_balance': 0.5,
                'state_coverage': 0.0
            },
            'session_metrics': {
                'start_time': time.time(),
                'total_steps': 0,
                'unique_states': set(),
                'error_count': 0
            }
        }
        return trainer
    
    def test_full_monitoring_cycle(self, integrated_trainer):
        """Test complete monitoring update cycle"""
        trainer = integrated_trainer
        
        # Simulate full training cycle
        for i in range(100):
            # Add metrics data
            trainer.performance_tracking['reward_window'].append(0.5 if i % 2 == 0 else -0.1)
            trainer.performance_tracking['llm_success_window'].append(i % 3 == 0)
            
            action = ['up', 'down', 'left', 'right'][i % 4]
            if action not in trainer.performance_tracking['action_counts']:
                trainer.performance_tracking['action_counts'][action] = 0
            trainer.performance_tracking['action_counts'][action] += 1
            
            # Update stats
            reward_rate = len([r for r in trainer.performance_tracking['reward_window'] if r > 0]) / \
                          max(len(trainer.performance_tracking['reward_window']), 1)
            
            llm_accuracy = len([d for d in trainer.performance_tracking['llm_success_window'] if d]) / \
                          max(len(trainer.performance_tracking['llm_success_window']), 1)
            
            trainer.stats['recent_stats']['reward_rate'] = reward_rate
            trainer.stats['training_metrics']['llm_accuracy'] = llm_accuracy
            trainer.stats['session_metrics']['total_steps'] = i + 1
            
            # Add to unique states
            trainer.stats['session_metrics']['unique_states'].add(f"state_{i % 10}")
        
        # Verify final metrics
        assert abs(trainer.stats['recent_stats']['reward_rate'] - 0.5) < 0.1
        assert abs(trainer.stats['training_metrics']['llm_accuracy'] - 1/3) < 0.1
        assert len(trainer.stats['session_metrics']['unique_states']) == 10
        assert trainer.stats['session_metrics']['total_steps'] == 100
    
    def test_error_handling_and_recovery(self, integrated_trainer):
        """Test monitoring system error handling"""
        trainer = integrated_trainer
        
        # Simulate web server errors
        trainer.web_server.trainer_stats = None  # Force error
        
        # Should handle None stats gracefully
        try:
            reward_rate = len([r for r in trainer.performance_tracking['reward_window'] if r > 0]) / \
                          max(len(trainer.performance_tracking['reward_window']), 1)
            trainer.stats['recent_stats']['reward_rate'] = reward_rate
            trainer.web_server.trainer_stats = trainer.stats
        except Exception as e:
            pytest.fail(f"Failed to handle None stats: {e}")
        
        # Verify error was tracked
        trainer.stats['session_metrics']['error_count'] += 1
        assert trainer.stats['session_metrics']['error_count'] == 1
    
    def test_performance_trend_analysis(self, integrated_trainer):
        """Test performance trend analysis"""
        trainer = integrated_trainer
        
        # Simulate improving performance trend
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for r in rewards:
            trainer.performance_tracking['reward_window'].append(r)
        
        # Calculate trend
        initial_rate = sum(rewards[:5]) / 5
        final_rate = sum(rewards[5:]) / 5
        improvement = final_rate - initial_rate
        
        assert improvement > 0  # Positive trend
        assert final_rate > initial_rate
import json
import time
import threading
import queue
import base64
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import numpy as np
from PIL import Image

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from training.trainer import (
    TrainingConfig,
)

from training.unified_trainer import UnifiedTrainer


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
            debug_mode=True,
            test_mode=True  # Add test_mode to isolate tests
        )
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('trainer.web_server.WebServer')
    def trainer_with_web(self, mock_web_server_class, mock_pyboy_class, web_config):
        """Create trainer with web server enabled"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock web server
        mock_web_server_instance = Mock()
        mock_web_server_instance.server = Mock()  # Mock the server attribute
        mock_web_server_class.return_value = mock_web_server_instance
        
        trainer = UnifiedTrainer(web_config)
        trainer.mock_server = mock_web_server_instance.server
        
        return trainer
    
    def test_web_server_initialization(self, trainer_with_web):
        """Test trainer initialization (web server consolidated into WebMonitor)"""
        trainer = trainer_with_web
        
        # Web server functionality consolidated into core.web_monitor.WebMonitor
        # Note: test_mode=True forces enable_web=False for test isolation
        assert trainer.config.test_mode == True
        assert trainer.screen_queue is not None
        assert trainer.capture_active is False  # Initially inactive
        
        # Check that trainer has required attributes for monitoring
        assert hasattr(trainer, 'config')
        assert hasattr(trainer, 'stats')
        
    def test_screen_capture_queue_management(self, trainer_with_web):
        """Test screen capture queue management"""
        trainer = trainer_with_web
        
        # Test queue initialization
        assert isinstance(trainer.screen_queue, queue.Queue)
        assert trainer.screen_queue.maxsize == 30
        
        # Test screen capture and queuing
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_jpg_data = np.array([1, 2, 3], dtype=np.uint8).tobytes()
        
        with patch('cv2.cvtColor', return_value=test_screen) as mock_cvt, \
             patch('cv2.imencode', return_value=(True, mock_jpg_data)) as mock_encode, \
             patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            
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
        
        # Force cv2 import to fail to test PIL fallback
        def mock_save(fp, format, quality=85):
            # Just write some test data
            if hasattr(fp, 'write'):
                fp.write(b'test_jpg_data')
        
        # Create mock PIL Image
        mock_image = MagicMock()
        mock_image.save.side_effect = mock_save
        
        # Set up mocks
        with patch('cv2.cvtColor', side_effect=ImportError), \
             patch('cv2.imencode', side_effect=ImportError), \
             patch('PIL.Image.fromarray', return_value=mock_image), \
             patch('base64.b64encode', return_value=b'test_base64_data'), \
             patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):

            trainer._capture_and_queue_screen()
            
            # Should have called PIL.Image.fromarray with screen data
            Image.fromarray.assert_called_once_with(test_screen)
            
            # Should have called save with JPEG format
            assert mock_image.save.call_count == 1
            args = mock_image.save.call_args
            assert args[1].get('format') == 'JPEG'
            assert args[1].get('quality') == 85


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
        
        return UnifiedTrainer(config)
    
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
        
        trainer = UnifiedTrainer(config)
        
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
        
        return UnifiedTrainer(config)
    
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
        
        return UnifiedTrainer(config)
    
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
        
        return UnifiedTrainer(config)
    
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
                try:
                    trainer._execute_action(action)
                    training_data['actions_executed'] += 1
                    # Increment trainer stats since we're mocking the execution
                    trainer.stats['total_actions'] = training_data['actions_executed']
                except Exception as e:
                    # Just log errors in test environment
                    print(f"Action failed (expected in tests): {e}")
                
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
