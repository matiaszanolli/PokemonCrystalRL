#!/usr/bin/env python3
"""
Test Suite for Configuration Module

Tests the unified configuration system for Pokemon Crystal RL including
TrainingConfig, MonitorConfig, VisionConfig, SystemConfig, and UnifiedConfig.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from config.config import (
    TrainingMode, LLMBackend, LogLevel,
    TrainingConfig, MonitorConfig, VisionConfig, SystemConfig, UnifiedConfig
)


class TestTrainingMode:
    """Test TrainingMode enum functionality."""

    def test_training_mode_values(self):
        """Test that all training mode enums have correct values."""
        assert TrainingMode.FAST.value == 'fast'
        assert TrainingMode.LLM.value == 'llm'
        assert TrainingMode.CURRICULUM.value == 'curriculum'
        assert TrainingMode.SYNCHRONIZED.value == 'synchronized'

    def test_training_mode_completeness(self):
        """Test that all expected training modes are defined."""
        expected_modes = ['fast', 'llm', 'curriculum', 'synchronized']
        actual_modes = [mode.value for mode in TrainingMode]
        assert set(actual_modes) == set(expected_modes)


class TestLLMBackend:
    """Test LLMBackend enum functionality."""

    def test_llm_backend_values(self):
        """Test that all LLM backend enums have correct values."""
        assert LLMBackend.NONE.value == 'none'
        assert LLMBackend.OLLAMA.value == 'ollama'
        assert LLMBackend.OPENAI.value == 'openai'
        assert LLMBackend.ANTHROPIC.value == 'anthropic'

    def test_llm_backend_completeness(self):
        """Test that all expected LLM backends are defined."""
        expected_backends = ['none', 'ollama', 'openai', 'anthropic']
        actual_backends = [backend.value for backend in LLMBackend]
        assert set(actual_backends) == set(expected_backends)


class TestLogLevel:
    """Test LogLevel enum functionality."""

    def test_log_level_values(self):
        """Test that all log level enums have correct values."""
        assert LogLevel.DEBUG.value == 'DEBUG'
        assert LogLevel.INFO.value == 'INFO'
        assert LogLevel.WARNING.value == 'WARNING'
        assert LogLevel.ERROR.value == 'ERROR'

    def test_log_level_completeness(self):
        """Test that all expected log levels are defined."""
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        actual_levels = [level.value for level in LogLevel]
        assert set(actual_levels) == set(expected_levels)


class TestTrainingConfig:
    """Test TrainingConfig dataclass functionality."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.mode == TrainingMode.FAST
        assert config.max_episodes == 1000
        assert config.max_actions == 10000
        assert config.target_fps == 30
        assert config.llm_backend is None
        assert config.llm_model == 'smollm2:1.7b'
        assert config.llm_interval == 10
        assert config.curriculum_stages == 5
        assert config.curriculum_step_size == 100
        assert config.batch_size == 32
        assert config.memory_limit == 2048
        assert config.device == 'cpu'

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            mode=TrainingMode.LLM,
            max_episodes=500,
            max_actions=5000,
            target_fps=60,
            llm_backend=LLMBackend.OLLAMA,
            llm_model='llama2:7b',
            llm_interval=20,
            curriculum_stages=10,
            curriculum_step_size=200,
            batch_size=64,
            memory_limit=4096,
            device='cuda'
        )

        assert config.mode == TrainingMode.LLM
        assert config.max_episodes == 500
        assert config.max_actions == 5000
        assert config.target_fps == 60
        assert config.llm_backend == LLMBackend.OLLAMA
        assert config.llm_model == 'llama2:7b'
        assert config.llm_interval == 20
        assert config.curriculum_stages == 10
        assert config.curriculum_step_size == 200
        assert config.batch_size == 64
        assert config.memory_limit == 4096
        assert config.device == 'cuda'

    def test_training_config_validation(self):
        """Test TrainingConfig field validation."""
        # Test that all fields are correctly typed
        config = TrainingConfig()
        assert isinstance(config.mode, TrainingMode)
        assert isinstance(config.max_episodes, int)
        assert isinstance(config.max_actions, int)
        assert isinstance(config.target_fps, int)
        assert isinstance(config.llm_model, str)
        assert isinstance(config.llm_interval, int)


class TestMonitorConfig:
    """Test MonitorConfig dataclass functionality."""

    def test_monitor_config_defaults(self):
        """Test MonitorConfig default values."""
        config = MonitorConfig()

        assert config.enable_web is True
        assert config.host == '127.0.0.1'
        assert config.port == 8080
        assert config.web_port is None
        assert config.update_interval == 0.1
        assert config.screenshot_fps == 10
        assert config.cache_size == 1000
        assert config.compression_level == 6
        assert config.db_path is None
        assert config.data_dir is None
        assert config.static_dir is None
        assert config.snapshot_interval == 1.0
        assert config.max_events == 10000
        assert config.max_snapshots == 100
        assert config.debug is False

    def test_monitor_config_custom_values(self):
        """Test MonitorConfig with custom values."""
        config = MonitorConfig(
            enable_web=False,
            host='0.0.0.0',
            port=9000,
            web_port=9001,
            update_interval=0.5,
            screenshot_fps=20,
            cache_size=2000,
            compression_level=9,
            db_path='/tmp/db.sqlite',
            data_dir='/tmp/data',
            static_dir='/tmp/static',
            snapshot_interval=2.0,
            max_events=20000,
            max_snapshots=200,
            debug=True
        )

        assert config.enable_web is False
        assert config.host == '0.0.0.0'
        assert config.port == 9000
        assert config.web_port == 9001
        assert config.update_interval == 0.5
        assert config.screenshot_fps == 20
        assert config.cache_size == 2000
        assert config.compression_level == 9
        assert config.db_path == '/tmp/db.sqlite'
        assert config.data_dir == '/tmp/data'
        assert config.static_dir == '/tmp/static'
        assert config.snapshot_interval == 2.0
        assert config.max_events == 20000
        assert config.max_snapshots == 200
        assert config.debug is True


class TestVisionConfig:
    """Test VisionConfig dataclass functionality."""

    def test_vision_config_defaults(self):
        """Test VisionConfig default values."""
        config = VisionConfig()

        assert config.enable_vision is True
        assert config.font_path is None
        assert config.template_path is None
        assert config.min_confidence == 0.6
        assert config.cache_size == 1000
        assert config.batch_size == 4

    def test_vision_config_custom_values(self):
        """Test VisionConfig with custom values."""
        config = VisionConfig(
            enable_vision=False,
            font_path='/path/to/font.ttf',
            template_path='/path/to/templates',
            min_confidence=0.8,
            cache_size=500,
            batch_size=8
        )

        assert config.enable_vision is False
        assert config.font_path == '/path/to/font.ttf'
        assert config.template_path == '/path/to/templates'
        assert config.min_confidence == 0.8
        assert config.cache_size == 500
        assert config.batch_size == 8


class TestSystemConfig:
    """Test SystemConfig dataclass functionality."""

    def test_system_config_defaults(self):
        """Test SystemConfig default values."""
        config = SystemConfig()

        assert config.debug_mode is False
        assert config.log_level == LogLevel.INFO
        assert config.log_file is None
        assert config.capture_screens is False
        assert config.save_states is True
        assert config.rom_path is None
        assert config.output_dir == 'outputs'

    def test_system_config_custom_values(self):
        """Test SystemConfig with custom values."""
        config = SystemConfig(
            debug_mode=True,
            log_level=LogLevel.DEBUG,
            log_file='/tmp/debug.log',
            capture_screens=True,
            save_states=False,
            rom_path='/path/to/rom.gbc',
            output_dir='/tmp/outputs'
        )

        assert config.debug_mode is True
        assert config.log_level == LogLevel.DEBUG
        assert config.log_file == '/tmp/debug.log'
        assert config.capture_screens is True
        assert config.save_states is False
        assert config.rom_path == '/path/to/rom.gbc'
        assert config.output_dir == '/tmp/outputs'


class TestUnifiedConfig:
    """Test UnifiedConfig dataclass functionality."""

    def test_unified_config_defaults(self):
        """Test UnifiedConfig default initialization."""
        config = UnifiedConfig()

        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.monitor, MonitorConfig)
        assert isinstance(config.vision, VisionConfig)
        assert isinstance(config.system, SystemConfig)

        # Test default values are propagated
        assert config.training.mode == TrainingMode.FAST
        assert config.monitor.enable_web is True
        assert config.vision.enable_vision is True
        assert config.system.debug_mode is False

    def test_unified_config_custom_components(self):
        """Test UnifiedConfig with custom component instances."""
        training = TrainingConfig(mode=TrainingMode.LLM, max_episodes=100)
        monitor = MonitorConfig(enable_web=False, port=9000)
        vision = VisionConfig(enable_vision=False, min_confidence=0.9)
        system = SystemConfig(debug_mode=True, log_level=LogLevel.DEBUG)

        config = UnifiedConfig(
            training=training,
            monitor=monitor,
            vision=vision,
            system=system
        )

        assert config.training == training
        assert config.monitor == monitor
        assert config.vision == vision
        assert config.system == system

    def test_from_dict_complete(self):
        """Test UnifiedConfig.from_dict with complete data."""
        data = {
            'training': {
                'mode': 'llm',
                'max_episodes': 500,
                'max_actions': 5000,
                'target_fps': 60,
                'llm_backend': 'ollama',
                'llm_model': 'llama2:7b'
            },
            'monitor': {
                'enable_web': False,
                'host': '0.0.0.0',
                'port': 9000
            },
            'vision': {
                'enable_vision': False,
                'min_confidence': 0.8
            },
            'system': {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'rom_path': '/path/to/rom.gbc'
            }
        }

        config = UnifiedConfig.from_dict(data)

        # Test training config
        assert config.training.mode == TrainingMode.LLM
        assert config.training.max_episodes == 500
        assert config.training.max_actions == 5000
        assert config.training.target_fps == 60
        assert config.training.llm_backend == LLMBackend.OLLAMA
        assert config.training.llm_model == 'llama2:7b'

        # Test monitor config
        assert config.monitor.enable_web is False
        assert config.monitor.host == '0.0.0.0'
        assert config.monitor.port == 9000

        # Test vision config
        assert config.vision.enable_vision is False
        assert config.vision.min_confidence == 0.8

        # Test system config
        assert config.system.debug_mode is True
        assert config.system.log_level == LogLevel.DEBUG
        assert config.system.rom_path == '/path/to/rom.gbc'

    def test_from_dict_partial(self):
        """Test UnifiedConfig.from_dict with partial data."""
        data = {
            'training': {
                'mode': 'curriculum',
                'max_episodes': 200
            },
            'monitor': {
                'port': 8090
            }
        }

        config = UnifiedConfig.from_dict(data)

        # Test specified values
        assert config.training.mode == TrainingMode.CURRICULUM
        assert config.training.max_episodes == 200
        assert config.monitor.port == 8090

        # Test default values for unspecified fields
        assert config.training.max_actions == 10000  # Default
        assert config.monitor.enable_web is True  # Default
        assert config.vision.enable_vision is True  # Default
        assert config.system.debug_mode is False  # Default

    def test_from_dict_empty(self):
        """Test UnifiedConfig.from_dict with empty data."""
        config = UnifiedConfig.from_dict({})

        # Should have all default values
        assert config.training.mode == TrainingMode.FAST
        assert config.monitor.enable_web is True
        assert config.vision.enable_vision is True
        assert config.system.debug_mode is False

    def test_from_file(self):
        """Test UnifiedConfig.from_file method."""
        test_data = {
            'training': {
                'mode': 'llm',
                'max_episodes': 100
            },
            'monitor': {
                'enable_web': True,
                'port': 8081
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            config_path = f.name

        try:
            config = UnifiedConfig.from_file(config_path)
            assert config.training.mode == TrainingMode.LLM
            assert config.training.max_episodes == 100
            assert config.monitor.port == 8081
        finally:
            Path(config_path).unlink()

    def test_to_dict(self):
        """Test UnifiedConfig.to_dict method."""
        training = TrainingConfig(
            mode=TrainingMode.LLM,
            max_episodes=500,
            llm_backend=LLMBackend.OLLAMA
        )
        monitor = MonitorConfig(enable_web=False, port=9000)
        vision = VisionConfig(enable_vision=False)
        system = SystemConfig(debug_mode=True, log_level=LogLevel.DEBUG)

        config = UnifiedConfig(
            training=training,
            monitor=monitor,
            vision=vision,
            system=system
        )

        result = config.to_dict()

        assert result['training']['mode'] == 'llm'
        assert result['training']['max_episodes'] == 500
        assert result['training']['llm_backend'] == 'ollama'
        assert result['monitor']['enable_web'] is False
        assert result['monitor']['port'] == 9000
        assert result['vision']['enable_vision'] is False
        assert result['system']['debug_mode'] is True

    def test_to_dict_with_none_backend(self):
        """Test UnifiedConfig.to_dict with None LLM backend."""
        config = UnifiedConfig()  # Default has None LLM backend
        result = config.to_dict()

        assert result['training']['llm_backend'] is None

    def test_config_roundtrip(self):
        """Test config serialization/deserialization roundtrip."""
        original = UnifiedConfig()
        original.training.mode = TrainingMode.CURRICULUM
        original.training.max_episodes = 300
        original.training.llm_backend = LLMBackend.OLLAMA  # Set non-None backend
        original.monitor.port = 8085
        original.system.debug_mode = True

        # Convert to dict and back
        data = original.to_dict()
        restored = UnifiedConfig.from_dict(data)

        assert restored.training.mode == original.training.mode
        assert restored.training.max_episodes == original.training.max_episodes
        assert restored.training.llm_backend == original.training.llm_backend
        assert restored.monitor.port == original.monitor.port
        assert restored.system.debug_mode == original.system.debug_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])