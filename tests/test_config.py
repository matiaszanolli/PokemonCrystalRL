"""Tests for the configuration module."""

import pytest
from pathlib import Path
import tempfile
import json

from pokemon_crystal_rl.core.config import (
    UnifiedConfig,
    TrainingMode,
    LLMBackend,
    LogLevel,
)

def test_default_config():
    """Test default configuration values."""
    config = UnifiedConfig()
    
    # Check training defaults
    assert config.training.mode == TrainingMode.FAST
    assert config.training.max_episodes == 1000
    assert config.training.max_actions == 10000
    assert config.training.target_fps == 30
    assert config.training.llm_backend is None
    assert config.training.llm_model == 'smollm2:1.7b'
    
    # Check monitor defaults
    assert config.monitor.enable_web is True
    assert config.monitor.host == '127.0.0.1'
    assert config.monitor.port == 8080
    
    # Check vision defaults
    assert config.vision.enable_vision is True
    assert config.vision.font_path is None
    assert config.vision.template_path is None
    assert config.vision.min_confidence == 0.6
    
    # Check system defaults
    assert config.system.debug_mode is False
    assert config.system.log_level == LogLevel.INFO
    assert config.system.log_file is None
    assert config.system.rom_path is None

def test_config_from_dict():
    """Test configuration creation from dictionary."""
    config_dict = {
        'training': {
            'mode': 'llm',
            'max_episodes': 500,
            'llm_backend': 'ollama',
            'llm_model': 'codellama:7b',
        },
        'monitor': {
            'enable_web': False,
            'port': 9090,
        },
        'vision': {
            'min_confidence': 0.8,
        },
        'system': {
            'debug_mode': True,
            'log_level': 'DEBUG',
            'rom_path': '/path/to/rom.gb',
        },
    }
    
    config = UnifiedConfig.from_dict(config_dict)
    
    # Check training values
    assert config.training.mode == TrainingMode.LLM
    assert config.training.max_episodes == 500
    assert config.training.llm_backend == LLMBackend.OLLAMA
    assert config.training.llm_model == 'codellama:7b'
    
    # Check monitor values
    assert config.monitor.enable_web is False
    assert config.monitor.port == 9090
    
    # Check vision values
    assert config.vision.min_confidence == 0.8
    
    # Check system values
    assert config.system.debug_mode is True
    assert config.system.log_level == LogLevel.DEBUG
    assert config.system.rom_path == '/path/to/rom.gb'

def test_config_save_load():
    """Test saving and loading configuration."""
    original_config = UnifiedConfig()
    original_config.training.mode = TrainingMode.LLM
    original_config.training.llm_backend = LLMBackend.OLLAMA
    original_config.system.rom_path = '/path/to/rom.gb'
    
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        # Save config
        original_config.save(tmp.name)
        
        # Load config
        loaded_config = UnifiedConfig.from_file(tmp.name)
        
        # Verify values
        assert loaded_config.training.mode == TrainingMode.LLM
        assert loaded_config.training.llm_backend == LLMBackend.OLLAMA
        assert loaded_config.system.rom_path == '/path/to/rom.gb'

def test_config_validation():
    """Test configuration validation."""
    config = UnifiedConfig()
    
    # Test with missing ROM path
    errors = config.validate()
    assert "ROM path not specified" in errors
    
    # Test with LLM mode but no backend
    config.training.mode = TrainingMode.LLM
    errors = config.validate()
    assert "LLM backend required for LLM/synchronized mode" in errors
    
    # Test with invalid port
    config.monitor.port = 80  # Below 1024
    errors = config.validate()
    assert any(error.startswith("Invalid port number") for error in errors)
    
    # Test with non-existent template path
    config.vision.template_path = "/nonexistent/path"
    errors = config.validate()
    assert any(error.startswith("Font template file not found") for error in errors)

def test_config_merge():
    """Test configuration merging."""
    base_config = UnifiedConfig()
    base_config.training.mode = TrainingMode.FAST
    base_config.monitor.port = 8080
    base_config.vision.min_confidence = 0.6
    
    other_config = UnifiedConfig()
    other_config.training.mode = TrainingMode.LLM
    other_config.training.llm_backend = LLMBackend.OLLAMA
    other_config.monitor.port = 9090
    other_config.vision.min_confidence = 0.8
    
    base_config.merge(other_config)
    
    assert base_config.training.mode == TrainingMode.LLM
    assert base_config.training.llm_backend == LLMBackend.OLLAMA
    assert base_config.monitor.port == 9090
    assert base_config.vision.min_confidence == 0.8
