"""
Configuration module for Pokemon Crystal RL.

Provides a unified configuration system for all components:
- Training settings
- Monitoring options
- Vision processing parameters
- LLM settings
- System configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import json
import logging


class TrainingMode(Enum):
    """Available training modes."""
    FAST = 'fast'  # Fast training without LLM
    LLM = 'llm'  # LLM-powered training
    CURRICULUM = 'curriculum'  # Progressive difficulty
    SYNCHRONIZED = 'synchronized'  # Frame-perfect LLM


class LLMBackend(Enum):
    """Supported LLM backends."""
    NONE = 'none'  # No LLM
    OLLAMA = 'ollama'  # Local Ollama models
    OPENAI = 'openai'  # OpenAI API
    ANTHROPIC = 'anthropic'  # Anthropic API


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    mode: TrainingMode = TrainingMode.FAST
    max_episodes: int = 1000
    max_actions: int = 10000
    target_fps: int = 30
    
    # LLM settings
    llm_backend: Optional[LLMBackend] = None
    llm_model: str = 'smollm2:1.7b'
    llm_interval: int = 10  # Steps between LLM calls
    
    # Curriculum settings
    curriculum_stages: int = 5
    curriculum_step_size: int = 100
    
    # Performance
    batch_size: int = 32
    memory_limit: int = 2048  # MB
    device: str = 'cpu'  # or 'cuda'


@dataclass
class MonitorConfig:
    """Web monitoring configuration."""
    enable_web: bool = True
    host: str = '127.0.0.1'
    port: int = 8080
    update_interval: float = 0.1
    screenshot_fps: int = 10
    cache_size: int = 1000
    compression_level: int = 6


@dataclass
class VisionConfig:
    """Vision processing configuration."""
    enable_vision: bool = True
    font_path: Optional[str] = None
    template_path: Optional[str] = None
    min_confidence: float = 0.6
    cache_size: int = 1000
    batch_size: int = 4


@dataclass
class SystemConfig:
    """System-wide configuration."""
    debug_mode: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    capture_screens: bool = False
    save_states: bool = True
    rom_path: Optional[str] = None
    output_dir: str = 'outputs'


@dataclass
class UnifiedConfig:
    """Unified configuration for all components."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'UnifiedConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Training config
        if 'training' in data:
            train_data = data['training']
            config.training.mode = TrainingMode(train_data.get('mode', 'fast'))
            config.training.max_episodes = train_data.get('max_episodes', 1000)
            config.training.max_actions = train_data.get('max_actions', 10000)
            config.training.target_fps = train_data.get('target_fps', 30)
            if 'llm_backend' in train_data:
                config.training.llm_backend = LLMBackend(train_data['llm_backend'])
            config.training.llm_model = train_data.get('llm_model', 'smollm2:1.7b')
        
        # Monitor config
        if 'monitor' in data:
            mon_data = data['monitor']
            config.monitor.enable_web = mon_data.get('enable_web', True)
            config.monitor.host = mon_data.get('host', '127.0.0.1')
            config.monitor.port = mon_data.get('port', 8080)
        
        # Vision config
        if 'vision' in data:
            vis_data = data['vision']
            config.vision.enable_vision = vis_data.get('enable_vision', True)
            config.vision.min_confidence = vis_data.get('min_confidence', 0.6)
        
        # System config
        if 'system' in data:
            sys_data = data['system']
            config.system.debug_mode = sys_data.get('debug_mode', False)
            config.system.log_level = LogLevel(sys_data.get('log_level', 'INFO'))
            config.system.rom_path = sys_data.get('rom_path')
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'training': {
                'mode': self.training.mode.value,
                'max_episodes': self.training.max_episodes,
                'max_actions': self.training.max_actions,
                'target_fps': self.training.target_fps,
                'llm_backend': self.training.llm_backend.value if self.training.llm_backend else None,
                'llm_model': self.training.llm_model,
                'llm_interval': self.training.llm_interval,
                'curriculum_stages': self.training.curriculum_stages,
                'curriculum_step_size': self.training.curriculum_step_size,
                'batch_size': self.training.batch_size,
                'memory_limit': self.training.memory_limit,
                'device': self.training.device
            },
            'monitor': {
                'enable_web': self.monitor.enable_web,
                'host': self.monitor.host,
                'port': self.monitor.port,
                'update_interval': self.monitor.update_interval,
                'screenshot_fps': self.monitor.screenshot_fps,
                'cache_size': self.monitor.cache_size,
                'compression_level': self.monitor.compression_level
            },
            'vision': {
                'enable_vision': self.vision.enable_vision,
                'font_path': self.vision.font_path,
                'template_path': self.vision.template_path,
                'min_confidence': self.vision.min_confidence,
                'cache_size': self.vision.cache_size,
                'batch_size': self.vision.batch_size
            },
            'system': {
                'debug_mode': self.system.debug_mode,
                'log_level': self.system.log_level.value,
                'log_file': self.system.log_file,
                'capture_screens': self.system.capture_screens,
                'save_states': self.system.save_states,
                'rom_path': self.system.rom_path,
                'output_dir': self.system.output_dir
            }
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check ROM path
        if not self.system.rom_path:
            errors.append("ROM path not specified")
        elif not Path(self.system.rom_path).exists():
            errors.append(f"ROM file not found: {self.system.rom_path}")
        
        # Check LLM config
        if self.training.mode in [TrainingMode.LLM, TrainingMode.SYNCHRONIZED]:
            if not self.training.llm_backend:
                errors.append("LLM backend required for LLM/synchronized mode")
            elif self.training.llm_backend != LLMBackend.NONE and not self.training.llm_model:
                errors.append("LLM model must be specified")
        
        # Check vision config
        if self.vision.enable_vision:
            if self.vision.template_path and not Path(self.vision.template_path).exists():
                errors.append(f"Font template file not found: {self.vision.template_path}")
        
        # Check monitor config
        if self.monitor.enable_web:
            if self.monitor.port < 1024 or self.monitor.port > 65535:
                errors.append(f"Invalid port number: {self.monitor.port}")
        
        return errors
    
    def merge(self, other: 'UnifiedConfig') -> None:
        """Merge another configuration into this one."""
        # Merge training config
        if other.training.mode != self.training.mode:
            self.training.mode = other.training.mode
        if other.training.llm_backend:
            self.training.llm_backend = other.training.llm_backend
        if other.training.llm_model:
            self.training.llm_model = other.training.llm_model
        
        # Merge monitor config
        if other.monitor.port != self.monitor.port:
            self.monitor.port = other.monitor.port
        if other.monitor.host != self.monitor.host:
            self.monitor.host = other.monitor.host
        
        # Merge vision config
        if other.vision.template_path:
            self.vision.template_path = other.vision.template_path
        if other.vision.min_confidence != self.vision.min_confidence:
            self.vision.min_confidence = other.vision.min_confidence
        
        # Merge system config
        if other.system.rom_path:
            self.system.rom_path = other.system.rom_path
        if other.system.debug_mode != self.system.debug_mode:
            self.system.debug_mode = other.system.debug_mode


def load_config(config_path: Optional[str] = None) -> UnifiedConfig:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        try:
            config = UnifiedConfig.from_file(config_path)
            logging.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
    
    logging.info("Using default configuration")
    return UnifiedConfig()
