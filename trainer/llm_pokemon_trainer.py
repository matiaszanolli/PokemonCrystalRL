#!/usr/bin/env python3
"""
LLM Pokemon Trainer Module

Advanced Pokemon Crystal trainer orchestrator with LLM integration, reward system,
and comprehensive training coordination capabilities.

This module provides the main LLMPokemonTrainer class which handles:
- PyBoy emulation setup and management
- Training loop coordination and orchestration  
- LLM-based decision making integration
- Multi-factor reward calculation and state tracking
- Web monitoring interface integration
- DQN hybrid training support
- Signal handling for graceful shutdown
- Training statistics and analytics
- Experience memory and pattern recognition
- Failsafe mechanisms and stuck detection

The trainer acts as the central coordinator for all Pokemon Crystal RL training
operations, managing the interaction between game emulation, AI decision making,
reward systems, and monitoring interfaces.
"""

import time
import numpy as np
from pyboy import PyBoy
import json
import signal
import sys
import os
import threading
import io
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
import requests
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

# Import core systems and monitoring (try/except for optional dependencies)
try:
    from core.web_monitor import WebMonitor, WebMonitorHandler, ScreenCapture
except ImportError:
    print("⚠️  WebMonitor not available - web monitoring disabled")
    WebMonitor = WebMonitorHandler = ScreenCapture = None

try:
    from core.memory.reader import GameState
except ImportError:
    print("⚠️  GameState not available")
    GameState = None

try:
    from core.state.analyzer import GameStateAnalyzer
except ImportError:
    print("⚠️  GameStateAnalyzer not available")
    GameStateAnalyzer = None

try:
    from trainer.web_server import WebServer, ServerConfig
except ImportError:
    print("⚠️  WebServer not available")
    WebServer = ServerConfig = None

# Import game intelligence and experience systems (optional)
try:
    from core.game_intelligence import GameIntelligence, GameContext, ActionPlan
except ImportError:
    print("⚠️  GameIntelligence not available")
    GameIntelligence = GameContext = ActionPlan = None

try:
    from core.experience_memory import ExperienceMemory
except ImportError:
    print("⚠️  ExperienceMemory not available")
    ExperienceMemory = None

# Import DQN agent for hybrid training (optional)
try:
    from core.dqn_agent import DQNAgent, HybridAgent
except ImportError:
    print("⚠️  DQN agents not available")
    DQNAgent = HybridAgent = None

from config.memory_addresses import MEMORY_ADDRESSES
from utils.memory_reader import build_observation

# Import core systems and monitoring (try/except for optional dependencies)
try:
    from core.web_monitor import WebMonitor, WebMonitorHandler, ScreenCapture
except ImportError:
    print("⚠️  WebMonitor not available - web monitoring disabled")
    WebMonitor = WebMonitorHandler = ScreenCapture = None

try:
    from core.memory.reader import GameState
except ImportError:
    print("⚠️  GameState not available")
    GameState = None
# Orphaned code removed - this was part of incomplete refactoring


class LLMPokemonTrainer:
    """Advanced Pokemon Crystal trainer with LLM integration and reward system"""
    
    def __init__(self, rom_path, max_actions=5000, save_state=None,
                 llm_model="smollm2:1.7b", llm_base_url="http://localhost:11434",
                 llm_interval=20, llm_temperature=0.7, enable_web=True, web_port=8080,
                 web_host="localhost", enable_dqn=True, dqn_model_path=None,
                 dqn_learning_rate=1e-4, dqn_batch_size=32, dqn_memory_size=50000,
                 dqn_training_frequency=4, dqn_save_frequency=500, log_dir="logs",
                 show_progress=True):
        # Core paths and configuration
        self.rom_path = rom_path
        self.save_state = save_state
        self.max_actions = max_actions
        self.log_dir = log_dir
        self.show_progress = show_progress
        
        # LLM configuration
        self.llm_interval = llm_interval
        self.llm_temperature = llm_temperature
        
        # Web server configuration
        self.enable_web = enable_web
        self.web_port = web_port
        self.web_host = web_host
        
        # DQN configuration
        self.enable_dqn = enable_dqn
        self.dqn_learning_rate = dqn_learning_rate
        self.dqn_batch_size = dqn_batch_size
        self.dqn_memory_size = dqn_memory_size
        self.dqn_training_frequency = dqn_training_frequency
        self.dqn_save_frequency = dqn_save_frequency
        
        # Core components - Import these locally to avoid circular dependencies
        try:
            from trainer.llm import LLMAgent
            self.llm_agent = LLMAgent(llm_model, llm_base_url)
        except ImportError:
            print("⚠️  LLMAgent not available - using mock agent")
            self.llm_agent = self._create_mock_llm_agent()
            
        try:
            from trainer.rewards import PokemonRewardCalculator
            self.reward_calculator = PokemonRewardCalculator()
        except ImportError:
            print("⚠️  PokemonRewardCalculator not available - using mock calculator")
            self.reward_calculator = self._create_mock_reward_calculator()
        
        self.pyboy = None
        
        # Initialize logging
        self.logger = logging.getLogger("LLMTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Web monitor setup
        self.web_monitor = None
        if self.enable_web and WebMonitor is not None:
            self.web_monitor = WebMonitor(self, self.web_port, self.web_host)
        elif self.enable_web:
            print("⚠️  Web monitoring disabled - WebMonitor not available")
            self.enable_web = False
        
        # DQN components
        self.dqn_agent = None
        self.hybrid_agent = None
        self.dqn_training_frequency = dqn_training_frequency  # Train DQN every N actions
        self.dqn_save_frequency = dqn_save_frequency  # Save DQN model every N actions
        self.dqn_memory_size = dqn_memory_size  # Size of experience replay buffer
        self.dqn_batch_size = dqn_batch_size  # Training batch size
        self.dqn_learning_rate = dqn_learning_rate  # Learning rate for optimizer
        
        if self.enable_dqn and DQNAgent is not None:
            # Initialize DQN agent
            self.dqn_agent = DQNAgent(
                state_size=32,
                action_size=8,
                learning_rate=self.dqn_learning_rate,
                gamma=0.99,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                memory_size=self.dqn_memory_size,
                batch_size=self.dqn_batch_size,
                target_update=1000
            )
            
            # Load existing model if provided
            if dqn_model_path and os.path.exists(dqn_model_path):
                self.dqn_agent.load_model(dqn_model_path)
                print(f"📥 Loaded DQN model from {dqn_model_path}")
            
            # Create hybrid agent combining LLM and DQN
            if HybridAgent is not None:
                self.hybrid_agent = HybridAgent(
                    dqn_agent=self.dqn_agent,
                    llm_agent=self.llm_agent,
                    dqn_weight=0.2,  # Start with low DQN influence
                    exploration_bonus=0.1
                )
                
            print(f"🧠 DQN Agent initialized with {self.dqn_agent.device}")
        elif self.enable_dqn:
            print("⚠️  DQN training disabled - DQN agents not available")
            self.enable_dqn = False
        
        # Experience tracking
        self.recent_situation_hashes = []
        self.recent_action_sequences = []
        self.experience_window = 10  # Track last N actions for experience recording
        
        # Training state and performance tracking
        self.actions_taken = 0
        self.start_time = time.time()
        self.previous_game_state = {}
        self.total_reward = 0.0
        
        # Performance logging
        self.decision_log = []
        self.performance_log = []
        self.last_llm_decision_action = 0
        
        # Failsafe mechanism for stuck detection
        self.last_positive_reward_action = 0
        self.actions_without_reward = 0
        self.stuck_threshold = 100  # Actions without reward before intervention
        self.location_stuck_tracker = {}  # Track how long we've been in same location
        
        # Statistics
        self.stats = {
            'actions_taken': 0,
            'training_time': 0,
            'actions_per_second': 0,
            'llm_decision_count': 0,
            'total_reward': 0.0,
            'player_level': 0,
            'badges_total': 0,
            'last_reward_breakdown': '',
            'recent_llm_decisions': [],
            'start_time': datetime.now().isoformat()
        }
        
        self.running = True
        self.recent_actions = []
        
        # LLM decision tracking for web monitor
        self.llm_decisions = deque(maxlen=10)  # Keep last 10 decisions
        
        # Web monitoring setup
        self.web_monitor = None
        if self.enable_web:
            self.web_monitor = WebMonitor(self, self.web_port, self.web_host)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        print(f"🤖 LLM Agent: {llm_model} {'✅' if self.llm_agent.available else '❌'}")
        if self.enable_web:
            print(f"🌐 Web monitor will start at http://{self.web_host}:{self.web_port}")
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        print(f"🤖 LLM Agent: {llm_model} {'✅' if self.llm_agent.available else '❌'}")
        
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler for the training system"""
        print(f"\n⏸️ Shutting down LLM training...")
        self.running = False
        
        # Stop web monitor first
        if self.web_monitor:
            print("🛑 Stopping web monitor...")
            try:
                self.web_monitor.stop()
                print("✅ Web monitor stopped")
            except Exception as e:
                print(f"⚠️ Error stopping web monitor: {e}")
        
        # Stop PyBoy
        if self.pyboy:
            print("🎮 Stopping PyBoy...")
            try:
                self.pyboy.stop()
                print("✅ PyBoy stopped")
            except Exception as e:
                print(f"⚠️ Error stopping PyBoy: {e}")
        
        # Save final training data
        try:
            self.save_training_data()
            print("💾 Training data saved")
        except Exception as e:
            print(f"⚠️ Error saving training data: {e}")
        
        print("✅ Training system shut down cleanly")
        
        # Only exit if called as signal handler
        if signum is not None:
            sys.exit(0)
    
    def setup_web_server(self):
        """Setup enhanced web monitoring server"""
        if not self.enable_web:
            return
        
        # Use the new WebMonitor if available
        if self.web_monitor:
            success = self.web_monitor.start()
            if success:
                print(f"🌐 Web monitor started at {self.web_monitor.get_url()}")
            else:
                print("⚠️ Failed to start web monitor")
            return
        
        # Legacy fallback (should not be reached normally)
        print("⚠️ Using legacy web server fallback")
        try:
            # Note: This refers to the old WebMonitor class that should be replaced
            from core.web_monitor import WebMonitorHandler
            self.web_server = HTTPServer(('localhost', self.web_port), WebMonitorHandler)
            
            self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_thread.start()
            
            # Initialize detailed stats tracking for web display
            self.web_stats_history = {
                'reward_history': [],  # Track reward trends
                'action_history': [],  # Track action frequencies
                'progression': [],     # Track game progression
                'performance': []      # Track system performance
            }
            
            # Initialize enhanced performance tracking
            self.performance_tracking = {
                'reward_window': [],           # Track rewards for rate calculation
                'llm_success_window': [],      # Track LLM success for accuracy
                'action_counts': {},          # Count of each action type
                'state_transitions': {},      # Track state changes
                'window_size': 100            # Size of sliding windows
            }
            
            # Set initial web stats with expanded metrics
            self.stats.update({
                'experience_stats': {
                    'total_experiences': 0,
                    'positive_patterns': 0,
                    'learning_rate': 1.0
                },
                'recent_stats': {
                    'reward_rate': 0.0,       # Per-action reward rate
                    'exploration_rate': 0.0,   # Rate of new area discovery
                    'stuck_rate': 0.0,        # Rate of stuck detection
                    'success_rate': 0.0       # Rate of positive outcomes
                },
                'training_metrics': {
                    'llm_accuracy': 0.0,      # LLM decision quality
                    'dqn_loss': 0.0,         # DQN training loss
                    'hybrid_balance': 0.5,    # LLM vs DQN balance
                    'state_coverage': 0.0     # Game state exploration
                },
                'session_metrics': {
                    'start_time': time.time(),
                    'last_save': time.time(),
                    'total_steps': 0,
                    'unique_states': set(),
                    'error_count': 0
                }
            })
            
            print(f"🌐 Enhanced web monitor: http://localhost:{self.web_port}")
            print("   Real-time metrics and visualization available")
            
        except Exception as e:
            print(f"⚠️ Failed to start web server: {e}")
            self.enable_web = False
    
    def get_current_stats(self):
        """Get current training statistics for web monitor"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Extract game state from latest observation
        current_state = {}
        if hasattr(self, 'pyboy') and self.pyboy:
            try:
                memory = self.pyboy.memory
                current_state = {
                    'party_count': memory[MEMORY_ADDRESSES['party_count']],
                    'player_map': memory[MEMORY_ADDRESSES['player_map']],
                    'player_x': memory[MEMORY_ADDRESSES['player_x']],
                    'player_y': memory[MEMORY_ADDRESSES['player_y']],
                    'money': (memory[MEMORY_ADDRESSES['money_low']] + 
                             (memory[MEMORY_ADDRESSES['money_mid']] << 8) +
                             (memory[MEMORY_ADDRESSES['money_high']] << 16)),
                    'badges': bin(memory[MEMORY_ADDRESSES['badges']]).count('1'),
                    'in_battle': memory[MEMORY_ADDRESSES['in_battle']],
                    'player_level': memory[MEMORY_ADDRESSES['player_level']] if memory[MEMORY_ADDRESSES['party_count']] > 0 else 0
                }
            except Exception as e:
                # Return empty state on error
                current_state = {}
        
        return {
            'total_actions': self.actions_taken,
            'actions_per_second': self.actions_taken / max(elapsed, 1),
            'llm_calls': getattr(self, 'llm_decision_count', 0),
            'total_episodes': 1,  # Single continuous session
            'session_duration': elapsed,
            'start_time': self.start_time,
            'total_reward': self.total_reward,
            'badges_earned': current_state.get('badges', 0),
            'current_map': current_state.get('player_map', 0),
            'player_position': {
                'x': current_state.get('player_x', 0),
                'y': current_state.get('player_y', 0)
            },
            'money': current_state.get('money', 0),
            'party': current_state.get('party_count', 0),
            'memory_data': current_state,
            'game_phase': 'Gameplay',
            'phase_progress': min((elapsed / 3600) * 100, 100)
        }
    
    def initialize_pyboy(self):
        """Initialize PyBoy emulator"""
        print(f"🎮 Initializing PyBoy with {self.rom_path}")
        self.pyboy = PyBoy(self.rom_path, window="null", debug=False)
        
        # Load save state if available
        save_state_path = self.rom_path + '.state'
        if os.path.exists(save_state_path):
            print(f"💾 Loading save state: {save_state_path}")
            with open(save_state_path, 'rb') as f:
                self.pyboy.load_state(f)
            print("✅ Save state loaded - starting from saved position")
        
        print("✅ PyBoy initialized successfully")
        
        # Start web monitor and update with PyBoy instance
        if self.web_monitor:
            self.web_monitor.update_pyboy(self.pyboy)
            if not self.web_monitor.running:
                success = self.web_monitor.start()
                if success:
                    print(f"🌐 Web monitor started at {self.web_monitor.get_url()}")
                else:
                    print("⚠️ Failed to start web monitor")
        
        # Get initial game state
        self.previous_game_state = self.get_game_state()
        
    def get_game_state(self) -> Dict:
        """Extract comprehensive game state from memory using validated addresses"""
        if not self.pyboy:
            return {}
        
        # Use the build_observation function with validated memory structure
        return build_observation(self.pyboy.memory)
    
    def analyze_screen(self) -> Dict:
        """Analyze current screen state with improved detection"""
        if not self.pyboy:
            return {'state': 'unknown', 'variance': 0, 'colors': 0}
            
        screen = self.pyboy.screen.ndarray
        variance = float(np.var(screen.astype(np.float32)))
        
        # Get additional screen analysis metrics
        unique_colors = len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        brightness = float(np.mean(screen.astype(np.float32)))
        
        # CRITICAL: Very few colors (2-3) almost always means menu/battle/evolution
        # This prevents false positives where menus are classified as overworld
        if unique_colors <= 3:
            # Very few colors - definitely not overworld
            if variance < 50:
                state = "loading"  # Solid colors or very simple screen
            elif brightness > 200:
                state = "dialogue"  # High brightness with few colors = dialogue box
            else:
                state = "menu"  # Low brightness with few colors = menu/battle/evolution
        # Very low variance = loading/transition screen
        elif variance < 50:
            state = "loading"
        # Very high variance with many colors = battle screen (lots of sprites/effects)
        elif variance > 20000 and unique_colors > 8:
            state = "battle"
        # Medium-high variance with many colors = overworld
        elif variance > 3000 and unique_colors > 10:
            state = "overworld"
        # Low variance patterns
        elif variance < 3000:
            # Further distinguish between menu and dialogue
            if brightness > 200 and unique_colors < 8:
                # Very bright with few colors = likely dialogue box
                state = "dialogue"
            elif unique_colors < 6:
                # Few colors = menu system
                state = "menu"
            elif variance > 500 and unique_colors >= 8:
                # Some variance with multiple colors = likely settings/menu
                state = "settings_menu"
            else:
                # Default to menu for low variance screens
                state = "menu"
        else:
            # Medium variance with reasonable colors - could be overworld
            if unique_colors > 8:
                state = "overworld"
            else:
                state = "menu"  # Conservative: few colors = likely menu
            
        return {
            'state': state,
            'variance': variance,
            'colors': unique_colors,
            'brightness': brightness
        }
    
    def get_next_action(self) -> Tuple[str, str]:
        """Get next action using hybrid DQN+LLM or fallback logic"""
        game_state = self.get_game_state()
        screen_analysis = self.analyze_screen()
        
        # CRITICAL: Forbid START and SELECT until first Pokemon is obtained
        party_count = game_state.get('party_count', 0)
        forbidden_actions = set()
        if party_count == 0:
            forbidden_actions.add('start')
            forbidden_actions.add('select')
        
        # If DQN is enabled and available, use hybrid approach
        if self.enable_dqn and self.hybrid_agent:
            # Use hybrid agent that combines LLM reasoning with DQN experience
            use_llm = (self.actions_taken - self.last_llm_decision_action) >= self.llm_interval
            
            if use_llm and self.llm_agent.available:
                # Get hybrid decision combining LLM and DQN
                action, reasoning = self.hybrid_agent.get_hybrid_action(
                    game_state, screen_analysis, self.recent_actions
                )
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                    reasoning = f"Forbidden {action} -> using allowed alternative"
                
                self.last_llm_decision_action = self.actions_taken
                self.stats['llm_decision_count'] += 1
                
                # Track recent decisions for web display
                self.stats['recent_llm_decisions'].append({
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp': time.time()
                })
                
                # Keep only last 5 decisions
                if len(self.stats['recent_llm_decisions']) > 5:
                    self.stats['recent_llm_decisions'].pop(0)
                
                return action, f"Hybrid: {reasoning[:50]}..."
            else:
                # Use DQN-only action selection between LLM decisions
                action, q_value = self.dqn_agent.get_action(game_state, screen_analysis, training=True)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                
                dqn_info = self.hybrid_agent.get_info()
                return action, f"DQN (Q={q_value:.3f}) - {dqn_info}"
        
        # Fallback to original LLM-only logic
        elif self.llm_agent.available:
            # Use LLM every N actions
            use_llm = (self.actions_taken - self.last_llm_decision_action) >= self.llm_interval
            
            if use_llm:
                action, reasoning = self.llm_agent.get_decision(game_state, screen_analysis, self.recent_actions)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                    reasoning = f"Forbidden action -> using {action}"
                
                self.last_llm_decision_action = self.actions_taken
                self.stats['llm_decision_count'] += 1
                
                # Track recent LLM decisions for web display
                decision_data = {
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp': time.time()
                }
                self.stats['recent_llm_decisions'].append(decision_data)
                self.llm_decisions.append(decision_data)  # For new web monitor
                
                # Keep only last 5 decisions
                if len(self.stats['recent_llm_decisions']) > 5:
                    self.stats['recent_llm_decisions'].pop(0)
                
                return action, f"LLM: {reasoning[:50]}..."
            else:
                # Fallback rule-based action
                action = self._get_rule_based_action(game_state, screen_analysis)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                
                return action, "Rule-based fallback"
        else:
            # No LLM available, use rule-based fallback
            action = self._get_rule_based_action(game_state, screen_analysis)
            
            # Override forbidden actions
            if action in forbidden_actions:
                action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
            
            return action, "Rule-based fallback"
    
    def _get_allowed_alternative_action(self, forbidden_action: str, game_state: Dict, screen_analysis: Dict) -> str:
        """Get an allowed alternative when an action is forbidden (START/SELECT before first Pokemon)"""
        state_type = screen_analysis.get('state', 'unknown')
        
        # Context-aware alternatives based on screen state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Always attack in battle
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'menu':
            return 'b'  # Exit menus when we can't use START
        elif state_type == 'loading':
            return 'a'  # Wait during loading
        else:
            # In overworld - focus on exploration and interaction
            # Priority order: interact with objects/NPCs, then explore
            exploration_priority = ['a', 'up', 'down', 'left', 'right']
            return exploration_priority[self.actions_taken % len(exploration_priority)]
    
    def _get_rule_based_action(self, game_state: Dict, screen_analysis: Dict) -> str:
        """Rule-based fallback action with improved screen state handling"""
        state_type = screen_analysis.get('state', 'unknown')
        
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'settings_menu':
            return 'b'  # Always exit settings menu immediately
        elif state_type == 'menu':
            # Smart menu handling - check recent actions
            recent_actions_str = ' '.join(self.recent_actions[-3:]) if self.recent_actions else ''
            if 'START' in recent_actions_str:
                return 'b'  # Exit menu if we recently opened one
            else:
                return 'b'  # Default to exiting menus unless we have a specific goal
        elif state_type == 'loading':
            return 'a'  # Wait during loading screens
        else:
            # Exploration pattern for overworld (avoid START button spam)
            exploration_actions = ['up', 'up', 'a', 'right', 'right', 'a', 'down', 'down', 'a', 'left', 'left', 'a']
            return exploration_actions[self.actions_taken % len(exploration_actions)]
    
    def _update_stuck_detection(self, current_state: Dict, reward: float):
        """Update stuck detection mechanism and trigger failsafe if needed"""
        # Track positive rewards
        if reward > 0.05:  # Any positive reward (excluding tiny dialogue rewards)
            self.last_positive_reward_action = self.actions_taken
            self.actions_without_reward = 0
        else:
            self.actions_without_reward += 1
        
        # Track location-based stuck detection
        current_location = (
            current_state.get('player_map', 0),
            current_state.get('player_x', 0),
            current_state.get('player_y', 0)
        )
        
        if current_location not in self.location_stuck_tracker:
            self.location_stuck_tracker[current_location] = 0
        self.location_stuck_tracker[current_location] += 1
        
        # Clean up old location tracking (keep only recent 50 actions worth)
        if len(self.location_stuck_tracker) > 50:
            # Remove locations with low visit counts
            min_visits = min(self.location_stuck_tracker.values())
            self.location_stuck_tracker = {
                loc: count for loc, count in self.location_stuck_tracker.items() 
                if count > min_visits
            }
        
        # Check if we're stuck and need intervention
        self._check_failsafe_intervention(current_state, current_location)
    
    def _check_failsafe_intervention(self, current_state: Dict, current_location: Tuple):
        """Check if failsafe intervention is needed and modify LLM prompts accordingly"""
        # Detect stuck conditions
        stuck_too_long = self.actions_without_reward >= self.stuck_threshold
        stuck_at_location = self.location_stuck_tracker.get(current_location, 0) > 20
        
        if stuck_too_long or stuck_at_location:
            # Modify LLM behavior to break out of stuck pattern
            self._trigger_failsafe_intervention(current_state, current_location, stuck_too_long, stuck_at_location)
    
    def _trigger_failsafe_intervention(self, current_state: Dict, current_location: Tuple, 
                                     stuck_too_long: bool, stuck_at_location: bool):
        """Trigger failsafe intervention to break stuck patterns"""
        
        # Force next LLM decision with explicit stuck-breaking prompt
        self.last_llm_decision_action = self.actions_taken - self.llm_interval + 1
        
        # Add failsafe context to the LLM agent for next decision
        if not hasattr(self.llm_agent, 'failsafe_context'):
            self.llm_agent.failsafe_context = {}
        
        self.llm_agent.failsafe_context = {
            'stuck_detected': True,
            'stuck_location': current_location,
            'actions_without_reward': self.actions_without_reward,
            'stuck_reason': 'no_reward' if stuck_too_long else 'same_location',
            'intervention_action': self.actions_taken
        }
        
        print(f"🚨 FAILSAFE: Detected stuck behavior!")
        print(f"   📍 Location: Map {current_location[0]}, Pos ({current_location[1]},{current_location[2]})")
        print(f"   ⏱️ Actions without reward: {self.actions_without_reward}")
        print(f"   🎯 INTERVENTION: Providing concrete movement instructions...")
        
        # Reset some stuck tracking to prevent immediate re-triggering
        self.actions_without_reward = max(0, self.actions_without_reward - 20)
        if current_location in self.location_stuck_tracker:
            self.location_stuck_tracker[current_location] = max(1, 
                self.location_stuck_tracker[current_location] - 10)
    
    def execute_action(self, action: str):
        """Execute action and calculate rewards with smart movement handling"""
        if not self.running or not self.pyboy:
            return
            
        # Store previous state for reward calculation and DQN training
        previous_state = self.previous_game_state.copy()
        previous_screen_analysis = self.analyze_screen()
        
        # Check if this is a directional movement in overworld
        is_directional = action.lower() in ['up', 'down', 'left', 'right']
        screen_state = previous_screen_analysis.get('state', 'unknown')
        is_overworld = screen_state == 'overworld'
        
        if is_directional and is_overworld:
            # Smart directional movement: try twice if first attempt doesn't move
            current_state = self._execute_smart_movement(action, previous_state)
            current_screen_analysis = self.analyze_screen()
        else:
            # Normal action execution for non-directional actions or non-overworld
            self._execute_single_action(action)
            current_state = self.get_game_state()
            current_screen_analysis = self.analyze_screen()
        
        # Pass screen state info to reward calculator for BOTH exploration AND progression reward filtering
        self.reward_calculator.last_screen_state = current_screen_analysis.get('state', 'unknown')
        self.reward_calculator.prev_screen_state = previous_screen_analysis.get('state', 'unknown')
        
        # Pass action info to prevent SELECT button false rewards
        self.reward_calculator.last_action = action
        
        # Pass screen state info to reward calculator for BOTH exploration AND progression reward filtering
        self.reward_calculator.last_screen_state = current_screen_analysis.get('state', 'unknown')
        self.reward_calculator.prev_screen_state = previous_screen_analysis.get('state', 'unknown')
        
        # Pass action info to prevent SELECT button false rewards
        self.reward_calculator.last_action = action
        
        # Calculate reward with enhanced state validation
        try:
            reward, reward_breakdown = self.reward_calculator.calculate_reward(current_state, previous_state)
            
            # Track reward history for analysis
            if not hasattr(self, 'reward_history'):
                self.reward_history = []
            
            # Keep detailed reward history
            reward_entry = {
                'action': action,
                'total_reward': reward,
                'breakdown': reward_breakdown,
                'state_changes': {
                    'party_count': (previous_state.get('party_count', 0), current_state.get('party_count', 0)),
                    'level': (previous_state.get('player_level', 0), current_state.get('player_level', 0)),
                    'position': (
                        (previous_state.get('player_x', 0), previous_state.get('player_y', 0)),
                        (current_state.get('player_x', 0), current_state.get('player_y', 0))
                    ),
                    'map': (previous_state.get('player_map', 0), current_state.get('player_map', 0)),
                    'screen_state': (
                        previous_screen_analysis.get('state', 'unknown'),
                        current_screen_analysis.get('state', 'unknown')
                    )
                },
                'timestamp': time.time()
            }
            
            self.reward_history.append(reward_entry)
            if len(self.reward_history) > 1000:  # Keep last 1000 rewards
                self.reward_history.pop(0)
            
            # Analyze significant rewards in detail
            if abs(reward) > 1.0:
                print(f"🔍 REWARD ANALYSIS | Action {self.actions_taken} | {action.upper()}")
                print(f"   💰 Total: {reward:+.2f}")
                
                # Show significant components
                significant_components = [
                    (cat, val) for cat, val in reward_breakdown.items()
                    if abs(val) > 0.01
                ]
                if significant_components:
                    print("   📊 Components:")
                    for category, value in significant_components:
                        prefix = '🟢' if value > 0 else '🔴'
                        print(f"      {prefix} {category}: {value:+.2f}")
                
                # Show relevant state changes
                print("   📈 State Changes:")
                for key, (old, new) in reward_entry['state_changes'].items():
                    if old != new:
                        print(f"      {key}: {old} → {new}")
                print()
            
        except Exception as e:
            # Handle reward calculation errors
            print(f"⚠️ Reward calculation error: {str(e)}")
            reward = 0.0
            reward_breakdown = {'error': 0.0}
            
            # Log error for debugging
            self.logger.error(f"Reward calculation failed: {str(e)}")
            self.logger.error(f"States: Previous={previous_state}, Current={current_state}")
            self.logger.error(f"Screen: Previous={previous_screen_analysis}, Current={current_screen_analysis}")
        
        # DQN experience storage and training
        if self.enable_dqn and self.dqn_agent:
            # Store experience in DQN replay buffer
            done = False  # We don't have episode termination in continuous play
            self.dqn_agent.store_experience(
                previous_state, previous_screen_analysis, action,
                reward, current_state, current_screen_analysis, done
            )
            
            # Train DQN periodically
            if self.actions_taken % self.dqn_training_frequency == 0:
                loss = self.dqn_agent.train_step()
                if loss > 0 and self.actions_taken % (self.dqn_training_frequency * 10) == 0:
                    print(f"🧠 DQN training: loss={loss:.4f}, ε={self.dqn_agent.epsilon:.3f}")
            
            # Record performance for hybrid agent adaptation
            if self.hybrid_agent:
                self.hybrid_agent.record_performance(reward)
            
            # Save DQN model periodically
            if self.actions_taken % self.dqn_save_frequency == 0:
                model_path = os.path.join("logs", f"dqn_model_{self.actions_taken}.pth")
                self.dqn_agent.save_model(model_path)
                print(f"🔄 DQN model saved at action {self.actions_taken}")
        
        # Track experience for learning
        self._track_experience(action, previous_state, current_state, reward)
        
        # Update tracking
        self.actions_taken += 1
        self.total_reward += reward
        self.previous_game_state = current_state
        
        # Update recent actions
        self.recent_actions.append(action.upper())
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # FAILSAFE: Track stuck detection
        self._update_stuck_detection(current_state, reward)
        
        # Update stats
        self.stats['actions_taken'] = self.actions_taken
        self.stats['total_reward'] = float(self.total_reward)
        self.stats['player_level'] = current_state.get('player_level', 0)
        self.stats['badges_total'] = current_state.get('badges_total', 0)
        self.stats['last_reward_breakdown'] = self.reward_calculator.get_reward_summary(reward_breakdown)
        
        # Add DQN stats if enabled
        if self.enable_dqn and self.dqn_agent:
            dqn_stats = self.dqn_agent.get_training_stats()
            self.stats['dqn_steps'] = dqn_stats['steps_trained']
            self.stats['dqn_epsilon'] = dqn_stats['epsilon']
            self.stats['dqn_memory_size'] = dqn_stats['memory_size']
        
        return reward, reward_breakdown
    
    def _execute_single_action(self, action: str):
        """Execute a single action with proper Pokemon Crystal timing"""
        screen_state = self.analyze_screen().get('state', 'unknown')
        
        # Different timing for different game states
        if screen_state == 'overworld':
            self._execute_overworld_action(action)
        elif screen_state == 'battle':
            self._execute_battle_action(action)
        elif screen_state in ['menu', 'dialogue', 'settings_menu']:
            self._execute_menu_action(action)
        else:
            # Default timing for unknown states
            self._execute_default_action(action)
    
    def _execute_overworld_action(self, action: str):
        """Execute action with overworld timing (16 frames per action)"""
        # Press button
        self.pyboy.button_press(action)
        
        # Hold for 2 frames (minimum for registration)
        for _ in range(2):
            if not self.running:
                break
            self.pyboy.tick()
        
        # Release button
        self.pyboy.button_release(action)
        
        # Wait for game to process (14 more frames for 16-frame alignment)
        for _ in range(14):
            if not self.running:
                break
            self.pyboy.tick()
    
    def _execute_battle_action(self, action: str):
        """Execute action in battle with input availability checking"""
        # In battle, we need to wait for input to be available
        # For now, use longer timing to ensure battle system processes
        self.pyboy.button_press(action)
        
        # Hold button for 4 frames
        for _ in range(4):
            if not self.running:
                break
            self.pyboy.tick()
        
        self.pyboy.button_release(action)
        
        # Wait longer for battle system to process (20 frames)
        for _ in range(20):
            if not self.running:
                break
            self.pyboy.tick()
    
    def _execute_menu_action(self, action: str):
        """Execute action in menus/dialogue with appropriate timing"""
        self.pyboy.button_press(action)
        
        # Hold for 3 frames
        for _ in range(3):
            if not self.running:
                break
            self.pyboy.tick()
        
        self.pyboy.button_release(action)
        
        # Wait for menu system to process (8 frames)
        for _ in range(8):
            if not self.running:
                break
            self.pyboy.tick()
    
    def _execute_default_action(self, action: str):
        """Default action timing for unknown states"""
        self.pyboy.button_press(action)
        
        # Hold button for several frames
        for _ in range(8):
            if not self.running:
                break
            self.pyboy.tick()
        
        self.pyboy.button_release(action)
        
        # Wait a few more frames
        for _ in range(4):
            if not self.running:
                break
            self.pyboy.tick()
    
    def _execute_smart_movement(self, direction: str, previous_state: Dict) -> Dict:
        """Execute directional movement with automatic retry logic"""
        # Get initial position
        prev_x = previous_state.get('player_x', 0)
        prev_y = previous_state.get('player_y', 0)
        prev_map = previous_state.get('player_map', 0)
        
        # First attempt: press direction key
        self._execute_single_action(direction)
        
        # Check if we moved
        intermediate_state = self.get_game_state()
        curr_x = intermediate_state.get('player_x', 0)
        curr_y = intermediate_state.get('player_y', 0)
        curr_map = intermediate_state.get('player_map', 0)
        
        # Check if position changed (either coordinates or map)
        position_changed = (curr_x != prev_x) or (curr_y != prev_y) or (curr_map != prev_map)
        
        if not position_changed:
            # First press likely just changed facing direction, try again
            self._execute_single_action(direction)
            
            # Get final state after second attempt
            final_state = self.get_game_state()
            final_x = final_state.get('player_x', 0)
            final_y = final_state.get('player_y', 0)
            final_map = final_state.get('player_map', 0)
            
            # Check if second attempt moved us
            second_attempt_moved = (final_x != curr_x) or (final_y != curr_y) or (final_map != curr_map)
            
            if not second_attempt_moved:
                # Still didn't move - likely blocked by wall or edge
                print(f"🚧 Movement blocked: {direction.upper()} at ({curr_x},{curr_y}) on Map {curr_map}")
            else:
                print(f"↔️ Smart movement: {direction.upper()} → ({prev_x},{prev_y}) → ({final_x},{final_y})")
            
            return final_state
        else:
            # First press actually moved us (unusual but possible)
            print(f"⚡ Direct movement: {direction.upper()} → ({prev_x},{prev_y}) → ({curr_x},{curr_y})")
            return intermediate_state
    
    def _track_experience(self, action: str, previous_state: Dict, current_state: Dict, reward: float):
        """Track experience for learning with enhanced pattern recognition"""
        try:
            # Create state analysis with deeper context
            screen_analysis = self.analyze_screen()
            
            # Get enhanced game context
            game_context = self.llm_agent.game_intelligence.analyze_game_context(previous_state, screen_analysis)
            
            # Create richer situation context including screen state
            situation_context = {
                'phase': game_context.phase.name,
                'location_type': game_context.location_type.name,
                'screen_state': screen_analysis.get('state', 'unknown'),
                'has_pokemon': previous_state.get('party_count', 0) > 0,
                'in_battle': previous_state.get('in_battle', 0) == 1,
                'location_progress': {
                    'map': previous_state.get('player_map', 0),
                    'xy': (previous_state.get('player_x', 0), previous_state.get('player_y', 0))
                }
            }
            
            # Create enhanced situation hash with more context
            situation_hash = self.llm_agent.experience_memory.get_situation_hash(
                previous_state, 
                screen_analysis,
                situation_context
            )
            
            # Track recent situations and actions
            self.recent_situation_hashes.append({
                'hash': situation_hash,
                'context': situation_context,
                'timestamp': time.time()
            })
            self.recent_action_sequences.append({
                'action': action.upper(),
                'reward': reward,
                'state_change': self._get_significant_state_changes(previous_state, current_state)
            })
            
            # Maintain manageable history window
            window_size = self.experience_window
            if len(self.recent_action_sequences) > window_size:
                self.recent_situation_hashes = self.recent_situation_hashes[-window_size:]
                self.recent_action_sequences = self.recent_action_sequences[-window_size:]
            
            # Analyze recent experience patterns
            if len(self.recent_action_sequences) >= 3:
                # Identify action patterns that led to good outcomes
                recent_actions = [a['action'] for a in self.recent_action_sequences[-5:]]
                recent_rewards = [a['reward'] for a in self.recent_action_sequences[-5:]]
                cumulative_reward = sum(recent_rewards)
                
                # Track significant experiences (with richer context)
                experience_significance = self._evaluate_experience_significance(
                    reward, cumulative_reward, current_state, previous_state
                )
                
                if experience_significance['is_significant']:
                    # Record enriched experience
                    self.llm_agent.experience_memory.record_experience(
                        situation_hash=situation_hash,
                        actions=recent_actions,
                        reward=reward,
                        context={
                            **situation_context,
                            'cumulative_reward': cumulative_reward,
                            'significance_type': experience_significance['type'],
                            'pattern_info': experience_significance['pattern'],
                            'state_changes': experience_significance['changes'],
                            'timestamp': time.time()
                        }
                    )
                    
                    # Log significant experience for debugging
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"Recorded significant experience: {experience_significance['type']} "
                            f"[Pattern: {' → '.join(recent_actions)}] "
                            f"Reward: {reward:+.2f} (Cumulative: {cumulative_reward:+.2f})"
                        )
            
        except Exception as e:
            self.logger.error(f"Experience tracking error: {str(e)}")
            self.logger.error("Recent actions:", self.recent_action_sequences[-5:] if self.recent_action_sequences else [])
    
    def _get_significant_state_changes(self, previous: Dict, current: Dict) -> Dict:
        """Identify significant state changes between states"""
        changes = {}
        
        # Check party changes
        if current.get('party_count', 0) != previous.get('party_count', 0):
            changes['party'] = {
                'from': previous.get('party_count', 0),
                'to': current.get('party_count', 0)
            }
        
        # Check level changes
        if current.get('player_level', 0) != previous.get('player_level', 0):
            changes['level'] = {
                'from': previous.get('player_level', 0),
                'to': current.get('player_level', 0)
            }
        
        # Check battle state changes
        prev_battle = bool(previous.get('in_battle', 0))
        curr_battle = bool(current.get('in_battle', 0))
        if prev_battle != curr_battle:
            changes['battle'] = {
                'from': prev_battle,
                'to': curr_battle,
                'enemy_level': current.get('enemy_level', 0) if curr_battle else None
            }
        
        # Check location changes
        prev_loc = (previous.get('player_map', 0), previous.get('player_x', 0), previous.get('player_y', 0))
        curr_loc = (current.get('player_map', 0), current.get('player_x', 0), current.get('player_y', 0))
        if prev_loc != curr_loc:
            changes['location'] = {
                'from': {'map': prev_loc[0], 'x': prev_loc[1], 'y': prev_loc[2]},
                'to': {'map': curr_loc[0], 'x': curr_loc[1], 'y': curr_loc[2]}
            }
        
        return changes
    
    def _evaluate_experience_significance(self, reward: float, cumulative_reward: float,
                                        current: Dict, previous: Dict) -> Dict:
        """Evaluate the significance of an experience with pattern recognition"""
        result = {
            'is_significant': False,
            'type': None,
            'pattern': None,
            'changes': self._get_significant_state_changes(previous, current)
        }
        
        # Check for significant immediate rewards
        if reward > 0.1 or abs(reward) > 10.0:
            result['is_significant'] = True
            result['type'] = 'reward'
        
        # Check for significant game progress
        state_changes = result['changes']
        if state_changes.get('party') or state_changes.get('level'):
            result['is_significant'] = True
            result['type'] = 'progression'
            
        # Check for strategic achievements
        if state_changes.get('location', {}).get('to', {}).get('map') != \
           state_changes.get('location', {}).get('from', {}).get('map'):
            result['is_significant'] = True
            result['type'] = 'exploration'
        
        # Check battle outcomes
        battle_change = state_changes.get('battle', {})
        if battle_change and not battle_change.get('to') and battle_change.get('from'):
            # Battle ended - check if it was significant
            if reward > 5.0:  # Significant positive reward suggests victory
                result['is_significant'] = True
                result['type'] = 'battle_victory'
            elif reward < -2.0:  # Significant negative reward suggests defeat/flee
                result['is_significant'] = True
                result['type'] = 'battle_defeat'
        
        # Check for cumulative success patterns
        if cumulative_reward > 1.0:
            result['is_significant'] = True
            result['type'] = 'success_pattern'
            
        # Add pattern analysis if significant
        if result['is_significant']:
            pattern_info = {
                'context': {
                    'has_pokemon': current.get('party_count', 0) > 0,
                    'in_battle': current.get('in_battle', 0) == 1,
                    'location': (current.get('player_map', 0), current.get('player_x', 0), current.get('player_y', 0))
                },
                'reward_scale': 'major' if abs(reward) > 10.0 else 'minor',
                'cumulative_impact': 'positive' if cumulative_reward > 0 else 'negative'
            }
            result['pattern'] = pattern_info
            
        return result
    
    def update_web_data(self):
        """Update data for enhanced web monitoring"""
        if not self.enable_web or not self.web_monitor:
            return
            
        try:
            # Update core performance stats
            elapsed = time.time() - self.start_time
            self.stats.update({
                'training_time': elapsed,
                'actions_per_second': self.actions_taken / elapsed if elapsed > 0 else 0,
                'memory_usage': self.dqn_agent.get_memory_usage() if self.enable_dqn else 0,
                'training_status': 'running' if self.running else 'stopped'
            })
        except Exception as e:
            self.logger.error(f"Stats update error: {str(e)}")
            self.stats['error'] = str(e)
            return
            
            # Get current state information
            current_game_state = self.get_game_state()
            screen_analysis = self.analyze_screen()
            
            # Calculate various game progress metrics
            progress_metrics = self._calculate_progress_metrics(current_game_state)
            
            # Get enhanced game context
            if hasattr(self.llm_agent, 'context_builder'):
                context = self.llm_agent.context_builder.build_context(
                    current_game_state,
                    self.recent_actions[-1] if self.recent_actions else None,
                    None
                )
                analysis = context.current_analysis
                
                # Update strategic information
                self.stats.update({
                    'game_phase': analysis.phase.name,
                    'criticality': analysis.criticality.value,
                    'phase_progress': analysis.progression_score,
                    'threats': analysis.immediate_threats,
                    'opportunities': analysis.opportunities,
                    'strategic_advice': self._get_strategic_advice(analysis)
                })
            
            # Update reward tracking information
            if hasattr(self, 'reward_history') and self.reward_history:
                recent_rewards = self.reward_history[-10:]
                self.stats.update({
                    'recent_rewards': [
                        {
                            'action': r['action'],
                            'reward': r['total_reward'],
                            'breakdown': r['breakdown'],
                            'timestamp': r['timestamp']
                        } for r in recent_rewards
                    ],
                    'reward_trends': self._calculate_reward_trends(recent_rewards)
                })
            
            # Update DQN metrics if enabled
            if self.enable_dqn and self.dqn_agent:
                dqn_stats = self.dqn_agent.get_training_stats()
                self.stats.update({
                    'dqn_stats': {
                        'steps_trained': dqn_stats['steps_trained'],
                        'epsilon': dqn_stats['epsilon'],
                        'memory_size': dqn_stats['memory_size'],
                        'recent_losses': dqn_stats.get('recent_losses', []),
                        'exploration_rate': dqn_stats.get('exploration_rate', 0.0)
                    }
                })
            
            # Calculate enhanced performance metrics
            reward_rate = len([r for r in self.performance_tracking['reward_window'] if r > 0]) / \
                         max(len(self.performance_tracking['reward_window']), 1)
                         
            llm_accuracy = len([d for d in self.performance_tracking['llm_success_window'] if d]) / \
                          max(len(self.performance_tracking['llm_success_window']), 1)
            
            # Calculate action distribution
            total_actions = sum(self.performance_tracking['action_counts'].values()) or 1
            action_distribution = {
                action: count/total_actions 
                for action, count in self.performance_tracking['action_counts'].items()
            }
            
            # Update performance metrics
            self.stats['recent_stats'].update({
                'reward_rate': reward_rate,
                'exploration_rate': len(getattr(self.reward_calculator, 'visited_maps', set())) / 255,
                'stuck_rate': self.actions_without_reward / max(self.actions_taken, 1),
                'success_rate': reward_rate
            })
            
            # Update training metrics
            self.stats['training_metrics'].update({
                'llm_accuracy': llm_accuracy,
                'dqn_loss': float(np.mean(dqn_stats.get('recent_losses', [0]))) if self.enable_dqn else 0.0,
                'hybrid_balance': self.hybrid_agent.get_balance() if hasattr(self.hybrid_agent, 'get_balance') else 0.5,
                'state_coverage': len(self.stats['session_metrics']['unique_states']) / (255 * 255) * 100
            })
            
            # Track history for trends
            self.web_stats_history['reward_history'].append({
                'timestamp': time.time(),
                'reward_rate': reward_rate,
                'total_reward': self.total_reward,
                'action_dist': action_distribution
            })
            
            # Keep history size manageable
            if len(self.web_stats_history['reward_history']) > 1000:
                self.web_stats_history['reward_history'] = self.web_stats_history['reward_history'][-1000:]
            
            # Update game state and progress
            self.stats.update({
                'final_game_state': current_game_state,
                'screen_state': screen_analysis.get('state', 'unknown'),
                'progress_metrics': progress_metrics,
                'recent_actions': self.recent_actions[-10:],
                'stuck_detection': {
                    'actions_without_reward': self.actions_without_reward,
                    'stuck_threshold': self.stuck_threshold,
                    'stuck_location': getattr(self.llm_agent, 'failsafe_context', {}).get('stuck_location')
                }
            })
            
            # Update screenshot with error handling
            self._update_screenshot()
            
            # Update live memory data for debugging
            if hasattr(self.web_monitor, 'update_game_state'):
                self.web_monitor.update_game_state(current_game_state.copy())
            
            # Update final server stats
            if hasattr(self.web_monitor, 'update_stats'):
                self.web_monitor.update_stats(self.stats)
            
        except Exception as e:
            self.logger.error(f"Web data update error: {str(e)}")
            # Ensure the web interface shows error state
            self.stats['error'] = str(e)
            if hasattr(self.web_monitor, 'update_stats'):
                self.web_monitor.update_stats(self.stats)
    
    def _calculate_progress_metrics(self, state: Dict) -> Dict:
        """Calculate detailed progress metrics"""
        return {
            'game_completion': {
                'badges': (state.get('badges_total', 0) / 16) * 100,
                'pokemon': min((state.get('party_count', 0) / 6) * 100, 100),
                'exploration': len(getattr(self.reward_calculator, 'visited_maps', set())) / 255 * 100
            },
            'current_status': {
                'has_pokemon': state.get('party_count', 0) > 0,
                'in_battle': state.get('in_battle', 0) == 1,
                'pokemon_health': state.get('health_percentage', 0),
                'location': {
                    'map': state.get('player_map', 0),
                    'position': (state.get('player_x', 0), state.get('player_y', 0))
                }
            },
            'milestones': {
                'first_pokemon': state.get('party_count', 0) > 0,
                'first_battle': any(h.get('type') == 'battle_victory' 
                                  for h in getattr(self, 'reward_history', [])),
                'first_badge': state.get('badges_total', 0) > 0
            }
        }
    
    def _get_strategic_advice(self, analysis) -> List[str]:
        """Generate strategic advice based on current analysis"""
        advice = []
        
        # Phase-specific advice
        advice.append(f"Current Phase: {analysis.phase.name}")
        
        # Add critical information
        if analysis.criticality.value >= 4:
            advice.append("⚠️ High-priority situation detected!")
        
        # Add immediate threats
        if analysis.immediate_threats:
            advice.append(f"🔥 Threats: {', '.join(analysis.immediate_threats)}")
        
        # Add opportunities
        if analysis.opportunities:
            advice.append(f"✨ Opportunities: {', '.join(analysis.opportunities)}")
        
        # Add progression advice
        if analysis.progression_score < 50:
            advice.append("🎯 Focus on main objectives to progress")
        elif analysis.progression_score >= 90:
            advice.append("🌟 Excellent progress! Ready for next phase")
        
        return advice
    
    def _calculate_reward_trends(self, recent_rewards: List[Dict]) -> Dict:
        """Calculate trends in recent rewards"""
        if not recent_rewards:
            return {}
            
        rewards = [r['total_reward'] for r in recent_rewards]
        return {
            'average': sum(rewards) / len(rewards),
            'trend': 'improving' if rewards[-1] > rewards[0] else 'declining',
            'consistency': abs(max(rewards) - min(rewards)) < 1.0,
            'peaks': {
                'positive': max(rewards),
                'negative': min(rewards)
            }
        }
    
    def _update_screenshot(self):
        """Update screenshot with error handling and headless support"""
        try:
            if not self.pyboy or not self.web_monitor:
                return
            
            # Force PyBoy to tick once to ensure screen buffer is updated
            # This is crucial for headless mode
            self.pyboy.tick()
            
            # Get screen data from PyBoy
            screen = self.pyboy.screen.ndarray
            
            # Handle different screen formats
            if screen is None or screen.size == 0:
                raise ValueError("Screen buffer is empty")
            
            # Ensure we have the right dimensions (Game Boy screen is 160x144)
            if len(screen.shape) == 2:  # Grayscale
                screen = np.stack([screen] * 3, axis=-1)  # Convert to RGB
            elif screen.shape[2] == 4:  # RGBA to RGB
                screen = screen[:, :, :3]
            elif screen.shape[2] == 1:  # Single channel to RGB
                screen = np.repeat(screen, 3, axis=2)
            
            # Ensure data type is correct for PIL
            if screen.dtype != np.uint8:
                # Convert to 0-255 range if needed
                if screen.max() <= 1.0:
                    screen = (screen * 255).astype(np.uint8)
                else:
                    screen = screen.astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(screen)
            
            # Scale up for better visibility (Game Boy 160x144 -> 480x432 = 3x scale)
            img = img.resize((480, 432), Image.NEAREST)  # Pixel art style scaling
            
            # Save to buffer
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            screenshot_data = buf.getvalue()
            
            # Update server data
            if hasattr(self.web_monitor, 'update_screenshot'):
                self.web_monitor.update_screenshot(screenshot_data)
            
            # Debug info for first few screenshots
            if not hasattr(self, '_screenshot_debug_count'):
                self._screenshot_debug_count = 0
            
            if self._screenshot_debug_count < 3:
                print(f"📷 Screenshot {self._screenshot_debug_count + 1}: {screen.shape}, dtype={screen.dtype}, size={len(screenshot_data)} bytes")
                self._screenshot_debug_count += 1
            
        except Exception as e:
            self.logger.warning(f"Screenshot update failed: {str(e)}")
            
            # Create error indicator image with debug info
            try:
                error_img = Image.new('RGB', (480, 432), (32, 16, 16))  # Dark red background
                draw = ImageDraw.Draw(error_img)
                
                # Add error text
                try:
                    # Try to use default font
                    font = ImageFont.load_default()
                except:
                    font = None
                
                error_text = f"Screenshot Error:\n{str(e)[:100]}"
                if font:
                    draw.text((20, 200), error_text, fill=(255, 255, 255), font=font)
                else:
                    draw.text((20, 200), error_text, fill=(255, 255, 255))
                
                buf = io.BytesIO()
                error_img.save(buf, format='PNG')
                if hasattr(self.web_monitor, 'update_screenshot'):
                    self.web_monitor.update_screenshot(buf.getvalue())
                
            except Exception as inner_e:
                self.logger.error(f"Failed to create error screenshot: {inner_e}")
                # Last resort - create minimal error image
                try:
                    minimal_error = Image.new('RGB', (480, 432), (64, 0, 0))
                    buf = io.BytesIO()
                    minimal_error.save(buf, format='PNG')
                    if hasattr(self.web_monitor, 'update_screenshot'):
                        self.web_monitor.update_screenshot(buf.getvalue())
                except:
                    pass  # Give up on screenshots
    
    def print_progress(self, action: str, decision_source: str, reward: float, reward_breakdown: Dict):
        """Print detailed training progress with strategic context"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        game_state = self.previous_game_state
        screen_analysis = self.analyze_screen()
        
        reward_summary = self.reward_calculator.get_reward_summary(reward_breakdown)
        
        # Enhanced screen analysis display
        screen_info = f"{screen_analysis['state']} (v:{screen_analysis['variance']:.0f}, b:{screen_analysis['brightness']:.0f}, c:{screen_analysis['colors']})"
        
        print(f"⚡ Action {self.actions_taken}/{self.max_actions} | {action.upper()} ({decision_source})")
        print(f"   📊 {aps:.1f} a/s | Screen: {screen_info} | Level: {game_state.get('player_level', 0)} | Badges: {game_state.get('badges_total', 0)}")
        print(f"   💰 Reward: {reward:+.2f} (Total: {self.total_reward:.2f}) | {reward_summary}")
        
        # Add strategic context if available
        if hasattr(self, 'context_builder') and self.context_builder:
            try:
                context = self.context_builder.build_context(game_state, action, reward)
                analysis = context.current_analysis
                
                # Show phase and criticality
                if hasattr(analysis, 'phase') and hasattr(analysis, 'criticality'):
                    print(f"   🎯 Phase: {analysis.phase.value} | Criticality: {analysis.criticality.value}")
                
                # Show immediate threats (limit to 2 for readability)
                if hasattr(analysis, 'immediate_threats') and analysis.immediate_threats:
                    threats_display = analysis.immediate_threats[:2]
                    print(f"   ⚠️ Threats: {', '.join(threats_display)}")
                
                # Show opportunities (limit to 2 for readability)  
                if hasattr(analysis, 'opportunities') and analysis.opportunities:
                    opps_display = analysis.opportunities[:2]
                    print(f"   🎯 Opportunities: {', '.join(opps_display)}")
                    
            except Exception as e:
                # Don't break progress display if context analysis fails
                pass
        
        print()
    
    def save_training_data(self):
        """Save comprehensive training data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save main stats
        stats_file = os.path.join(logs_dir, f"llm_training_stats_{timestamp}.json")
        final_stats = self.stats.copy()
        final_stats['final_game_state'] = self.get_game_state()
        final_stats['llm_decisions'] = len(self.llm_agent.decision_history)
        
        # Convert any sets to lists for JSON serialization
        def convert_sets_to_lists(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            else:
                return obj
        
        final_stats = convert_sets_to_lists(final_stats)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Save detailed decision log with strategic context (from smart trainer)
        if self.decision_log:
            decision_file = os.path.join(logs_dir, f"enhanced_decisions_{timestamp}.json")
            enhanced_decisions = []
            
            for decision in self.decision_log:
                # Enhance with any available strategic context
                enhanced_decision = decision.copy()
                if hasattr(self, 'context_builder') and self.context_builder:
                    try:
                        # Add strategic analysis if available
                        enhanced_decision['enhanced'] = True
                    except:
                        enhanced_decision['enhanced'] = False
                
                enhanced_decisions.append(enhanced_decision)
            
            with open(decision_file, 'w') as f:
                json.dump(enhanced_decisions, f, indent=2)
            
            print(f"📊 Enhanced decisions saved: {decision_file}")
        
        # Save performance log (from smart trainer)
        if self.performance_log:
            performance_file = os.path.join(logs_dir, f"performance_metrics_{timestamp}.json")
            with open(performance_file, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
            
            print(f"📈 Performance metrics saved: {performance_file}")
        
        # Generate training summary (from smart trainer)
        summary = {
            'total_actions': self.actions_taken,
            'total_reward': self.total_reward,
            'training_time': time.time() - self.start_time,
            'avg_reward_per_action': self.total_reward / max(self.actions_taken, 1),
            'llm_decisions_made': len(self.decision_log),
            'actions_per_second': self.actions_taken / max(time.time() - self.start_time, 1),
            'final_performance': self.performance_log[-1] if self.performance_log else {},
            'hybrid_training': self.enable_dqn,
            'web_monitoring': self.enable_web
        }
        
        summary_file = os.path.join(logs_dir, f"training_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed LLM decisions
        if self.llm_agent.decision_history:
            llm_file = os.path.join(logs_dir, f"llm_decisions_{timestamp}.json")
            with open(llm_file, 'w') as f:
                json.dump(self.llm_agent.decision_history, f, indent=2)
            print(f"🧠 LLM decisions saved to {llm_file}")
        
        # Save final DQN model if enabled
        if self.enable_dqn and self.dqn_agent:
            final_model_path = os.path.join(logs_dir, f"dqn_final_model_{timestamp}.pth")
            self.dqn_agent.save_model(final_model_path)
            print(f"🧠 Final DQN model saved to {final_model_path}")
        
        # Save experience memory
        self.llm_agent.experience_memory.save_memory()
        memory_stats = self.llm_agent.experience_memory.get_memory_stats()
        print(f"📚 Experience memory saved: {memory_stats['total_experiences']} experiences, {memory_stats['total_patterns']} patterns")
        
        print(f"📊 Training stats saved to {stats_file}")
    
    def start_training(self):
        """Start the LLM-enhanced training process"""
        print("🚀 Starting LLM-Enhanced Pokemon Crystal RL Training")
        print("=" * 80)
        print(f"🤖 LLM Model: {self.llm_agent.model_name}")
        print(f"🧠 LLM Decision Interval: Every {self.llm_interval} actions")
        print(f"💰 Reward System: Multi-factor Pokemon progress tracking")
        print()
        
        try:
            # Setup components
            self.setup_web_server()
            self.initialize_pyboy()
            
            print(f"🎯 Starting training loop ({self.max_actions} actions)")
            if self.enable_web:
                print(f"🌐 Monitor at: http://localhost:{self.web_port}")
            print("🔄 Press Ctrl+C to stop training gracefully")
            print("=" * 80)
            print()
            
            # Main training loop
            while self.running and self.actions_taken < self.max_actions:
                # Get next action
                action, decision_source = self.get_next_action()
                
                # Execute action and get reward
                reward, reward_breakdown = self.execute_action(action)
                
                # Update web monitoring
                if self.actions_taken % 5 == 0:
                    self.update_web_data()
                
                # Print progress every 50 actions or on LLM decisions
                if (self.actions_taken % 50 == 0 or "LLM" in decision_source):
                    self.print_progress(action, decision_source, reward, reward_breakdown)
            
            # Training completed
            if self.actions_taken >= self.max_actions:
                print(f"✅ LLM training completed! {self.actions_taken} actions executed")
                print(f"🏆 Final reward: {self.total_reward:.2f}")
                print(f"🧠 LLM decisions made: {self.stats['llm_decision_count']}")
                
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Cleanup
            if self.web_monitor:
                try:
                    self.web_monitor.stop()
                except:
                    pass
                    
            if self.pyboy:
                self.pyboy.stop()
            
            self.save_training_data()
    
    # Mock implementations for missing dependencies
    def _create_mock_llm_agent(self):
        """Create a mock LLM agent when the real one is not available"""
        class MockLLMAgent:
            def __init__(self):
                self.available = False
                self.model_name = "mock-agent"
                self.decision_history = []
                self.experience_memory = self._create_mock_experience_memory()
                
            def get_decision(self, game_state, screen_analysis, recent_actions):
                # Simple fallback logic
                if game_state.get('in_battle', 0) == 1:
                    return 'a', "Mock: Attack in battle"
                elif screen_analysis.get('state') == 'dialogue':
                    return 'a', "Mock: Progress dialogue"
                elif screen_analysis.get('state') == 'menu':
                    return 'b', "Mock: Exit menu"
                else:
                    # Simple exploration
                    actions = ['up', 'down', 'left', 'right', 'a']
                    import random
                    action = random.choice(actions)
                    return action, f"Mock: Random {action}"
                    
            def _create_mock_experience_memory(self):
                class MockExperienceMemory:
                    def get_situation_hash(self, *args):
                        return "mock_hash"
                    def record_experience(self, *args, **kwargs):
                        pass
                    def save_memory(self):
                        pass
                    def get_memory_stats(self):
                        return {'total_experiences': 0, 'total_patterns': 0}
                return MockExperienceMemory()
        
        return MockLLMAgent()
    
    def _create_mock_reward_calculator(self):
        """Create a mock reward calculator when the real one is not available"""
        class MockRewardCalculator:
            def __init__(self):
                self.last_screen_state = 'unknown'
                self.prev_screen_state = 'unknown'
                self.last_action = None
                
            def calculate_reward(self, current_state, previous_state):
                # Simple reward logic
                reward = 0.0
                breakdown = {}
                
                # Give small reward for moving
                if (current_state.get('player_x', 0) != previous_state.get('player_x', 0) or
                    current_state.get('player_y', 0) != previous_state.get('player_y', 0)):
                    reward += 0.1
                    breakdown['movement'] = 0.1
                
                # Give big reward for getting pokemon
                if current_state.get('party_count', 0) > previous_state.get('party_count', 0):
                    reward += 10.0
                    breakdown['pokemon'] = 10.0
                
                # Give reward for level up
                if current_state.get('player_level', 0) > previous_state.get('player_level', 0):
                    reward += 5.0
                    breakdown['level'] = 5.0
                
                return reward, breakdown
                
            def get_reward_summary(self, breakdown):
                if not breakdown:
                    return "no rewards"
                return " | ".join([f"{k}: {v:+.2f}" for k, v in breakdown.items() if abs(v) > 0.01])
        
        return MockRewardCalculator()