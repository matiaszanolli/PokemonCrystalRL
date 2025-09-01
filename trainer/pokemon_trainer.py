"""
Main Pokemon Crystal RL trainer implementation.

This module contains the main trainer class that coordinates the LLM agent,
reward system, and monitoring components.
"""

import time
import numpy as np
import signal
import sys
import os
import threading
from datetime import datetime
from typing import Dict, List, Tuple
from pyboy import PyBoy
from http.server import HTTPServer
import json

from core.memory_map_new import (
    MEMORY_ADDRESSES,
    DERIVED_VALUES,
    get_badges_earned
)
from core.dqn_agent import DQNAgent, HybridAgent
from .llm import LLMAgent
from .rewards import PokemonRewardCalculator
from .monitoring import WebMonitor

class LLMPokemonTrainer:
    """Advanced Pokemon Crystal trainer with LLM integration and reward system"""
    
    def __init__(self, rom_path, max_actions=5000, llm_model="smollm2:1.7b", 
                 llm_interval=20, enable_web=True, web_port=8080, enable_dqn=True, 
                 dqn_model_path=None):
        self.rom_path = rom_path
        self.data_dir = os.path.join(os.path.dirname(rom_path), 'training_data')
        self.max_actions = max_actions
        self.llm_interval = llm_interval
        self.enable_web = enable_web
        self.web_port = web_port
        self.enable_dqn = enable_dqn
        
        # Core components
        self.pyboy = None
        self.llm_agent = LLMAgent(llm_model)
        self.reward_calculator = PokemonRewardCalculator()
        
        # DQN components
        self.dqn_agent = None
        self.hybrid_agent = None
        self.dqn_training_frequency = 4  # Train DQN every N actions
        self.dqn_save_frequency = 500  # Save DQN model every N actions
        
        if self.enable_dqn:
            # Initialize DQN agent
            self.dqn_agent = DQNAgent(
                state_size=32,
                action_size=8,
                learning_rate=1e-4,
                gamma=0.99,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                memory_size=50000,
                batch_size=32,
                target_update=1000
            )
            
            # Load existing model if provided
            if dqn_model_path and os.path.exists(dqn_model_path):
                self.dqn_agent.load_model(dqn_model_path)
            
            # Create hybrid agent combining LLM and DQN
            self.hybrid_agent = HybridAgent(
                dqn_agent=self.dqn_agent,
                llm_agent=self.llm_agent,
                dqn_weight=0.2,  # Start with low DQN influence
                exploration_bonus=0.1
            )
            
            print(f"🧠 DQN Agent initialized with {self.dqn_agent.device}")
        
        # Experience tracking
        self.recent_situation_hashes = []
        self.recent_action_sequences = []
        self.experience_window = 10  # Track last N actions for experience recording
        
        # Training state
        self.actions_taken = 0
        self.start_time = time.time()
        self.previous_game_state = {}
        self.total_reward = 0.0
        self.last_llm_decision_action = 0
        
        # Failsafe mechanism for stuck detection
        self.last_positive_reward_action = 0
        self.actions_without_reward = 0
        self.stuck_threshold = 100  # Actions without reward before intervention
        self.location_stuck_tracker = {}  # Track how long we've been in same location
        
        # Movement history tracking for intelligent decisions
        self.movement_history = []  # Track recent movement attempts and their success/failure
        self.movement_history_limit = 20  # Keep last N movement attempts
        
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
        
        # Web server setup
        self.web_server = None
        self.web_thread = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        print(f"🤖 LLM Agent: {llm_model} {'✅' if self.llm_agent.available else '❌'}")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n⏸️ Shutting down LLM training...")
        self.running = False
        
        if self.web_server:
            print("Stopping web server...")
            try:
                self.web_server.shutdown()
            except:
                pass
            
        if self.pyboy:
            self.pyboy.stop()
            
        self.save_training_data()
        print("✅ Training stopped cleanly")
        sys.exit(0)
    
    def setup_web_server(self):
        """Setup enhanced web monitoring server"""
        if not self.enable_web:
            return
            
        try:
            self.web_server = HTTPServer(('localhost', self.web_port), WebMonitor)
            self.web_server.trainer_stats = self.stats
            self.web_server.screenshot_data = None
            
            self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_thread.start()
            
            print(f"🌐 Enhanced web monitor: http://localhost:{self.web_port}")
            
        except Exception as e:
            print(f"⚠️ Failed to start web server: {e}")
            self.enable_web = False
    
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
        
        # Get initial game state
        self.previous_game_state = self.get_game_state()
    
    def get_game_state(self) -> Dict:
        """Extract comprehensive game state from memory"""
        if not self.pyboy:
            return {}
        
        state = {}
        
        # Read all memory addresses
        for name, addr in MEMORY_ADDRESSES.items():
            try:
                if name in ['money']:  # Special BCD handling
                    # Read 3 bytes for money in BCD format
                    byte1 = self.pyboy.memory[addr]
                    byte2 = self.pyboy.memory[addr + 1]
                    byte3 = self.pyboy.memory[addr + 2]
                    
                    # Convert BCD to decimal
                    def bcd_to_decimal(byte):
                        high = (byte >> 4) & 0xF
                        low = byte & 0xF
                        return high * 10 + low if high <= 9 and low <= 9 else 0
                    
                    money = bcd_to_decimal(byte1) * 10000 + bcd_to_decimal(byte2) * 100 + bcd_to_decimal(byte3)
                    state['money'] = money
                else:
                    state[name] = self.pyboy.memory[addr]
            except:
                state[name] = 0
        
        # Use alt coordinates if main coordinates are 0
        if state.get('player_x', 0) == 0 and state.get('alt_x', 0) != 0:
            state['player_x'] = state['alt_x']
        if state.get('player_y', 0) == 0 and state.get('alt_y', 0) != 0:
            state['player_y'] = state['alt_y']
        
        # Calculate derived values
        for name, func in DERIVED_VALUES.items():
            try:
                state[name] = func(state)
            except:
                state[name] = 0
        
        # Add badge parsing with comprehensive sanitization for uninitialized memory
        johto_badges = state.get('badges', 0)
        kanto_badges = state.get('kanto_badges', 0)
        player_level = state.get('player_level', 0)
        party_count = state.get('party_count', 0)
        
        # Sanitize implausible values that indicate uninitialized memory
        early_game_indicators = party_count == 0 or player_level == 0
        invalid_badge_values = johto_badges == 0xFF or kanto_badges == 0xFF or johto_badges > 0x80 or kanto_badges > 0x80
        
        if early_game_indicators and invalid_badge_values:
            johto_badges = 0
            kanto_badges = 0
        
        # Additional sanity check: if level > 100 (impossible in Pokemon), sanitize everything
        if player_level > 100:
            state['player_level'] = 0
            johto_badges = 0
            kanto_badges = 0
        
        state['badges_earned'] = get_badges_earned(johto_badges, kanto_badges)
        state['badges_total'] = len(state['badges_earned'])
        
        return state
    
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
                self.stats['recent_llm_decisions'].append({
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp': time.time()
                })
                
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
        """Rule-based fallback action with improved screen state awareness"""
        state_type = screen_analysis.get('state', 'unknown')
        
        # Screen state priorities
        if state_type == 'settings_menu':
            return 'b'  # Always exit settings menu
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'menu' and game_state.get('menu_state', 0) != 0:
            return 'b'  # Exit unwanted menus
        elif game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        else:
            # In overworld or unknown - cycle through movement/interaction
            actions = ['up', 'right', 'down', 'left', 'a']
            return actions[int(time.time()) % len(actions)]
    
    def save_training_data(self):
        """Save training data and statistics"""
        # Create training data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Save final statistics
        stats_path = os.path.join(self.data_dir, 'final_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save DQN model if enabled
        if self.enable_dqn and self.dqn_agent:
            dqn_path = os.path.join(self.data_dir, 'dqn_model.pt')
            self.dqn_agent.save_model(dqn_path)
            
        # Save game save state if needed
        if self.pyboy:
            save_state_path = self.rom_path + '.state'
            try:
                with open(save_state_path, 'wb') as f:
                    self.pyboy.save_state(f)
            except Exception as e:
                print(f"⚠️ Failed to save game state: {e}")
