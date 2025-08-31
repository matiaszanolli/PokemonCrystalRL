#!/usr/bin/env python3
"""
LLM-Enhanced Pokemon Crystal RL Training Script

An advanced training script that combines:
- Local LLM integration for intelligent decision making
- Sophisticated reward function based on Pokemon game progress
- Memory map integration for game state analysis
- Web monitoring with LLM decision tracking
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
from datetime import datetime
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from typing import Dict, List, Tuple, Optional

# Import our memory mapping system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.memory_map import (
    MEMORY_ADDRESSES, 
    DERIVED_VALUES,
    IMPORTANT_LOCATIONS,
    POKEMON_SPECIES,
    STATUS_CONDITIONS,
    BADGE_MASKS,
    get_badges_earned
)

class LLMAgent:
    """Local LLM agent for Pokemon Crystal decision making"""
    
    def __init__(self, model_name="smollm2:1.7b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.decision_history = []
        self.last_decision_time = 0
        
        # Test LLM availability
        self.available = self._test_llm_connection()
        
    def _test_llm_connection(self) -> bool:
        """Test if LLM is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_decision(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str]) -> Tuple[str, str]:
        """Get LLM decision based on game state"""
        if not self.available:
            return self._fallback_decision(game_state), "LLM unavailable - using fallback"
        
        try:
            prompt = self._build_prompt(game_state, screen_analysis, recent_actions)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 50
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                action = self._parse_llm_response(result.get('response', ''))
                reasoning = result.get('response', '').strip()
                
                # Track decision
                self.decision_history.append({
                    'timestamp': time.time(),
                    'action': action,
                    'reasoning': reasoning,
                    'game_state': game_state.copy()
                })
                
                return action, reasoning
            else:
                return self._fallback_decision(game_state), "LLM request failed"
                
        except Exception as e:
            return self._fallback_decision(game_state), f"LLM error: {str(e)}"
    
    def _build_prompt(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str]) -> str:
        """Build context-aware prompt for LLM"""
        
        # Game state summary
        player_info = f"Player: Level {game_state.get('player_level', '?')}, HP {game_state.get('player_hp', 0)}/{game_state.get('player_max_hp', 1)}"
        location_info = f"Map: {game_state.get('player_map', 'unknown')} at ({game_state.get('player_x', 0)}, {game_state.get('player_y', 0)})"
        badges_info = f"Badges: {game_state.get('badges_total', 0)}/16"
        money_info = f"Money: ¥{game_state.get('money', 0)}"
        party_info = f"Party: {game_state.get('party_count', 0)} Pokemon"
        
        # Screen analysis
        screen_state = screen_analysis.get('state', 'unknown')
        screen_variance = screen_analysis.get('variance', 0)
        
        # Recent actions context
        recent = " → ".join(recent_actions[-5:]) if recent_actions else "None"
        
        # Battle context
        battle_context = ""
        if game_state.get('in_battle', 0) == 1:
            enemy_level = game_state.get('enemy_level', 0)
            enemy_species = game_state.get('enemy_species', 0)
            battle_context = f"\n🔥 IN BATTLE: Enemy Level {enemy_level} (Species {enemy_species})"
        
        prompt = f"""You are an AI playing Pokemon Crystal. Make the best action choice based on the current situation.

CURRENT STATUS:
{player_info}
{location_info}
{badges_info}
{money_info}
{party_info}
Screen State: {screen_state} (variance: {screen_variance:.1f})
Recent Actions: {recent}
{battle_context}

AVAILABLE ACTIONS:
up, down, left, right - Movement
a - Interact/Confirm/Attack
b - Cancel/Back
start - Menu
select - Select button

STRATEGY PRIORITIES:
1. If in battle: Use effective attacks (mostly 'a')
2. If in dialogue: Progress with 'a'  
3. If in menu: Navigate with directional keys, confirm with 'a'
4. If in overworld: Explore new areas, talk to NPCs, find items
5. Avoid getting stuck - vary actions if repeating

Choose ONE action and briefly explain why. Format: ACTION: [action]
Reasoning: [brief explanation]

Your choice:"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> str:
        """Parse LLM response to extract action"""
        response = response.lower().strip()
        
        # Look for ACTION: pattern
        if "action:" in response:
            action_part = response.split("action:")[1].split("\n")[0].strip()
            # Extract first word that looks like an action
            words = action_part.split()
            for word in words:
                clean_word = word.strip('.,!?()[]{}').lower()
                if clean_word in ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']:
                    return clean_word
        
        # Look for common action words in the response
        valid_actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
        for action in valid_actions:
            if action in response:
                return action
        
        # Fallback to 'a' if nothing found
        return 'a'
    
    def _fallback_decision(self, game_state: Dict) -> str:
        """Fallback decision when LLM is unavailable"""
        # Smart fallback based on game state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif game_state.get('menu_state', 0) != 0:
            return 'a'  # Navigate menus
        else:
            # Exploration pattern
            actions = ['up', 'right', 'down', 'left', 'a']
            return actions[int(time.time()) % len(actions)]

class PokemonRewardCalculator:
    """Sophisticated reward calculation for Pokemon Crystal"""
    
    def __init__(self):
        self.previous_state = {}
        self.exploration_bonus = {}
        self.last_reward_time = time.time()
        
    def calculate_reward(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward based on game progress"""
        rewards = {}
        total_reward = 0.0
        
        # 1. Health and survival rewards
        hp_reward = self._calculate_hp_reward(current_state, previous_state)
        rewards['health'] = hp_reward
        total_reward += hp_reward
        
        # 2. Level progression rewards
        level_reward = self._calculate_level_reward(current_state, previous_state)
        rewards['level'] = level_reward
        total_reward += level_reward
        
        # 3. Badge progression rewards (major milestone)
        badge_reward = self._calculate_badge_reward(current_state, previous_state)
        rewards['badges'] = badge_reward
        total_reward += badge_reward
        
        # 4. Money and item rewards
        money_reward = self._calculate_money_reward(current_state, previous_state)
        rewards['money'] = money_reward
        total_reward += money_reward
        
        # 5. Exploration rewards
        exploration_reward = self._calculate_exploration_reward(current_state, previous_state)
        rewards['exploration'] = exploration_reward
        total_reward += exploration_reward
        
        # 6. Battle performance rewards
        battle_reward = self._calculate_battle_reward(current_state, previous_state)
        rewards['battle'] = battle_reward
        total_reward += battle_reward
        
        # 7. Progress and efficiency penalties
        efficiency_penalty = self._calculate_efficiency_penalty(current_state)
        rewards['efficiency'] = efficiency_penalty
        total_reward += efficiency_penalty
        
        # 7. Early game progression rewards (getting first Pokemon, etc.)
        progression_reward = self._calculate_progression_reward(current_state, previous_state)
        rewards['progression'] = progression_reward
        total_reward += progression_reward
        
        # 8. Time-based small negative reward to encourage efficiency
        time_penalty = -0.01  # Small penalty each step
        rewards['time'] = time_penalty
        total_reward += time_penalty
        
        return total_reward, rewards
    
    def _calculate_hp_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for maintaining/improving health"""
        # Only calculate health rewards if player has Pokemon
        party_count = current.get('party_count', 0)
        if party_count == 0:
            return 0.0  # No Pokemon = no health rewards/penalties
            
        curr_hp = current.get('player_hp', 0)
        curr_max_hp = current.get('player_max_hp', 1)
        prev_hp = previous.get('player_hp', curr_hp)
        prev_max_hp = previous.get('player_max_hp', curr_max_hp)
        
        # Skip if no valid HP data
        if curr_max_hp == 0 or prev_max_hp == 0:
            return 0.0
        
        curr_hp_pct = curr_hp / curr_max_hp
        prev_hp_pct = prev_hp / prev_max_hp
        
        # Reward health improvement, penalize health loss
        hp_change = curr_hp_pct - prev_hp_pct
        
        if hp_change > 0:
            return hp_change * 5.0  # Reward healing
        elif hp_change < 0:
            return hp_change * 10.0  # Penalty for taking damage
        
        # Small bonus for staying healthy
        if curr_hp_pct > 0.8:
            return 0.1
        elif curr_hp_pct < 0.2:
            return -0.5  # Penalty for being low on health
            
        return 0.0
    
    def _calculate_level_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for leveling up Pokemon"""
        curr_level = current.get('player_level', 0)
        prev_level = previous.get('player_level', curr_level)
        
        if curr_level > prev_level:
            level_gain = curr_level - prev_level
            return level_gain * 50.0  # Big reward for leveling up
            
        return 0.0
    
    def _calculate_badge_reward(self, current: Dict, previous: Dict) -> float:
        """Huge reward for earning badges (major milestones)"""
        curr_badges = current.get('badges_total', 0)
        prev_badges = previous.get('badges_total', curr_badges)
        
        if curr_badges > prev_badges:
            badge_gain = curr_badges - prev_badges
            return badge_gain * 500.0  # Huge reward for badge progress!
            
        return 0.0
    
    def _calculate_money_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for earning money"""
        curr_money = current.get('money', 0)
        prev_money = previous.get('money', curr_money)
        
        money_change = curr_money - prev_money
        if money_change > 0:
            return min(money_change * 0.01, 5.0)  # Cap money rewards
        elif money_change < 0:
            return max(money_change * 0.005, -2.0)  # Small penalty for spending
            
        return 0.0
    
    def _calculate_exploration_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for exploring new areas"""
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # New map reward
        if curr_map != prev_map:
            return 10.0  # Reward for entering new area
        
        # Movement reward (small)
        distance = abs(curr_x - prev_x) + abs(curr_y - prev_y)
        if distance > 0:
            return min(distance * 0.1, 1.0)  # Small reward for moving
            
        return 0.0
    
    def _calculate_battle_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for battle performance"""
        curr_in_battle = current.get('in_battle', 0)
        prev_in_battle = previous.get('in_battle', 0)
        
        # Entered battle
        if curr_in_battle == 1 and prev_in_battle == 0:
            return 2.0  # Small reward for engaging in battle
            
        # Exited battle (assuming victory if health didn't drop significantly)
        if curr_in_battle == 0 and prev_in_battle == 1:
            curr_hp_pct = current.get('player_hp', 0) / max(current.get('player_max_hp', 1), 1)
            if curr_hp_pct > 0.5:  # Likely won the battle
                return 20.0  # Good reward for winning battle
            else:
                return -5.0  # Penalty for losing/fleeing
                
        return 0.0
    
    def _calculate_efficiency_penalty(self, current: Dict) -> float:
        """Penalty for inefficient actions"""
        # Penalty for being stuck (no progress)
        # This could be enhanced with more sophisticated stuck detection
        return 0.0  # Placeholder for now
    
    def _calculate_progression_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for early game progression milestones"""
        curr_party_count = current.get('party_count', 0)
        prev_party_count = previous.get('party_count', 0)
        
        # Huge reward for getting first Pokemon (major early game milestone)
        if curr_party_count > prev_party_count:
            if prev_party_count == 0 and curr_party_count == 1:
                return 100.0  # First Pokemon is a huge milestone!
            else:
                return 25.0  # Additional Pokemon also rewarded
        
        return 0.0
    
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get human-readable reward summary"""
        summary_parts = []
        for category, value in rewards.items():
            if abs(value) > 0.01:  # Only show significant rewards
                summary_parts.append(f"{category}: {value:+.2f}")
        
        return " | ".join(summary_parts) if summary_parts else "no rewards"

class WebMonitor(BaseHTTPRequestHandler):
    """Enhanced web server with LLM decision tracking"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pokemon Crystal LLM RL Training Monitor</title>
                <meta http-equiv="refresh" content="2">
                <style>
                    body { font-family: Arial; margin: 20px; background: #0a0a0a; color: #fff; }
                    .header { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .stats { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
                    .stat-box { background: #1a1a1a; padding: 15px; border-radius: 8px; min-width: 150px; border: 1px solid #333; }
                    .stat-value { font-size: 24px; font-weight: bold; color: #00ff88; }
                    .stat-label { color: #aaa; font-size: 12px; text-transform: uppercase; }
                    .screen { text-align: center; background: #1a1a1a; padding: 20px; border-radius: 8px; border: 1px solid #333; }
                    .game-screen { border: 2px solid #00ff88; border-radius: 4px; }
                    .decisions { background: #1a1a1a; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #333; }
                    .llm-decision { background: #2a2a2a; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 4px solid #00ff88; }
                    .reward-info { background: #1a1a1a; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #333; }
                    .positive-reward { color: #00ff88; }
                    .negative-reward { color: #ff4444; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Pokemon Crystal LLM RL Training Monitor</h1>
                    <p>Advanced RL training with local LLM decision making and sophisticated reward system</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value" id="actions">-</div>
                        <div class="stat-label">Actions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="aps">-</div>
                        <div class="stat-label">Actions/Sec</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="llm-decisions">-</div>
                        <div class="stat-label">LLM Decisions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="total-reward">-</div>
                        <div class="stat-label">Total Reward</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="level">-</div>
                        <div class="stat-label">Player Level</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="badges">-</div>
                        <div class="stat-label">Badges</div>
                    </div>
                </div>
                
                <div class="screen">
                    <h3>🖼️ Live Game Screen</h3>
                    <img id="gameScreen" class="game-screen" src="/screenshot" width="320" height="288" alt="Game Screen">
                </div>
                
                <div class="reward-info">
                    <h3>💰 Current Reward Breakdown</h3>
                    <div id="reward-breakdown">Loading...</div>
                </div>
                
                <div class="decisions">
                    <h3>🧠 Recent LLM Decisions</h3>
                    <div id="llm-decisions-list">Loading...</div>
                </div>
                
                <script>
                    async function updateStats() {
                        try {
                            const response = await fetch('/stats');
                            const stats = await response.json();
                            
                            document.getElementById('actions').textContent = stats.actions_taken || '-';
                            document.getElementById('aps').textContent = (stats.actions_per_second || 0).toFixed(1);
                            document.getElementById('llm-decisions').textContent = stats.llm_decision_count || '-';
                            document.getElementById('total-reward').textContent = (stats.total_reward || 0).toFixed(2);
                            document.getElementById('level').textContent = stats.player_level || '-';
                            document.getElementById('badges').textContent = (stats.badges_total || 0) + '/16';
                            
                            // Update reward breakdown
                            const rewardDiv = document.getElementById('reward-breakdown');
                            if (stats.last_reward_breakdown) {
                                rewardDiv.innerHTML = stats.last_reward_breakdown;
                            }
                            
                            // Update LLM decisions
                            const decisionsDiv = document.getElementById('llm-decisions-list');
                            if (stats.recent_llm_decisions) {
                                decisionsDiv.innerHTML = stats.recent_llm_decisions.map(d => 
                                    `<div class="llm-decision">
                                        <strong>Action:</strong> ${d.action} | 
                                        <strong>Reasoning:</strong> ${d.reasoning.substring(0, 100)}...
                                    </div>`
                                ).join('');
                            }
                            
                        } catch (e) {
                            console.error('Failed to update stats:', e);
                        }
                    }
                    
                    setInterval(updateStats, 1000);
                    updateStats();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats = getattr(self.server, 'trainer_stats', {})
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path == '/screenshot':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            screenshot_data = getattr(self.server, 'screenshot_data', None)
            if screenshot_data:
                self.wfile.write(screenshot_data)
            else:
                # Send empty 1x1 PNG if no screenshot
                empty_img = Image.new('RGB', (1, 1), (0, 0, 0))
                buf = io.BytesIO()
                empty_img.save(buf, format='PNG')
                self.wfile.write(buf.getvalue())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs

class LLMPokemonTrainer:
    """Advanced Pokemon Crystal trainer with LLM integration and reward system"""
    
    def __init__(self, rom_path, max_actions=5000, llm_model="smollm2:1.7b", 
                 llm_interval=20, enable_web=True, web_port=8080):
        self.rom_path = rom_path
        self.max_actions = max_actions
        self.llm_interval = llm_interval
        self.enable_web = enable_web
        self.web_port = web_port
        
        # Core components
        self.pyboy = None
        self.llm_agent = LLMAgent(llm_model)
        self.reward_calculator = PokemonRewardCalculator()
        
        # Training state
        self.actions_taken = 0
        self.start_time = time.time()
        self.previous_game_state = {}
        self.total_reward = 0.0
        self.last_llm_decision_action = 0
        
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
                    byte1 = self.pyboy.get_memory_value(addr)
                    byte2 = self.pyboy.get_memory_value(addr + 1)
                    byte3 = self.pyboy.get_memory_value(addr + 2)
                    
                    # Convert BCD to decimal
                    def bcd_to_decimal(byte):
                        high = (byte >> 4) & 0xF
                        low = byte & 0xF
                        return high * 10 + low if high <= 9 and low <= 9 else 0
                    
                    money = bcd_to_decimal(byte1) * 10000 + bcd_to_decimal(byte2) * 100 + bcd_to_decimal(byte3)
                    state['money'] = money
                else:
                    state[name] = self.pyboy.get_memory_value(addr)
            except:
                state[name] = 0
        
        # Calculate derived values
        for name, func in DERIVED_VALUES.items():
            try:
                state[name] = func(state)
            except:
                state[name] = 0
        
        # Add badge parsing
        johto_badges = state.get('badges', 0)
        kanto_badges = state.get('kanto_badges', 0)
        state['badges_earned'] = get_badges_earned(johto_badges, kanto_badges)
        state['badges_total'] = len(state['badges_earned'])
        
        return state
    
    def analyze_screen(self) -> Dict:
        """Analyze current screen state"""
        if not self.pyboy:
            return {'state': 'unknown', 'variance': 0, 'colors': 0}
            
        screen = self.pyboy.screen.ndarray
        variance = float(np.var(screen.astype(np.float32)))
        
        # Get additional screen analysis metrics
        unique_colors = len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        brightness = float(np.mean(screen.astype(np.float32)))
        
        # Improved state detection with multiple metrics
        if variance < 50:
            state = "loading"
        elif variance < 1000:
            state = "menu"
        elif brightness > 200 and unique_colors < 10:
            state = "dialogue"  # Dialogue often has high brightness, few colors
        elif variance > 20000:
            state = "battle"  # Battles have very high variance
        else:
            state = "overworld"  # Default to overworld for exploration
            
        return {
            'state': state,
            'variance': variance,
            'colors': unique_colors,
            'brightness': brightness
        }
    
    def get_next_action(self) -> Tuple[str, str]:
        """Get next action using LLM or fallback logic"""
        game_state = self.get_game_state()
        screen_analysis = self.analyze_screen()
        
        # Use LLM every N actions
        use_llm = (self.actions_taken - self.last_llm_decision_action) >= self.llm_interval
        
        if use_llm and self.llm_agent.available:
            action, reasoning = self.llm_agent.get_decision(game_state, screen_analysis, self.recent_actions)
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
            return self._get_rule_based_action(game_state, screen_analysis), "Rule-based fallback"
    
    def _get_rule_based_action(self, game_state: Dict, screen_analysis: Dict) -> str:
        """Rule-based fallback action"""
        state_type = screen_analysis.get('state', 'unknown')
        
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'menu':
            # Simple menu navigation
            menu_actions = ['up', 'down', 'a', 'b']
            return menu_actions[self.actions_taken % len(menu_actions)]
        else:
            # Exploration pattern
            exploration_actions = ['up', 'up', 'a', 'right', 'right', 'a', 'down', 'down', 'a', 'left', 'left', 'a']
            return exploration_actions[self.actions_taken % len(exploration_actions)]
    
    def execute_action(self, action: str):
        """Execute action and calculate rewards"""
        if not self.running or not self.pyboy:
            return
            
        # Store previous state for reward calculation
        previous_state = self.previous_game_state.copy()
        
        # Execute the action
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
        
        # Get new state and calculate rewards
        current_state = self.get_game_state()
        reward, reward_breakdown = self.reward_calculator.calculate_reward(current_state, previous_state)
        
        # Update tracking
        self.actions_taken += 1
        self.total_reward += reward
        self.previous_game_state = current_state
        
        # Update recent actions
        self.recent_actions.append(action.upper())
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # Update stats
        self.stats['actions_taken'] = self.actions_taken
        self.stats['total_reward'] = float(self.total_reward)
        self.stats['player_level'] = current_state.get('player_level', 0)
        self.stats['badges_total'] = current_state.get('badges_total', 0)
        self.stats['last_reward_breakdown'] = self.reward_calculator.get_reward_summary(reward_breakdown)
        
        return reward, reward_breakdown
    
    def update_web_data(self):
        """Update data for web monitoring"""
        if not self.enable_web or not self.web_server:
            return
            
        # Update performance stats
        elapsed = time.time() - self.start_time
        self.stats['training_time'] = elapsed
        self.stats['actions_per_second'] = self.actions_taken / elapsed if elapsed > 0 else 0
        
        # Update screenshot
        try:
            screen = self.pyboy.screen.ndarray
            if screen.shape[2] == 4:  # RGBA to RGB
                screen = screen[:, :, :3]
            
            img = Image.fromarray(screen)
            img = img.resize((320, 288), Image.NEAREST)
            
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            self.web_server.screenshot_data = buf.getvalue()
            
        except Exception as e:
            pass  # Ignore screenshot errors
        
        # Update server stats
        self.web_server.trainer_stats = self.stats.copy()
    
    def print_progress(self, action: str, decision_source: str, reward: float, reward_breakdown: Dict):
        """Print detailed training progress"""
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
        print()
    
    def save_training_data(self):
        """Save comprehensive training data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main stats
        stats_file = f"llm_training_stats_{timestamp}.json"
        final_stats = self.stats.copy()
        final_stats['final_game_state'] = self.get_game_state()
        final_stats['llm_decisions'] = len(self.llm_agent.decision_history)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Save detailed LLM decisions
        if self.llm_agent.decision_history:
            llm_file = f"llm_decisions_{timestamp}.json"
            with open(llm_file, 'w') as f:
                json.dump(self.llm_agent.decision_history, f, indent=2)
            print(f"🧠 LLM decisions saved to {llm_file}")
        
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
            if self.web_server:
                try:
                    self.web_server.shutdown()
                except:
                    pass
                    
            if self.pyboy:
                self.pyboy.stop()
            
            self.save_training_data()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Enhanced Pokemon Crystal RL Training")
    parser.add_argument("--rom", default="roms/pokemon_crystal.gbc", help="ROM file path")
    parser.add_argument("--actions", type=int, default=2000, help="Number of actions to execute")
    parser.add_argument("--llm-model", default="smollm2:1.7b", 
                       choices=["smollm2:1.7b", "llama3.2:1b", "llama3.2:3b", "deepseek-coder:latest"],
                       help="LLM model to use")
    parser.add_argument("--llm-interval", type=int, default=20, 
                       help="Actions between LLM decisions")
    parser.add_argument("--web-port", type=int, default=8080, help="Web monitoring port")
    parser.add_argument("--no-web", action="store_true", help="Disable web monitoring")
    
    args = parser.parse_args()
    
    # Validate ROM file
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        return 1
    
    # Create and start trainer
    trainer = LLMPokemonTrainer(
        rom_path=args.rom,
        max_actions=args.actions,
        llm_model=args.llm_model,
        llm_interval=args.llm_interval,
        enable_web=not args.no_web,
        web_port=args.web_port
    )
    
    success = trainer.start_training()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
