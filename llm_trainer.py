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
b - Cancel/Back/Exit Menu
start - Open Menu
select - Select button

CRITICAL SCREEN STATE RULES:
🔥 BATTLE: Use 'a' to attack
💬 DIALOGUE: Use 'a' to progress text
⚙️ SETTINGS_MENU: Use 'b' to exit (you're stuck in settings!)
📋 MENU: Use 'b' to exit unless you have a specific goal
🌍 OVERWORLD: Explore with movement + 'a' to interact
⏳ LOADING: Wait (any action is fine)

IMPORTANT GUIDELINES:
- If screen_state is 'settings_menu': ALWAYS use 'b' to escape
- If screen_state is 'menu' and you didn't intend to open it: use 'b'
- If screen_state is 'dialogue': ALWAYS use 'a' to progress
- If recent actions show 'START' but you're in 'settings_menu': use 'b' to exit
- Only use 'start' when you specifically need to access menu for healing/items
- 'b' is your escape key - use it liberally to exit unwanted screens

STRATEGY PRIORITIES:
1. If stuck in settings_menu: Press 'b' immediately
2. If in battle: Use 'a' to attack
3. If in dialogue: Use 'a' to progress
4. If in unwanted menu: Use 'b' to exit
5. If in overworld: Explore and interact

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
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Pokemon Crystal LLM RL Training Dashboard</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
                        color: #fff;
                        min-height: 100vh;
                        overflow-x: hidden;
                    }
                    
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                        position: sticky;
                        top: 0;
                        z-index: 100;
                    }
                    
                    .header h1 {
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 8px;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    }
                    
                    .header p {
                        opacity: 0.9;
                        font-size: 16px;
                    }
                    
                    .status-indicator {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #00ff88;
                        margin-left: 10px;
                        animation: pulse 2s infinite;
                    }
                    
                    @keyframes pulse {
                        0% { opacity: 1; }
                        50% { opacity: 0.5; }
                        100% { opacity: 1; }
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 0 20px;
                        display: grid;
                        grid-template-columns: 2fr 1fr;
                        gap: 20px;
                    }
                    
                    .main-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .side-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .stat-card {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .stat-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
                    }
                    
                    .stat-card::before {
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 3px;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                    }
                    
                    .stat-value {
                        font-size: 32px;
                        font-weight: 800;
                        color: #00ff88;
                        margin-bottom: 5px;
                        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
                    }
                    
                    .stat-label {
                        font-size: 14px;
                        color: #aaa;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        font-weight: 500;
                    }
                    
                    .stat-change {
                        font-size: 12px;
                        margin-top: 5px;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }
                    
                    .stat-change.positive { color: #00ff88; }
                    .stat-change.negative { color: #ff4757; }
                    
                    .game-screen-container {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        text-align: center;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .game-screen-container h3 {
                        margin-bottom: 15px;
                        font-size: 20px;
                        color: #fff;
                    }
                    
                    .game-screen {
                        border-radius: 8px;
                        border: 2px solid #00ff88;
                        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                        max-width: 100%;
                        height: auto;
                        image-rendering: pixelated;
                        image-rendering: crisp-edges;
                    }
                    
                    .screen-info {
                        display: flex;
                        justify-content: space-between;
                        margin-top: 15px;
                        font-size: 12px;
                        color: #aaa;
                    }
                    
                    .panel {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    }
                    
                    .panel h3 {
                        font-size: 18px;
                        margin-bottom: 15px;
                        color: #fff;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    .reward-bars {
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }
                    
                    .reward-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 6px;
                        padding: 8px 12px;
                        display: flex;
                        justify-content: between;
                        align-items: center;
                        font-size: 14px;
                    }
                    
                    .reward-bar.positive {
                        background: linear-gradient(90deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
                        border-left: 3px solid #00ff88;
                    }
                    
                    .reward-bar.negative {
                        background: linear-gradient(90deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
                        border-left: 3px solid #ff4757;
                    }
                    
                    .llm-decision {
                        background: rgba(255,255,255,0.05);
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        border-left: 4px solid #00ff88;
                        position: relative;
                    }
                    
                    .llm-decision::before {
                        content: '🧠';
                        position: absolute;
                        top: 12px;
                        right: 12px;
                        font-size: 16px;
                    }
                    
                    .llm-action {
                        font-weight: 700;
                        color: #00ff88;
                        font-size: 16px;
                        margin-bottom: 5px;
                    }
                    
                    .llm-reasoning {
                        color: #ccc;
                        font-size: 13px;
                        line-height: 1.4;
                    }
                    
                    .llm-timestamp {
                        color: #666;
                        font-size: 11px;
                        margin-top: 5px;
                    }
                    
                    .progress-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 10px;
                        height: 8px;
                        overflow: hidden;
                        margin-top: 8px;
                    }
                    
                    .progress-fill {
                        height: 100%;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                        transition: width 0.3s ease;
                        border-radius: 10px;
                    }
                    
                    .loading-spinner {
                        display: inline-block;
                        width: 16px;
                        height: 16px;
                        border: 2px solid #333;
                        border-radius: 50%;
                        border-top: 2px solid #00ff88;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    .error-state {
                        color: #ff4757;
                        text-align: center;
                        padding: 20px;
                    }
                    
                    @media (max-width: 1024px) {
                        .container {
                            grid-template-columns: 1fr;
                            max-width: 800px;
                        }
                        
                        .stats-grid {
                            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        }
                    }
                    
                    @media (max-width: 768px) {
                        .stats-grid {
                            grid-template-columns: repeat(2, 1fr);
                        }
                        
                        .header h1 {
                            font-size: 24px;
                        }
                        
                        .container {
                            padding: 0 15px;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Pokemon Crystal LLM RL Training Dashboard<span class="status-indicator" id="status-indicator"></span></h1>
                    <p>Advanced reinforcement learning with local LLM decision making and real-time monitoring</p>
                </div>
                
                <div class="container">
                    <div class="main-panel">
                        <!-- Stats Grid -->
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value" id="actions">-</div>
                                <div class="stat-label">Total Actions</div>
                                <div class="stat-change" id="actions-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="aps">-</div>
                                <div class="stat-label">Actions/Second</div>
                                <div class="stat-change" id="aps-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="llm-decisions">-</div>
                                <div class="stat-label">LLM Decisions</div>
                                <div class="stat-change" id="llm-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="total-reward">-</div>
                                <div class="stat-label">Total Reward</div>
                                <div class="stat-change" id="reward-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="level">-</div>
                                <div class="stat-label">Player Level</div>
                                <div class="stat-change" id="level-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="badges">-</div>
                                <div class="stat-label">Badges Earned</div>
                                <div class="progress-bar"><div class="progress-fill" id="badge-progress"></div></div>
                            </div>
                        </div>
                        
                        <!-- Game Screen -->
                        <div class="game-screen-container">
                            <h3>🖼️ Live Game Screen</h3>
                            <img id="gameScreen" class="game-screen" src="/screenshot?t=0" width="480" height="432" alt="Game Screen">
                            <div class="screen-info">
                                <span>Screen State: <span id="screen-state">-</span></span>
                                <span>Refresh Rate: <span id="refresh-rate">500ms</span></span>
                                <span>Last Updated: <span id="last-update">-</span></span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="side-panel">
                        <!-- Game State Info -->
                        <div class="panel">
                            <h3>🎮 Game State</h3>
                            <div class="reward-bars" id="game-state-info">
                                <div class="reward-bar">Map: <span id="player-map">-</span></div>
                                <div class="reward-bar">Position: <span id="player-position">-</span></div>
                                <div class="reward-bar">Money: ¥<span id="player-money">-</span></div>
                                <div class="reward-bar">Party: <span id="party-count">-</span> Pokemon</div>
                                <div class="reward-bar">HP: <span id="player-hp">-</span></div>
                            </div>
                        </div>
                        
                        <!-- Reward Breakdown -->
                        <div class="panel">
                            <h3>💰 Reward Analysis</h3>
                            <div class="reward-bars" id="reward-breakdown">
                                <div class="loading-spinner"></div> Loading...
                            </div>
                        </div>
                        
                        <!-- LLM Decisions -->
                        <div class="panel">
                            <h3>🧠 Recent LLM Decisions</h3>
                            <div id="llm-decisions-list">
                                <div class="loading-spinner"></div> Loading decisions...
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    let lastStats = {};
                    let updateCount = 0;
                    let startTime = Date.now();
                    
                    function formatNumber(num) {
                        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
                        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
                        return num.toString();
                    }
                    
                    function formatChange(current, previous) {
                        if (previous === undefined) return '';
                        const change = current - previous;
                        if (change === 0) return '';
                        const sign = change > 0 ? '+' : '';
                        const className = change > 0 ? 'positive' : 'negative';
                        return `<span class="${className}">${sign}${change.toFixed(2)}</span>`;
                    }
                    
                    function updateGameScreen() {
                        const screen = document.getElementById('gameScreen');
                        const timestamp = Date.now();
                        screen.src = `/screenshot?t=${timestamp}`;
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    }
                    
                    async function updateStats() {
                        try {
                            updateCount++;
                            const response = await fetch('/stats');
                            const stats = await response.json();
                            
                            // Update status indicator
                            document.getElementById('status-indicator').style.background = '#00ff88';
                            
                            // Update main stats with change indicators
                            document.getElementById('actions').textContent = formatNumber(stats.actions_taken || 0);
                            document.getElementById('aps').textContent = (stats.actions_per_second || 0).toFixed(1);
                            document.getElementById('llm-decisions').textContent = formatNumber(stats.llm_decision_count || 0);
                            document.getElementById('total-reward').textContent = (stats.total_reward || 0).toFixed(2);
                            document.getElementById('level').textContent = stats.player_level || 0;
                            document.getElementById('badges').textContent = `${stats.badges_total || 0}/16`;
                            
                            // Update change indicators
                            document.getElementById('actions-change').innerHTML = formatChange(stats.actions_taken, lastStats.actions_taken);
                            document.getElementById('aps-change').innerHTML = formatChange(stats.actions_per_second, lastStats.actions_per_second);
                            document.getElementById('llm-change').innerHTML = formatChange(stats.llm_decision_count, lastStats.llm_decision_count);
                            document.getElementById('reward-change').innerHTML = formatChange(stats.total_reward, lastStats.total_reward);
                            document.getElementById('level-change').innerHTML = formatChange(stats.player_level, lastStats.player_level);
                            
                            // Update badge progress
                            const badgeProgress = ((stats.badges_total || 0) / 16) * 100;
                            document.getElementById('badge-progress').style.width = badgeProgress + '%';
                            
                            // Update game state info
                            document.getElementById('player-map').textContent = stats.final_game_state?.player_map || '-';
                            document.getElementById('player-position').textContent = 
                                stats.final_game_state ? 
                                `(${stats.final_game_state.player_x}, ${stats.final_game_state.player_y})` : '-';
                            document.getElementById('player-money').textContent = formatNumber(stats.final_game_state?.money || 0);
                            document.getElementById('party-count').textContent = stats.final_game_state?.party_count || 0;
                            document.getElementById('player-hp').textContent = 
                                stats.final_game_state ? 
                                `${stats.final_game_state.player_hp}/${stats.final_game_state.player_max_hp}` : '-';
                            
                            // Update screen state
                            document.getElementById('screen-state').textContent = stats.screen_state || 'Unknown';
                            
                            // Update reward breakdown with bars
                            const rewardDiv = document.getElementById('reward-breakdown');
                            if (stats.last_reward_breakdown && stats.last_reward_breakdown !== 'no rewards') {
                                const rewards = stats.last_reward_breakdown.split(' | ');
                                rewardDiv.innerHTML = rewards.map(reward => {
                                    const [category, value] = reward.split(': ');
                                    const numValue = parseFloat(value);
                                    const className = numValue > 0 ? 'positive' : 'negative';
                                    return `<div class="reward-bar ${className}">${category}: <strong>${value}</strong></div>`;
                                }).join('');
                            } else {
                                rewardDiv.innerHTML = '<div class="reward-bar">No active rewards</div>';
                            }
                            
                            // Update LLM decisions with enhanced formatting
                            const decisionsDiv = document.getElementById('llm-decisions-list');
                            if (stats.recent_llm_decisions && stats.recent_llm_decisions.length > 0) {
                                decisionsDiv.innerHTML = stats.recent_llm_decisions.map((d, index) => {
                                    const timestamp = new Date(d.timestamp * 1000);
                                    return `<div class="llm-decision">
                                        <div class="llm-action">Action: ${d.action.toUpperCase()}</div>
                                        <div class="llm-reasoning">${d.reasoning.substring(0, 150)}${d.reasoning.length > 150 ? '...' : ''}</div>
                                        <div class="llm-timestamp">${timestamp.toLocaleTimeString()}</div>
                                    </div>`;
                                }).join('');
                            } else {
                                decisionsDiv.innerHTML = '<div class="llm-decision">No LLM decisions yet...</div>';
                            }
                            
                            lastStats = {...stats};
                            
                        } catch (e) {
                            console.error('Failed to update stats:', e);
                            document.getElementById('status-indicator').style.background = '#ff4757';
                        }
                    }
                    
                    // Update stats every 500ms
                    setInterval(updateStats, 500);
                    
                    // Update game screen every 250ms for smoother visuals
                    setInterval(updateGameScreen, 250);
                    
                    // Initial load
                    updateStats();
                    updateGameScreen();
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
            
        elif self.path.startswith('/screenshot'):
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
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
        """Analyze current screen state with improved detection"""
        if not self.pyboy:
            return {'state': 'unknown', 'variance': 0, 'colors': 0}
            
        screen = self.pyboy.screen.ndarray
        variance = float(np.var(screen.astype(np.float32)))
        
        # Get additional screen analysis metrics
        unique_colors = len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        brightness = float(np.mean(screen.astype(np.float32)))
        
        # More sophisticated state detection
        # Check for common UI patterns by analyzing screen regions
        
        # Very low variance = loading/transition screen
        if variance < 50:
            state = "loading"
        # Very high variance = battle screen (lots of sprites/effects)
        elif variance > 20000:
            state = "battle"
        # Medium-high variance with many colors = overworld
        elif variance > 3000 and unique_colors > 10:
            state = "overworld"
        # Low variance with high brightness = likely a menu or dialogue
        elif variance < 3000:
            # Further distinguish between menu and dialogue
            # Dialogue typically has more uniform color distribution
            # Menus often have more structured patterns
            
            # Check brightness patterns - dialogue boxes tend to have consistent bright areas
            if brightness > 200 and unique_colors < 8:
                # Very bright with few colors = likely dialogue box
                state = "dialogue"
            elif variance > 500 and unique_colors >= 8:
                # Some variance with multiple colors = likely settings/menu
                state = "settings_menu"
            elif variance < 500:
                # Low variance = simple menu
                state = "menu"
            else:
                # Default case
                state = "menu"
        else:
            # Default to overworld for anything else
            state = "overworld"
            
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
        
        # Update current game state and screen analysis
        current_game_state = self.get_game_state()
        screen_analysis = self.analyze_screen()
        
        self.stats['final_game_state'] = current_game_state
        self.stats['screen_state'] = screen_analysis.get('state', 'unknown')
        
        # Update screenshot
        try:
            screen = self.pyboy.screen.ndarray
            if screen.shape[2] == 4:  # RGBA to RGB
                screen = screen[:, :, :3]
            
            img = Image.fromarray(screen)
            img = img.resize((480, 432), Image.NEAREST)  # Match the display size
            
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
