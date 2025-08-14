"""
terminal_training_monitor.py - Terminal-based real-time training monitor

This module provides a text-based view of what the emulator is doing during training,
displaying game state, visual analysis, and LLM decisions in the terminal.
"""

import time
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import deque, Counter

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from vision_processor import VisualContext


class TerminalTrainingMonitor:
    """
    Real-time terminal-based monitor for Pokemon Crystal training
    """
    
    def __init__(self):
        """Initialize the terminal monitor"""
        self.stats = {
            'start_time': time.time(),
            'total_steps': 0,
            'episodes': 0,
            'actions_taken': deque(maxlen=100),
            'rewards': deque(maxlen=50),
            'screen_types': deque(maxlen=50),
            'visual_analyses': 0,
            'last_screenshot_time': 0
        }
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_header(self):
        """Create the header display"""
        elapsed = time.time() - self.stats['start_time']
        
        header = [
            "🎮 POKEMON CRYSTAL VISION-ENHANCED RL TRAINING MONITOR",
            "=" * 70,
            f"⏰ Runtime: {elapsed/60:.1f}m | Episodes: {self.stats['episodes']} | Steps: {self.stats['total_steps']}",
            f"📊 Visual Analyses: {self.stats['visual_analyses']} | Steps/min: {self.stats['total_steps']/(elapsed/60):.1f}" if elapsed > 60 else f"📊 Visual Analyses: {self.stats['visual_analyses']} | Steps/min: --",
            "=" * 70,
        ]
        return "\n".join(header)
    
    def format_game_screen_ascii(self, screenshot, width=40, height=20):
        """Convert screenshot to ASCII representation"""
        if screenshot is None or screenshot.size == 0:
            return ["[No Screenshot Available]"] + [" " * width] * (height-1)
        
        try:
            import cv2
            import numpy as np
            
            # Resize to ASCII dimensions
            resized = cv2.resize(screenshot, (width, height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            
            # ASCII characters from darkest to lightest
            chars = "@%#*+=-:. "
            
            ascii_lines = []
            for row in gray:
                line = ""
                for pixel in row:
                    # Map pixel intensity to ASCII character
                    char_index = min(int(pixel / 255 * (len(chars) - 1)), len(chars) - 1)
                    line += chars[char_index]
                ascii_lines.append(line)
            
            return ascii_lines
            
        except Exception as e:
            return ["[Screenshot processing error]"] + [" " * width] * (height-1)
    
    def format_visual_analysis(self, visual_context: Optional[VisualContext]):
        """Format visual analysis information"""
        if not visual_context:
            return ["👁️ VISUAL ANALYSIS", "-" * 20, "No visual analysis available"]
        
        lines = [
            "👁️ VISUAL ANALYSIS",
            "-" * 20,
            f"Screen Type: {visual_context.screen_type.upper()}",
            f"Game Phase: {visual_context.game_phase}",
            f"Summary: {visual_context.visual_summary}",
        ]
        
        if visual_context.detected_text:
            lines.append("\nDetected Text:")
            for i, text_obj in enumerate(visual_context.detected_text[:3]):
                lines.append(f"  • '{text_obj.text}' (conf: {text_obj.confidence:.2f})")
        
        if visual_context.ui_elements:
            ui_types = [elem.element_type for elem in visual_context.ui_elements]
            lines.append(f"\nUI Elements: {', '.join(set(ui_types))}")
        
        return lines
    
    def format_game_state(self, game_state: Dict[str, Any]):
        """Format game state information"""
        lines = [
            "📊 GAME STATE",
            "-" * 20,
        ]
        
        player = game_state.get('player', {})
        party = game_state.get('party', [])
        
        # Player info
        lines.extend([
            f"Location: Map {player.get('map', 0)} ({player.get('x', 0)}, {player.get('y', 0)})",
            f"Money: ${player.get('money', 0)}",
            f"Badges: {player.get('badges', 0)}",
            "",
            "PARTY:",
        ])
        
        if party:
            for i, pokemon in enumerate(party[:3]):
                species = pokemon.get('species', 0)
                level = pokemon.get('level', 0)
                hp = pokemon.get('hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                
                hp_percent = (hp / max_hp * 100) if max_hp > 0 else 0
                hp_bar = "█" * int(hp_percent / 10) + "░" * (10 - int(hp_percent / 10))
                
                lines.append(f"  {i+1}. Species #{species} Lv{level}")
                lines.append(f"     HP: {hp_bar} {hp_percent:.0f}%")
        else:
            lines.append("  No Pokemon in party")
        
        return lines
    
    def format_action_stats(self):
        """Format action distribution and statistics"""
        lines = [
            "🎯 ACTION STATISTICS",
            "-" * 20,
        ]
        
        if len(self.stats['actions_taken']) > 0:
            action_counts = Counter(list(self.stats['actions_taken'])[-20:])  # Last 20 actions
            
            for action, count in action_counts.most_common(5):
                bar = "█" * min(count, 10) + "░" * (10 - min(count, 10))
                lines.append(f"  {action:6s}: {bar} ({count})")
        else:
            lines.append("  No actions recorded yet")
        
        # Recent rewards
        if len(self.stats['rewards']) > 0:
            recent_rewards = list(self.stats['rewards'])[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            lines.extend([
                "",
                "💰 RECENT REWARDS:",
                f"  Average: {avg_reward:.2f}",
                f"  Range: {min(recent_rewards):.1f} to {max(recent_rewards):.1f}",
            ])
        
        # Screen types
        if len(self.stats['screen_types']) > 0:
            screen_counts = Counter(list(self.stats['screen_types']))
            lines.extend([
                "",
                "📺 SCREEN TYPES:",
            ])
            for screen_type, count in screen_counts.most_common(3):
                lines.append(f"  {screen_type}: {count}")
        
        return lines
    
    def display_training_frame(self, episode: int, step: int, screenshot, visual_context: Optional[VisualContext], 
                              game_state: Dict[str, Any], action: str, reward: float, llm_reasoning: str = ""):
        """Display a complete training frame in the terminal"""
        
        # Update stats
        self.stats['total_steps'] += 1
        self.stats['episodes'] = episode
        self.stats['actions_taken'].append(action)
        self.stats['rewards'].append(reward)
        
        if visual_context:
            self.stats['visual_analyses'] += 1
            self.stats['screen_types'].append(visual_context.screen_type)
            self.stats['last_screenshot_time'] = time.time()
        
        # Clear screen and build display
        self.clear_screen()
        
        # Create columns layout
        ascii_art = self.format_game_screen_ascii(screenshot, 30, 15)
        visual_info = self.format_visual_analysis(visual_context)
        game_info = self.format_game_state(game_state)
        action_stats = self.format_action_stats()
        
        # Print header
        print(self.format_header())
        print()
        
        # Print current action and reward prominently
        reward_color = "🟢" if reward >= 0 else "🔴"
        print(f"🎯 CURRENT ACTION: {action.upper()} | {reward_color} REWARD: {reward:.2f}")
        print(f"🤖 EPISODE {episode} - STEP {step}")
        print("-" * 70)
        print()
        
        # Create three-column layout
        max_lines = max(len(ascii_art), len(visual_info), len(game_info))
        
        for i in range(max_lines):
            # Column 1: ASCII game screen
            screen_line = ascii_art[i] if i < len(ascii_art) else " " * 30
            
            # Column 2: Visual analysis
            visual_line = visual_info[i] if i < len(visual_info) else ""
            visual_line = (visual_line[:25] + "...") if len(visual_line) > 28 else visual_line
            visual_line = visual_line.ljust(28)
            
            # Column 3: Game state
            game_line = game_info[i] if i < len(game_info) else ""
            game_line = (game_line[:25] + "...") if len(game_line) > 28 else game_line
            
            print(f"{screen_line} | {visual_line} | {game_line}")
        
        print("\n" + "-" * 70)
        
        # Print action statistics below
        for line in action_stats:
            print(line)
        
        # Print LLM reasoning if available
        if llm_reasoning and len(llm_reasoning.strip()) > 0:
            print("\n🧠 LLM REASONING:")
            print("-" * 20)
            # Wrap text to fit terminal
            wrapped_reasoning = llm_reasoning[:200] + "..." if len(llm_reasoning) > 200 else llm_reasoning
            print(f"  {wrapped_reasoning}")
        
        print(f"\n⏰ Last updated: {datetime.now().strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop training")
        print("=" * 70)


class TerminalMonitoredTrainingSession:
    """
    Training session with terminal-based monitoring
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 model_name: str = "llama3.2:3b",
                 max_steps_per_episode: int = 100,
                 screenshot_interval: int = 3):
        """Initialize terminal monitored training session"""
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.max_steps_per_episode = max_steps_per_episode
        self.screenshot_interval = screenshot_interval
        
        # Initialize monitor
        self.monitor = TerminalTrainingMonitor()
        
        # Initialize components
        print("🎮 Initializing PyBoy environment...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            debug_mode=True
        )
        
        print("🤖 Initializing Enhanced LLM Agent...")
        self.agent = EnhancedLLMPokemonAgent(
            model_name=model_name,
            use_vision=True
        )
        
        print("✅ Terminal monitored training session initialized")
        print("\nStarting visual training monitor in 3 seconds...")
        time.sleep(3)
    
    def run_episode(self, episode_num: int):
        """Run a single monitored episode"""
        obs = self.env.reset()
        done = False
        step = 0
        
        while not done and step < self.max_steps_per_episode:
            # Get current state and screenshot
            game_state = self.env.get_game_state()
            screenshot = None
            visual_context = None
            
            if step % self.screenshot_interval == 0:
                screenshot = self.env.get_screenshot()
                
                if self.agent.vision_processor:
                    visual_context = self.agent.vision_processor.process_screenshot(screenshot)
            
            # Make decision
            action = self.agent.decide_next_action(
                state=game_state,
                screenshot=screenshot,
                recent_history=[]
            )
            
            # Execute action
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            # Display current frame in terminal
            self.monitor.display_training_frame(
                episode=episode_num,
                step=step,
                screenshot=screenshot,
                visual_context=visual_context,
                game_state=game_state,
                action=self.agent.action_map[action],
                reward=reward,
                llm_reasoning="LLM decision based on visual and game context"
            )
            
            step += 1
            
            # Pause to make monitoring visible
            time.sleep(1.0)  # 1 second per step for visibility
    
    def run_training(self, num_episodes: int = 3):
        """Run terminal monitored training session"""
        print(f"🎯 Starting terminal monitored training: {num_episodes} episodes")
        
        try:
            for episode in range(1, num_episodes + 1):
                self.run_episode(episode)
                
                if episode < num_episodes:
                    # Show pause screen
                    self.monitor.clear_screen()
                    print("🎮 POKEMON CRYSTAL TRAINING MONITOR")
                    print("=" * 50)
                    print(f"Episode {episode} completed!")
                    print(f"Starting Episode {episode + 1} in 3 seconds...")
                    print("Press Ctrl+C to stop training")
                    print("=" * 50)
                    time.sleep(3)
                    
        except KeyboardInterrupt:
            print("\n\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()
            
            # Final summary
            self.monitor.clear_screen()
            elapsed = time.time() - self.monitor.stats['start_time']
            
            print("🎉 TRAINING SESSION COMPLETE!")
            print("=" * 50)
            print(f"Total Runtime: {elapsed/60:.1f} minutes")
            print(f"Episodes Completed: {self.monitor.stats['episodes']}")
            print(f"Total Steps: {self.monitor.stats['total_steps']}")
            print(f"Visual Analyses: {self.monitor.stats['visual_analyses']}")
            print(f"Average Steps/Episode: {self.monitor.stats['total_steps']/max(1, self.monitor.stats['episodes']):.1f}")
            
            if len(self.monitor.stats['actions_taken']) > 0:
                action_counts = Counter(self.monitor.stats['actions_taken'])
                print("\nMost Used Actions:")
                for action, count in action_counts.most_common(5):
                    print(f"  {action}: {count}")
            
            if len(self.monitor.stats['rewards']) > 0:
                avg_reward = sum(self.monitor.stats['rewards']) / len(self.monitor.stats['rewards'])
                print(f"\nAverage Reward: {avg_reward:.2f}")
            
            print("=" * 50)


def main():
    """Run terminal monitored training session"""
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please add a Pokemon Crystal ROM file to continue")
        return
    
    # Start terminal monitored training
    session = TerminalMonitoredTrainingSession(
        rom_path=rom_path,
        save_state_path=None,
        model_name="llama3.2:3b",
        max_steps_per_episode=20,   # Short episodes for demo
        screenshot_interval=2       # Screenshot every 2 steps
    )
    
    session.run_training(num_episodes=2)


if __name__ == "__main__":
    main()
