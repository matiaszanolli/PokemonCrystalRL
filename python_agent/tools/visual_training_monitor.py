"""
visual_training_monitor.py - Real-time visual training monitor

This module provides a live view of what the emulator is doing during training,
showing screenshots, visual analysis, and LLM decisions in real-time.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import threading
import queue
from dataclasses import dataclass

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from vision_processor import VisualContext


@dataclass
class TrainingFrame:
    """Container for a single training frame with all context"""
    timestamp: float
    step: int
    screenshot: np.ndarray
    visual_context: Optional[VisualContext]
    game_state: Dict[str, Any]
    action: int
    action_name: str
    reward: float
    llm_reasoning: str
    episode: int


class VisualTrainingMonitor:
    """
    Real-time visual monitor for Pokemon Crystal training
    """
    
    def __init__(self, figsize=(16, 12)):
        """Initialize the visual monitor"""
        self.figsize = figsize
        self.frame_queue = queue.Queue(maxsize=100)
        self.current_frame: Optional[TrainingFrame] = None
        self.is_running = False
        self.fig = None
        self.axes = {}
        
        # Training statistics
        self.stats = {
            'total_steps': 0,
            'current_episode': 0,
            'actions_taken': [],
            'screen_types_seen': [],
            'rewards': [],
            'start_time': time.time()
        }
        
    def setup_display(self):
        """Setup the matplotlib display with subplots"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle('🎮 Pokemon Crystal Vision-Enhanced RL Training Monitor', 
                         fontsize=16, fontweight='bold')
        
        # Create subplot layout
        gs = self.fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1, 1])
        
        # Main game screenshot
        self.axes['screenshot'] = self.fig.add_subplot(gs[0, 0])
        self.axes['screenshot'].set_title('🖥️ Game Screen', fontweight='bold')
        self.axes['screenshot'].axis('off')
        
        # Visual analysis
        self.axes['visual'] = self.fig.add_subplot(gs[0, 1])
        self.axes['visual'].set_title('👁️ Visual Analysis', fontweight='bold')
        self.axes['visual'].axis('off')
        
        # Game state
        self.axes['state'] = self.fig.add_subplot(gs[0, 2])
        self.axes['state'].set_title('📊 Game State', fontweight='bold')
        self.axes['state'].axis('off')
        
        # LLM Decision
        self.axes['decision'] = self.fig.add_subplot(gs[0, 3])
        self.axes['decision'].set_title('🤖 LLM Decision', fontweight='bold')
        self.axes['decision'].axis('off')
        
        # Action distribution
        self.axes['actions'] = self.fig.add_subplot(gs[1, 0])
        self.axes['actions'].set_title('🎯 Action Distribution')
        
        # Reward over time
        self.axes['rewards'] = self.fig.add_subplot(gs[1, 1:3])
        self.axes['rewards'].set_title('💰 Reward Over Time')
        
        # Training stats
        self.axes['stats'] = self.fig.add_subplot(gs[1, 3])
        self.axes['stats'].set_title('📈 Stats')
        self.axes['stats'].axis('off')
        
        # Screen types
        self.axes['screens'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['screens'].set_title('📺 Screen Types Encountered')
        
        # Progress timeline
        self.axes['progress'] = self.fig.add_subplot(gs[2, 2:4])
        self.axes['progress'].set_title('⏱️ Training Progress')
        self.axes['progress'].axis('off')
        
        plt.tight_layout()
        
    def add_frame(self, frame: TrainingFrame):
        """Add a new training frame to the display queue"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Remove oldest frame if queue is full
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    
    def update_display(self):
        """Update the visual display with the latest frame"""
        if self.frame_queue.empty():
            return
            
        # Get latest frame
        try:
            self.current_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return
        
        # Update statistics
        self._update_stats(self.current_frame)
        
        # Clear all axes
        for ax_name, ax in self.axes.items():
            ax.clear()
            
        # Update each subplot
        self._update_screenshot()
        self._update_visual_analysis()
        self._update_game_state()
        self._update_llm_decision()
        self._update_action_distribution()
        self._update_reward_plot()
        self._update_training_stats()
        self._update_screen_types()
        self._update_progress_timeline()
        
        # Restore titles
        self.axes['screenshot'].set_title('🖥️ Game Screen', fontweight='bold')
        self.axes['visual'].set_title('👁️ Visual Analysis', fontweight='bold')
        self.axes['state'].set_title('📊 Game State', fontweight='bold')
        self.axes['decision'].set_title('🤖 LLM Decision', fontweight='bold')
        self.axes['actions'].set_title('🎯 Action Distribution')
        self.axes['rewards'].set_title('💰 Reward Over Time')
        self.axes['stats'].set_title('📈 Stats')
        self.axes['screens'].set_title('📺 Screen Types Encountered')
        self.axes['progress'].set_title('⏱️ Training Progress')
        
        plt.draw()
        plt.pause(0.01)  # Small pause to update display
    
    def _update_stats(self, frame: TrainingFrame):
        """Update internal statistics"""
        self.stats['total_steps'] += 1
        self.stats['current_episode'] = frame.episode
        self.stats['actions_taken'].append(frame.action_name)
        self.stats['rewards'].append(frame.reward)
        
        if frame.visual_context:
            self.stats['screen_types_seen'].append(frame.visual_context.screen_type)
    
    def _update_screenshot(self):
        """Update the main game screenshot display"""
        if self.current_frame and self.current_frame.screenshot is not None:
            screenshot = self.current_frame.screenshot
            
            # Scale up the Game Boy screen for better visibility
            scaled = cv2.resize(screenshot, (320, 288), interpolation=cv2.INTER_NEAREST)
            
            self.axes['screenshot'].imshow(scaled)
            self.axes['screenshot'].set_title(f'🖥️ Game Screen (Step {self.current_frame.step})', 
                                           fontweight='bold')
            
            # Add action overlay
            action_text = f"Action: {self.current_frame.action_name}"
            self.axes['screenshot'].text(5, 15, action_text, 
                                      color='white', fontweight='bold',
                                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        self.axes['screenshot'].axis('off')
    
    def _update_visual_analysis(self):
        """Update visual analysis display"""
        self.axes['visual'].axis('off')
        
        if not self.current_frame or not self.current_frame.visual_context:
            self.axes['visual'].text(0.5, 0.5, "No visual\nanalysis", 
                                  ha='center', va='center', fontsize=12)
            return
        
        vc = self.current_frame.visual_context
        
        # Display visual analysis info
        y_pos = 0.9
        line_height = 0.15
        
        # Screen type
        self.axes['visual'].text(0.05, y_pos, f"Screen: {vc.screen_type.upper()}", 
                              fontweight='bold', fontsize=10, transform=self.axes['visual'].transAxes)
        y_pos -= line_height
        
        # Game phase
        self.axes['visual'].text(0.05, y_pos, f"Phase: {vc.game_phase}", 
                              fontsize=9, transform=self.axes['visual'].transAxes)
        y_pos -= line_height
        
        # Detected text
        if vc.detected_text:
            self.axes['visual'].text(0.05, y_pos, "Text detected:", 
                                  fontweight='bold', fontsize=9, transform=self.axes['visual'].transAxes)
            y_pos -= line_height * 0.7
            
            for text_obj in vc.detected_text[:3]:  # Show first 3
                text_line = f"• '{text_obj.text}' ({text_obj.confidence:.2f})"
                self.axes['visual'].text(0.1, y_pos, text_line, 
                                      fontsize=8, transform=self.axes['visual'].transAxes)
                y_pos -= line_height * 0.7
        
        # UI elements
        if vc.ui_elements:
            ui_types = [elem.element_type for elem in vc.ui_elements]
            ui_text = f"UI: {', '.join(set(ui_types))}"
            self.axes['visual'].text(0.05, y_pos, ui_text, 
                                  fontsize=9, transform=self.axes['visual'].transAxes)
            y_pos -= line_height
        
        # Dominant colors
        if vc.dominant_colors:
            color_names = []
            for r, g, b in vc.dominant_colors[:2]:
                if r > g and r > b:
                    color_names.append("red")
                elif g > r and g > b:
                    color_names.append("green")
                elif b > r and b > g:
                    color_names.append("blue")
                else:
                    color_names.append("mixed")
            
            colors_text = f"Colors: {', '.join(color_names)}"
            self.axes['visual'].text(0.05, y_pos, colors_text, 
                                  fontsize=9, transform=self.axes['visual'].transAxes)
    
    def _update_game_state(self):
        """Update game state display"""
        self.axes['state'].axis('off')
        
        if not self.current_frame:
            return
            
        state = self.current_frame.game_state
        player = state.get('player', {})
        party = state.get('party', [])
        
        y_pos = 0.9
        line_height = 0.12
        
        # Player info
        self.axes['state'].text(0.05, y_pos, "PLAYER", fontweight='bold', fontsize=11, 
                             transform=self.axes['state'].transAxes)
        y_pos -= line_height
        
        location = f"Map {player.get('map', 0)} ({player.get('x', 0)}, {player.get('y', 0)})"
        self.axes['state'].text(0.05, y_pos, f"Location: {location}", fontsize=9, 
                             transform=self.axes['state'].transAxes)
        y_pos -= line_height
        
        self.axes['state'].text(0.05, y_pos, f"Money: ${player.get('money', 0)}", fontsize=9, 
                             transform=self.axes['state'].transAxes)
        y_pos -= line_height
        
        self.axes['state'].text(0.05, y_pos, f"Badges: {player.get('badges', 0)}", fontsize=9, 
                             transform=self.axes['state'].transAxes)
        y_pos -= line_height * 1.5
        
        # Party info
        if party:
            self.axes['state'].text(0.05, y_pos, "PARTY", fontweight='bold', fontsize=11, 
                                 transform=self.axes['state'].transAxes)
            y_pos -= line_height
            
            for i, pokemon in enumerate(party[:3]):  # Show first 3
                species = pokemon.get('species', 0)
                level = pokemon.get('level', 0)
                hp = pokemon.get('hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                
                hp_percent = (hp / max_hp * 100) if max_hp > 0 else 0
                hp_status = "OK" if hp_percent > 50 else "LOW" if hp_percent > 20 else "CRIT"
                
                pokemon_text = f"#{species} Lv{level} HP:{hp_percent:.0f}% ({hp_status})"
                self.axes['state'].text(0.05, y_pos, pokemon_text, fontsize=8, 
                                     transform=self.axes['state'].transAxes)
                y_pos -= line_height * 0.8
        else:
            self.axes['state'].text(0.05, y_pos, "PARTY: None", fontsize=9, 
                                 transform=self.axes['state'].transAxes)
    
    def _update_llm_decision(self):
        """Update LLM decision display"""
        self.axes['decision'].axis('off')
        
        if not self.current_frame:
            return
        
        y_pos = 0.9
        line_height = 0.15
        
        # Action taken
        self.axes['decision'].text(0.05, y_pos, f"ACTION: {self.current_frame.action_name}", 
                                fontweight='bold', fontsize=11, color='red',
                                transform=self.axes['decision'].transAxes)
        y_pos -= line_height
        
        # Reward
        reward_color = 'green' if self.current_frame.reward >= 0 else 'red'
        self.axes['decision'].text(0.05, y_pos, f"Reward: {self.current_frame.reward:.1f}", 
                                fontsize=10, color=reward_color,
                                transform=self.axes['decision'].transAxes)
        y_pos -= line_height
        
        # LLM reasoning (truncated)
        if self.current_frame.llm_reasoning:
            reasoning = self.current_frame.llm_reasoning[:100] + "..." if len(self.current_frame.llm_reasoning) > 100 else self.current_frame.llm_reasoning
            
            # Split into multiple lines
            words = reasoning.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 20:
                    current_line += word + " "
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            
            self.axes['decision'].text(0.05, y_pos, "Reasoning:", fontweight='bold', fontsize=9,
                                    transform=self.axes['decision'].transAxes)
            y_pos -= line_height * 0.7
            
            for line in lines[:4]:  # Show max 4 lines
                self.axes['decision'].text(0.05, y_pos, line, fontsize=8,
                                        transform=self.axes['decision'].transAxes)
                y_pos -= line_height * 0.6
    
    def _update_action_distribution(self):
        """Update action distribution chart"""
        if len(self.stats['actions_taken']) < 2:
            self.axes['actions'].text(0.5, 0.5, "Collecting data...", ha='center', va='center')
            return
        
        # Count recent actions (last 50)
        recent_actions = self.stats['actions_taken'][-50:]
        from collections import Counter
        action_counts = Counter(recent_actions)
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
        bars = self.axes['actions'].bar(actions, counts, color=colors)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.axes['actions'].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   str(count), ha='center', va='bottom', fontsize=8)
        
        self.axes['actions'].set_xlabel('Actions')
        self.axes['actions'].set_ylabel('Frequency')
        self.axes['actions'].tick_params(axis='x', rotation=45, labelsize=8)
    
    def _update_reward_plot(self):
        """Update reward over time plot"""
        if len(self.stats['rewards']) < 2:
            self.axes['rewards'].text(0.5, 0.5, "Collecting data...", ha='center', va='center')
            return
        
        rewards = self.stats['rewards'][-100:]  # Last 100 rewards
        steps = list(range(max(1, len(self.stats['rewards']) - len(rewards) + 1), len(self.stats['rewards']) + 1))
        
        self.axes['rewards'].plot(steps, rewards, 'b-', alpha=0.7, linewidth=1)
        
        # Add moving average
        if len(rewards) > 10:
            window_size = min(10, len(rewards))
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            avg_steps = steps[window_size-1:]
            self.axes['rewards'].plot(avg_steps, moving_avg, 'r-', linewidth=2, label='Moving Average')
            self.axes['rewards'].legend()
        
        self.axes['rewards'].set_xlabel('Step')
        self.axes['rewards'].set_ylabel('Reward')
        self.axes['rewards'].grid(True, alpha=0.3)
    
    def _update_training_stats(self):
        """Update training statistics display"""
        self.axes['stats'].axis('off')
        
        elapsed_time = time.time() - self.stats['start_time']
        
        y_pos = 0.9
        line_height = 0.15
        
        stats_text = [
            f"Episode: {self.stats['current_episode']}",
            f"Total Steps: {self.stats['total_steps']}",
            f"Time: {elapsed_time/60:.1f}m",
            f"Steps/min: {self.stats['total_steps']/(elapsed_time/60):.1f}" if elapsed_time > 60 else "Steps/min: --"
        ]
        
        for stat in stats_text:
            self.axes['stats'].text(0.05, y_pos, stat, fontsize=10, 
                                 transform=self.axes['stats'].transAxes)
            y_pos -= line_height
        
        # Add current reward
        if self.stats['rewards']:
            avg_recent_reward = np.mean(self.stats['rewards'][-10:])
            reward_text = f"Avg Reward: {avg_recent_reward:.1f}"
            self.axes['stats'].text(0.05, y_pos, reward_text, fontsize=10,
                                 color='green' if avg_recent_reward >= 0 else 'red',
                                 transform=self.axes['stats'].transAxes)
    
    def _update_screen_types(self):
        """Update screen types encountered"""
        if len(self.stats['screen_types_seen']) < 2:
            self.axes['screens'].text(0.5, 0.5, "Collecting data...", ha='center', va='center')
            return
        
        from collections import Counter
        screen_counts = Counter(self.stats['screen_types_seen'][-100:])  # Last 100
        
        screens = list(screen_counts.keys())
        counts = list(screen_counts.values())
        
        colors = {'overworld': 'green', 'battle': 'red', 'menu': 'blue', 
                 'dialogue': 'orange', 'intro': 'purple'}
        bar_colors = [colors.get(screen, 'gray') for screen in screens]
        
        bars = self.axes['screens'].bar(screens, counts, color=bar_colors)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.axes['screens'].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   str(count), ha='center', va='bottom', fontsize=8)
        
        self.axes['screens'].set_xlabel('Screen Types')
        self.axes['screens'].set_ylabel('Frequency')
        self.axes['screens'].tick_params(axis='x', rotation=45, labelsize=8)
    
    def _update_progress_timeline(self):
        """Update progress timeline"""
        self.axes['progress'].axis('off')
        
        if not self.current_frame:
            return
        
        # Show recent key events
        y_pos = 0.9
        line_height = 0.2
        
        timestamp = datetime.fromtimestamp(self.current_frame.timestamp).strftime('%H:%M:%S')
        
        events = [
            f"⏰ {timestamp}",
            f"🎮 Episode {self.current_frame.episode}, Step {self.current_frame.step}",
            f"🎯 Last Action: {self.current_frame.action_name}",
        ]
        
        # Add visual context if available
        if self.current_frame.visual_context:
            events.append(f"👁️ Screen: {self.current_frame.visual_context.screen_type}")
        
        for event in events:
            self.axes['progress'].text(0.05, y_pos, event, fontsize=9,
                                     transform=self.axes['progress'].transAxes)
            y_pos -= line_height
    
    def start_monitoring(self):
        """Start the visual monitoring loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.setup_display()
        
        # Start update loop in a separate thread
        def update_loop():
            while self.is_running:
                try:
                    self.update_display()
                    time.sleep(0.1)  # Update 10 times per second
                except Exception as e:
                    print(f"Monitor update error: {e}")
                    time.sleep(1)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
        print("🖥️ Visual training monitor started!")
        print("Close the matplotlib window to stop monitoring.")
    
    def stop_monitoring(self):
        """Stop the visual monitoring"""
        self.is_running = False
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=1)
        plt.close('all')
        print("🖥️ Visual training monitor stopped.")


class MonitoredTrainingSession:
    """
    Training session with integrated visual monitoring
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 model_name: str = "llama3.2:3b",
                 max_steps_per_episode: int = 1000,
                 screenshot_interval: int = 5,
                 show_monitor: bool = True):
        """Initialize monitored training session"""
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.max_steps_per_episode = max_steps_per_episode
        self.screenshot_interval = screenshot_interval
        self.show_monitor = show_monitor
        
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
        
        # Initialize monitor
        if self.show_monitor:
            self.monitor = VisualTrainingMonitor()
            self.monitor.start_monitoring()
        else:
            self.monitor = None
        
        print("✅ Monitored training session initialized")
    
    def run_episode(self, episode_num: int):
        """Run a single monitored episode"""
        print(f"\n🚀 Starting Episode {episode_num}")
        
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
            
            # Create training frame for monitor
            if self.monitor and screenshot is not None:
                frame = TrainingFrame(
                    timestamp=time.time(),
                    step=step,
                    screenshot=screenshot,
                    visual_context=visual_context,
                    game_state=game_state,
                    action=action,
                    action_name=self.agent.action_map[action],
                    reward=reward,
                    llm_reasoning="Strategic decision based on visual context",  # Could get actual reasoning
                    episode=episode_num
                )
                self.monitor.add_frame(frame)
            
            step += 1
            
            # Small delay to make monitoring visible
            time.sleep(0.2)
        
        print(f"📋 Episode {episode_num} completed: {step} steps")
    
    def run_training(self, num_episodes: int = 5):
        """Run monitored training session"""
        print(f"🎯 Starting monitored training: {num_episodes} episodes")
        
        try:
            for episode in range(1, num_episodes + 1):
                self.run_episode(episode)
                
                if episode < num_episodes:
                    print("⏸️ Pausing between episodes...")
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.monitor:
                input("\nPress Enter to stop monitoring and close...")
                self.monitor.stop_monitoring()
            self.env.close()
            print("🎉 Monitored training session complete!")


def main():
    """Run monitored training session"""
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please add a Pokemon Crystal ROM file to continue")
        return
    
    # Start monitored training
    session = MonitoredTrainingSession(
        rom_path=rom_path,
        save_state_path=None,
        model_name="llama3.2:3b",
        max_steps_per_episode=50,  # Short episodes for demo
        screenshot_interval=3,     # Frequent screenshots
        show_monitor=True
    )
    
    session.run_training(num_episodes=3)


if __name__ == "__main__":
    main()
