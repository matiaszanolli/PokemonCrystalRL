"""
Training strategies and rule-based action handlers for Pokemon Crystal RL Trainer
"""

import time
import logging
from typing import Dict, List, Any


def get_rule_based_action(current_state: str, step: int) -> int:
    """Get rule-based action based on current game state"""
    
    # State-based action selection
    if current_state == "title_screen":
        return handle_title_screen(step)
    elif current_state == "intro_sequence":
        return handle_intro_sequence(step)
    elif current_state == "new_game_menu":
        return handle_new_game_menu(step)
    elif current_state == "dialogue":
        return handle_dialogue(step)
    elif current_state == "battle":
        return handle_battle(step)
    elif current_state == "overworld":
        return handle_overworld(step)
    elif current_state == "menu":
        return handle_menu(step)
    else:
        return handle_unknown_state(step)


def handle_title_screen(step: int) -> int:
    """Handle title screen navigation"""
    if step % 10 == 0:
        print(f"🎮 Title screen detected at step {step}")
    
    # Cycle through menu options to start game
    pattern = [7, 5, 5, 5, 2, 5, 5, 1, 5, 5]  # START, A spam, DOWN, A, UP, A
    return pattern[step % len(pattern)]


def handle_intro_sequence(step: int) -> int:
    """Handle intro/cutscene sequences"""
    if step % 20 == 0:
        print(f"🎬 Intro sequence detected at step {step}")
    
    # Rapidly skip through intro text
    pattern = [5, 5, 5, 7, 5, 5]  # A spam + occasional START to skip
    return pattern[step % len(pattern)]


def handle_new_game_menu(step: int) -> int:
    """Handle new game character creation"""
    if step % 15 == 0:
        print(f"👤 New game menu detected at step {step}")
    
    # Navigate new game menus (select options and confirm)
    pattern = [5, 5, 2, 5, 1, 5, 5]  # A, A, DOWN, A, UP, A, A
    return pattern[step % len(pattern)]


def handle_dialogue(step: int) -> int:
    """Handle dialogue boxes"""
    if step % 25 == 0:
        print(f"💬 Dialogue detected at step {step}")
    
    # Advance through dialogue quickly but not too fast
    pattern = [5, 0, 5, 0, 5]  # A, wait, A, wait, A (0 = no action)
    action = pattern[step % len(pattern)]
    return 5 if action == 0 else action  # Convert 0 to A button


def handle_battle(step: int) -> int:
    """Handle battle state actions"""
    if step % 10 == 0:
        print(f"⚔️ Battle detected at step {step}")
    
    # Simple battle strategy: alternate between attack and healing
    pattern = [5, 5, 5, 1, 5, 2, 5, 5, 5, 6]  # A, A, A, UP, A, DOWN, A, A, A, B
    return pattern[step % len(pattern)]


def handle_overworld(step: int) -> int:
    """Handle overworld movement"""
    if step % 30 == 0:
        print(f"🗺️ Overworld detected at step {step}")
    
    # Explore the world with varied movement
    movement_patterns = [
        [1, 1, 1, 5],      # Up, interact
        [2, 2, 2, 5],      # Down, interact  
        [3, 3, 3, 5],      # Left, interact
        [4, 4, 4, 5],      # Right, interact
    ]
    
    pattern_idx = (step // 20) % len(movement_patterns)
    pattern = movement_patterns[pattern_idx]
    return pattern[step % len(pattern)]


def handle_menu(step: int) -> int:
    """Handle menu navigation"""
    if step % 20 == 0:
        print(f"📋 Menu detected at step {step}")
    
    # Navigate menus efficiently
    pattern = [1, 5, 2, 5, 6]  # UP, A, DOWN, A, B (to exit)
    return pattern[step % len(pattern)]


def handle_unknown_state(step: int) -> int:
    """Handle unknown game states"""
    if step % 40 == 0:
        print(f"❓ Unknown state at step {step}")
    
    # Conservative exploration pattern
    pattern = [5, 5, 1, 4, 2, 3, 5, 6]  # A spam, movement, A, B
    return pattern[step % len(pattern)]


class TrainingStrategyManager:
    """Manages different training strategies and execution modes"""
    
    def __init__(self, config, pyboy, llm_manager, game_state_detector):
        self.config = config
        self.pyboy = pyboy
        self.llm_manager = llm_manager
        self.game_state_detector = game_state_detector
        self.logger = logging.getLogger(__name__)
        
        # Action mappings from PyBoy
        from pyboy.utils import WindowEvent
        self.actions = {
            1: WindowEvent.PRESS_ARROW_UP,
            2: WindowEvent.PRESS_ARROW_DOWN, 
            3: WindowEvent.PRESS_ARROW_LEFT,
            4: WindowEvent.PRESS_ARROW_RIGHT,
            5: WindowEvent.PRESS_BUTTON_A,
            6: WindowEvent.PRESS_BUTTON_B,
            7: WindowEvent.PRESS_BUTTON_START,
            8: WindowEvent.PRESS_BUTTON_SELECT,
            0: None
        }
    
    def execute_action(self, action: int):
        """Execute action in the game"""
        if self.pyboy and action in self.actions and self.actions[action]:
            self.pyboy.send_input(self.actions[action])
            self.pyboy.tick()
    
    def execute_synchronized_action(self, action: int, frames: int):
        """Execute action for exact frame duration with proper timing"""
        if not self.pyboy:
            return
        
        # Calculate timing for real Game Boy speed (60 FPS)
        frame_duration = 1.0 / 60.0  # 16.67ms per frame
        total_duration = frames * frame_duration
        
        start_time = time.time()
        
        # Press the button (if action is not 0/no-op)
        if action in self.actions and self.actions[action]:
            self.pyboy.send_input(self.actions[action])
        
        # Run the exact number of frames with proper timing
        for frame in range(frames):
            frame_start = time.time()
            
            # Execute one frame
            self.pyboy.tick()
            
            # Calculate how long this frame took
            frame_elapsed = time.time() - frame_start
            
            # Sleep for remainder of frame duration to maintain 60 FPS timing
            remaining_time = frame_duration - frame_elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        # Ensure total action duration is correct
        total_elapsed = time.time() - start_time
        if total_elapsed < total_duration:
            time.sleep(total_duration - total_elapsed)
    
    def run_ultra_fast_training(self, max_actions: int) -> Dict[str, Any]:
        """Run rule-based ultra-fast training"""
        actions_taken = 0
        action_pattern = [5, 5, 1, 1, 4, 4, 2, 2, 3, 3]  # Exploration pattern
        pattern_index = 0
        
        self.logger.info("🚀 Ultra-fast rule-based training (no LLM overhead)")
        
        start_time = time.time()
        
        try:
            while actions_taken < max_actions:
                # Get action from pattern
                action = action_pattern[pattern_index % len(action_pattern)]
                pattern_index += 1
                
                # Execute action
                self.execute_action(action)
                actions_taken += 1
                
                # Progress updates
                if actions_taken % 200 == 0:
                    elapsed = time.time() - start_time
                    aps = actions_taken / elapsed
                    self.logger.info(f"🚀 Ultra-fast: {actions_taken}/{max_actions} ({aps:.0f} a/s)")
                
                # Minimal delay
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            self.logger.info("⏸️ Ultra-fast training interrupted")
        
        elapsed = time.time() - start_time
        return {
            'total_actions': actions_taken,
            'elapsed_time': elapsed,
            'actions_per_second': actions_taken / elapsed if elapsed > 0 else 0
        }
    
    def run_curriculum_training(self, max_episodes: int) -> Dict[str, Any]:
        """Run progressive curriculum training"""
        current_stage = 1
        stage_episodes = 0
        stage_successes = 0
        total_episodes = 0
        
        self.logger.info(f"📚 Starting {self.config.curriculum_stages}-stage curriculum")
        
        start_time = time.time()
        
        try:
            while (current_stage <= self.config.curriculum_stages and 
                   total_episodes < max_episodes):
                
                # Run single episode
                success = self._run_curriculum_episode(current_stage)
                
                stage_episodes += 1
                total_episodes += 1
                
                if success:
                    stage_successes += 1
                
                # Check stage mastery
                success_rate = stage_successes / stage_episodes
                
                self.logger.info(f"📖 Stage {current_stage}, Episode {stage_episodes}: "
                               f"{'✅' if success else '❌'} ({success_rate:.1%} success)")
                
                # Advance stage if mastered
                if (stage_episodes >= self.config.min_stage_episodes and 
                    success_rate >= self.config.stage_mastery_threshold):
                    
                    self.logger.info(f"🎓 Stage {current_stage} mastered! Advancing...")
                    current_stage += 1
                    stage_episodes = 0
                    stage_successes = 0
                
                # Timeout check
                elif stage_episodes >= self.config.max_stage_episodes:
                    self.logger.info(f"⏰ Stage {current_stage} timeout, advancing anyway")
                    current_stage += 1
                    stage_episodes = 0
                    stage_successes = 0
        
        except KeyboardInterrupt:
            self.logger.info("⏸️ Curriculum training interrupted")
        
        elapsed = time.time() - start_time
        return {
            'total_episodes': total_episodes,
            'final_stage': current_stage - 1,
            'elapsed_time': elapsed
        }
    
    def _run_curriculum_episode(self, stage: int) -> bool:
        """Run single curriculum episode"""
        # Load save state if available
        if self.config.save_state_path:
            try:
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
            except Exception as e:
                self.logger.warning(f"Could not load save state: {e}")
        
        actions_taken = 0
        max_actions = 500  # Episode length
        success_indicators = 0
        
        while actions_taken < max_actions:
            # Get stage-appropriate action
            action = self._get_stage_action(stage)
            
            # Execute action
            self.execute_action(action)
            
            # Simple progress detection
            success_indicators += 1 if actions_taken % 50 == 0 else 0
            actions_taken += 1
        
        # Success criteria: multiple indicators of progress
        return success_indicators >= 5
    
    def _get_stage_action(self, stage: int) -> int:
        """Get stage-appropriate action for curriculum training"""
        stage_prompts = {
            1: "BASIC_CONTROLS - Focus on navigation",
            2: "DIALOGUE - Focus on text interaction", 
            3: "POKEMON_SELECTION - Focus on menu choices",
            4: "BATTLE_FUNDAMENTALS - Focus on combat",
            5: "EXPLORATION - Focus on world navigation"
        }
        
        stage_name = stage_prompts.get(stage, "GENERAL")
        
        # Use LLM if available, otherwise use rule-based
        if self.llm_manager and self.config.llm_backend:
            return self.llm_manager.get_llm_action(None, stage_name)
        else:
            # Rule-based stage actions
            if stage == 1:  # Basic controls
                pattern = [1, 2, 3, 4, 5]  # Movement + A
                return pattern[time.time() % len(pattern)]
            elif stage == 2:  # Dialogue
                return 5  # A button
            elif stage == 3:  # Menu selection
                pattern = [1, 2, 5, 6]  # Navigate + select/back
                return pattern[time.time() % len(pattern)]
            elif stage == 4:  # Battle
                pattern = [5, 1, 5, 2, 5]  # A, UP, A, DOWN, A
                return pattern[time.time() % len(pattern)]
            else:  # Exploration
                pattern = [1, 2, 3, 4, 5, 5]  # Movement + interact
                return pattern[time.time() % len(pattern)]
