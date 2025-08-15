#!/usr/bin/env python3
"""
ultimate_training.py - Pokemon Crystal RL Training with ROM Font System

The most advanced Pokemon Crystal RL training system featuring:
- ROM-extracted font recognition for perfect text understanding
- Enhanced LLM agent with deep game knowledge
- Intelligent action planning and execution
- Comprehensive progress tracking and debugging
- Web monitoring with real-time insights
- Game completion optimization

Goal: Train an agent capable of completing Pokemon Crystal from start to finish.
"""

import time
import threading
import signal
import sys
import numpy as np
import json
import os
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import our core components
from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from vision_processor import PokemonVisionProcessor
from enhanced_font_decoder import ROMFontDecoder
from gameboy_color_palette import GameBoyColorPalette

# Import web monitoring (if available)
try:
    from web_monitor import PokemonRLWebMonitor, create_dashboard_templates
    WEB_MONITOR_AVAILABLE = True
except ImportError:
    WEB_MONITOR_AVAILABLE = False
    print("⚠️ Web monitor not available - training will run without web interface")


class GameStateAnalyzer:
    """Analyzes game state for intelligent decision making"""
    
    def __init__(self):
        """Initialize the game state analyzer"""
        self.previous_states = deque(maxlen=50)
        self.location_visit_count = defaultdict(int)
        self.stuck_detection_threshold = 10
        self.progress_milestones = {
            'started_game': False,
            'got_starter': False,
            'first_battle': False,
            'first_catch': False,
            'first_gym': False,
            'second_gym': False,
            'elite_four': False,
            'champion': False
        }
        
    def analyze_progress(self, game_state: Dict[str, Any], visual_context=None) -> Dict[str, Any]:
        """Analyze current game progress and state"""
        analysis = {
            'current_phase': self._determine_game_phase(game_state, visual_context),
            'is_stuck': self._detect_stuck_state(game_state),
            'progress_score': self._calculate_progress_score(game_state),
            'recommended_strategy': self._recommend_strategy(game_state, visual_context),
            'priority_actions': self._get_priority_actions(game_state, visual_context)
        }
        
        self.previous_states.append(game_state.copy())
        
        # Update milestones
        self._update_milestones(game_state, visual_context)
        
        return analysis
    
    def _determine_game_phase(self, game_state: Dict[str, Any], visual_context=None) -> str:
        """Determine what phase of the game we're in"""
        party_size = game_state.get('party_size', 0)
        badges = game_state.get('badges', 0)
        money = game_state.get('money', 0)
        
        # Use visual context if available
        if visual_context:
            screen_type = visual_context.screen_type
            detected_text = [t.text for t in visual_context.detected_text]
            
            if screen_type == 'intro':
                return 'game_intro'
            elif screen_type == 'dialogue':
                if any('PROF' in text.upper() for text in detected_text):
                    return 'lab_sequence'
            elif screen_type == 'battle':
                return 'in_battle'
        
        # Determine phase based on game state
        if party_size == 0:
            return 'pre_starter'
        elif party_size >= 1 and badges == 0:
            return 'early_exploration'
        elif badges == 1:
            return 'post_first_gym'
        elif badges <= 3:
            return 'mid_game'
        elif badges <= 6:
            return 'late_game'
        else:
            return 'end_game'
    
    def _detect_stuck_state(self, game_state: Dict[str, Any]) -> bool:
        """Detect if the agent is stuck in the same state"""
        if len(self.previous_states) < self.stuck_detection_threshold:
            return False
        
        # Check if position hasn't changed much
        recent_positions = [(s.get('player_x', 0), s.get('player_y', 0)) 
                           for s in list(self.previous_states)[-self.stuck_detection_threshold:]]
        
        unique_positions = set(recent_positions)
        return len(unique_positions) <= 2  # Stuck if only 1-2 unique positions
    
    def _calculate_progress_score(self, game_state: Dict[str, Any]) -> float:
        """Calculate overall progress score (0-100)"""
        score = 0.0
        
        # Basic progression
        party_size = game_state.get('party_size', 0)
        badges = game_state.get('badges', 0)
        money = game_state.get('money', 0)
        
        # Starter Pokemon (20 points)
        if party_size >= 1:
            score += 20
        
        # Money/items (10 points)
        score += min(10, money / 1000)
        
        # Badges (major progression - 50 points)
        score += badges * 6.25  # 8 badges * 6.25 = 50 points
        
        # Milestone bonuses (20 points)
        milestone_score = sum(10 if achieved else 0 for achieved in self.progress_milestones.values())
        score += min(20, milestone_score)
        
        return min(100.0, score)
    
    def _recommend_strategy(self, game_state: Dict[str, Any], visual_context=None) -> str:
        """Recommend high-level strategy based on current state"""
        phase = self._determine_game_phase(game_state, visual_context)
        
        strategies = {
            'game_intro': "Follow intro sequence, press A to advance text",
            'pre_starter': "Navigate to Professor Elm's lab, choose starter Pokemon",
            'lab_sequence': "Complete lab tutorial, learn basic mechanics",
            'early_exploration': "Catch Pokemon on Route 29, level up starter, visit Cherrygrove",
            'in_battle': "Use type advantages, heal when HP low, catch new Pokemon",
            'post_first_gym': "Explore new areas, train Pokemon to level 15-20",
            'mid_game': "Balance training, gym challenges, and story progression",
            'late_game': "Prepare for Elite Four, train team to level 40+",
            'end_game': "Challenge Elite Four and Champion"
        }
        
        return strategies.get(phase, "Explore and progress through the game")
    
    def _get_priority_actions(self, game_state: Dict[str, Any], visual_context=None) -> List[str]:
        """Get prioritized list of recommended actions"""
        phase = self._determine_game_phase(game_state, visual_context)
        is_stuck = self._detect_stuck_state(game_state)
        
        if is_stuck:
            return ["Try B to exit menus", "Try different movement direction", "Open START menu"]
        
        phase_actions = {
            'game_intro': ["Press A to advance", "Watch for name input"],
            'pre_starter': ["Move towards lab", "Talk to Professor Elm", "Choose starter"],
            'early_exploration': ["Catch Pokemon", "Level up starter", "Visit Pokemon Center"],
            'in_battle': ["Use effective moves", "Switch if needed", "Catch if wild"],
            'post_first_gym': ["Train team", "Explore new routes", "Prepare for next gym"]
        }
        
        return phase_actions.get(phase, ["Explore", "Progress story", "Train Pokemon"])
    
    def _update_milestones(self, game_state: Dict[str, Any], visual_context=None):
        """Update progress milestones"""
        party_size = game_state.get('party_size', 0)
        badges = game_state.get('badges', 0)
        
        if party_size >= 1 and not self.progress_milestones['got_starter']:
            self.progress_milestones['got_starter'] = True
            print("🎉 Milestone: Got starter Pokemon!")
        
        if badges >= 1 and not self.progress_milestones['first_gym']:
            self.progress_milestones['first_gym'] = True
            print("🏆 Milestone: Defeated first gym!")
        
        if badges >= 2 and not self.progress_milestones['second_gym']:
            self.progress_milestones['second_gym'] = True
            print("🏆 Milestone: Defeated second gym!")


class UltimateTrainingSession:
    """Ultimate Pokemon Crystal RL training with ROM font system"""
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 llm_interval: int = 5,          # More frequent LLM calls for intelligence
                 visual_interval: int = 10,      # Regular visual analysis
                 web_update_interval: int = 3,   # Frequent web updates
                 enable_web_monitor: bool = True,
                 model_name: str = "llama3.2:3b",
                 target_completion_time: int = 7200):  # 2 hours target
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.llm_interval = llm_interval
        self.visual_interval = visual_interval
        self.web_update_interval = web_update_interval
        self.enable_web_monitor = enable_web_monitor and WEB_MONITOR_AVAILABLE
        self.model_name = model_name
        self.target_completion_time = target_completion_time
        
        print("🎮 Ultimate Pokemon Crystal RL Training System")
        print("=" * 60)
        
        # Initialize core components
        self._initialize_environment()
        self._initialize_agent()
        self._initialize_vision_system()
        self._initialize_monitoring()
        
        # Training state
        self.training_active = False
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "llm_calls": 0,
            "visual_analyses": 0,
            "font_recognitions": 0,
            "actions_per_second": 0,
            "start_time": None,
            "completion_target": target_completion_time,
            "estimated_completion": None
        }
        
        # Intelligent action management
        self.action_sequence = deque()
        self.action_history = deque(maxlen=50)
        self.decision_history = deque(maxlen=20)
        
        # Game state analysis
        self.state_analyzer = GameStateAnalyzer()
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.last_performance_check = time.time()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Ultimate training system ready!")
        print(f"🎯 Target completion time: {target_completion_time/3600:.1f} hours")
    
    def _initialize_environment(self):
        """Initialize the PyBoy environment"""
        print("🎮 Initializing enhanced PyBoy environment...")
        
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=self.rom_path,
            save_state_path=self.save_state_path,
            headless=True,  # Headless for speed
            debug_mode=False
        )
        
        print("✅ Environment initialized")
    
    def _initialize_agent(self):
        """Initialize the enhanced LLM agent"""
        print(f"🤖 Initializing Enhanced LLM Agent ({self.model_name})...")
        
        self.agent = EnhancedLLMPokemonAgent(
            model_name=self.model_name,
            use_vision=True,
            memory_db="outputs/ultimate_training_memory.db"
        )
        
        print("✅ Enhanced agent initialized with ROM font support")
    
    def _initialize_vision_system(self):
        """Initialize the advanced vision system"""
        print("👁️ Initializing ROM-based vision system...")
        
        # Initialize vision processor with ROM fonts
        self.vision_processor = PokemonVisionProcessor()
        
        # Initialize ROM font decoder
        self.font_decoder = ROMFontDecoder()
        
        # Initialize Game Boy Color palette support
        self.gbc_palette = GameBoyColorPalette()
        
        print("✅ Vision system initialized with ROM font extraction")
        print(f"📚 Font decoder has {len(self.font_decoder.font_templates)} character templates")
    
    def _initialize_monitoring(self):
        """Initialize monitoring systems"""
        if self.enable_web_monitor:
            print("🌐 Initializing web monitoring...")
            self.web_monitor = PokemonRLWebMonitor()
            self.web_server_thread = None
        else:
            self.web_monitor = None
            print("📊 Using console monitoring only")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_web_server(self, host='127.0.0.1', port=5000) -> Optional[str]:
        """Start the web monitoring server"""
        if not self.enable_web_monitor:
            print("⚠️ Web monitoring disabled")
            return None
        
        print(f"🚀 Starting web server on {host}:{port}")
        
        # Create templates
        create_dashboard_templates()
        
        # Start web server in separate thread
        def run_web_server():
            self.web_monitor.run(host=host, port=port, debug=False)
        
        self.web_server_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_server_thread.start()
        
        # Give the server a moment to start
        time.sleep(2)
        
        url = f"http://{host}:{port}"
        print(f"📊 Web monitoring dashboard: {url}")
        return url
    
    def intelligent_action_decision(self, step: int, game_state: Dict[str, Any]) -> int:
        """Make intelligent action decisions using all available systems"""
        
        # Get visual context with ROM font analysis
        visual_context = None
        if step % self.visual_interval == 0:
            visual_context = self._get_enhanced_visual_context()
        
        # Analyze game state
        state_analysis = self.state_analyzer.analyze_progress(game_state, visual_context)
        
        # Make decision based on context
        if step % self.llm_interval == 0 or len(self.action_sequence) == 0:
            action = self._get_intelligent_action(game_state, visual_context, state_analysis, step)
        else:
            action = self._get_cached_action(game_state, state_analysis)
        
        # Record decision
        self._record_decision(action, game_state, visual_context, state_analysis, step)
        
        return action
    
    def _get_enhanced_visual_context(self):
        """Get enhanced visual context using ROM font system"""
        screenshot = self.env.get_screenshot()
        if screenshot is None:
            return None
        
        try:
            # Process with enhanced vision system
            visual_context = self.vision_processor.process_screenshot(screenshot)
            
            # Enhance text recognition with ROM fonts
            if visual_context.detected_text:
                for text_region in visual_context.detected_text:
                    # Use ROM font decoder for better accuracy
                    enhanced_text = self.font_decoder.decode_text_region_with_palette(
                        screenshot[text_region.bbox[1]:text_region.bbox[3], 
                                 text_region.bbox[0]:text_region.bbox[2]],
                        visual_context.screen_type
                    )
                    
                    if enhanced_text and len(enhanced_text.strip()) > len(text_region.text.strip()):
                        text_region.text = enhanced_text
                        text_region.confidence = 0.95  # High confidence for ROM fonts
            
            self.training_stats["visual_analyses"] += 1
            self.training_stats["font_recognitions"] += len(visual_context.detected_text)
            
            return visual_context
            
        except Exception as e:
            print(f"⚠️ Visual analysis error: {e}")
            return None
    
    def _get_intelligent_action(self, game_state: Dict[str, Any], visual_context, 
                               state_analysis: Dict[str, Any], step: int) -> int:
        """Get intelligent action using LLM with full context"""
        try:
            start_time = time.time()
            
            # Enhanced context for LLM decision
            enhanced_context = {
                'game_state': game_state,
                'visual_context': {
                    'screen_type': visual_context.screen_type if visual_context else 'unknown',
                    'detected_text': [t.text for t in visual_context.detected_text] if visual_context else [],
                    'ui_elements': [e.element_type for e in visual_context.ui_elements] if visual_context else [],
                    'game_phase': visual_context.game_phase if visual_context else 'unknown'
                },
                'state_analysis': state_analysis,
                'recent_actions': list(self.action_history)[-10:],
                'recent_decisions': list(self.decision_history)[-5:],
                'performance_trend': self._get_performance_trend()
            }
            
            # Get LLM decision - convert action integers to strings
            recent_history_str = [self.agent.action_map.get(act, str(act)) for act in list(self.action_history)[-10:]]
            action = self.agent.decide_next_action(
                state=game_state,
                screenshot=self.env.get_screenshot(),
                recent_history=recent_history_str
            )
            
            # Generate intelligent follow-up actions
            follow_up_actions = self._generate_intelligent_sequence(
                action, game_state, visual_context, state_analysis
            )
            self.action_sequence.extend(follow_up_actions)
            
            llm_time = time.time() - start_time
            self.training_stats["llm_calls"] += 1
            
            if step % 50 == 0:
                print(f"🧠 Intelligent decision: {self.agent.action_map.get(action, action)} "
                      f"(took {llm_time:.2f}s, phase: {state_analysis['current_phase']})")
            
            return action
            
        except Exception as e:
            print(f"⚠️ LLM decision error at step {step}: {e}")
            return self._get_fallback_action(game_state, state_analysis)
    
    def _get_cached_action(self, game_state: Dict[str, Any], state_analysis: Dict[str, Any]) -> int:
        """Get cached action with safety checks"""
        if self.action_sequence:
            return self.action_sequence.popleft()
        
        return self._get_fallback_action(game_state, state_analysis)
    
    def _get_fallback_action(self, game_state: Dict[str, Any], state_analysis: Dict[str, Any]) -> int:
        """Intelligent fallback action selection"""
        if state_analysis['is_stuck']:
            # Anti-stuck actions
            return np.random.choice([6, 7, 2, 3, 4])  # B, START, or movement
        
        # Phase-based fallback
        phase = state_analysis['current_phase']
        if phase in ['game_intro', 'lab_sequence']:
            return 5  # A button for dialogue/intro
        elif phase == 'in_battle':
            return 5  # A button for battle actions
        else:
            # Exploration movement
            return np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2])
    
    def _generate_intelligent_sequence(self, primary_action: int, game_state: Dict[str, Any],
                                     visual_context, state_analysis: Dict[str, Any]) -> List[int]:
        """Generate intelligent follow-up action sequence"""
        sequence = []
        phase = state_analysis['current_phase']
        
        # Phase-specific sequences
        if phase == 'game_intro':
            sequence.extend([5, 5, 5])  # Multiple A presses for intro
        elif phase == 'in_battle':
            sequence.extend([5, 0, 5])  # A, wait, A for battle
        elif phase in ['early_exploration', 'post_first_gym']:
            # Exploration with interaction
            if primary_action in [1, 2, 3, 4]:  # Movement
                sequence.extend([primary_action, 0, 5])  # Move, wait, interact
        
        return sequence[:3]  # Limit sequence length
    
    def _record_decision(self, action: int, game_state: Dict[str, Any], visual_context,
                        state_analysis: Dict[str, Any], step: int):
        """Record decision for analysis and monitoring"""
        
        decision_record = {
            'step': step,
            'action': self.agent.action_map.get(action, action),
            'game_phase': state_analysis['current_phase'],
            'progress_score': state_analysis['progress_score'],
            'is_stuck': state_analysis['is_stuck'],
            'timestamp': time.time()
        }
        
        self.decision_history.append(decision_record)
        self.action_history.append(action)
        
        # Update web monitor
        if self.web_monitor and step % self.web_update_interval == 0:
            self._update_web_monitor(decision_record, game_state, visual_context, state_analysis)
            
            # Stream game screen to web monitor
            screenshot = self.env.get_screenshot()
            if screenshot is not None:
                self.web_monitor.update_screenshot(screenshot)
    
    def _update_web_monitor(self, decision_record: Dict[str, Any], game_state: Dict[str, Any],
                           visual_context, state_analysis: Dict[str, Any]):
        """Update web monitoring dashboard"""
        try:
            # Update current stats for web monitor
            self.web_monitor.current_stats = {
                'player_x': game_state.get('player_x', 0),
                'player_y': game_state.get('player_y', 0),
                'player_map': game_state.get('player_map', 0),
                'party_size': game_state.get('party_size', 0),
                'money': game_state.get('money', 0),
                'badges': game_state.get('badges', 0),
                'game_phase': state_analysis['current_phase'],
                'progress_score': state_analysis['progress_score'],
                'action': decision_record['action'],
                'reasoning': f"Phase: {decision_record['game_phase']}, Progress: {decision_record['progress_score']:.1f}%",
                'confidence': 0.9 if not state_analysis['is_stuck'] else 0.5,
                'screen_type': visual_context.screen_type if visual_context else 'unknown',
                'detected_text_count': len(visual_context.detected_text) if visual_context else 0,
                'training_stats': self.training_stats
            }
            
            # Update action history in web monitor
            action_name = self.agent.action_map.get(decision_record['step'] % 9, 'NONE')
            self.web_monitor.update_action(decision_record['action'], decision_record.get('reasoning', ''))
            
            # Update decision data in web monitor  
            self.web_monitor.update_decision({
                'decision': decision_record['action'],
                'reasoning': decision_record.get('reasoning', ''),
                'confidence': decision_record.get('confidence', 0.8),
                'visual_context': {
                    'screen_type': visual_context.screen_type if visual_context else 'unknown',
                    'detected_text_count': len(visual_context.detected_text) if visual_context else 0
                }
            })
            
            # Emit real-time update via WebSocket
            if hasattr(self.web_monitor, 'socketio'):
                self.web_monitor.socketio.emit('game_update', self.web_monitor.current_stats)
            
        except Exception as e:
            print(f"⚠️ Web monitor update error: {e}")
    
    def _get_performance_trend(self) -> str:
        """Analyze performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent_scores = [p['progress_score'] for p in list(self.performance_history)[-10:]]
        trend = np.mean(np.diff(recent_scores))
        
        if trend > 0.5:
            return "improving"
        elif trend < -0.5:
            return "declining"
        else:
            return "stable"
    
    def _update_performance_tracking(self, step: int, game_state: Dict[str, Any], 
                                   state_analysis: Dict[str, Any]):
        """Update performance tracking metrics"""
        current_time = time.time()
        
        if current_time - self.last_performance_check >= 60:  # Update every minute
            elapsed_time = current_time - self.training_stats["start_time"]
            actions_per_second = step / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate completion time based on progress
            progress_score = state_analysis['progress_score']
            if progress_score > 0:
                estimated_total_time = elapsed_time / (progress_score / 100)
                self.training_stats["estimated_completion"] = estimated_total_time
            
            self.training_stats["actions_per_second"] = actions_per_second
            
            performance_record = {
                'timestamp': current_time,
                'step': step,
                'progress_score': progress_score,
                'actions_per_second': actions_per_second,
                'phase': state_analysis['current_phase']
            }
            
            self.performance_history.append(performance_record)
            self.last_performance_check = current_time
            
            # Progress logging
            if step % 1000 == 0:
                hours_elapsed = elapsed_time / 3600
                estimated_hours = self.training_stats.get("estimated_completion", 0) / 3600
                
                print(f"📊 Step {step:,} | Progress: {progress_score:.1f}% | "
                      f"Speed: {actions_per_second:.1f} act/s | "
                      f"Time: {hours_elapsed:.1f}h / ~{estimated_hours:.1f}h")
    
    def run_training(self, max_steps: int = 100000, save_interval: int = 5000) -> Dict[str, Any]:
        """Run the ultimate training session"""
        
        print(f"\n🚀 Starting Ultimate Pokemon Crystal RL Training")
        print(f"🎯 Target: Complete Pokemon Crystal in {self.target_completion_time/3600:.1f} hours")
        print(f"📊 Max steps: {max_steps:,}")
        print("=" * 60)
        
        # Start web server if enabled
        web_url = None
        if self.enable_web_monitor:
            web_url = self.start_web_server()
            # Start web monitoring
            self.web_monitor.start_monitoring()
        
        # Initialize training
        self.training_active = True
        self.training_stats["start_time"] = time.time()
        
        # Reset environment
        initial_state = self.env.reset()
        step = 0
        episode = 0
        
        try:
            while self.training_active and step < max_steps:
                # Get current game state
                game_state = self.env.get_game_state()
                
                # Make intelligent action decision
                action = self.intelligent_action_decision(step, game_state)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Update training stats
                self.training_stats["total_steps"] = step
                
                # Analyze state and update performance
                visual_context = None
                if step % self.visual_interval == 0:
                    visual_context = self._get_enhanced_visual_context()
                
                state_analysis = self.state_analyzer.analyze_progress(game_state, visual_context)
                self._update_performance_tracking(step, game_state, state_analysis)
                
                # Check for episode completion
                if done:
                    episode += 1
                    self.training_stats["episodes"] = episode
                    
                    completion_score = state_analysis['progress_score']
                    if completion_score >= 90:
                        print(f"🏆 GAME COMPLETED! Score: {completion_score:.1f}% in {step:,} steps!")
                        break
                    
                    print(f"📋 Episode {episode} completed at step {step:,} (Progress: {completion_score:.1f}%)")
                    initial_state = self.env.reset()
                
                # Save periodically
                if step % save_interval == 0 and step > 0:
                    self._save_checkpoint(step, game_state, state_analysis)
                
                step += 1
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user at step {step:,}")
        
        except Exception as e:
            print(f"\n❌ Training error at step {step:,}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final save and cleanup
            self._save_final_results(step, game_state if 'game_state' in locals() else {})
            self.stop()
        
        # Return training results
        elapsed_time = time.time() - self.training_stats["start_time"]
        final_progress = state_analysis['progress_score'] if 'state_analysis' in locals() else 0
        
        results = {
            'total_steps': step,
            'episodes': episode,
            'final_progress': final_progress,
            'elapsed_time': elapsed_time,
            'actions_per_second': step / elapsed_time if elapsed_time > 0 else 0,
            'llm_calls': self.training_stats["llm_calls"],
            'visual_analyses': self.training_stats["visual_analyses"],
            'font_recognitions': self.training_stats["font_recognitions"],
            'web_url': web_url,
            'completion_achieved': final_progress >= 90
        }
        
        print(f"\n🎯 Training Results:")
        print(f"   Steps: {results['total_steps']:,}")
        print(f"   Progress: {results['final_progress']:.1f}%")
        print(f"   Time: {results['elapsed_time']/3600:.2f} hours")
        print(f"   Speed: {results['actions_per_second']:.1f} actions/second")
        print(f"   LLM calls: {results['llm_calls']:,}")
        print(f"   Font recognitions: {results['font_recognitions']:,}")
        
        if results['completion_achieved']:
            print("🏆 POKEMON CRYSTAL COMPLETED SUCCESSFULLY! 🏆")
        
        return results
    
    def _save_checkpoint(self, step: int, game_state: Dict[str, Any], state_analysis: Dict[str, Any]):
        """Save training checkpoint"""
        try:
            checkpoint_data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'game_state': game_state,
                'state_analysis': state_analysis,
                'training_stats': self.training_stats,
                'milestones': self.state_analyzer.progress_milestones
            }
            
            os.makedirs("outputs/checkpoints", exist_ok=True)
            checkpoint_file = f"outputs/checkpoints/checkpoint_{step:06d}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"💾 Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            print(f"⚠️ Checkpoint save error: {e}")
    
    def _save_final_results(self, final_step: int, final_game_state: Dict[str, Any]):
        """Save final training results"""
        try:
            results = {
                'training_session': {
                    'final_step': final_step,
                    'total_time': time.time() - self.training_stats["start_time"],
                    'completion_time': datetime.now().isoformat(),
                    'training_stats': self.training_stats,
                    'performance_history': list(self.performance_history),
                    'milestones_achieved': self.state_analyzer.progress_milestones
                },
                'final_game_state': final_game_state,
                'system_info': {
                    'rom_path': self.rom_path,
                    'model_name': self.model_name,
                    'font_decoder_chars': len(self.font_decoder.font_templates),
                    'vision_enabled': True,
                    'web_monitor_enabled': self.enable_web_monitor
                }
            }
            
            results_file = f"outputs/training_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📊 Final results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Results save error: {e}")
    
    def stop(self):
        """Stop the training session gracefully"""
        self.training_active = False
        
        if hasattr(self, 'env'):
            self.env.close()
        
        print("🛑 Training session stopped")


def main():
    """Main training execution"""
    
    # Configuration
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("💡 Please place your Pokemon Crystal ROM in the current directory")
        return
    
    print("🎮 Pokemon Crystal Ultimate RL Training System")
    print("🚀 Featuring ROM font extraction for perfect text recognition")
    print("=" * 70)
    
    # Initialize training session
    training_session = UltimateTrainingSession(
        rom_path=rom_path,
        save_state_path=None,  # Start from beginning
        llm_interval=5,        # Intelligent decisions every 5 steps
        visual_interval=10,    # Visual analysis every 10 steps
        enable_web_monitor=True,
        model_name="llama3.2:3b",
        target_completion_time=7200  # 2 hours target
    )
    
    try:
        # Run training
        results = training_session.run_training(
            max_steps=200000,    # Increased for full game completion
            save_interval=10000  # Save every 10k steps
        )
        
        # Display final results
        print("\n" + "=" * 70)
        print("🎯 ULTIMATE TRAINING COMPLETED!")
        
        if results['completion_achieved']:
            print("🏆 POKEMON CRYSTAL SUCCESSFULLY COMPLETED! 🏆")
            print(f"⏱️ Completion time: {results['elapsed_time']/3600:.2f} hours")
            print(f"🎮 Total steps: {results['total_steps']:,}")
        else:
            print(f"📊 Final progress: {results['final_progress']:.1f}%")
            print(f"⏱️ Training time: {results['elapsed_time']/3600:.2f} hours")
        
        print(f"🧠 LLM decisions: {results['llm_calls']:,}")
        print(f"👁️ Visual analyses: {results['visual_analyses']:,}")
        print(f"📚 Font recognitions: {results['font_recognitions']:,}")
        print(f"⚡ Average speed: {results['actions_per_second']:.1f} actions/second")
        
        if results.get('web_url'):
            print(f"📊 Web dashboard: {results['web_url']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
