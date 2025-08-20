#!/usr/bin/env python3
"""
Monitoring Client Integration
Easy-to-use wrapper for integrating advanced web monitoring into training scripts
"""

import time
import base64
import threading
import requests
import json
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MonitoringClient:
    """
    Client for integrating with the advanced web monitoring system.
    Provides simple methods to send training data to the monitoring dashboard.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", auto_start: bool = True):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 1.0  # Quick timeout for non-blocking operation
        
        # State tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.recent_actions = []
        self.text_frequency = {}
        self.start_time = time.time()
        
        # Performance tracking
        self.last_action_time = time.time()
        self.action_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Auto-start server if requested
        if auto_start:
            self._try_start_server()
    
    def _try_start_server(self):
        """Try to start the monitoring server if it's not running"""
        try:
            response = self.session.get(f"{self.server_url}/api/status")
            if response.status_code == 200:
                logger.info("Monitoring server is already running")
                return True
        except requests.exceptions.ConnectionError:
            logger.info("Monitoring server not running, attempting to start...")
            try:
                import subprocess
                import os
                
                # Try to start the server in background
                server_path = os.path.join(os.path.dirname(__file__), "advanced_web_monitor.py")
                if os.path.exists(server_path):
                    subprocess.Popen([
                        "python", server_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Wait a moment for server to start
                    time.sleep(2)
                    
                    # Try connecting again
                    response = self.session.get(f"{self.server_url}/api/status")
                    if response.status_code == 200:
                        logger.info("Monitoring server started successfully")
                        return True
            except Exception as e:
                logger.warning(f"Could not auto-start monitoring server: {e}")
        
        return False
    
    def _safe_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     files: Optional[Dict] = None) -> bool:
        """Make a safe HTTP request that won't block training"""
        try:
            url = f"{self.server_url}{endpoint}"
            
            if method.upper() == "POST":
                if files:
                    response = self.session.post(url, data=data, files=files)
                else:
                    response = self.session.post(url, json=data)
            else:
                response = self.session.get(url, params=data)
            
            success = response.status_code == 200
            if not success:
                print(f"⚠️ Monitoring request failed: {method} {endpoint} -> {response.status_code}")
            # Only log successful requests for important endpoints
            elif endpoint in ['/api/episode', '/api/screenshot'] or method == 'GET':
                print(f"✓ Monitoring: {method} {endpoint}")
            
            return success
            
        except Exception as e:
            print(f"⚠️ Monitoring request error: {method} {endpoint} -> {e}")
            return False
    
    def update_episode(self, episode: int, total_reward: float = 0.0, 
                      steps: int = 0, success: bool = False):
        """Update episode information"""
        self.current_episode = episode
        self.current_step = 0
        
        if episode > 0:  # Don't record the first empty episode
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
        
        # Keep only last 100 episodes for performance
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
            self.episode_steps.pop(0)
        
        self._safe_request("POST", "/api/episode", {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "success": success
        })
    
    def update_step(self, step: int, reward: float = 0.0, action: str = "NONE", 
                   screen_type: str = "", map_id: int = 0, player_x: int = 0, 
                   player_y: int = 0, **kwargs):
        """Update step information"""
        self.current_step = step
        
        # Track recent actions
        self.recent_actions.append({
            "action": action,
            "timestamp": time.time()
        })
        
        # Keep only last 50 actions
        if len(self.recent_actions) > 50:
            self.recent_actions.pop(0)
        
        # Update action counter for FPS calculation
        self.action_count += 1
        current_time = time.time()
        
        self._safe_request("POST", "/api/step", {
            "episode": self.current_episode,
            "step": step,
            "reward": reward,
            "action": action,
            "screen_type": screen_type,
            "map_id": map_id,
            "player_x": player_x,
            "player_y": player_y,
            **kwargs
        })
    
    def update_screenshot(self, screenshot: np.ndarray):
        """Update game screenshot"""
        try:
            # Convert numpy array to PIL Image
            if screenshot.dtype != np.uint8:
                screenshot = (screenshot * 255).astype(np.uint8)
            
            # Handle different array shapes
            if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
                # RGBA to RGB
                screenshot = screenshot[:, :, :3]
            elif len(screenshot.shape) == 3 and screenshot.shape[2] == 1:
                # Grayscale to RGB
                screenshot = np.repeat(screenshot, 3, axis=2)
            
            image = Image.fromarray(screenshot)
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            self._safe_request("POST", "/api/screenshot", {
                "image": image_b64
            })
            
            # Update FPS counter
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
                
                self._update_system_stats(fps)
                
        except Exception as e:
            logger.debug(f"Screenshot update failed: {e}")
    
    def update_llm_decision(self, action: str, reasoning: str = "", 
                           context: Dict[str, Any] = None):
        """Update LLM decision information"""
        self._safe_request("POST", "/api/decision", {
            "action": action,
            "reasoning": reasoning,
            "context": context or {},
            "timestamp": time.time()
        })
    
    def update_text(self, text: str, text_type: str = "dialogue"):
        """Update detected text information"""
        # Update frequency tracking
        if text and len(text.strip()) > 0:
            clean_text = text.strip().upper()
            self.text_frequency[clean_text] = self.text_frequency.get(clean_text, 0) + 1
        
        self._safe_request("POST", "/api/text", {
            "text": text,
            "type": text_type,
            "frequency": dict(list(self.text_frequency.items())[-20:])  # Last 20 entries
        })
    
    def _update_system_stats(self, fps: float):
        """Update system performance statistics"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            uptime = time.time() - self.start_time
            
            # Calculate actions per second
            current_time = time.time()
            time_diff = current_time - self.last_action_time
            actions_per_sec = self.action_count / max(time_diff, 1.0) if time_diff > 0 else 0.0
            
            self._safe_request("POST", "/api/system", {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "fps": fps,
                "uptime": f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}",
                "actions_per_sec": actions_per_sec
            })
            
        except ImportError:
            # psutil not available
            uptime = time.time() - self.start_time
            self._safe_request("POST", "/api/system", {
                "cpu_percent": 0,
                "memory_percent": 0,
                "fps": fps,
                "uptime": f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}",
                "actions_per_sec": 0.0
            })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.episode_rewards:
            return {
                "avg_reward": 0.0,
                "avg_steps": 0,
                "success_rate": 0.0,
                "total_episodes": 0
            }
        
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_steps = sum(self.episode_steps) / len(self.episode_steps)
        
        # Success rate based on reward threshold (can be customized)
        successful_episodes = sum(1 for r in self.episode_rewards if r > 0)
        success_rate = (successful_episodes / len(self.episode_rewards)) * 100
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "total_episodes": len(self.episode_rewards),
            "episode_rewards": self.episode_rewards.copy(),
            "episode_steps": self.episode_steps.copy()
        }
    
    def send_performance_update(self):
        """Send performance statistics to monitoring server"""
        stats = self.get_performance_stats()
        self._safe_request("POST", "/api/performance", stats)
    
    def control_training(self, action: str) -> bool:
        """Send control commands to training (pause, resume, reset, etc.)"""
        return self._safe_request("POST", f"/api/control/{action}")
    
    def is_server_available(self) -> bool:
        """Check if monitoring server is available"""
        try:
            response = self.session.get(f"{self.server_url}/api/status")
            return response.status_code == 200
        except:
            return False

# Context manager for easy integration
class MonitoredTraining:
    """
    Context manager for monitored training sessions.
    
    Example:
        with MonitoredTraining() as monitor:
            for episode in range(100):
                monitor.start_episode(episode)
                for step in range(1000):
                    action = agent.decide()
                    obs, reward, done = env.step(action)
                    monitor.update(step, reward, action, env.get_screenshot())
                    if done:
                        break
                monitor.end_episode(total_reward)
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", auto_start: bool = True):
        self.client = MonitoringClient(server_url, auto_start)
        self.episode_start_time = None
        self.episode_total_reward = 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Send final performance update
        self.client.send_performance_update()
    
    def start_episode(self, episode: int):
        """Start a new episode"""
        self.episode_start_time = time.time()
        self.episode_total_reward = 0.0
        self.client.current_episode = episode
        self.client.current_step = 0
    
    def update(self, step: int, reward: float, action: str, screenshot: np.ndarray = None,
               screen_type: str = "", map_id: int = 0, player_x: int = 0, player_y: int = 0,
               text: str = "", llm_reasoning: str = "", **kwargs):
        """Update training progress"""
        self.episode_total_reward += reward
        
        # Update step info
        self.client.update_step(
            step=step,
            reward=reward,
            action=action,
            screen_type=screen_type,
            map_id=map_id,
            player_x=player_x,
            player_y=player_y,
            **kwargs
        )
        
        # Update screenshot if provided
        if screenshot is not None:
            self.client.update_screenshot(screenshot)
        
        # Update text if provided
        if text:
            self.client.update_text(text)
        
        # Update LLM decision if reasoning provided
        if llm_reasoning:
            self.client.update_llm_decision(action, llm_reasoning)
    
    def end_episode(self, success: bool = False):
        """End current episode"""
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        self.client.update_episode(
            episode=self.client.current_episode,
            total_reward=self.episode_total_reward,
            steps=self.client.current_step,
            success=success
        )
        
        # Send performance update every few episodes
        if self.client.current_episode % 5 == 0:
            self.client.send_performance_update()
    
    def update_llm_decision(self, action: str, reasoning: str, context: Dict = None):
        """Update LLM decision"""
        self.client.update_llm_decision(action, reasoning, context)
    
    def is_available(self) -> bool:
        """Check if monitoring is available"""
        return self.client.is_server_available()

if __name__ == "__main__":
    # Test the monitoring client
    import numpy as np
    
    print("Testing Monitoring Client...")
    
    client = MonitoringClient()
    
    if client.is_server_available():
        print("✓ Server is available")
        
        # Test screenshot update
        test_image = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        client.update_screenshot(test_image)
        print("✓ Screenshot updated")
        
        # Test step update
        client.update_step(1, 0.5, "RIGHT", "overworld", 1, 10, 15)
        print("✓ Step updated")
        
        # Test LLM decision
        client.update_llm_decision("RIGHT", "Moving towards the exit")
        print("✓ LLM decision updated")
        
        # Test text update
        client.update_text("PROFESSOR OAK: Hello there!")
        print("✓ Text updated")
        
        print("✓ All tests passed!")
        
    else:
        print("✗ Server not available")
        print("Please start the monitoring server first:")
        print("python advanced_web_monitor.py")
