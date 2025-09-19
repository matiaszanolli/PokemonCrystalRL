#!/usr/bin/env python3
"""
Integration test script for the Unified Web Dashboard.

This script provides a quick way to test the new unified dashboard
system without needing a full training session.
"""

import sys
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MockGameState:
    """Mock game state for testing."""
    current_map: int = 4
    player_x: int = 3
    player_y: int = 0
    money: int = 1500
    badges: int = 2
    party_count: int = 3
    level: int = 25


class MockMemoryReader:
    """Mock memory reader for testing."""

    def __init__(self):
        self.game_state = MockGameState()

    def read_game_state(self) -> Dict[str, Any]:
        """Return mock game state data."""
        return {
            'PARTY_COUNT': self.game_state.party_count,
            'PLAYER_MAP': self.game_state.current_map,
            'PLAYER_X': self.game_state.player_x,
            'PLAYER_Y': self.game_state.player_y,
            'MONEY': self.game_state.money,
            'BADGES': self.game_state.badges,
            'IN_BATTLE': 0,
            'PLAYER_LEVEL': self.game_state.level,
            'HP_CURRENT': 45,
            'HP_MAX': 50,
            'FACING_DIRECTION': 1,
            'STEP_COUNTER': 0,
            'TIME_HOURS': 12,
            'COORDS': [self.game_state.player_x, self.game_state.player_y],
            'HP_PERCENTAGE': 90,
            'HAS_POKEMON': True,
            'memory_read_success': True,
            'timestamp': time.time()
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Return mock debug info."""
        return {
            'cache_age_seconds': 0.001,
            'cache_duration': 0.1,
            'pyboy_available': True,
            'memory_addresses_count': 13
        }


class MockPyBoy:
    """Mock PyBoy instance for testing."""

    def __init__(self):
        self.screen_array = None

    def botsupport_manager(self):
        """Return mock botsupport manager."""
        return MockBotSupportManager()


class MockBotSupportManager:
    """Mock botsupport manager."""

    def screen(self):
        """Return mock screen."""
        return MockScreen()


class MockScreen:
    """Mock screen for testing."""

    def screen_ndarray(self):
        """Return mock screen array."""
        import numpy as np
        # Create a simple test pattern
        screen = np.zeros((144, 160, 3), dtype=np.uint8)

        # Create a simple test pattern
        screen[20:40, 20:40] = [255, 0, 0]    # Red square
        screen[60:80, 60:80] = [0, 255, 0]    # Green square
        screen[100:120, 100:120] = [0, 0, 255]  # Blue square

        # Add some text-like pattern
        for i in range(10, 130, 20):
            screen[i:i+5, 10:150] = [255, 255, 255]  # White lines

        return screen


class MockEmulationManager:
    """Mock emulation manager for testing."""

    def __init__(self):
        self.pyboy = MockPyBoy()

    def get_instance(self):
        """Return mock PyBoy instance."""
        return self.pyboy


class MockStatisticsTracker:
    """Mock statistics tracker for testing."""

    def __init__(self):
        self.start_time = time.time()
        self.actions = 0
        self.rewards = []
        self.llm_calls = 0

        # Start updating stats in background
        self.running = True
        self.update_thread = threading.Thread(target=self._update_stats, daemon=True)
        self.update_thread.start()

    def _update_stats(self):
        """Update stats in background to simulate training."""
        while self.running:
            self.actions += 1
            # Simulate some reward variation
            if self.actions % 50 == 0:
                reward = 10.0  # Occasional positive reward
            else:
                reward = -0.2  # Small negative reward

            self.rewards.append(reward)
            if len(self.rewards) > 100:
                self.rewards.pop(0)

            if self.actions % 20 == 0:
                self.llm_calls += 1

            time.sleep(0.5)  # Simulate action interval

    def get_current_stats(self) -> Dict[str, Any]:
        """Return mock training statistics."""
        current_time = time.time()
        duration = current_time - self.start_time
        actions_per_second = self.actions / duration if duration > 0 else 0
        total_reward = sum(self.rewards)

        return {
            'total_actions': self.actions,
            'actions_per_second': actions_per_second,
            'llm_calls': self.llm_calls,
            'total_reward': total_reward,
            'session_duration': duration,
            'success_rate': 0.85,
            'exploration_rate': 0.65,
            'recent_rewards': self.rewards[-20:],
            'current_map': 4,
            'player_position': {'x': 3, 'y': 0},
            'money': 1500,
            'badges': 2,
            'party_count': 3,
            'level': 25
        }

    def stop(self):
        """Stop the background stats updates."""
        self.running = False


class MockTrainer:
    """Mock trainer for testing the unified web dashboard."""

    def __init__(self):
        logger.info("🤖 Creating mock trainer for dashboard testing")

        # Initialize components
        self.emulation_manager = MockEmulationManager()
        self.memory_reader = MockMemoryReader()
        self.statistics_tracker = MockStatisticsTracker()

        # LLM decisions storage
        self.llm_decisions = deque(maxlen=20)

        # Training state
        self.training_active = True

        # Web monitor placeholder
        self.web_monitor = None

        # Start generating mock LLM decisions
        self._start_llm_decisions()

        logger.info("✅ Mock trainer initialized successfully")

    def _start_llm_decisions(self):
        """Generate mock LLM decisions for testing."""
        def generate_decisions():
            decision_count = 0
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B']
            reasonings = [
                'Moving towards the next area to explore',
                'Trying to interact with an NPC',
                'Navigating around an obstacle',
                'Attempting to enter a building',
                'Engaging in battle with wild Pokemon',
                'Opening the menu to check items'
            ]

            while self.training_active:
                decision_count += 1
                action = actions[decision_count % len(actions)]
                reasoning = reasonings[decision_count % len(reasonings)]

                decision_record = {
                    'action': decision_count % 8,  # 0-7 action range
                    'action_name': action,
                    'reasoning': reasoning,
                    'confidence': 0.7 + (decision_count % 3) * 0.1,  # 0.7-0.9
                    'response_time_ms': 50 + (decision_count % 10) * 10,  # 50-140ms
                    'timestamp': time.time(),
                    'game_state': {
                        'map': 4,
                        'position': (3, 0),
                        'badges': 2
                    }
                }

                self.llm_decisions.append(decision_record)
                time.sleep(3)  # New decision every 3 seconds

        thread = threading.Thread(target=generate_decisions, daemon=True)
        thread.start()

    def stop(self):
        """Stop the mock trainer."""
        logger.info("🛑 Stopping mock trainer")
        self.training_active = False
        if self.statistics_tracker:
            self.statistics_tracker.stop()


def test_unified_dashboard():
    """Test the unified web dashboard with mock data."""
    try:
        # Import the unified dashboard
        from web_dashboard import create_web_server

        logger.info("🎮 Starting Unified Web Dashboard Integration Test")

        # Create mock trainer
        trainer = MockTrainer()

        # Create web server
        logger.info("🌐 Creating unified web server")
        web_server = create_web_server(
            trainer=trainer,
            host='localhost',
            http_port=8080,
            ws_port=8081
        )

        # Start web server
        logger.info("🚀 Starting web server")
        web_server.start()

        logger.info("✅ Web server started successfully!")
        logger.info("📊 Dashboard: http://localhost:8080")
        logger.info("📡 WebSocket: ws://localhost:8081")
        logger.info("")
        logger.info("🧪 Integration test running...")
        logger.info("   - Mock training data is being generated")
        logger.info("   - LLM decisions are being simulated")
        logger.info("   - Game state is being updated")
        logger.info("   - Memory debug data is available")
        logger.info("")
        logger.info("🔍 Test the following:")
        logger.info("   1. Open http://localhost:8080 in your browser")
        logger.info("   2. Verify all dashboard sections load with data")
        logger.info("   3. Check real-time updates are working")
        logger.info("   4. Test API endpoints directly:")
        logger.info("      - curl http://localhost:8080/api/dashboard")
        logger.info("      - curl http://localhost:8080/api/memory_debug")
        logger.info("      - curl http://localhost:8080/health")
        logger.info("")
        logger.info("⏹️  Press Ctrl+C to stop the test")

        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping integration test")

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure the web_dashboard package is properly installed")
        return False

    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return False

    finally:
        # Cleanup
        try:
            if 'web_server' in locals():
                web_server.stop()
            if 'trainer' in locals():
                trainer.stop()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    return True


def test_api_endpoints():
    """Test API endpoints independently."""
    try:
        import requests
        import json

        base_url = "http://localhost:8080"
        endpoints = [
            "/health",
            "/api/dashboard",
            "/api/game_state",
            "/api/training_stats",
            "/api/memory_debug",
            "/api/llm_decisions",
            "/api/system_status"
        ]

        logger.info("🔗 Testing API endpoints")

        for endpoint in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ {endpoint}: OK ({len(json.dumps(data))} bytes)")
                else:
                    logger.warning(f"⚠️  {endpoint}: HTTP {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ {endpoint}: {e}")

    except ImportError:
        logger.warning("⚠️  requests library not available, skipping API endpoint tests")
        logger.info("   Install with: pip install requests")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-api":
        test_api_endpoints()
    else:
        success = test_unified_dashboard()
        sys.exit(0 if success else 1)